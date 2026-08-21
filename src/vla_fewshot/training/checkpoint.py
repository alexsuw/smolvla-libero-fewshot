"""Atomic checkpoint save/load with COMPLETED.json and fresh-instance verify."""

from __future__ import annotations

import json
import os
import random
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from vla_fewshot.config import TrainConfig
from vla_fewshot.reproducibility import atomic_write_json, atomic_write_text
from vla_fewshot.storage.checksums import (
    decode_floats,
    encode_floats,
    file_checksums,
    sha256_json,
    verify_file_checksums,
)
from vla_fewshot.storage.layout import (
    CHECKPOINT_COMPLETED_NAME,
    CHECKPOINT_CHECKSUMS_NAME,
    CHECKPOINT_FORMAT_VERSION,
    CHECKPOINT_OPTIMIZER_NAME,
    CHECKPOINT_OPTIMIZER_PT_NAME,
    CHECKPOINT_RNG_NAME,
    CHECKPOINT_TRAIN_STATE_NAME,
    CHECKPOINT_WEIGHTS_NAME,
    CHECKPOINT_WEIGHTS_PT_NAME,
    CHECKPOINTS_INDEX_NAME,
    LATEST_POINTER_NAME,
    RESOLVED_CONFIG_NAME,
    TRAINABLE_PARAMETERS_NAME,
    checkpoints_root,
    step_directory,
    step_directory_name,
)
from vla_fewshot.training.optim import ToyAdamW
from vla_fewshot.training.sampler import DeterministicSampler
from vla_fewshot.training.toy import ToyPolicy


class CheckpointError(RuntimeError):
    """Raised when a checkpoint is incomplete or fails verification."""


def _fsync_directory(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def encode_weights(policy: ToyPolicy) -> dict[str, list[str]]:
    return {name: encode_floats(values) for name, values in policy.state_dict().items()}


def decode_weights(payload: dict[str, list[str]]) -> dict[str, list[float]]:
    return {name: decode_floats(values) for name, values in payload.items()}


def capture_rng() -> dict[str, Any]:
    return {
        "python": list(random.getstate()[1]),
        "python_gauss": random.getstate()[2],
        "numpy": None,
        "torch_cpu": None,
        "torch_cuda": [],
    }


def restore_rng(payload: dict[str, Any]) -> None:
    version = random.getstate()[0]
    python_state = tuple(payload["python"])
    gauss = payload.get("python_gauss", 0)
    random.setstate((version, python_state, gauss))


def train_state_payload(
    *,
    global_step: int,
    samples_seen: int,
    accumulation_position: int,
    epoch_fraction: float,
    metrics_cursor: int,
    sampler: DeterministicSampler,
    sample_order: list[int],
) -> dict[str, Any]:
    return {
        "global_step": global_step,
        "samples_seen": samples_seen,
        "accumulation_position": accumulation_position,
        "epoch_fraction": epoch_fraction,
        "metrics_cursor": metrics_cursor,
        "sampler": sampler.state_dict(),
        "sample_order": sample_order,
        "resolved_precision": "fp32",
        "amp_grad_scaler": None,
        "checkpoint_format_version": CHECKPOINT_FORMAT_VERSION,
    }


def load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise CheckpointError(f"{path} is not a JSON object")
    return payload


def is_complete_checkpoint(directory: Path) -> bool:
    if not directory.is_dir():
        return False
    common = (
        CHECKPOINT_COMPLETED_NAME,
        CHECKPOINT_CHECKSUMS_NAME,
        CHECKPOINT_RNG_NAME,
        CHECKPOINT_TRAIN_STATE_NAME,
    )
    if not all((directory / name).is_file() for name in common):
        return False
    if (directory / CHECKPOINT_WEIGHTS_PT_NAME).is_file():
        return (directory / CHECKPOINT_OPTIMIZER_PT_NAME).is_file()
    toy = (
        CHECKPOINT_WEIGHTS_NAME,
        CHECKPOINT_OPTIMIZER_NAME,
    )
    return all((directory / name).is_file() for name in toy)


def verify_checkpoint_dir(directory: Path) -> dict[str, Any]:
    if (directory / CHECKPOINT_WEIGHTS_PT_NAME).is_file():
        from vla_fewshot.training.torch_checkpoint import verify_torch_checkpoint_dir

        return verify_torch_checkpoint_dir(directory)
    if not (directory / CHECKPOINT_COMPLETED_NAME).is_file():
        raise CheckpointError(f"incomplete checkpoint (missing COMPLETED.json): {directory}")
    checksums = load_json(directory / CHECKPOINT_CHECKSUMS_NAME)
    files = checksums.get("files")
    if not isinstance(files, dict):
        raise CheckpointError("checksums.json missing files map")
    verify_file_checksums(directory, {str(key): str(value) for key, value in files.items()})
    fresh = ToyPolicy(seed=0)
    weights = decode_weights(load_json(directory / CHECKPOINT_WEIGHTS_NAME))
    fresh.load_state_dict(weights)
    encoded = encode_weights(fresh)
    expected = sha256_json(load_json(directory / CHECKPOINT_WEIGHTS_NAME))
    observed = sha256_json(encoded)
    if expected != observed:
        raise CheckpointError("fresh-instance weight checksum mismatch")
    completed = load_json(directory / CHECKPOINT_COMPLETED_NAME)
    return {
        "directory": str(directory),
        "complete": True,
        "step": completed.get("global_step"),
        "weights_sha256": expected,
        "fresh_load_verified": True,
        "checkpoint_format_version": completed.get("checkpoint_format_version"),
    }


def _write_checkpoint_payload(
    directory: Path,
    *,
    config: TrainConfig,
    policy: ToyPolicy,
    optimizer: ToyAdamW,
    sampler: DeterministicSampler,
    train_state: dict[str, Any],
    trainable_names: list[str],
) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    atomic_write_json(directory / CHECKPOINT_WEIGHTS_NAME, encode_weights(policy))
    atomic_write_json(directory / CHECKPOINT_OPTIMIZER_NAME, optimizer.state_dict())
    atomic_write_json(directory / CHECKPOINT_RNG_NAME, capture_rng())
    atomic_write_json(directory / CHECKPOINT_TRAIN_STATE_NAME, train_state)
    atomic_write_json(
        directory / RESOLVED_CONFIG_NAME,
        config.model_dump(mode="json"),
        overwrite=True,
    )
    atomic_write_text(
        directory / TRAINABLE_PARAMETERS_NAME,
        "\n".join(trainable_names) + ("\n" if trainable_names else ""),
        overwrite=True,
    )
    hashed = file_checksums(
        directory,
        exclude=(CHECKPOINT_CHECKSUMS_NAME, CHECKPOINT_COMPLETED_NAME),
    )
    atomic_write_json(
        directory / CHECKPOINT_CHECKSUMS_NAME,
        {"schema_version": 1, "files": hashed},
    )
    verify_checkpoint_payload(directory, policy_seed=config.training.seed)
    atomic_write_json(
        directory / CHECKPOINT_COMPLETED_NAME,
        {
            "schema_version": 1,
            "created_at_utc": datetime.now(UTC).isoformat(),
            "global_step": train_state["global_step"],
            "checkpoint_format_version": CHECKPOINT_FORMAT_VERSION,
            "weights_sha256": sha256_json(encode_weights(policy)),
        },
    )
    _fsync_directory(directory)


def verify_checkpoint_payload(directory: Path, *, policy_seed: int) -> None:
    checksums = load_json(directory / CHECKPOINT_CHECKSUMS_NAME)
    files = {str(key): str(value) for key, value in checksums["files"].items()}
    verify_file_checksums(directory, files)
    fresh = ToyPolicy(seed=policy_seed)
    fresh.load_state_dict(decode_weights(load_json(directory / CHECKPOINT_WEIGHTS_NAME)))
    if encode_weights(fresh) != load_json(directory / CHECKPOINT_WEIGHTS_NAME):
        raise CheckpointError("fresh-instance load did not restore exact weights")


def save_checkpoint(
    run_dir: Path,
    *,
    step: int,
    config: TrainConfig,
    policy: ToyPolicy,
    optimizer: ToyAdamW,
    sampler: DeterministicSampler,
    train_state: dict[str, Any],
    trainable_names: list[str],
) -> Path:
    final_dir = step_directory(run_dir, step)
    if final_dir.exists():
        raise FileExistsError(f"refusing to overwrite checkpoint {final_dir}")
    tmp_dir = checkpoints_root(run_dir) / f"{step_directory_name(step)}.tmp-{uuid.uuid4().hex}"
    checkpoints_root(run_dir).mkdir(parents=True, exist_ok=True)
    try:
        _write_checkpoint_payload(
            tmp_dir,
            config=config,
            policy=policy,
            optimizer=optimizer,
            sampler=sampler,
            train_state=train_state,
            trainable_names=trainable_names,
        )
        os.replace(tmp_dir, final_dir)
        _fsync_directory(checkpoints_root(run_dir))
    except Exception:
        # Leave tmp dir for forensics if rename failed after COMPLETED.
        if tmp_dir.exists() and not final_dir.exists():
            pass
        raise
    pointer = {
        "step": step,
        "directory": step_directory_name(step),
        "path": str(final_dir),
        "updated_at_utc": datetime.now(UTC).isoformat(),
    }
    atomic_write_json(checkpoints_root(run_dir) / LATEST_POINTER_NAME, pointer, overwrite=True)
    index_path = run_dir / CHECKPOINTS_INDEX_NAME
    index: dict[str, Any]
    if index_path.exists():
        index = load_json(index_path)
    else:
        index = {"schema_version": 1, "checkpoints": []}
    index["checkpoints"].append(
        {
            "step": step,
            "directory": str(final_dir),
            "weights_sha256": sha256_json(encode_weights(policy)),
        }
    )
    atomic_write_json(index_path, index, overwrite=True)
    return final_dir


def load_checkpoint(
    directory: Path,
    *,
    policy: ToyPolicy,
    optimizer: ToyAdamW,
    sampler: DeterministicSampler,
) -> dict[str, Any]:
    report = verify_checkpoint_dir(directory)
    policy.load_state_dict(decode_weights(load_json(directory / CHECKPOINT_WEIGHTS_NAME)))
    optimizer.load_state_dict(load_json(directory / CHECKPOINT_OPTIMIZER_NAME))
    restore_rng(load_json(directory / CHECKPOINT_RNG_NAME))
    train_state = load_json(directory / CHECKPOINT_TRAIN_STATE_NAME)
    sampler.load_state_dict(train_state["sampler"])
    report["train_state"] = train_state
    return report
