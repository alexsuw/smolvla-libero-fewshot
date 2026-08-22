"""Torch weight checkpoints. JSON toy checkpoints stay on the static path."""

from __future__ import annotations

import os
import shutil
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from vla_fewshot.config import TrainConfig
from vla_fewshot.reproducibility import atomic_write_json, atomic_write_text
from vla_fewshot.model.peft import maybe_save_adapter_sidecar
from vla_fewshot.storage.checksums import file_checksums, sha256_file, verify_file_checksums
from vla_fewshot.storage.layout import (
    CHECKPOINT_CHECKSUMS_NAME,
    CHECKPOINT_COMPLETED_NAME,
    CHECKPOINT_FORMAT_VERSION,
    CHECKPOINT_OPTIMIZER_PT_NAME,
    CHECKPOINT_RNG_NAME,
    CHECKPOINT_RNG_PT_NAME,
    CHECKPOINT_SCALER_PT_NAME,
    CHECKPOINT_TRAIN_STATE_NAME,
    CHECKPOINT_WEIGHTS_PT_NAME,
    CHECKPOINTS_INDEX_NAME,
    LATEST_POINTER_NAME,
    NORMALIZATION_STATS_NAME,
    RESOLVED_CONFIG_NAME,
    TRAINABLE_PARAMETERS_NAME,
    checkpoints_root,
    step_directory,
    step_directory_name,
)
from vla_fewshot.training.checkpoint import (
    CheckpointError,
    _fsync_directory,
    capture_rng,
    load_json,
)


def is_torch_checkpoint(directory: Path) -> bool:
    return (directory / CHECKPOINT_WEIGHTS_PT_NAME).is_file()


def _atomic_torch_save(path: Path, payload: Any) -> None:
    import torch

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{uuid.uuid4().hex}")
    try:
        torch.save(payload, temporary)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def verify_torch_checkpoint_dir(directory: Path) -> dict[str, Any]:
    if not (directory / CHECKPOINT_COMPLETED_NAME).is_file():
        raise CheckpointError(f"incomplete checkpoint (missing COMPLETED.json): {directory}")
    if not (directory / CHECKPOINT_WEIGHTS_PT_NAME).is_file():
        raise CheckpointError(f"missing {CHECKPOINT_WEIGHTS_PT_NAME}: {directory}")
    checksums = load_json(directory / CHECKPOINT_CHECKSUMS_NAME)
    files = checksums.get("files")
    if not isinstance(files, dict):
        raise CheckpointError("checksums.json missing files map")
    verify_file_checksums(directory, {str(key): str(value) for key, value in files.items()})
    completed = load_json(directory / CHECKPOINT_COMPLETED_NAME)
    return {
        "directory": str(directory),
        "complete": True,
        "format": "torch",
        "step": completed.get("global_step"),
        "weights_sha256": sha256_file(directory / CHECKPOINT_WEIGHTS_PT_NAME),
        "fresh_load_verified": False,
        "checkpoint_format_version": completed.get("checkpoint_format_version"),
    }


def save_torch_checkpoint(
    run_dir: Path,
    *,
    step: int,
    config: TrainConfig,
    policy: Any,
    optimizer: Any,
    train_state: dict[str, Any],
    trainable_names: list[str],
    scaler: Any | None = None,
) -> Path:
    import torch

    final_dir = step_directory(run_dir, step)
    if final_dir.exists():
        raise FileExistsError(f"refusing to overwrite checkpoint {final_dir}")
    tmp_dir = checkpoints_root(run_dir) / f"{step_directory_name(step)}.tmp-{uuid.uuid4().hex}"
    checkpoints_root(run_dir).mkdir(parents=True, exist_ok=True)
    tmp_dir.mkdir(parents=True, exist_ok=False)
    try:
        _atomic_torch_save(tmp_dir / CHECKPOINT_WEIGHTS_PT_NAME, policy.state_dict())
        maybe_save_adapter_sidecar(tmp_dir, policy=policy, peft=config.peft)
        _atomic_torch_save(tmp_dir / CHECKPOINT_OPTIMIZER_PT_NAME, optimizer.state_dict())
        rng_payload: dict[str, Any] = {
            "torch_cpu": torch.get_rng_state(),
            "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        }
        _atomic_torch_save(tmp_dir / CHECKPOINT_RNG_PT_NAME, rng_payload)
        if scaler is not None:
            _atomic_torch_save(tmp_dir / CHECKPOINT_SCALER_PT_NAME, scaler.state_dict())
        atomic_write_json(tmp_dir / CHECKPOINT_RNG_NAME, capture_rng())
        atomic_write_json(tmp_dir / CHECKPOINT_TRAIN_STATE_NAME, train_state)
        atomic_write_json(
            tmp_dir / RESOLVED_CONFIG_NAME,
            config.model_dump(mode="json"),
            overwrite=True,
        )
        atomic_write_text(
            tmp_dir / TRAINABLE_PARAMETERS_NAME,
            "\n".join(trainable_names) + ("\n" if trainable_names else ""),
            overwrite=True,
        )
        sidecar = run_dir / NORMALIZATION_STATS_NAME
        if sidecar.is_file():
            shutil.copy2(sidecar, tmp_dir / NORMALIZATION_STATS_NAME)
        hashed = file_checksums(
            tmp_dir,
            exclude=(CHECKPOINT_CHECKSUMS_NAME, CHECKPOINT_COMPLETED_NAME),
        )
        atomic_write_json(
            tmp_dir / CHECKPOINT_CHECKSUMS_NAME,
            {"schema_version": 1, "files": hashed},
        )
        verify_file_checksums(tmp_dir, hashed)
        completed = {
            "schema_version": 1,
            "created_at_utc": datetime.now(UTC).isoformat(),
            "global_step": train_state["global_step"],
            "checkpoint_format_version": CHECKPOINT_FORMAT_VERSION,
            "format": "torch",
            "weights_sha256": sha256_file(tmp_dir / CHECKPOINT_WEIGHTS_PT_NAME),
            "peft_merged": False if config.peft is not None else None,
        }
        atomic_write_json(
            tmp_dir / CHECKPOINT_COMPLETED_NAME,
            {key: value for key, value in completed.items() if value is not None},
        )
        _fsync_directory(tmp_dir)
        os.replace(tmp_dir, final_dir)
        _fsync_directory(checkpoints_root(run_dir))
    except Exception:
        raise

    pointer = {
        "step": step,
        "directory": step_directory_name(step),
        "path": str(final_dir),
        "updated_at_utc": datetime.now(UTC).isoformat(),
        "format": "torch",
    }
    atomic_write_json(checkpoints_root(run_dir) / LATEST_POINTER_NAME, pointer, overwrite=True)
    index_path = run_dir / CHECKPOINTS_INDEX_NAME
    if index_path.exists():
        index = load_json(index_path)
    else:
        index = {"schema_version": 1, "checkpoints": []}
    index["checkpoints"].append(
        {
            "step": step,
            "directory": str(final_dir),
            "format": "torch",
            "weights_sha256": sha256_file(final_dir / CHECKPOINT_WEIGHTS_PT_NAME),
        }
    )
    atomic_write_json(index_path, index, overwrite=True)
    return final_dir


def load_policy_weights(
    directory: Path,
    *,
    policy: Any,
    expected_sha256: str | None = None,
) -> dict[str, Any]:
    """Load weights.pt only. Fresh optimizer is created by the caller."""

    import torch

    report = verify_torch_checkpoint_dir(directory)
    if expected_sha256 is not None and report["weights_sha256"] != expected_sha256:
        raise CheckpointError(
            f"origin checkpoint hash {report['weights_sha256']} != {expected_sha256}"
        )
    map_location = next(policy.parameters()).device
    weights = torch.load(
        directory / CHECKPOINT_WEIGHTS_PT_NAME,
        map_location=map_location,
        weights_only=True,
    )
    policy.load_state_dict(weights)
    report["fresh_load_verified"] = True
    return report


def load_torch_checkpoint(
    directory: Path,
    *,
    policy: Any,
    optimizer: Any,
    scaler: Any | None = None,
) -> dict[str, Any]:
    import torch

    report = verify_torch_checkpoint_dir(directory)
    map_location = next(policy.parameters()).device
    weights = torch.load(directory / CHECKPOINT_WEIGHTS_PT_NAME, map_location=map_location, weights_only=True)
    policy.load_state_dict(weights)
    opt_state = torch.load(
        directory / CHECKPOINT_OPTIMIZER_PT_NAME, map_location=map_location, weights_only=False
    )
    optimizer.load_state_dict(opt_state)
    rng_path = directory / CHECKPOINT_RNG_PT_NAME
    if rng_path.is_file():
        rng_payload = torch.load(rng_path, map_location="cpu", weights_only=False)
        torch.set_rng_state(rng_payload["torch_cpu"])
        if torch.cuda.is_available() and rng_payload.get("torch_cuda"):
            torch.cuda.set_rng_state_all(rng_payload["torch_cuda"])
    if scaler is not None and (directory / CHECKPOINT_SCALER_PT_NAME).is_file():
        scaler.load_state_dict(
            torch.load(directory / CHECKPOINT_SCALER_PT_NAME, map_location="cpu", weights_only=False)
        )
    report["train_state"] = load_json(directory / CHECKPOINT_TRAIN_STATE_NAME)
    report["fresh_load_verified"] = True
    return report
