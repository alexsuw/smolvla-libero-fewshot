"""Fail-closed resume compatibility checks."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from vla_fewshot.config import TrainConfig
from vla_fewshot.training.checkpoint import load_json
from vla_fewshot.storage.layout import RESOLVED_CONFIG_NAME

RESUME_OVERRIDE_ALLOWLIST = frozenset(
    {"log_freq", "destination", "stop_after", "backup_dir", "output_dir"}
)


class ResumeError(ValueError):
    """Raised when a resume would change a frozen training contract."""


def frozen_training_contract(config: TrainConfig) -> dict[str, Any]:
    return {
        "dataset_repo_id": config.dataset.repo_id,
        "dataset_revision": config.dataset.revision,
        "dataset_suite": config.dataset.suite,
        "trainable_scope": config.trainable_scope.model_dump(mode="json"),
        "optimizer": config.optimizer.model_dump(mode="json"),
        "scheduler": config.scheduler.model_dump(mode="json"),
        "effective_batch_size": config.training.effective_batch_size,
        "physical_batch_size": config.training.physical_batch_size,
        "gradient_accumulation": config.training.gradient_accumulation,
        "num_workers": config.training.num_workers,
        "seed": config.training.seed,
        "max_steps": config.training.max_steps,
        "model_repo_id": config.model.repo_id,
        "model_revision": config.model.revision,
    }


def assert_resume_compatible(checkpoint_dir: Path, config: TrainConfig) -> TrainConfig:
    saved_raw = load_json(checkpoint_dir / RESOLVED_CONFIG_NAME)
    saved = TrainConfig.model_validate(saved_raw)
    current = frozen_training_contract(config)
    expected = frozen_training_contract(saved)
    # YAML still has auto_fit/auto after the first run froze integers. Re-fitting
    # on resume can pick a different batch if GPU memory changed; use the saved
    # contract instead of treating auto_fit as a contract mismatch.
    if current["physical_batch_size"] == "auto_fit":
        current["physical_batch_size"] = expected["physical_batch_size"]
    if current["gradient_accumulation"] == "auto":
        current["gradient_accumulation"] = expected["gradient_accumulation"]
    if current != expected:
        raise ResumeError(
            "resume forbids changing dataset revision, split, trainable scope, "
            f"optimizer, scheduler, batch, workers, or seed: {current!r} != {expected!r}"
        )
    return saved


def assert_override_allowlist(overrides: Mapping[str, Any]) -> None:
    unknown = [key for key, value in overrides.items() if value is not None and key not in RESUME_OVERRIDE_ALLOWLIST]
    if unknown:
        raise ResumeError(
            "resume overrides not on the allowlist "
            f"{sorted(RESUME_OVERRIDE_ALLOWLIST)}: {unknown}"
        )
