"""Canonical run-directory filenames. Paths come from the caller, never hosts."""

from __future__ import annotations

from pathlib import Path

MANIFEST_NAME = "manifest.json"
RESOLVED_CONFIG_NAME = "config.resolved.yaml"
ENVIRONMENT_MANIFEST_NAME = "environment_manifest.json"
TRAINABLE_PARAMETERS_NAME = "trainable_parameters.txt"
TRAIN_LOG_NAME = "train.log"
METRICS_CSV_NAME = "metrics.csv"
EVENTS_JSONL_NAME = "events.jsonl"
TENSORBOARD_DIRNAME = "tensorboard"
CHECKPOINTS_INDEX_NAME = "checkpoints.json"
CHECKPOINTS_DIRNAME = "checkpoints"
LATEST_POINTER_NAME = "latest.json"
BACKUP_STATUS_NAME = "backup_status.json"
RUN_LOCK_NAME = "run.lock"

CHECKPOINT_COMPLETED_NAME = "COMPLETED.json"
CHECKPOINT_CHECKSUMS_NAME = "checksums.json"
CHECKPOINT_WEIGHTS_NAME = "weights.json"
CHECKPOINT_WEIGHTS_PT_NAME = "weights.pt"
CHECKPOINT_OPTIMIZER_NAME = "optimizer.json"
CHECKPOINT_OPTIMIZER_PT_NAME = "optimizer.pt"
CHECKPOINT_SCALER_PT_NAME = "scaler.pt"
CHECKPOINT_RNG_NAME = "rng.json"
CHECKPOINT_RNG_PT_NAME = "rng.pt"
CHECKPOINT_TRAIN_STATE_NAME = "train_state.json"
CHECKPOINT_ADAPTER_DIRNAME = "adapter"
CHECKPOINT_ADAPTER_CONFIG_NAME = "adapter_config.json"
CHECKPOINT_ADAPTER_WEIGHTS_NAME = "adapter_model.pt"
CHECKPOINT_FORMAT_VERSION = 1


def run_lock_path(run_dir: Path) -> Path:
    return run_dir / RUN_LOCK_NAME


def checkpoints_root(run_dir: Path) -> Path:
    return run_dir / CHECKPOINTS_DIRNAME


def step_directory_name(step: int) -> str:
    if step < 0:
        raise ValueError("checkpoint step must be non-negative")
    return f"step_{step:06d}"


def step_directory(run_dir: Path, step: int) -> Path:
    return checkpoints_root(run_dir) / step_directory_name(step)
