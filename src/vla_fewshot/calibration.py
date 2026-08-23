"""Assert frozen pseudo-target hyperparameters match tracked train configs."""

from __future__ import annotations

import os
from pathlib import Path

from vla_fewshot.config import (
    CalibrationConfig,
    SelectedCheckpointConfig,
    TrainConfig,
    load_config,
)
from vla_fewshot.data.pseudo import load_pseudo_target_splits
from vla_fewshot.predictions import require_frozen_predictions

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CALIBRATION = ROOT / "configs" / "calibration.yaml"
DEFAULT_SELECTED = ROOT / "configs" / "selected_seen_checkpoint.yaml"
DEFAULT_PSEUDO = ROOT / "configs" / "splits" / "pseudo_target_splits.json"


def load_calibration(path: str | Path = DEFAULT_CALIBRATION) -> CalibrationConfig:
    loaded = load_config(path)
    if not isinstance(loaded, CalibrationConfig):
        raise TypeError(f"{path} is not a calibration config")
    return loaded


def load_selected_checkpoint(
    path: str | Path = DEFAULT_SELECTED,
) -> SelectedCheckpointConfig:
    loaded = load_config(path)
    if not isinstance(loaded, SelectedCheckpointConfig):
        raise TypeError(f"{path} is not a selected-checkpoint config")
    return loaded


def resolve_selected_checkpoint_path(
    selected: SelectedCheckpointConfig,
    *,
    checkpoint: Path | None = None,
) -> Path:
    """Resolve the frozen origin. Tracked YAML stores a run-relative uri."""

    if checkpoint is not None:
        return Path(checkpoint)
    if selected.uri is None:
        raise ValueError("selected checkpoint uri is missing")
    raw = Path(selected.uri)
    if raw.is_absolute():
        return raw
    runs = os.environ.get("VLA_RUNS_DIR")
    if runs and selected.run_id:
        return Path(runs) / selected.run_id / raw
    return raw


def _close(left: float, right: float) -> bool:
    return abs(left - right) <= 1e-12


def assert_train_matches_calibration(
    train: TrainConfig,
    cal: CalibrationConfig,
) -> None:
    """Fail if a tracked train YAML drifted from the frozen calibration."""

    if train.stage == "seen" and train.method == "expert":
        if not _close(train.optimizer.lr, cal.seen_pretrain.lr):
            raise ValueError("seen expert lr drifted from frozen calibration")
        if train.training.max_steps != cal.seen_max_steps:
            raise ValueError("seen max_steps drifted from frozen calibration")
        if train.training.effective_batch_size != cal.seen_effective_batch_size:
            raise ValueError("seen batch size drifted from frozen calibration")
        return
    if train.stage == "target" and train.method in {
        "baseline",
        "frozen_stats",
        "anchored_l2sp",
    }:
        if not _close(train.optimizer.lr, cal.target_baseline.lr):
            raise ValueError("target baseline lr drifted from frozen calibration")
        if train.training.max_steps != cal.target_max_steps:
            raise ValueError("target max_steps drifted from frozen calibration")
        if train.training.epochs != cal.target_epochs:
            raise ValueError("target epochs drifted from frozen calibration")
        return
    if train.stage in {"seen", "target"} and train.method in {"lora", "replay_lora"}:
        if train.peft is None:
            raise ValueError(f"{train.method} is missing peft")
        if train.peft.r != cal.lora_r or train.peft.lora_alpha != cal.lora_alpha:
            raise ValueError("LoRA rank/alpha drifted from frozen calibration")
        if not _close(train.peft.lora_dropout, cal.lora_dropout):
            raise ValueError("LoRA dropout drifted from frozen calibration")
        if not _close(train.optimizer.lr, cal.lora_lr):
            raise ValueError("LoRA lr drifted from frozen calibration")
        if train.method == "replay_lora":
            if train.replay is None:
                raise ValueError("replay_lora is missing replay")
            if not _close(train.replay.target_fraction, cal.replay_target_fraction):
                raise ValueError("replay target fraction drifted")
            if train.replay.seen_suite != cal.replay_seen_suite:
                raise ValueError("replay suite drifted from frozen calibration")
        return


def assert_frozen_calibration(*, root: Path = ROOT) -> None:
    """Load the committed freeze and confirm train YAMLs still match."""

    cal = load_calibration(root / "configs" / "calibration.yaml")
    require_frozen_predictions(root=root)
    selected = load_selected_checkpoint(root / "configs" / "selected_seen_checkpoint.yaml")
    splits = load_pseudo_target_splits(root / cal.pseudo_target_splits)
    if list(splits.slugs) != list(cal.seen_probe_slugs):
        raise ValueError("seen-probe slugs must equal the frozen pseudo-target set")
    if selected.status != "pending_seen_pretrain":
        if selected.sha256 is None:
            raise ValueError("selected checkpoint status/hash pair is inconsistent")
    train_files = [
        root / "configs" / "train" / "seen_expert.yaml",
        root / "configs" / "train" / "seen_lora.yaml",
        root / "configs" / "train" / "target_baseline.yaml",
        root / "configs" / "train" / "target_lora.yaml",
        root / "configs" / "train" / "target_replay_lora.yaml",
        root / "configs" / "train" / "target_frozen_stats.yaml",
        root / "configs" / "train" / "target_anchored_l2sp.yaml",
    ]
    for path in train_files:
        loaded = load_config(path)
        if not isinstance(loaded, TrainConfig):
            raise TypeError(f"{path} is not a train config")
        assert_train_matches_calibration(loaded, cal)
