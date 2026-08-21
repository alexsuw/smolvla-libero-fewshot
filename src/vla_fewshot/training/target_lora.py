"""Target LoRA ablation contracts. Same origin/episodes as baseline; no replay."""

from __future__ import annotations

from pathlib import Path

from vla_fewshot.calibration import assert_train_matches_calibration, load_calibration
from vla_fewshot.config import TrainConfig
from vla_fewshot.training.baseline import (
    TARGET_SLUGS,
    TRAIN_SEEDS,
    apply_cell_overrides,
    baseline_command,
    baseline_grid,
    build_target_run_id,
    cap_optimizer_steps,
    episode_ids_for_cell,
    require_frozen_seen_origin,
)
from vla_fewshot.training.trainer import TrainError

__all__ = [
    "TARGET_SLUGS",
    "TRAIN_SEEDS",
    "apply_cell_overrides",
    "assert_target_lora_train_config",
    "assert_target_train_config",
    "baseline_grid",
    "build_target_run_id",
    "cap_optimizer_steps",
    "episode_ids_for_cell",
    "lora_command",
    "require_frozen_seen_origin",
    "target_train_command",
]


def assert_target_lora_train_config(config: TrainConfig) -> None:
    if config.stage != "target" or config.method != "lora":
        raise TrainError(
            "train_target LoRA path requires stage=target method=lora. "
            "no GPU training was started."
        )
    if config.peft is None:
        raise TrainError("target LoRA requires peft. no GPU training was started.")
    if config.replay is not None and config.replay.enabled:
        raise TrainError(
            "target LoRA ablation forbids seen replay. no GPU training was started."
        )
    if config.dataset.suite != "libero_goal":
        raise TrainError("target LoRA trains only libero_goal selected episodes")
    if not config.training.sample_with_replacement:
        raise TrainError("target LoRA requires sample_with_replacement: true")
    if config.trainable_scope.train_action_expert:
        raise TrainError(
            "target LoRA keeps Action Expert frozen except adapters. "
            "no GPU training was started."
        )
    if not config.trainable_scope.freeze_vlm_backbone or not config.trainable_scope.freeze_vision_encoder:
        raise TrainError("target LoRA keeps VLM/vision frozen. no GPU training was started.")
    assert_train_matches_calibration(config, load_calibration())


def assert_target_train_config(config: TrainConfig) -> None:
    if config.method == "baseline":
        from vla_fewshot.training.baseline import assert_baseline_train_config

        assert_baseline_train_config(config)
        return
    if config.method == "lora":
        assert_target_lora_train_config(config)
        return
    if config.method == "replay_lora":
        from vla_fewshot.training.replay_mixer import assert_replay_lora_train_config

        assert_replay_lora_train_config(config)
        return
    raise TrainError(
        "unknown target method. no GPU training was started."
    )


def lora_command(
    *,
    task: str,
    n_demos: int,
    seed: int,
    config: Path = Path("configs/train/target_lora.yaml"),
) -> list[str]:
    return baseline_command(task=task, n_demos=n_demos, seed=seed, config=config)


def target_train_command(
    *,
    task: str,
    n_demos: int,
    seed: int,
    config: Path,
) -> list[str]:
    return baseline_command(task=task, n_demos=n_demos, seed=seed, config=config)
