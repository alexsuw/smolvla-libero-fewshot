"""Naive target baseline contracts. No LoRA, no replay, no target early stopping."""

from __future__ import annotations

from pathlib import Path

from vla_fewshot.calibration import load_selected_checkpoint, resolve_selected_checkpoint_path
from vla_fewshot.config import TrainConfig
from vla_fewshot.data.expected import DEMO_BUDGETS, TARGET_TASKS
from vla_fewshot.data.splits import TargetSplits
from vla_fewshot.data.subset import nested_ids
from vla_fewshot.training.trainer import TrainError

TARGET_SLUGS = tuple(TARGET_TASKS)
TRAIN_SEEDS = (42, 123)
BASELINE_METHODS = frozenset({"baseline"})


def assert_baseline_train_config(config: TrainConfig) -> None:
    if config.stage != "target" or config.method != "baseline":
        raise TrainError(
            "train_target baseline path only (no seen LoRA, no target LoRA, "
            "no replay). no GPU training was started."
        )
    if config.peft is not None:
        raise TrainError("baseline forbids LoRA/PEFT. no GPU training was started.")
    if config.replay is not None and config.replay.enabled:
        raise TrainError("baseline forbids seen replay. no GPU training was started.")
    if config.dataset.suite != "libero_goal":
        raise TrainError("baseline trains only libero_goal selected episodes")
    if not config.training.sample_with_replacement:
        raise TrainError("baseline requires sample_with_replacement: true")


def require_frozen_seen_origin(*, checkpoint: Path | None = None) -> tuple[Path, str, str | None]:
    """Return (checkpoint dir, sha256, run_id) from the frozen selected YAML."""

    selected = load_selected_checkpoint()
    if selected.status != "frozen" or not selected.sha256 or selected.uri is None:
        raise TrainError(
            "target training waits until configs/selected_seen_checkpoint.yaml "
            "is frozen from seen probes. no GPU training was started."
        )
    origin = resolve_selected_checkpoint_path(selected, checkpoint=checkpoint)
    return origin, selected.sha256, selected.run_id


def episode_ids_for_cell(splits: TargetSplits, *, task_slug: str, n_demos: int) -> list[int]:
    if task_slug not in splits.tasks:
        raise TrainError(f"unknown target task {task_slug!r}")
    return nested_ids(splits.tasks[task_slug].episode_ids_first_25, n_demos)


def cap_optimizer_steps(
    *,
    max_steps: int,
    epochs: int | None,
    n_samples: int,
    effective_batch_size: int,
) -> int:
    """min(max_steps, epochs × ceil(frames / effective batch))."""

    stop = max_steps
    if epochs:
        per_epoch = max(1, (n_samples + effective_batch_size - 1) // effective_batch_size)
        stop = min(stop, epochs * per_epoch)
    if stop < 1:
        raise TrainError("resolved max optimizer steps must be positive")
    return stop


def apply_throughput_batch(config: TrainConfig, batch_size: int) -> TrainConfig:
    """Set physical=effective batch. Epoch cap then uses the new batch."""

    if batch_size < 1:
        raise TrainError("batch_size must be positive")
    return config.model_copy(
        update={
            "training": config.training.model_copy(
                update={
                    "effective_batch_size": batch_size,
                    "physical_batch_size": batch_size,
                    "gradient_accumulation": 1,
                }
            )
        }
    )


def apply_cell_overrides(
    config: TrainConfig,
    *,
    seed: int,
) -> TrainConfig:
    if seed not in TRAIN_SEEDS:
        raise TrainError(f"train seed must be one of {TRAIN_SEEDS}")
    return config.model_copy(
        update={"training": config.training.model_copy(update={"seed": seed})}
    )


def baseline_grid() -> list[tuple[str, int, int]]:
    return [
        (task, n_demos, seed)
        for task in TARGET_SLUGS
        for n_demos in DEMO_BUDGETS
        for seed in TRAIN_SEEDS
    ]


def baseline_command(
    *,
    task: str,
    n_demos: int,
    seed: int,
    config: Path = Path("configs/train/target_baseline.yaml"),
) -> list[str]:
    return [
        "python",
        "scripts/train_target.py",
        "--config",
        str(config),
        "--task",
        task,
        "--n-demos",
        str(n_demos),
        "--seed",
        str(seed),
    ]


def build_target_run_id(
    config: TrainConfig,
    *,
    task_slug: str,
    n_demos: int,
    project_root: Path,
    created_at: str | None = None,
) -> str:
    from vla_fewshot.logging.manifest import git_sha7, utc_timestamp

    stamp = created_at or utc_timestamp()
    return "__".join(
        [
            config.stage,
            config.method,
            task_slug,
            f"n{n_demos:02d}",
            f"s{config.training.seed}",
            stamp,
            f"g{git_sha7(project_root)}",
        ]
    )
