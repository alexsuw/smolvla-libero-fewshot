"""Seen-retention eval of naive target-adapted finals. No retraining."""

from __future__ import annotations

from pathlib import Path

from vla_fewshot.calibration import load_calibration
from vla_fewshot.data.expected import PREFIX_BUDGETS
from vla_fewshot.evaluation.protocol import FINAL_SEED_VALUES
from vla_fewshot.evaluation.select import list_complete_checkpoints
from vla_fewshot.storage.layout import step_directory_name
from vla_fewshot.training.baseline import TARGET_SLUGS, TRAIN_SEEDS
from vla_fewshot.training.trainer import TrainError

RETENTION_BUDGETS = PREFIX_BUDGETS
PROBE_SEED_COUNT = 10
PROBE_SEEDS = FINAL_SEED_VALUES[:PROBE_SEED_COUNT]
FROZEN_SEEN_SHA256 = "2cd510a594a87580f7368b782ca9b37332c0e5002d807093c759e95fbfb57c88"
FROZEN_SEEN_SUCCESSES = 24
FROZEN_SEEN_ROLLOUTS = 30
FROZEN_SEEN_RATE = FROZEN_SEEN_SUCCESSES / FROZEN_SEEN_ROLLOUTS


def seen_probe_slugs() -> tuple[str, ...]:
    return tuple(load_calibration().seen_probe_slugs)


def retention_grid(
    budgets: tuple[int, ...] = RETENTION_BUDGETS,
) -> list[tuple[str, int, int]]:
    unknown = [item for item in budgets if item not in PREFIX_BUDGETS]
    if unknown:
        raise TrainError(f"unsupported retention budgets {unknown}")
    return [
        (task, n_demos, seed)
        for task in TARGET_SLUGS
        for n_demos in budgets
        for seed in TRAIN_SEEDS
    ]


def cell_name(task: str, n_demos: int, seed: int) -> str:
    return f"{task}_n{n_demos:02d}_s{seed}"


def adapted_run_dir(
    *,
    task: str,
    n_demos: int,
    seed: int,
    official_runs: Path,
    n12_runs: Path,
) -> Path:
    root = n12_runs if n_demos in (1, 2) else official_runs
    return Path(root) / cell_name(task, n_demos, seed)


def require_final_checkpoint(run_dir: Path) -> tuple[int, Path]:
    listed = list_complete_checkpoints(run_dir)
    if not listed:
        raise TrainError(f"no complete checkpoints under {run_dir}")
    return listed[-1]


def retention_cell_dir(
    eval_root: Path,
    *,
    task: str,
    n_demos: int,
    seed: int,
    step: int,
    probe: str,
) -> Path:
    return (
        Path(eval_root)
        / cell_name(task, n_demos, seed)
        / step_directory_name(step)
        / probe
    )


def retention_command(
    *,
    task: str,
    n_demos: int,
    seed: int,
    run_dir: Path,
    output_dir: Path,
) -> list[str]:
    return [
        "python",
        "scripts/eval_seen_retention.py",
        "--task",
        task,
        "--n-demos",
        str(n_demos),
        "--seed",
        str(seed),
        "--run-dir",
        str(run_dir),
        "--output-dir",
        str(output_dir),
        "--skip-videos",
        "--skip-traces",
    ]
