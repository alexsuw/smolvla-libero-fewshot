"""2×2 weights × stats control. Does not retrain or rerun the 900 sweep."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from vla_fewshot.evaluation.protocol import FINAL_SEED_VALUES
from vla_fewshot.evaluation.seen_retention import cell_name

CONTROL_SEEDS = FINAL_SEED_VALUES[:5]
CONTROL_ADAPTED = (
    ("drawer_middle", 1, 42),
    ("drawer_middle", 25, 42),
    ("wine_cabinet", 1, 123),
    ("wine_cabinet", 25, 123),
)
WeightsKind = Literal["frozen_seen", "target_adapted"]
StatsKind = Literal["libero_90", "target_overlay"]


def control_jobs() -> list[dict[str, object]]:
    """Four adapted+libero_90 cells and four paired frozen+overlay cells."""

    jobs: list[dict[str, object]] = []
    for task, n_demos, seed in CONTROL_ADAPTED:
        name = cell_name(task, n_demos, seed)
        jobs.append(
            {
                "name": f"adapted_libero90__{name}",
                "weights": "target_adapted",
                "stats": "libero_90",
                "task": task,
                "n_demos": n_demos,
                "seed": seed,
            }
        )
        jobs.append(
            {
                "name": f"frozen_overlay__{name}",
                "weights": "frozen_seen",
                "stats": "target_overlay",
                "task": task,
                "n_demos": n_demos,
                "seed": seed,
            }
        )
    return jobs


def control_command(
    *,
    weights: WeightsKind,
    stats: StatsKind,
    task: str,
    n_demos: int,
    seed: int,
    run_dir: Path,
    output_dir: Path,
) -> list[str]:
    return [
        "python",
        "scripts/eval_retention_control.py",
        "--weights",
        weights,
        "--stats",
        stats,
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
