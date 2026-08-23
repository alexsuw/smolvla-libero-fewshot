from pathlib import Path

from vla_fewshot.config import load_config
from vla_fewshot.evaluation.normalization import uses_suite_stats_only
from vla_fewshot.evaluation.protocol import FINAL_SEED_VALUES, seeds_for_config
from vla_fewshot.evaluation.runner import plan_eval_rollouts
from vla_fewshot.evaluation.seen_retention import (
    FROZEN_SEEN_RATE,
    FROZEN_SEEN_SHA256,
    PROBE_SEEDS,
    adapted_run_dir,
    cell_name,
    retention_command,
    retention_grid,
    seen_probe_slugs,
)
from vla_fewshot.training.baseline import episode_ids_for_cell
from vla_fewshot.data.splits import load_target_splits


ROOT = Path(__file__).resolve().parents[2]


def test_retention_grid_is_30_adapted_finals() -> None:
    grid = retention_grid()
    assert len(grid) == 30
    assert len(set(grid)) == 30
    assert ("drawer_middle", 1, 42) in grid
    assert ("wine_cabinet", 25, 123) in grid
    assert all(n_demos in {1, 2, 5, 10, 25} for _, n_demos, _ in grid)
    assert all(cell[0] in {"drawer_middle", "bowl_stove", "wine_cabinet"} for cell in grid)


def test_retention_uses_same_10_probe_seeds_as_seen_probe() -> None:
    probe = load_config(ROOT / "configs" / "eval" / "seen_probe.yaml")
    assert PROBE_SEEDS == FINAL_SEED_VALUES[:10] == list(range(1000, 1010))
    assert seeds_for_config(probe, project_root=ROOT) == list(PROBE_SEEDS)
    assert probe.protocol.protocol_id == "seen_probe_v1"
    assert probe.protocol.max_horizon == 300
    assert tuple(seen_probe_slugs()) == ("black_bowl_plate", "drawer_bowl", "book_caddy")
    assert uses_suite_stats_only(probe) is True
    assert uses_suite_stats_only(load_config(ROOT / "configs" / "eval" / "final.yaml")) is False


def test_retention_plans_libero_90_probes_not_target_tasks() -> None:
    probe = load_config(ROOT / "configs" / "eval" / "seen_probe.yaml")
    planned = plan_eval_rollouts(
        probe,
        task_slug="book_caddy",
        n_demos=5,
        train_seed=42,
        project_root=ROOT,
        language_control=False,
        seed_values=list(PROBE_SEEDS),
    )
    assert len(planned) == 10
    assert planned[0].suite == "libero_90"
    assert planned[0].task_index == 72
    assert planned[0].eval_seed == 1000
    assert planned[0].rollout_index == 0
    assert planned[-1].eval_seed == 1009
    assert planned[0].protocol_id == "seen_probe_v1"
    assert planned[0].n_demos == 5
    assert planned[0].train_seed == 42


def test_n12_and_official_run_roots_stay_split() -> None:
    official = Path("/tmp/official")
    n12 = Path("/tmp/n12")
    low = adapted_run_dir(
        task="bowl_stove", n_demos=2, seed=123, official_runs=official, n12_runs=n12
    )
    high = adapted_run_dir(
        task="bowl_stove", n_demos=5, seed=123, official_runs=official, n12_runs=n12
    )
    assert low == n12 / "bowl_stove_n02_s123"
    assert high == official / "bowl_stove_n05_s123"
    assert cell_name("drawer_middle", 1, 42) == "drawer_middle_n01_s42"


def test_retention_command_skips_video_and_does_not_point_at_frozen_seen() -> None:
    command = retention_command(
        task="wine_cabinet",
        n_demos=10,
        seed=42,
        run_dir=Path("/tmp/run"),
        output_dir=Path("/tmp/eval"),
    )
    assert "--skip-videos" in command
    assert "--skip-traces" in command
    assert "step_100000" not in " ".join(command)
    assert FROZEN_SEEN_SHA256 not in " ".join(command)
    assert FROZEN_SEEN_RATE == 0.8


def test_retention_episode_ids_come_from_the_target_cell_not_the_probe() -> None:
    splits = load_target_splits(ROOT / "configs" / "splits" / "target_splits.json")
    ids = episode_ids_for_cell(splits, task_slug="drawer_middle", n_demos=2)
    assert ids == [20, 26]
