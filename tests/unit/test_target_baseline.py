from pathlib import Path
import os
import subprocess
import sys

import pytest

from vla_fewshot.config import load_config
from vla_fewshot.data.splits import load_target_splits
from vla_fewshot.training.baseline import (
    apply_cell_overrides,
    assert_baseline_train_config,
    baseline_grid,
    cap_optimizer_steps,
    episode_ids_for_cell,
    require_frozen_seen_origin,
)
from vla_fewshot.training.stats import collect_state_action_rows, mean_std
from vla_fewshot.training.trainer import TrainError


ROOT = Path(__file__).resolve().parents[2]
SPLITS = ROOT / "configs" / "splits" / "target_splits.json"


def test_baseline_grid_is_18_independent_cells() -> None:
    grid = baseline_grid()
    assert len(grid) == 18
    assert len(set(grid)) == 18
    assert ("bowl_stove", 5, 42) in grid
    assert ("wine_cabinet", 25, 123) in grid


def test_nested_episode_prefixes_match_tracked_splits() -> None:
    splits = load_target_splits(SPLITS)
    one = episode_ids_for_cell(splits, task_slug="bowl_stove", n_demos=1)
    two = episode_ids_for_cell(splits, task_slug="bowl_stove", n_demos=2)
    five = episode_ids_for_cell(splits, task_slug="bowl_stove", n_demos=5)
    ten = episode_ids_for_cell(splits, task_slug="bowl_stove", n_demos=10)
    twenty_five = episode_ids_for_cell(splits, task_slug="bowl_stove", n_demos=25)
    assert one == [13]
    assert two == [13, 15]
    assert five == [13, 15, 16, 22, 36]
    assert one == two[:1] == five[:1]
    assert two == five[:2]
    assert five == ten[:5]
    assert ten == twenty_five[:10]


def test_low_n_grid_is_12_independent_cells() -> None:
    from vla_fewshot.training.baseline import low_n_baseline_grid

    grid = low_n_baseline_grid()
    assert len(grid) == 12
    assert len(set(grid)) == 12
    assert ("drawer_middle", 1, 42) in grid
    assert ("wine_cabinet", 2, 123) in grid
    assert all(n_demos in {1, 2} for _, n_demos, _ in grid)
    assert baseline_grid() == [
        cell for cell in baseline_grid() if cell[1] in {5, 10, 25}
    ]
    assert len(baseline_grid()) == 18


def test_cap_optimizer_steps_is_min_of_epochs_and_max_steps() -> None:
    assert (
        cap_optimizer_steps(
            max_steps=12000, epochs=100, n_samples=32, effective_batch_size=32
        )
        == 100
    )
    assert (
        cap_optimizer_steps(
            max_steps=12000, epochs=100, n_samples=50_000, effective_batch_size=32
        )
        == 12000
    )
    assert (
        cap_optimizer_steps(
            max_steps=100000, epochs=None, n_samples=1000, effective_batch_size=32
        )
        == 100000
    )


def test_baseline_yaml_is_accepted_and_lora_is_refused() -> None:
    baseline = load_config(ROOT / "configs" / "train" / "target_baseline.yaml")
    assert_baseline_train_config(baseline)
    seeded = apply_cell_overrides(baseline, seed=123)
    assert seeded.training.seed == 123
    lora = load_config(ROOT / "configs" / "train" / "target_lora.yaml")
    with pytest.raises(TrainError, match="LoRA"):
        assert_baseline_train_config(lora)
    seen = load_config(ROOT / "configs" / "train" / "seen_expert.yaml")
    with pytest.raises(TrainError, match="baseline path only"):
        assert_baseline_train_config(seen)


def test_frozen_selected_checkpoint_resolves_target_origin() -> None:
    origin, digest, run_id = require_frozen_seen_origin()
    assert digest
    assert origin.name.startswith("step_")
    assert run_id


def test_subset_mean_std_and_chunked_actions() -> None:
    stats = mean_std([[0.0, 2.0], [2.0, 2.0]])
    assert stats["mean"] == [1.0, 2.0]
    assert stats["std"][0] == pytest.approx(1.0)
    states, actions = collect_state_action_rows(
        [
            {
                "observation.state": [1.0, 0.0],
                "action": [[0.0] * 7, [1.0] * 7],
            }
        ]
    )
    assert states == [[1.0, 0.0]]
    assert len(actions) == 2
    assert actions[1][0] == 1.0


def test_print_grid_lists_eighteen_commands() -> None:
    completed = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "train_target.py"), "--print-grid"],
        cwd=ROOT,
        capture_output=True,
        check=False,
        text=True,
        timeout=20,
    )
    assert completed.returncode == 0, completed.stderr
    lines = [line for line in completed.stdout.splitlines() if line.strip()]
    assert len(lines) == 18
    assert "train_target.py" in lines[0]
    assert "--task drawer_middle --n-demos 5 --seed 42" in lines[0]


def test_train_target_refuses_unfrozen_seen_and_replay_lora(tmp_path: Path) -> None:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = ""
    replay = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "train_target.py"),
            "--config",
            str(ROOT / "configs" / "train" / "target_replay_lora.yaml"),
            "--task",
            "drawer_middle",
            "--n-demos",
            "5",
            "--seed",
            "42",
            "--output-dir",
            str(tmp_path / "replay"),
        ],
        cwd=ROOT,
        capture_output=True,
        check=False,
        text=True,
        timeout=20,
        env=env,
    )
    assert replay.returncode == 1
    combined_replay = replay.stdout + replay.stderr
    assert "no GPU training was started" in combined_replay
    assert "not wired" not in combined_replay

    baseline = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "train_target.py"),
            "--task",
            "drawer_middle",
            "--n-demos",
            "5",
            "--seed",
            "42",
            "--output-dir",
            str(tmp_path / "base"),
        ],
        cwd=ROOT,
        capture_output=True,
        check=False,
        text=True,
        timeout=20,
        env=env,
    )
    assert baseline.returncode == 1
    combined = baseline.stdout + baseline.stderr
    assert "no GPU training was started" in combined
