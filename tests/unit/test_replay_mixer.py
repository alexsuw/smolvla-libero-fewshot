from pathlib import Path
import subprocess
import sys

import pytest

from vla_fewshot.config import load_config
from vla_fewshot.data.splits import load_target_splits
from vla_fewshot.training.baseline import episode_ids_for_cell
from vla_fewshot.training.replay_mixer import (
    ReplayMixer,
    assert_replay_lora_train_config,
    assert_replay_pool,
    assert_replay_sample_not_goal,
    split_batch_counts,
)
from vla_fewshot.training.target_lora import assert_target_train_config
from vla_fewshot.training.trainer import TrainError
from vla_fewshot.training.full_loop import _collate


ROOT = Path(__file__).resolve().parents[2]


def test_replay_lora_yaml_matches_calibration() -> None:
    config = load_config(ROOT / "configs" / "train" / "target_replay_lora.yaml")
    assert_replay_lora_train_config(config)
    assert_target_train_config(config)
    assert config.replay is not None
    assert config.replay.target_fraction == 0.75
    assert config.replay.seen_fraction == 0.25
    assert config.replay.seen_suite == "libero_90"


def test_replay_pool_rejects_goal_suite_and_target_text() -> None:
    with pytest.raises(TrainError, match="libero_90"):
        assert_replay_pool(suite="libero_goal", task_texts=["open the middle drawer of the cabinet"])
    with pytest.raises(TrainError, match="leaked"):
        assert_replay_pool(suite="libero_90", task_texts=["put the bowl on the stove"])
    assert_replay_pool(suite="libero_90", task_texts=["put the middle black bowl on the plate"])


def test_replay_sample_goal_task_fails_closed() -> None:
    with pytest.raises(TrainError, match="libero_goal sample"):
        assert_replay_sample_not_goal({"task": "open the middle drawer of the cabinet"})
    assert_replay_sample_not_goal({"task": "put the book in the caddy"})


def test_mixer_keeps_075_025_over_a_window_and_is_seeded() -> None:
    first = ReplayMixer(
        n_target=8,
        n_replay=40,
        target_fraction=0.75,
        seen_fraction=0.25,
        seed=42,
        with_replacement=True,
    )
    second = ReplayMixer(
        n_target=8,
        n_replay=40,
        target_fraction=0.75,
        seen_fraction=0.25,
        seed=42,
        with_replacement=True,
    )
    n_target = 0
    n_replay = 0
    for _ in range(50):
        draw = first.next_draw(4)
        assert draw.n_target + draw.n_replay == 4
        assert second.next_draw(4).sources == draw.sources
        n_target += draw.n_target
        n_replay += draw.n_replay
    total = n_target + n_replay
    assert abs(n_target / total - 0.75) < 0.02
    assert abs(n_replay / total - 0.25) < 0.02
    cum_t, cum_s = first.cumulative_fractions()
    assert cum_t == pytest.approx(n_target / total)
    assert cum_s == pytest.approx(n_replay / total)


def test_mixer_resume_replays_the_same_stream() -> None:
    mixer = ReplayMixer(
        n_target=5,
        n_replay=20,
        target_fraction=0.75,
        seen_fraction=0.25,
        seed=123,
        with_replacement=True,
    )
    mixer.next_draw(4)
    state = mixer.state_dict()
    resumed = ReplayMixer(
        n_target=5,
        n_replay=20,
        target_fraction=0.75,
        seen_fraction=0.25,
        seed=123,
        with_replacement=True,
    )
    resumed.load_state_dict(state)
    assert mixer.next_draw(4).sources == resumed.next_draw(4).sources


def test_split_remainder_hits_three_quarters() -> None:
    remainder = 0.0
    n_target = 0
    n_replay = 0
    for _ in range(8):
        t, r, remainder = split_batch_counts(1, target_fraction=0.75, remainder=remainder)
        n_target += t
        n_replay += r
    assert n_target == 6
    assert n_replay == 2


def test_replay_does_not_change_target_episode_ids() -> None:
    splits = load_target_splits(ROOT / "configs" / "splits" / "target_splits.json")
    assert episode_ids_for_cell(splits, task_slug="bowl_stove", n_demos=5) == [
        13,
        15,
        16,
        22,
        36,
    ]


def test_mixed_collate_drops_suite_specific_optional_fields() -> None:
    torch = pytest.importorskip("torch")

    common = {
        "observation.images.image": torch.zeros(3, 2, 2),
        "observation.images.wrist_image": torch.zeros(3, 2, 2),
        "observation.state": torch.zeros(8),
        "action": torch.zeros(7),
        "task": "instruction",
    }
    target = {**common, "observation.states.ee_state": torch.zeros(3)}
    replay = dict(common)
    batch = _collate([target, replay])
    assert "observation.states.ee_state" not in batch
    assert batch["action"].shape == (2, 7)


def test_mixed_collate_fails_when_a_required_policy_field_differs() -> None:
    torch = pytest.importorskip("torch")

    target = {
        "observation.images.image": torch.zeros(3, 2, 2),
        "observation.images.wrist_image": torch.zeros(3, 2, 2),
        "observation.state": torch.zeros(8),
        "action": torch.zeros(7),
        "task": "instruction",
    }
    replay = dict(target)
    replay.pop("observation.state")
    with pytest.raises(TrainError, match="required SmolVLA fields"):
        _collate([target, replay])


def test_print_grid_replay_lora_lists_eighteen_commands() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "train_target.py"),
            "--config",
            str(ROOT / "configs" / "train" / "target_replay_lora.yaml"),
            "--print-grid",
        ],
        cwd=ROOT,
        capture_output=True,
        check=False,
        text=True,
        timeout=20,
    )
    assert completed.returncode == 0, completed.stderr
    lines = [line for line in completed.stdout.splitlines() if line.strip()]
    assert len(lines) == 18
    assert "target_replay_lora.yaml" in lines[0]


def test_eval_print_grid_replay_passes_train_config() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "eval_target.py"),
            "--train-config",
            str(ROOT / "configs" / "train" / "target_replay_lora.yaml"),
            "--print-grid",
        ],
        cwd=ROOT,
        capture_output=True,
        check=False,
        text=True,
        timeout=20,
    )
    assert completed.returncode == 0, completed.stderr
    lines = [line for line in completed.stdout.splitlines() if line.strip()]
    assert len(lines) == 18
    assert "target_replay_lora.yaml" in lines[0]
