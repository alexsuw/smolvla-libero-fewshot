from pathlib import Path

import pytest

from vla_fewshot.data.splits import load_target_splits
from vla_fewshot.env.replay import (
    load_episode_actions_from_rows,
    load_replay_gate,
    replay_actions_through_env,
)

ROOT = Path(__file__).resolve().parents[2]
SPLITS = ROOT / "configs" / "splits" / "target_splits.json"


class FakeEnv:
    def __init__(self) -> None:
        self.steps = 0

    def reset(self, seed: int = 0):
        return {"observation.state": [0.0] * 8}, {"is_success": False}

    def step(self, action):
        self.steps += 1
        success = self.steps >= 2
        return (
            {"observation.state": [0.0] * 8},
            1.0 if success else 0.0,
            success,
            False,
            {"is_success": success},
        )

    def close(self) -> None:
        return None


def test_replay_gate_has_six_required_episodes() -> None:
    gate = load_replay_gate(ROOT / "configs" / "splits" / "replay_gate.json")
    assert len(gate.episodes) == 6
    targets = {item.slug for item in gate.episodes if item.suite == "libero_goal"}
    seen = {item.suite for item in gate.episodes if item.suite == "libero_90"}
    assert targets == {"drawer_middle", "bowl_stove", "wine_cabinet"}
    assert seen == {"libero_90"}
    bowl = next(item for item in gate.episodes if item.slug == "bowl_stove")
    wine = next(item for item in gate.episodes if item.slug == "wine_cabinet")
    assert bowl.episode_id == 13
    assert bowl.env_task_id == 1
    assert wine.env_task_id == 2
    assert wine.env_init_state_id == 1
    book = next(item for item in gate.episodes if item.slug == "seen_book_caddy")
    assert book.episode_id == 139
    assert book.env_task_id == 73
    assert book.task_local_index == 1
    assert all(
        item.task_local_index == 0
        for item in gate.episodes
        if item.slug != "seen_book_caddy"
    )
    seen_items = [item for item in gate.episodes if item.suite == "libero_90"]
    assert {item.env_task_id for item in seen_items} == {8, 14, 73}
    splits = load_target_splits(SPLITS)
    for slug, spec in splits.tasks.items():
        gate_item = next(item for item in gate.episodes if item.slug == slug)
        assert gate_item.episode_id == spec.episode_ids_first_25[0]


def test_replay_uses_production_gripper_and_writes_dual_trace(tmp_path: Path) -> None:
    env = FakeEnv()
    actions = [
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
    ]
    result = replay_actions_through_env(
        env=env,
        dataset_actions=actions,
        output_dir=tmp_path / "replay",
        task_text="put the bowl on the stove",
        suite="libero_goal",
        episode_id=13,
        seed=0,
        save_video=True,
        save_frame=lambda _obs: [[[0, 0, 0], [255, 0, 0]], [[0, 255, 0], [0, 0, 255]]],
    )
    assert result.success is True
    trace = (tmp_path / "replay" / "trace.jsonl").read_text(encoding="utf-8")
    assert '"gripper_dataset": 0.0' in trace
    assert '"gripper_env": 1.0' in trace
    assert '"gripper_dataset": 1.0' in trace
    assert '"gripper_env": -1.0' in trace
    assert (tmp_path / "replay" / "frames" / "frame-00000.ppm").exists()


def test_replay_rejects_nan_actions(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="illegal dataset action"):
        replay_actions_through_env(
            env=FakeEnv(),
            dataset_actions=[[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, float("nan")]],
            output_dir=tmp_path / "bad",
            task_text="x",
            suite="libero_goal",
            episode_id=0,
            seed=0,
            save_video=False,
            save_frame=None,
        )


def test_configured_env_task_id_is_used_without_libero(monkeypatch: pytest.MonkeyPatch) -> None:
    from vla_fewshot.env import libero_env as libero_mod

    monkeypatch.setattr(
        libero_mod,
        "require_libero_runtime",
        lambda: (_ for _ in ()).throw(
            RuntimeError("LIBERO runtime requires `uv sync --frozen --extra gpu` on Linux")
        ),
    )
    assert (
        libero_mod.resolve_env_task_id(
            suite="libero_goal",
            task_text="put the bowl on the stove",
            configured=7,
        )
        == 7
    )


def test_language_match_overrides_dataset_task_index() -> None:
    from vla_fewshot.env.libero_env import require_libero_runtime, resolve_env_task_id

    try:
        require_libero_runtime()
    except RuntimeError:
        pytest.skip("LIBERO runtime is not installed")
    assert (
        resolve_env_task_id(
            suite="libero_goal",
            task_text="put the bowl on the stove",
            configured=7,
        )
        == 1
    )


def test_load_episode_actions_from_rows_is_ordered() -> None:
    rows = [
        {"index": 11, "action": [0, 0, 0, 0, 0, 0, 1]},
        {"index": 10, "action": [0.1, 0, 0, 0, 0, 0, 0]},
    ]
    actions = load_episode_actions_from_rows(rows, start_index=10, length=2)
    assert actions[0][0] == 0.1
    assert actions[1][6] == 1.0
