from pathlib import Path
import os
import subprocess
import sys

import pytest

from vla_fewshot.config import load_config
from vla_fewshot.evaluation.cli import run_eval_cli
from vla_fewshot.evaluation.protocol import ProtocolError, training_episode_ids
from vla_fewshot.evaluation.store import RolloutStore
from vla_fewshot.evaluation.zero_shot import (
    ZERO_SHOT_SLUGS,
    assert_frozen_checkpoint_hash,
    assert_zero_shot_cell,
    assert_zero_shot_config,
    resolve_frozen_eval_checkpoint,
    zero_shot_commands,
)


ROOT = Path(__file__).resolve().parents[2]


def test_zero_shot_yaml_and_empty_episode_list() -> None:
    config = load_config(ROOT / "configs" / "eval" / "zero_shot.yaml")
    assert_zero_shot_config(config, profile="full")
    assert config.protocol.rollouts_per_cell >= 20
    assert training_episode_ids(None, task_slug="bowl_stove", n_demos=0) == []
    assert_zero_shot_cell(n_demos=0, train_seed=None, episode_ids=[])
    with pytest.raises(ProtocolError, match="0 target demonstrations"):
        assert_zero_shot_cell(n_demos=5, train_seed=None, episode_ids=[])
    with pytest.raises(ProtocolError, match="empty"):
        assert_zero_shot_cell(n_demos=0, train_seed=None, episode_ids=[13])
    final = load_config(ROOT / "configs" / "eval" / "final.yaml")
    with pytest.raises(ProtocolError, match="zero_shot"):
        assert_zero_shot_config(final, profile="full")


def test_pending_seen_checkpoint_blocks_zero_shot_origin() -> None:
    with pytest.raises(RuntimeError, match="frozen"):
        resolve_frozen_eval_checkpoint(None)


def test_checkpoint_hash_must_match_frozen_digest(tmp_path: Path) -> None:
    path = tmp_path / "weights.bin"
    path.write_bytes(b"not-the-frozen-seen-checkpoint")
    with pytest.raises(RuntimeError, match="frozen seen"):
        assert_frozen_checkpoint_hash(path, "0" * 64)


def test_print_grid_lists_three_target_tasks() -> None:
    commands = zero_shot_commands()
    assert len(commands) == 3
    assert [row[5] for row in commands] == list(ZERO_SHOT_SLUGS)
    completed = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "eval_zero_shot.py"), "--print-grid"],
        cwd=ROOT,
        capture_output=True,
        check=False,
        text=True,
        timeout=20,
    )
    assert completed.returncode == 0, completed.stderr
    lines = [line for line in completed.stdout.splitlines() if line.strip()]
    assert len(lines) == 3
    assert "--task drawer_middle" in lines[0]


def test_eval_zero_shot_full_profile_fails_before_compute() -> None:
    env = os.environ.copy()
    completed = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "eval_zero_shot.py")],
        cwd=ROOT,
        capture_output=True,
        check=False,
        text=True,
        timeout=20,
        env=env,
    )
    assert completed.returncode == 1
    assert "no GPU evaluation was started" in completed.stdout + completed.stderr


def test_zero_shot_static_cli_runs_three_tasks_with_empty_train_list(
    tmp_path: Path,
) -> None:
    code = run_eval_cli(
        "zero_shot",
        [
            "--profile",
            "static",
            "--output-dir",
            str(tmp_path / "zs"),
        ],
    )
    assert code == 0
    for slug in ZERO_SHOT_SLUGS:
        store = RolloutStore(tmp_path / "zs" / slug / "rollouts.jsonl")
        assert len(store) == 3
        for record in store.records():
            assert record["stage"] == "zero_shot"
            assert int(record["n_demos"]) == 0
            assert record["training_episode_ids"] == []
            assert record["train_seed"] is None
            assert record["method"] == "seen"
