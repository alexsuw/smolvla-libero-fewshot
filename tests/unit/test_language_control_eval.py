from pathlib import Path
import os
import subprocess
import sys

import pytest

from vla_fewshot.config import load_config
from vla_fewshot.env.libero_env import apply_libero_init_state_id
from vla_fewshot.evaluation.cli import run_eval_cli
from vla_fewshot.evaluation.language_control import (
    LANGUAGE_CONTROL_SLUGS,
    assert_language_control_cell,
    assert_language_control_config,
    language_control_commands,
)
from vla_fewshot.evaluation.protocol import ProtocolError
from vla_fewshot.evaluation.runner import (
    assert_completed_language_pair_fingerprints,
    plan_eval_rollouts,
    static_smoke_config,
)
from vla_fewshot.evaluation.store import RolloutStore
from vla_fewshot.evaluation.zero_shot import resolve_frozen_eval_checkpoint


ROOT = Path(__file__).resolve().parents[2]


def test_language_control_yaml_and_empty_train_list() -> None:
    config = load_config(ROOT / "configs" / "eval" / "language_control.yaml")
    assert_language_control_config(config, profile="full")
    assert config.protocol.rollouts_per_cell >= 20
    assert_language_control_cell(n_demos=0, train_seed=None, episode_ids=[])
    with pytest.raises(ProtocolError, match="0 target demonstrations"):
        assert_language_control_cell(n_demos=5, train_seed=None, episode_ids=[])
    zero = load_config(ROOT / "configs" / "eval" / "zero_shot.yaml")
    with pytest.raises(ProtocolError, match="language_control"):
        assert_language_control_config(zero, profile="full")


def test_frozen_seen_checkpoint_resolves_language_control_origin() -> None:
    origin, digest = resolve_frozen_eval_checkpoint(None, purpose="language control")
    assert digest
    assert origin.name.startswith("step_")


def test_print_grid_lists_three_language_control_tasks() -> None:
    commands = language_control_commands()
    assert len(commands) == 3
    assert [row[5] for row in commands] == list(LANGUAGE_CONTROL_SLUGS)
    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "eval_language_control.py"),
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
    assert len(lines) == 3
    assert "--task drawer_middle" in lines[0]


def test_eval_language_control_full_profile_fails_before_compute() -> None:
    env = os.environ.copy()
    completed = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "eval_language_control.py")],
        cwd=ROOT,
        capture_output=True,
        check=False,
        text=True,
        timeout=20,
        env=env,
    )
    assert completed.returncode in {1, 2}
    text = completed.stdout + completed.stderr
    assert "no GPU evaluation was started" in text or "--output-dir is required" in text


def test_language_control_static_cli_runs_three_paired_tasks(tmp_path: Path) -> None:
    code = run_eval_cli(
        "language_control",
        [
            "--profile",
            "static",
            "--output-dir",
            str(tmp_path / "lc"),
        ],
    )
    assert code == 0
    for slug in LANGUAGE_CONTROL_SLUGS:
        store = RolloutStore(tmp_path / "lc" / slug / "rollouts.jsonl")
        assert len(store) == 6
        pairs_path = tmp_path / "lc" / slug / "language_pairs.json"
        assert pairs_path.is_file()
        fingerprints: dict[int, set[str]] = {}
        hashes: set[str] = set()
        for record in store.records():
            assert record["stage"] == "language_control"
            assert int(record["n_demos"]) == 0
            assert record["training_episode_ids"] == []
            fingerprints.setdefault(int(record["eval_seed"]), set()).add(
                record["initial_state_fingerprint"]
            )
            hashes.add(record["checkpoint_sha256"])
        assert all(len(value) == 1 for value in fingerprints.values())
        assert len(hashes) == 1
        conditions = [record["instruction_condition"] for record in store.records()]
        assert conditions == ["correct", "wrong", "correct", "wrong", "correct", "wrong"]


def test_apply_libero_init_state_id_pins_next_reset() -> None:
    class _Inner:
        init_state_id = 99

    class _Env:
        envs = [_Inner()]

    env = _Env()
    apply_libero_init_state_id(env, None)
    assert env.envs[0].init_state_id == 99
    apply_libero_init_state_id(env, 3)
    assert env.envs[0].init_state_id == 3


def test_language_control_plan_interleaves_same_seed_pairs() -> None:
    config = static_smoke_config(load_config(ROOT / "configs" / "eval" / "language_control.yaml"))
    planned = plan_eval_rollouts(
        config,
        task_slug="drawer_middle",
        n_demos=0,
        train_seed=None,
        project_root=ROOT,
        language_control=True,
    )
    assert [item.instruction_condition for item in planned] == [
        "correct",
        "wrong",
        "correct",
        "wrong",
        "correct",
        "wrong",
    ]
    assert planned[0].eval_seed == planned[1].eval_seed
    assert planned[0].rollout_index == planned[1].rollout_index == 0
    assert planned[0].instruction_text_used != planned[1].instruction_text_used


def test_completed_language_pair_fingerprint_drift_fails_closed() -> None:
    records = [
        {
            "eval_seed": 1000,
            "instruction_condition": "correct",
            "checkpoint_sha256": "abc",
            "initial_state_fingerprint": "fp-a",
        },
        {
            "eval_seed": 1000,
            "instruction_condition": "wrong",
            "checkpoint_sha256": "abc",
            "initial_state_fingerprint": "fp-b",
        },
    ]
    with pytest.raises(ProtocolError, match="fingerprints drifted"):
        assert_completed_language_pair_fingerprints(records)
    records[1]["initial_state_fingerprint"] = "fp-a"
    assert_completed_language_pair_fingerprints(records)
