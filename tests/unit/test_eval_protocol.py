from pathlib import Path

import pytest

from vla_fewshot.config import load_config
from vla_fewshot.data.splits import load_target_splits
from vla_fewshot.evaluation.language_control import (
    action_cosine_divergence,
    action_l2_divergence,
    instruction_for,
    wrong_instruction_map,
)
from vla_fewshot.evaluation.metrics import wilson_interval
from vla_fewshot.evaluation.protocol import (
    ProtocolError,
    assert_hard_reset,
    load_eval_seeds,
    rollout_key,
    seeds_for_config,
    training_episode_ids,
)
from vla_fewshot.evaluation.store import DuplicateConflictError, RolloutStore


ROOT = Path(__file__).resolve().parents[2]


def test_tracked_eval_seeds_are_1000_to_1019() -> None:
    seeds = load_eval_seeds(ROOT / "configs" / "eval" / "final_seeds.json")
    assert seeds == list(range(1000, 1020))
    config = load_config(ROOT / "configs" / "eval" / "final.yaml")
    assert seeds_for_config(config, project_root=ROOT) == seeds
    probe = load_config(ROOT / "configs" / "eval" / "seen_probe.yaml")
    assert seeds_for_config(probe, project_root=ROOT) == seeds[:10]


def test_unique_rollout_key_matches_spec() -> None:
    key = rollout_key(
        checkpoint_sha256="abc",
        task_slug="bowl_stove",
        n_demos=5,
        train_seed=42,
        eval_seed=1000,
        instruction_condition="correct",
        protocol_id="final_v1",
    )
    assert key == ("abc", "bowl_stove", 5, 42, 1000, "correct", "final_v1")


def test_wilson_interval_known_cases() -> None:
    rate, low, high = wilson_interval(0, 20)
    assert rate == 0.0
    assert low == 0.0
    assert 0.15 < high < 0.18
    rate, low, high = wilson_interval(20, 20)
    assert rate == 1.0
    assert high == 1.0
    assert 0.82 < low < 0.85
    rate, low, high = wilson_interval(10, 20)
    assert rate == 0.5
    assert low < 0.5 < high


def test_hard_reset_false_is_rejected_for_final_protocol() -> None:
    config = load_config(ROOT / "configs" / "eval" / "final.yaml")
    mutated = config.model_dump(mode="json")
    mutated["protocol"]["hard_reset"] = False
    from vla_fewshot.config import EvalConfig

    soft = EvalConfig.model_validate(mutated)
    with pytest.raises(ProtocolError, match="hard_reset"):
        assert_hard_reset(soft)


def test_wrong_instruction_map_is_cyclic_and_exact() -> None:
    config = load_config(ROOT / "configs" / "eval" / "language_control.yaml")
    mapping = wrong_instruction_map(config)
    assert mapping["drawer_middle"] == "put the bowl on the stove"
    assert mapping["bowl_stove"] == "put the wine bottle on top of the cabinet"
    assert mapping["wine_cabinet"] == "open the middle drawer of the cabinet"
    assert instruction_for(
        task_slug="bowl_stove", condition="correct", mapping=mapping
    ) == "put the bowl on the stove"
    assert instruction_for(
        task_slug="bowl_stove", condition="wrong", mapping=mapping
    ) == "put the wine bottle on top of the cabinet"


def test_jsonl_resume_skips_identical_and_refuses_conflicts(tmp_path: Path) -> None:
    store = RolloutStore(tmp_path / "rollouts.jsonl")
    record = {
        "checkpoint_sha256": "abc",
        "task_slug": "bowl_stove",
        "n_demos": 5,
        "train_seed": 42,
        "eval_seed": 1000,
        "instruction_condition": "correct",
        "protocol_id": "final_v1",
        "success": 0,
        "terminated": False,
        "truncated": True,
        "episode_length": 8,
        "instruction_text_used": "put the bowl on the stove",
        "initial_state_fingerprint": "sha256:1",
    }
    assert store.append(record) == "written"
    assert store.append(record) == "skipped"
    conflict = dict(record)
    conflict["success"] = 1
    with pytest.raises(DuplicateConflictError, match="conflicting"):
        store.append(conflict)
    reloaded = RolloutStore(tmp_path / "rollouts.jsonl")
    assert len(reloaded) == 1


def test_nested_training_episode_ids_come_from_tracked_splits() -> None:
    splits = load_target_splits(ROOT / "configs" / "splits" / "target_splits.json")
    ids = training_episode_ids(splits, task_slug="bowl_stove", n_demos=5)
    assert ids == [13, 15, 16, 22, 36]
    assert training_episode_ids(splits, task_slug="bowl_stove", n_demos=0) == []


def test_action_divergence_is_zero_for_identical_traces() -> None:
    trace = [[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]]
    assert action_l2_divergence(trace, trace) == 0.0
    assert action_cosine_divergence(trace, trace) == 0.0
