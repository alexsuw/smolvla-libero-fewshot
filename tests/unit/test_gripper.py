import pytest

from vla_fewshot.env.action_adapter import dataset_action_to_env, dual_space_trace_record
from vla_fewshot.env.gripper import (
    binary_dataset_gripper_to_env,
    dataset_gripper_to_env,
    env_gripper_to_dataset,
)


def test_dataset_gripper_endpoints_match_spec() -> None:
    assert dataset_gripper_to_env(0.0) == 1.0
    assert dataset_gripper_to_env(1.0) == -1.0
    assert dataset_gripper_to_env(0.5) == 0.0


def test_binary_runtime_postprocessor() -> None:
    assert binary_dataset_gripper_to_env(0.0) == 1.0
    assert binary_dataset_gripper_to_env(0.49) == 1.0
    assert binary_dataset_gripper_to_env(0.5) == -1.0
    assert binary_dataset_gripper_to_env(1.0) == -1.0


def test_env_gripper_roundtrip() -> None:
    assert env_gripper_to_dataset(dataset_gripper_to_env(0.25)) == pytest.approx(0.25)


def test_env_space_gripper_is_not_accepted_as_dataset_action() -> None:
    with pytest.raises(ValueError, match="double conversion"):
        dataset_action_to_env([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0])


def test_nan_gripper_is_rejected() -> None:
    with pytest.raises(ValueError, match="not finite"):
        dataset_gripper_to_env(float("nan"))


def test_dataset_action_converts_only_gripper() -> None:
    env = dataset_action_to_env([0.1, -0.2, 0.3, 0.0, 0.0, 0.0, 0.0], binary=True)
    assert env[:6] == [0.1, -0.2, 0.3, 0.0, 0.0, 0.0]
    assert env[6] == 1.0
    continuous = dataset_action_to_env(
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5], binary=False
    )
    assert continuous[6] == 0.0


def test_dual_space_trace_keeps_both_actions() -> None:
    dataset = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
    env = dataset_action_to_env(dataset)
    record = dual_space_trace_record(step=0, dataset_action=dataset, env_action=env)
    assert record["policy_action_dataset_space"][6] == 1.0
    assert record["env_action"][6] == -1.0
    assert record["finite"] is True
