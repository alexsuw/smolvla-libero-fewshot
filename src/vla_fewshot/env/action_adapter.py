"""7D action conversion. First six channels stay in dataset/control units."""

from __future__ import annotations

from typing import Any, Sequence

from vla_fewshot.env.gripper import (
    action_is_finite,
    binary_dataset_gripper_to_env,
    dataset_gripper_to_env,
)

ACTION_DIM = 7
ACTION_ABS_LIMIT = 1.0


def _as_action(action: Sequence[float]) -> list[float]:
    values = [float(item) for item in action]
    if len(values) != ACTION_DIM:
        raise ValueError(f"action must have {ACTION_DIM} values, got {len(values)}")
    if not action_is_finite(values):
        raise ValueError("action contains NaN or Inf")
    return values


def dataset_action_to_env(
    action: Sequence[float],
    *,
    binary: bool = True,
    threshold: float = 0.5,
) -> list[float]:
    """Convert dataset-space action to env.step space. Gripper last."""

    values = _as_action(action)
    gripper_dataset = values[6]
    if not 0.0 - 1e-6 <= gripper_dataset <= 1.0 + 1e-6:
        raise ValueError(
            "dataset gripper must stay in [0, 1]; refusing possible double conversion"
        )
    gripper = (
        binary_dataset_gripper_to_env(gripper_dataset, threshold)
        if binary
        else dataset_gripper_to_env(gripper_dataset)
    )
    return values[:6] + [gripper]


def dual_space_trace_record(
    *,
    step: int,
    dataset_action: Sequence[float],
    env_action: Sequence[float],
) -> dict[str, Any]:
    dataset = _as_action(dataset_action)
    env = _as_action(env_action)
    out_of_range = any(abs(item) > ACTION_ABS_LIMIT + 1e-6 for item in env[:6])
    return {
        "step": step,
        "policy_action_dataset_space": dataset,
        "env_action": env,
        "gripper_dataset": dataset[6],
        "gripper_env": env[6],
        "finite": True,
        "out_of_range": out_of_range,
        "max_abs_delta": max(abs(item) for item in env[:6]),
    }


def assert_env_action_stepable(action: Sequence[float]) -> None:
    values = _as_action(action)
    if not (-ACTION_ABS_LIMIT - 1e-6 <= values[6] <= ACTION_ABS_LIMIT + 1e-6):
        raise ValueError(f"env gripper out of [-1, 1]: {values[6]}")
