"""Dataset-space vs LIBERO env-space gripper conversion.

Dataset convention (pinned NVIDIA conversion): 0 closed, 1 open.
LIBERO env convention in this project: +1 close, -1 open.

    g_env = 1 - 2 * g_dataset
"""

from __future__ import annotations

import math
from typing import Sequence


def dataset_gripper_to_env(gripper: float) -> float:
    """Continuous conversion used by unit tests and traces."""

    value = float(gripper)
    if not math.isfinite(value):
        raise ValueError(f"gripper is not finite: {gripper!r}")
    return 1.0 - 2.0 * value


def env_gripper_to_dataset(gripper: float) -> float:
    value = float(gripper)
    if not math.isfinite(value):
        raise ValueError(f"gripper is not finite: {gripper!r}")
    return (1.0 - value) / 2.0


def binary_dataset_gripper_to_env(gripper: float, threshold: float = 0.5) -> float:
    """Runtime postprocessor applied immediately before env.step."""

    if not 0.0 < float(threshold) < 1.0:
        raise ValueError(f"threshold must be in (0, 1), got {threshold}")
    value = float(gripper)
    if not math.isfinite(value):
        raise ValueError(f"gripper is not finite: {gripper!r}")
    if value < threshold:
        return 1.0
    return -1.0


def action_is_finite(action: Sequence[float]) -> bool:
    return all(math.isfinite(float(item)) for item in action)
