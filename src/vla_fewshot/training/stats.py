"""Subset-local MEAN_STD stats from selected episode vectors. No GPU import."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Iterable, Sequence


def mean_std(rows: Sequence[Sequence[float]]) -> dict[str, list[float]]:
    if not rows:
        raise ValueError("cannot compute stats over zero rows")
    dim = len(rows[0])
    if dim < 1:
        raise ValueError("stat vectors must be non-empty")
    count = float(len(rows))
    means: list[float] = []
    stds: list[float] = []
    for index in range(dim):
        values = [float(row[index]) for row in rows]
        mean = sum(values) / count
        variance = sum((value - mean) ** 2 for value in values) / count
        std = math.sqrt(variance)
        if not math.isfinite(std) or std < 1e-6:
            std = 1e-6
        means.append(mean)
        stds.append(std)
    return {"mean": means, "std": stds}


def overlay_state_action_stats(
    base: dict[str, Any],
    *,
    state: dict[str, list[float]],
    action: dict[str, list[float]],
) -> dict[str, Any]:
    """Replace state/action MEAN_STD; keep other keys (images stay IDENTITY)."""

    updated = dict(base)
    updated["observation.state"] = {**dict(updated.get("observation.state") or {}), **state}
    updated["action"] = {**dict(updated.get("action") or {}), **action}
    return updated


def _as_rows(value: Any) -> list[list[float]]:
    if hasattr(value, "detach"):
        tensor = value.detach().cpu()
        if int(tensor.ndim) <= 1:
            return [[float(item) for item in tensor.reshape(-1).tolist()]]
        reshaped = tensor.reshape(-1, int(tensor.shape[-1]))
        return [[float(item) for item in row.tolist()] for row in reshaped]
    if hasattr(value, "tolist"):
        value = value.tolist()
    if not isinstance(value, (list, tuple)):
        return [[float(value)]]
    if value and isinstance(value[0], (list, tuple)):
        return [[float(item) for item in row] for row in value]
    return [[float(item) for item in value]]


def collect_state_action_rows(
    samples: Iterable[dict[str, Any]],
    *,
    state_key: str = "observation.state",
    action_key: str = "action",
) -> tuple[list[list[float]], list[list[float]]]:
    states: list[list[float]] = []
    actions: list[list[float]] = []
    for sample in samples:
        states.extend(_as_rows(sample[state_key]))
        actions.extend(_as_rows(sample[action_key]))
    return states, actions
