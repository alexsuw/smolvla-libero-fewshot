"""Subset-local MEAN_STD stats from selected episode vectors. No GPU import."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable, Sequence

from vla_fewshot.reproducibility import atomic_write_json
from vla_fewshot.storage.layout import NORMALIZATION_STATS_NAME


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


def overlay_dataset_state_action_stats(base: dict[str, Any], dataset: Any) -> dict[str, Any]:
    """Same overlay the trainer applies: every `dataset[i]` state/action row."""

    states, actions = collect_state_action_rows(dataset[index] for index in range(len(dataset)))
    return overlay_state_action_stats(base, state=mean_std(states), action=mean_std(actions))


def jsonable_stats(stats: dict[str, Any]) -> dict[str, Any]:
    """LeRobot `dataset.meta.stats` keeps numpy arrays; processors accept lists."""

    def _convert(value: Any) -> Any:
        if isinstance(value, dict):
            return {str(key): _convert(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [_convert(item) for item in value]
        if hasattr(value, "tolist"):
            return _convert(value.tolist())
        if hasattr(value, "item") and not isinstance(value, (bytes, str)):
            try:
                return value.item()
            except (ValueError, AttributeError):
                pass
        return value

    converted = _convert(stats)
    if not isinstance(converted, dict):
        raise TypeError("normalization stats must be a mapping")
    return converted


def stats_digest(stats: dict[str, Any]) -> str:
    payload = json.dumps(jsonable_stats(stats), sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )
    return hashlib.sha256(payload).hexdigest()


def write_normalization_stats(path: Path, stats: dict[str, Any]) -> str:
    if path.name != NORMALIZATION_STATS_NAME:
        raise ValueError(f"normalization sidecar must be named {NORMALIZATION_STATS_NAME}")
    payload = jsonable_stats(stats)
    atomic_write_json(path, payload, overwrite=path.exists())
    return stats_digest(payload)


def load_normalization_stats(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not payload:
        raise ValueError(f"invalid normalization sidecar: {path}")
    return payload


def find_normalization_sidecar(checkpoint: Path, run_dir: Path | None = None) -> Path | None:
    candidates = [checkpoint / NORMALIZATION_STATS_NAME]
    if run_dir is not None:
        candidates.append(Path(run_dir) / NORMALIZATION_STATS_NAME)
    elif checkpoint.parent.name == "checkpoints":
        candidates.append(checkpoint.parent.parent / NORMALIZATION_STATS_NAME)
    for path in candidates:
        if path.is_file():
            return path
    return None
