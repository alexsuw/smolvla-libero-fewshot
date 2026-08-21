"""Wilson interval for per-cell success rates. No extra scientific deps."""

from __future__ import annotations

import math
from typing import Any

Z_95 = 1.959963984540054


def wilson_interval(
    successes: int,
    n: int,
    *,
    z: float = Z_95,
) -> tuple[float, float, float]:
    """Return (rate, low, high) for a Wilson score interval."""

    if n < 1:
        raise ValueError("n must be positive")
    if successes < 0 or successes > n:
        raise ValueError("successes must be in [0, n]")
    p = successes / n
    z2 = z * z
    denom = 1.0 + z2 / n
    center = (p + z2 / (2.0 * n)) / denom
    margin = z * math.sqrt((p * (1.0 - p) + z2 / (4.0 * n)) / n) / denom
    low = max(0.0, center - margin)
    high = min(1.0, center + margin)
    return p, low, high


def cell_summary(
    *,
    method: str,
    task_slug: str,
    n_demos: int | None,
    train_seed: int | None,
    records: list[dict[str, Any]],
    checkpoint_sha256: str,
    protocol_id: str,
) -> dict[str, Any]:
    n = len(records)
    successes = sum(int(record["success"]) for record in records)
    rate, low, high = wilson_interval(successes, n) if n else (0.0, 0.0, 0.0)
    return {
        "method": method,
        "task_slug": task_slug,
        "n_demos": 0 if n_demos is None else n_demos,
        "train_seed": train_seed,
        "n_rollouts": n,
        "n_successes": successes,
        "success_rate": rate,
        "wilson_ci_low": low,
        "wilson_ci_high": high,
        "checkpoint_sha256": checkpoint_sha256,
        "protocol_id": protocol_id,
    }
