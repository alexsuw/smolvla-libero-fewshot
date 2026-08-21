"""Per-rollout action/state traces written immediately to disk."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from vla_fewshot.evaluation.protocol import key_slug
from vla_fewshot.reproducibility import atomic_write_text


def write_trace(
    output_dir: Path,
    key: tuple[Any, ...],
    records: Sequence[Mapping[str, Any]],
) -> str:
    directory = output_dir / "traces"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{key_slug(key)}.jsonl"
    atomic_write_text(
        path,
        "".join(json.dumps(dict(record), sort_keys=True) + "\n" for record in records),
        overwrite=True,
    )
    return str(path)


def load_actions(path: Path) -> list[list[float]]:
    actions: list[list[float]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        actions.append([float(item) for item in record["action"]])
    return actions
