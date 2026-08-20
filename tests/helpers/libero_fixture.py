"""Build tiny LeRobot v3 metadata trees for unit tests."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from vla_fewshot.data.expected import TARGET_TASKS
from vla_fewshot.data.splits import load_target_splits


def _arrow():
    import pyarrow as pa
    import pyarrow.parquet as pq

    return pa, pq


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_parquet(path: Path, rows: list[dict[str, Any]]) -> None:
    pa, pq = _arrow()
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows), path)


def _features() -> dict[str, Any]:
    video = {
        "dtype": "video",
        "shape": [256, 256, 3],
        "info": {
            "video.codec": "av1",
            "video.pix_fmt": "yuv420p",
            "video.fps": 20.0,
            "video.is_depth_map": False,
        },
    }
    return {
        "action": {"dtype": "float32", "shape": [7]},
        "observation.state": {"dtype": "float32", "shape": [8]},
        "observation.images.image": video,
        "observation.images.wrist_image": video,
        "timestamp": {"dtype": "float32", "shape": [1]},
        "frame_index": {"dtype": "int64", "shape": [1]},
        "episode_index": {"dtype": "int64", "shape": [1]},
        "task_index": {"dtype": "int64", "shape": [1]},
        "index": {"dtype": "int64", "shape": [1]},
    }


def _stats() -> dict[str, Any]:
    zeros7 = [0.0] * 6 + [0.0]
    ones7 = [1.0] * 6 + [1.0]
    zeros8 = [0.0] * 8
    ones8 = [1.0] * 8
    return {
        "action": {"min": zeros7, "max": ones7, "mean": [0.1] * 7, "std": [0.2] * 7},
        "observation.state": {
            "min": zeros8,
            "max": ones8,
            "mean": [0.3] * 8,
            "std": [0.4] * 8,
        },
    }


def _suite_meta(
    root: Path,
    suite: str,
    *,
    episodes: int,
    frames: int,
    tasks: list[str],
    episode_rows: list[dict[str, Any]],
) -> None:
    meta = root / suite / "meta"
    _write_json(
        meta / "info.json",
        {
            "codebase_version": "v3.0",
            "fps": 20,
            "total_episodes": episodes,
            "total_frames": frames,
            "total_tasks": len(tasks),
            "features": _features(),
        },
    )
    _write_json(meta / "stats.json", _stats())
    _write_parquet(
        meta / "tasks.parquet",
        [{"task_index": index, "task": text} for index, text in enumerate(tasks)],
    )
    _write_parquet(meta / "episodes" / "chunk-000" / "file-000.parquet", episode_rows)


def build_pinned_metadata_fixture(root: Path, splits_path: Path) -> Path:
    """Create a metadata tree whose counts and target IDs match the spec."""

    splits = load_target_splits(splits_path)
    used: set[int] = set()
    goal_rows: list[dict[str, Any]] = []
    extra_id = 1000
    for slug, spec in TARGET_TASKS.items():
        tracked = splits.tasks[slug]
        ids = list(tracked.episode_ids_first_25)
        while len(ids) < int(spec["available_count"]):
            if extra_id not in used and extra_id not in ids:
                ids.append(extra_id)
            extra_id += 1
        used.update(ids)
        for episode_id in sorted(ids):
            goal_rows.append(
                {
                    "episode_index": episode_id,
                    "tasks": [spec["task_text"]],
                    "length": 10,
                }
            )

    fillers = [
        "open the top drawer of the cabinet",
        "open the bottom drawer of the cabinet",
        "put the cream cheese in the bowl",
        "turn on the stove",
        "put the wine bottle in the bowl",
        "put the bowl on the plate",
        "push the plate to the front of the stove",
    ]
    goal_tasks = [""] * 10
    fill_index = 0
    for index in range(10):
        match = next(
            (
                spec
                for spec in TARGET_TASKS.values()
                if spec["task_index"] == index
            ),
            None,
        )
        if match is not None:
            goal_tasks[index] = str(match["task_text"])
        else:
            goal_tasks[index] = fillers[fill_index]
            fill_index += 1
    filler_id = 0
    while len(goal_rows) < 428:
        if filler_id not in used:
            goal_rows.append(
                {
                    "episode_index": filler_id,
                    "tasks": [fillers[len(goal_rows) % len(fillers)]],
                    "length": 8,
                }
            )
            used.add(filler_id)
        filler_id += 1
    _suite_meta(
        root,
        "libero_goal",
        episodes=428,
        frames=52042,
        tasks=goal_tasks,
        episode_rows=goal_rows,
    )

    seen_tasks = [f"seen task {index:02d}" for index in range(73)]
    seen_rows = [
        {"episode_index": index, "tasks": [seen_tasks[index % 73]], "length": 5}
        for index in range(3921)
    ]
    _suite_meta(
        root,
        "libero_90",
        episodes=3921,
        frames=569249,
        tasks=seen_tasks,
        episode_rows=seen_rows,
    )
    return root
