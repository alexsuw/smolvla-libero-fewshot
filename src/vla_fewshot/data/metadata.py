"""Read LeRobot v3 suite metadata without decoding videos."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from vla_fewshot.data.layout import metadata_root
from vla_fewshot.data.task_text import normalize_task_text, task_text_matches


_TASK_TEXT_KEYS = ("task", "task_text", "__index_level_0__")


def _require_pyarrow():
    try:
        import pyarrow.parquet as pq
    except ImportError as error:
        raise RuntimeError(
            "pyarrow is required for dataset metadata. "
            "Install with: uv sync --frozen --extra data"
        ) from error
    return pq


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def read_parquet_rows(paths: Iterable[Path]) -> list[dict[str, Any]]:
    pq = _require_pyarrow()
    files = sorted(paths)
    if not files:
        return []
    rows: list[dict[str, Any]] = []
    for path in files:
        table = pq.read_table(path)
        for batch in table.to_batches():
            columns = batch.to_pydict()
            length = batch.num_rows
            for index in range(length):
                rows.append({name: values[index] for name, values in columns.items()})
    return rows


def _feature_shape(feature: dict[str, Any]) -> list[int] | None:
    shape = feature.get("shape")
    if isinstance(shape, list) and all(isinstance(item, int) for item in shape):
        return shape
    return None


def _feature_dtype(feature: dict[str, Any]) -> str | None:
    dtype = feature.get("dtype")
    return str(dtype) if dtype is not None else None


def _video_info(feature: dict[str, Any]) -> dict[str, Any]:
    info = feature.get("info")
    if not isinstance(info, dict):
        info = feature.get("video_info")
    if not isinstance(info, dict):
        return {}
    return {
        "codec": info.get("video.codec") or info.get("codec"),
        "pix_fmt": info.get("video.pix_fmt") or info.get("pix_fmt"),
        "fps": info.get("video.fps") or info.get("fps"),
        "is_depth_map": info.get("video.is_depth_map"),
    }


def _row_task_text(row: dict[str, Any]) -> str:
    """Read a task string from LeRobot v3 tasks.parquet rows.

    Pinned `nvidia/LIBERO_LeRobot_v3` stores the text in pandas leftover
    column `__index_level_0__` rather than `task`.
    """

    for key in _TASK_TEXT_KEYS:
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return value
    return ""


def _task_texts_from_episode(row: dict[str, Any]) -> list[str]:
    raw = row.get("tasks", row.get("task"))
    if raw is None:
        return []
    if isinstance(raw, str):
        return [normalize_task_text(raw)]
    if isinstance(raw, list):
        return [normalize_task_text(str(item)) for item in raw]
    return [normalize_task_text(str(raw))]


def _vector_stats(stats: dict[str, Any], key: str) -> dict[str, Any] | None:
    payload = stats.get(key)
    if not isinstance(payload, dict):
        return None
    extracted: dict[str, Any] = {}
    for name in ("min", "max", "mean", "std"):
        if name in payload:
            extracted[name] = payload[name]
    return extracted or None


@dataclass(frozen=True)
class SuiteMetadata:
    suite: str
    root: Path
    info: dict[str, Any]
    stats: dict[str, Any]
    tasks: list[dict[str, Any]]
    episodes: list[dict[str, Any]]

    @property
    def fps(self) -> int | float | None:
        return self.info.get("fps")

    @property
    def total_episodes(self) -> int:
        if "total_episodes" in self.info:
            return int(self.info["total_episodes"])
        return len(self.episodes)

    @property
    def total_frames(self) -> int:
        if "total_frames" in self.info:
            return int(self.info["total_frames"])
        return int(sum(int(row.get("length") or 0) for row in self.episodes))

    @property
    def unique_task_texts(self) -> list[str]:
        texts: list[str] = []
        seen: set[str] = set()
        for row in self.tasks:
            text = normalize_task_text(_row_task_text(row))
            if text and text not in seen:
                seen.add(text)
                texts.append(text)
        if texts:
            return texts
        for episode in self.episodes:
            for text in _task_texts_from_episode(episode):
                if text not in seen:
                    seen.add(text)
                    texts.append(text)
        return texts

    @property
    def features(self) -> dict[str, Any]:
        raw = self.info.get("features")
        return raw if isinstance(raw, dict) else {}

    def feature_summary(self) -> dict[str, Any]:
        summary: dict[str, Any] = {}
        for name, feature in self.features.items():
            if not isinstance(feature, dict):
                continue
            item: dict[str, Any] = {
                "dtype": _feature_dtype(feature),
                "shape": _feature_shape(feature),
            }
            if name.startswith("observation.images.") or feature.get("dtype") == "video":
                item["video"] = _video_info(feature)
            summary[name] = item
        return summary

    def action_state_stats(self) -> dict[str, Any]:
        return {
            "action": _vector_stats(self.stats, "action"),
            "observation.state": _vector_stats(self.stats, "observation.state"),
        }

    def gripper_stats(self) -> dict[str, Any] | None:
        action = _vector_stats(self.stats, "action")
        if not action:
            return None
        extracted: dict[str, Any] = {}
        for name, values in action.items():
            if isinstance(values, list) and values:
                extracted[name] = values[-1]
        return extracted or None

    def task_index_for_text(self, task_text: str) -> int | None:
        for row in self.tasks:
            text = _row_task_text(row)
            if task_text_matches(text, task_text):
                if "task_index" in row:
                    return int(row["task_index"])
                if "index" in row:
                    return int(row["index"])
        return None

    def episode_ids_for_task(self, task_text: str) -> list[int]:
        ids: list[int] = []
        for row in self.episodes:
            if any(task_text_matches(text, task_text) for text in _task_texts_from_episode(row)):
                ids.append(int(row["episode_index"]))
        return sorted(ids)

    def contains_task_text(self, task_text: str) -> bool:
        return any(task_text_matches(text, task_text) for text in self.unique_task_texts)


def load_suite_metadata(revision_root: Path, suite: str) -> SuiteMetadata:
    root = metadata_root(revision_root, suite)
    info_path = root / "info.json"
    stats_path = root / "stats.json"
    if not info_path.exists():
        raise FileNotFoundError(f"missing {info_path}; run metadata download first")
    tasks_path = root / "tasks.parquet"
    episode_files = sorted((root / "episodes").rglob("*.parquet"))
    if not tasks_path.exists():
        raise FileNotFoundError(f"missing {tasks_path}")
    if not episode_files:
        raise FileNotFoundError(f"missing episode parquet under {root / 'episodes'}")
    return SuiteMetadata(
        suite=suite,
        root=root,
        info=load_json(info_path),
        stats=load_json(stats_path) if stats_path.exists() else {},
        tasks=read_parquet_rows([tasks_path]),
        episodes=read_parquet_rows(episode_files),
    )
