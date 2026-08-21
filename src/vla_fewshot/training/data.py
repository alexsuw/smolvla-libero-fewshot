"""Episode selection and SmolVLA action-chunk delta timestamps. No GPU import."""

from __future__ import annotations

from pathlib import Path

from vla_fewshot.config import TrainConfig, TrainDatasetConfig
from vla_fewshot.data.layout import suite_root
from vla_fewshot.data.metadata import SuiteMetadata, load_suite_metadata


def action_delta_timestamps(*, fps: float, chunk_size: int) -> dict[str, list[float]]:
    """Pinned SmolVLAConfig.action_delta_indices = range(chunk_size)."""

    if fps <= 0:
        raise ValueError("fps must be positive")
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")
    return {"action": [index / float(fps) for index in range(chunk_size)]}


def select_episode_ids(meta: SuiteMetadata, dataset: TrainDatasetConfig) -> list[int] | None:
    """Return None to load every episode; otherwise a deterministic ID list."""

    if dataset.episodes == "all" and dataset.max_episodes is None and dataset.max_tasks is None:
        return None
    ids = sorted(int(row["episode_index"]) for row in meta.episodes)
    if dataset.max_tasks is not None:
        texts = meta.unique_task_texts[: dataset.max_tasks]
        allowed: set[int] = set()
        for text in texts:
            allowed.update(meta.episode_ids_for_task(text))
        ids = [item for item in ids if item in allowed]
    if dataset.max_episodes is not None:
        ids = ids[: dataset.max_episodes]
    if not ids:
        raise ValueError("episode filter selected zero episodes")
    return ids


def suite_dataset_root(revision_root: Path, suite: str) -> Path:
    return suite_root(revision_root, suite)


def assert_suite_videos(suite_dir: Path) -> None:
    videos = suite_dir / "videos"
    if not videos.is_dir() or not any(videos.rglob("*.mp4")):
        raise FileNotFoundError(
            f"dataset videos missing under {videos}; "
            "re-run scripts/download_dataset.py --include-videos "
            f"--suite {suite_dir.name}. no GPU training was started."
        )


def load_suite_for_train(revision_root: Path, config: TrainConfig) -> tuple[SuiteMetadata, list[int] | None]:
    meta = load_suite_metadata(revision_root, config.dataset.suite)
    return meta, select_episode_ids(meta, config.dataset)
