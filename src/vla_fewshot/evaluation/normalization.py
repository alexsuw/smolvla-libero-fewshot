"""Train MEAN_STD and eval unnormalization must be hash-identical."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

from vla_fewshot.config import EvalConfig, TrainConfig
from vla_fewshot.data.layout import dataset_revision_root, suite_root
from vla_fewshot.data.metadata import load_suite_metadata
from vla_fewshot.data.splits import load_target_splits
from vla_fewshot.training.anchored import uses_frozen_seen_stats
from vla_fewshot.training.baseline import episode_ids_for_cell
from vla_fewshot.training.data import action_delta_timestamps
from vla_fewshot.training.stats import (
    find_normalization_sidecar,
    load_normalization_stats,
    overlay_dataset_state_action_stats,
    stats_digest,
)

TARGET_CHUNK_SIZE = 50
NormSource = Literal[
    "suite",
    "sidecar",
    "subset",
    "sidecar+subset",
    "sidecar+suite",
]


class NormalizationError(RuntimeError):
    """Train/eval MEAN_STD provenance does not match."""


def normalization_stats_suite(
    eval_config: EvalConfig,
    train_config: TrainConfig,
) -> str:
    """Use the statistics that trained the evaluated policy.

    A frozen seen policy must never be normalized with held-out target-suite
    statistics. Target checkpoints keep their target training suite here;
    the subset overlay is applied on top of that suite `stats.json`.
    """

    if eval_config.stage in {"zero_shot", "language_control"}:
        if train_config.stage != "seen":
            raise RuntimeError(
                f"{eval_config.stage} requires a seen train config for normalization; "
                f"got stage={train_config.stage}"
            )
        if train_config.dataset.suite != eval_config.dataset.suite_seen:
            raise RuntimeError(
                f"{eval_config.stage} normalization suite "
                f"{train_config.dataset.suite!r} != configured seen suite "
                f"{eval_config.dataset.suite_seen!r}"
            )
        return train_config.dataset.suite
    if eval_config.stage == "seen_probe":
        return eval_config.dataset.suite_seen
    if uses_frozen_seen_stats(train_config):
        if train_config.normalization is None:
            raise NormalizationError("frozen-stat method is missing normalization config")
        return train_config.normalization.suite
    return train_config.dataset.suite


def uses_suite_stats_only(eval_config: EvalConfig) -> bool:
    return eval_config.stage in {"zero_shot", "language_control", "seen_probe"}


def _canonical_digest(stats: dict[str, Any]) -> str:
    return stats_digest(json.loads(json.dumps(stats, sort_keys=True)))


def choose_normalization_stats(
    *,
    use_suite_only: bool,
    suite: dict[str, Any],
    sidecar: dict[str, Any] | None,
    subset: dict[str, Any] | None,
) -> tuple[dict[str, Any], NormSource]:
    """Pick eval stats. Sidecar and recomputed subset must match when both exist."""

    if use_suite_only:
        return suite, "suite"
    if sidecar is not None and subset is not None:
        if _canonical_digest(sidecar) != _canonical_digest(subset):
            raise NormalizationError(
                "normalization sidecar hash does not match the recomputed "
                "subset overlay. refusing to evaluate with mismatched MEAN_STD"
            )
        return sidecar, "sidecar+subset"
    if sidecar is not None:
        return sidecar, "sidecar"
    if subset is not None:
        return subset, "subset"
    raise NormalizationError(
        "target eval requires a normalization sidecar or a subset overlay; "
        "suite-wide libero_goal stats are not the training MEAN_STD"
    )


def recompute_subset_overlay_stats(
    *,
    datasets_dir: Path,
    repo_id: str,
    revision: str,
    suite: str,
    episode_ids: list[int],
    chunk_size: int = TARGET_CHUNK_SIZE,
) -> dict[str, Any]:
    """Rebuild the trainer overlay from the same LeRobot items."""

    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    if not episode_ids:
        raise NormalizationError("subset overlay requires selected episode IDs")
    revision_root = dataset_revision_root(datasets_dir, repo_id, revision)
    meta = load_suite_metadata(revision_root, suite)
    if not meta.stats:
        raise NormalizationError(f"missing stats.json for {suite}")
    fps = float(meta.fps or 20)
    dataset = LeRobotDataset(
        repo_id=repo_id,
        root=suite_root(revision_root, suite),
        episodes=episode_ids,
        revision=revision,
        download_videos=False,
        delta_timestamps=action_delta_timestamps(fps=fps, chunk_size=chunk_size),
    )
    return overlay_dataset_state_action_stats(meta.stats, dataset)


def resolve_live_normalization(
    *,
    eval_config: EvalConfig,
    train_config: TrainConfig,
    checkpoint: Path,
    datasets_dir: Path,
    run_dir: Path | None = None,
    task_slug: str | None = None,
    n_demos: int | None = None,
    split_path: Path | None = None,
    chunk_size: int = TARGET_CHUNK_SIZE,
) -> tuple[dict[str, Any], str, str, NormSource]:
    """Return (stats, suite_name, digest, source) for live processors."""

    suite_name = normalization_stats_suite(eval_config, train_config)
    meta = load_suite_metadata(
        dataset_revision_root(
            datasets_dir, eval_config.dataset.repo_id, eval_config.dataset.revision
        ),
        suite_name,
    )
    if not meta.stats:
        raise NormalizationError(
            f"missing stats.json for {suite_name}; identity stats are forbidden for eval"
        )
    suite = meta.stats
    if uses_frozen_seen_stats(train_config):
        if train_config.normalization is None:
            raise NormalizationError("frozen-stat method is missing normalization config")
        sidecar = find_normalization_sidecar(checkpoint, run_dir)
        if sidecar is None:
            raise NormalizationError(
                "frozen-stat target checkpoint requires normalization_stats.json"
            )
        sidecar_stats = load_normalization_stats(sidecar)
        suite_digest = stats_digest(suite)
        sidecar_digest = stats_digest(sidecar_stats)
        expected_digest = train_config.normalization.expected_sha256
        if suite_digest != expected_digest or sidecar_digest != expected_digest:
            raise NormalizationError(
                "frozen libero_90 normalization hash mismatch: "
                f"suite={suite_digest} sidecar={sidecar_digest} "
                f"expected={expected_digest}"
            )
        return sidecar_stats, suite_name, sidecar_digest, "sidecar+suite"

    sidecar_stats: dict[str, Any] | None = None
    subset_stats: dict[str, Any] | None = None
    if not uses_suite_stats_only(eval_config):
        sidecar = find_normalization_sidecar(checkpoint, run_dir)
        if sidecar is not None:
            sidecar_stats = load_normalization_stats(sidecar)
        if n_demos not in (None, 0) and task_slug and split_path is not None:
            splits = load_target_splits(split_path)
            episode_ids = episode_ids_for_cell(splits, task_slug=task_slug, n_demos=n_demos)
            subset_stats = recompute_subset_overlay_stats(
                datasets_dir=datasets_dir,
                repo_id=train_config.dataset.repo_id,
                revision=train_config.dataset.revision,
                suite=train_config.dataset.suite,
                episode_ids=episode_ids,
                chunk_size=chunk_size,
            )
    stats, source = choose_normalization_stats(
        use_suite_only=uses_suite_stats_only(eval_config),
        suite=suite,
        sidecar=sidecar_stats,
        subset=subset_stats,
    )
    return stats, suite_name, stats_digest(stats), source
