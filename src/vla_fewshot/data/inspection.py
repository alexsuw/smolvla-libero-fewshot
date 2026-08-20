"""Machine-readable dataset inspection without video decode."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from vla_fewshot.data.expected import (
    EXPECTED_FEATURES,
    EXPECTED_SUITE_COUNTS,
    SEEN_SUITE,
    TARGET_SUITE,
    TARGET_TASKS,
)
from vla_fewshot.data.metadata import SuiteMetadata, load_suite_metadata
from vla_fewshot.data.splits import (
    TargetSplits,
    load_target_splits,
    verify_splits_against_metadata,
)
from vla_fewshot.data.task_text import normalize_task_text


def _check(name: str, passed: bool, detail: str) -> dict[str, Any]:
    return {"name": name, "status": "pass" if passed else "fail", "detail": detail}


def _feature_checks(suite: SuiteMetadata) -> list[dict[str, Any]]:
    summary = suite.feature_summary()
    checks: list[dict[str, Any]] = []
    for name, expected in EXPECTED_FEATURES.items():
        observed = summary.get(name)
        if observed is None:
            checks.append(_check(f"{suite.suite}:feature:{name}", False, "missing"))
            continue
        shape_ok = expected.get("shape") is None or observed.get("shape") == expected["shape"]
        dtype_ok = (
            expected.get("dtype") is None
            or observed.get("dtype") == expected["dtype"]
            or (
                name.startswith("observation.images.")
                and observed.get("dtype") in {"video", "image", expected.get("dtype")}
            )
        )
        checks.append(
            _check(
                f"{suite.suite}:feature:{name}",
                bool(shape_ok and dtype_ok),
                f"observed={observed}",
            )
        )
    return checks


def _suite_count_checks(suite: SuiteMetadata) -> list[dict[str, Any]]:
    expected = EXPECTED_SUITE_COUNTS[suite.suite]
    unique = suite.unique_task_texts
    return [
        _check(
            f"{suite.suite}:episodes",
            suite.total_episodes == expected["episodes"],
            f"expected={expected['episodes']}, observed={suite.total_episodes}",
        ),
        _check(
            f"{suite.suite}:frames",
            suite.total_frames == expected["frames"],
            f"expected={expected['frames']}, observed={suite.total_frames}",
        ),
        _check(
            f"{suite.suite}:unique_task_texts",
            len(unique) == expected["unique_task_texts"],
            f"expected={expected['unique_task_texts']}, observed={len(unique)}",
        ),
        _check(
            f"{suite.suite}:fps",
            suite.fps == expected["fps"],
            f"expected={expected['fps']}, observed={suite.fps}",
        ),
        _check(
            f"{suite.suite}:episode_parquet_rows",
            len(suite.episodes) == suite.total_episodes,
            f"info={suite.total_episodes}, parquet={len(suite.episodes)}",
        ),
    ]


def _target_checks(
    target: SuiteMetadata, splits: TargetSplits
) -> tuple[list[dict[str, Any]], dict[str, list[int]]]:
    checks: list[dict[str, Any]] = []
    episode_map: dict[str, list[int]] = {}
    for slug, spec in TARGET_TASKS.items():
        task_text = str(spec["task_text"])
        ids = target.episode_ids_for_task(task_text)
        episode_map[slug] = ids
        task_index = target.task_index_for_text(task_text)
        checks.append(
            _check(
                f"target:{slug}:index",
                task_index == spec["task_index"],
                f"expected={spec['task_index']}, observed={task_index}",
            )
        )
        checks.append(
            _check(
                f"target:{slug}:count",
                len(ids) == spec["available_count"],
                f"expected={spec['available_count']}, observed={len(ids)}",
            )
        )
        tracked = splits.tasks[slug]
        checks.append(
            _check(
                f"target:{slug}:first_25",
                ids[:25] == tracked.episode_ids_first_25,
                f"expected={tracked.episode_ids_first_25}, observed={ids[:25]}",
            )
        )
        checks.append(
            _check(
                f"target:{slug}:nested_prefixes",
                tracked.ids_for_budget(5) == tracked.ids_for_budget(10)[:5]
                and tracked.ids_for_budget(10) == tracked.ids_for_budget(25)[:10],
                "N=5/10/25 prefixes are nested",
            )
        )
    return checks, episode_map


def inspect_revision(
    *,
    revision_root: Path,
    repo_id: str,
    revision: str,
    splits: TargetSplits,
) -> dict[str, Any]:
    seen = load_suite_metadata(revision_root, SEEN_SUITE)
    target = load_suite_metadata(revision_root, TARGET_SUITE)
    checks = []
    checks.extend(_suite_count_checks(seen))
    checks.extend(_suite_count_checks(target))
    checks.extend(_feature_checks(seen))
    checks.extend(_feature_checks(target))
    target_checks, episode_map = _target_checks(target, splits)
    checks.extend(target_checks)
    try:
        verify_splits_against_metadata(splits, episode_map)
        checks.append(
            _check("split:tracked_prefixes", True, "tracked first-25 IDs match metadata")
        )
    except ValueError as error:
        checks.append(_check("split:tracked_prefixes", False, str(error)))
    for slug, spec in TARGET_TASKS.items():
        present = seen.contains_task_text(str(spec["task_text"]))
        checks.append(
            _check(
                f"leakage:{slug}:in_seen",
                not present,
                f"target text present in {SEEN_SUITE}={present}",
            )
        )

    suites = {}
    for suite in (seen, target):
        features = suite.feature_summary()
        catalog = []
        for text in suite.unique_task_texts:
            ids = suite.episode_ids_for_task(text)
            catalog.append(
                {
                    "task_index": suite.task_index_for_text(text),
                    "task_text": text,
                    "episode_count": len(ids),
                    "episode_ids_first_25": ids[:25],
                }
            )
        suites[suite.suite] = {
            "episodes": suite.total_episodes,
            "frames": suite.total_frames,
            "unique_task_texts": suite.unique_task_texts,
            "unique_task_text_count": len(suite.unique_task_texts),
            "fps": suite.fps,
            "features": features,
            "video": {
                name: item.get("video")
                for name, item in features.items()
                if item.get("video")
            },
            "task_catalog": catalog,
            "action_state_stats": suite.action_state_stats(),
            "gripper_stats": suite.gripper_stats(),
            "episode_parquet_rows": len(suite.episodes),
            "task_parquet_rows": len(suite.tasks),
        }

    failed = [check for check in checks if check["status"] != "pass"]
    return {
        "schema_version": 1,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "dataset_repo_id": repo_id,
        "dataset_revision": revision,
        "local_root": str(revision_root),
        "videos_decoded": False,
        "suites": suites,
        "target_episode_ids": episode_map,
        "normalized_targets": {
            slug: {
                "raw": spec["task_text"],
                "normalized": normalize_task_text(str(spec["task_text"])),
            }
            for slug, spec in TARGET_TASKS.items()
        },
        "checks": checks,
        "acceptance_complete": not failed,
    }


def inspection_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Dataset inspection",
        "",
        f"- Repo: `{report['dataset_repo_id']}`",
        f"- Revision: `{report['dataset_revision']}`",
        f"- Local root: `{report['local_root']}`",
        f"- Videos decoded: `{report['videos_decoded']}`",
        f"- Acceptance complete: `{report['acceptance_complete']}`",
        "",
    ]
    for suite, payload in report["suites"].items():
        lines.extend(
            [
                f"## {suite}",
                "",
                f"- Episodes: `{payload['episodes']}`",
                f"- Frames: `{payload['frames']}`",
                f"- Unique task texts: `{payload['unique_task_text_count']}`",
                f"- FPS: `{payload['fps']}`",
                f"- Episode parquet rows: `{payload['episode_parquet_rows']}`",
                "",
            ]
        )
        video = payload.get("video") or {}
        if video:
            lines.append(f"- Video metadata (no decode): `{video}`")
            lines.append("")
        gripper = payload.get("gripper_stats") or {}
        if gripper:
            lines.append(f"- Gripper stats (dataset space): `{gripper}`")
            lines.append("")
    lines.extend(["## Checks", ""])
    for check in report["checks"]:
        lines.append(
            f"- `{check['status']}` **{check['name']}**: {check['detail']}"
        )
    return "\n".join(lines) + "\n"


def inspect_and_write(
    *,
    revision_root: Path,
    repo_id: str,
    revision: str,
    splits_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    from vla_fewshot.reproducibility import atomic_write_json, atomic_write_text

    splits = load_target_splits(splits_path)
    report = inspect_revision(
        revision_root=revision_root,
        repo_id=repo_id,
        revision=revision,
        splits=splits,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_json(output_dir / "inspection.json", report, overwrite=True)
    atomic_write_text(
        output_dir / "inspection.md",
        inspection_markdown(report),
        overwrite=True,
    )
    atomic_write_json(
        output_dir / "target_episode_ids.json",
        report["target_episode_ids"],
        overwrite=True,
    )
    return report
