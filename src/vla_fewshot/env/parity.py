"""Observation parity evidence. Chosen transform is identity after LeRobot rot180."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Mapping

from vla_fewshot.env.observation_adapter import (
    CAMERA_MANIFEST,
    IDENTITY,
    apply_canonical_image_keys,
    apply_hwc_transform,
    assert_single_orientation_processor,
    candidate_transforms,
    env_pixels_to_policy,
)
from vla_fewshot.env.replay import _write_ppm
from vla_fewshot.reproducibility import atomic_write_json, atomic_write_text


CHOSEN_PROJECT_TRANSFORM = IDENTITY
LEROBOT_PROCESSOR_ROT180 = True


def frozen_orientation_contract() -> dict[str, Any]:
    assert_single_orientation_processor(
        lerobot_libero_processor_rot180=LEROBOT_PROCESSOR_ROT180,
        project_transform=CHOSEN_PROJECT_TRANSFORM,
    )
    return {
        "lerobot_libero_processor_rot180": LEROBOT_PROCESSOR_ROT180,
        "project_transform": CHOSEN_PROJECT_TRANSFORM,
        "note": (
            "Pinned LiberoProcessorStep flips H and W (rot180). "
            "The project adapter only remaps image2 -> wrist_image."
        ),
    }


def write_parity_bundle(
    *,
    output_dir: Path,
    dataset_main: Any | None = None,
    dataset_wrist: Any | None = None,
    env_main: Any | None = None,
    env_wrist: Any | None = None,
    env_main_processed: Any | None = None,
    env_wrist_processed: Any | None = None,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Save side-by-side frames and candidate transforms. Does not decode a video corpus."""

    assert_single_orientation_processor(
        lerobot_libero_processor_rot180=LEROBOT_PROCESSOR_ROT180,
        project_transform=CHOSEN_PROJECT_TRANSFORM,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    frames_dir = output_dir / "frames"
    frames_dir.mkdir(exist_ok=True)
    named = {
        "dataset_main": dataset_main,
        "dataset_wrist": dataset_wrist,
        "env_main_raw": env_main,
        "env_wrist_raw": env_wrist,
        "env_main_after_lerobot_processor": env_main_processed,
        "env_wrist_after_lerobot_processor": env_wrist_processed,
    }
    written = []
    for name, frame in named.items():
        if frame is None:
            continue
        path = frames_dir / f"{name}.ppm"
        _write_ppm(path, frame)
        written.append(path.name)
        if name.startswith("dataset_") and frame is not None:
            for transform in candidate_transforms():
                variant = apply_hwc_transform(frame, transform)
                variant_path = frames_dir / f"{name}_{transform}.ppm"
                _write_ppm(variant_path, variant)
                written.append(variant_path.name)

    report = {
        "schema_version": 1,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "videos_decoded_corpus": False,
        "camera_manifest": CAMERA_MANIFEST,
        "orientation": frozen_orientation_contract(),
        "frames": written,
        "human_confirmation_required_for_env_frames": env_main is None,
        "extra": dict(extra or {}),
    }
    atomic_write_json(output_dir / "parity.json", report, overwrite=True)
    atomic_write_text(
        output_dir / "parity.md",
        _parity_markdown(report),
        overwrite=True,
    )
    return report


def canonical_policy_observation_from_env(
    *,
    pixels: Mapping[str, Any],
    processed_observation: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if processed_observation is not None:
        return apply_canonical_image_keys(processed_observation)
    return env_pixels_to_policy(pixels)


def _parity_markdown(report: dict[str, Any]) -> str:
    orientation = report["orientation"]
    lines = [
        "# Observation parity",
        "",
        f"- LeRobot LiberoProcessorStep rot180: `{orientation['lerobot_libero_processor_rot180']}`",
        f"- Project transform: `{orientation['project_transform']}`",
        f"- Human confirmation pending env frames: `{report['human_confirmation_required_for_env_frames']}`",
        "",
        orientation["note"],
        "",
        "## Camera map",
        "",
        "```json",
        str(report["camera_manifest"]),
        "```",
        "",
        "## Frames",
        "",
    ]
    for name in report["frames"]:
        lines.append(f"- `{name}`")
    return "\n".join(lines) + "\n"
