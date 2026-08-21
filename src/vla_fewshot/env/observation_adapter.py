"""Explicit camera/state mapping. Geometry lives in exactly one processor."""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence


ENV_MAIN_KEY = "image"
ENV_WRIST_KEY = "image2"
POLICY_MAIN_KEY = "observation.images.image"
POLICY_WRIST_KEY = "observation.images.wrist_image"
DATASET_WRIST_KEY = "observation.images.wrist_image"
LEROBOT_WRIST_KEY = "observation.images.image2"

CAMERA_MANIFEST = {
    "main": {
        "source": "agentview_image",
        "env_raw_key": ENV_MAIN_KEY,
        "dataset_key": POLICY_MAIN_KEY,
        "policy_key": POLICY_MAIN_KEY,
    },
    "wrist": {
        "source": "robot0_eye_in_hand_image",
        "env_raw_key": ENV_WRIST_KEY,
        "dataset_key": DATASET_WRIST_KEY,
        "policy_key": POLICY_WRIST_KEY,
        "lerobot_key": LEROBOT_WRIST_KEY,
    },
}

IDENTITY = "identity"
ROT180 = "rot180"
FLIP_H = "flip_h"
FLIP_W = "flip_w"
ALLOWED_PROJECT_TRANSFORMS = {IDENTITY}


class OrientationError(ValueError):
    """Raised when two processors would both rotate or flip images."""


def assert_single_orientation_processor(
    *,
    lerobot_libero_processor_rot180: bool,
    project_transform: str,
) -> None:
    """Fail closed on a double 180° / flip configuration."""

    project_changes_geometry = project_transform != IDENTITY
    if lerobot_libero_processor_rot180 and project_changes_geometry:
        raise OrientationError(
            "LiberoProcessorStep already applies rot180 (flip H and W); "
            f"project_transform={project_transform!r} would double-transform images"
        )
    if project_transform not in ALLOWED_PROJECT_TRANSFORMS:
        raise OrientationError(
            f"unsupported project_transform {project_transform!r}; "
            f"allowed={sorted(ALLOWED_PROJECT_TRANSFORMS)}"
        )


def apply_hwc_transform(image: Sequence[Sequence[Sequence[Any]]], transform: str) -> list:
    """Apply a named transform to an HWC image nested-sequence."""

    rows = [list(row) for row in image]
    if transform == IDENTITY:
        return rows
    if transform == FLIP_H:
        return list(reversed(rows))
    if transform == FLIP_W:
        return [list(reversed(row)) for row in rows]
    if transform == ROT180:
        return [list(reversed(row)) for row in reversed(rows)]
    raise OrientationError(f"unknown image transform {transform!r}")


def candidate_transforms() -> tuple[str, ...]:
    return (IDENTITY, ROT180, FLIP_H, FLIP_W)


def apply_canonical_image_keys(observation: Mapping[str, Any]) -> dict[str, Any]:
    """Map LeRobot env wrist key `image2` onto the dataset/policy `wrist_image`."""

    mapped = dict(observation)
    has_lerobot_wrist = LEROBOT_WRIST_KEY in mapped
    has_canonical_wrist = POLICY_WRIST_KEY in mapped
    if has_lerobot_wrist and has_canonical_wrist:
        raise OrientationError(
            "observation contains both image2 and wrist_image; refusing double alias"
        )
    if has_lerobot_wrist:
        mapped[POLICY_WRIST_KEY] = mapped.pop(LEROBOT_WRIST_KEY)
    if POLICY_MAIN_KEY not in mapped or POLICY_WRIST_KEY not in mapped:
        raise KeyError(
            "canonical observation requires "
            f"{POLICY_MAIN_KEY} and {POLICY_WRIST_KEY}; got {sorted(mapped)}"
        )
    return mapped


def env_pixels_to_policy(pixels: Mapping[str, Any]) -> dict[str, Any]:
    if ENV_MAIN_KEY not in pixels or ENV_WRIST_KEY not in pixels:
        raise KeyError(
            f"env pixels must contain {ENV_MAIN_KEY!r} and {ENV_WRIST_KEY!r}; "
            f"got {sorted(pixels)}"
        )
    return apply_canonical_image_keys(
        {
            POLICY_MAIN_KEY: pixels[ENV_MAIN_KEY],
            LEROBOT_WRIST_KEY: pixels[ENV_WRIST_KEY],
        }
    )


def quat_xyzw_to_axis_angle(quat: Sequence[float]) -> list[float]:
    """Match pinned LeRobot LiberoProcessorStep._quat2axisangle for one sample."""

    if len(quat) != 4:
        raise ValueError(f"quaternion must have 4 values, got {len(quat)}")
    x, y, z, w = (float(item) for item in quat)
    if not all(math.isfinite(item) for item in (x, y, z, w)):
        raise ValueError("quaternion is not finite")
    w = max(-1.0, min(1.0, w))
    den = math.sqrt(max(0.0, 1.0 - w * w))
    if den <= 1e-10:
        return [0.0, 0.0, 0.0]
    angle = 2.0 * math.acos(w)
    return [x / den * angle, y / den * angle, z / den * angle]


def flatten_libero_robot_state(robot_state: Mapping[str, Any]) -> list[float]:
    """8D [eef_pos(3), axis_angle(3), gripper_qpos(2)] in dataset/policy order."""

    try:
        pos = [float(item) for item in robot_state["eef"]["pos"]]
        quat = [float(item) for item in robot_state["eef"]["quat"]]
        gripper = [float(item) for item in robot_state["gripper"]["qpos"]]
    except (KeyError, TypeError) as error:
        raise KeyError("robot_state must contain eef.pos, eef.quat, gripper.qpos") from error
    if len(pos) != 3 or len(gripper) != 2:
        raise ValueError(f"unexpected state sizes pos={len(pos)} gripper={len(gripper)}")
    state = pos + quat_xyzw_to_axis_angle(quat) + gripper
    if len(state) != 8:
        raise ValueError(f"expected 8D state, got {len(state)}")
    if not all(math.isfinite(item) for item in state):
        raise ValueError("flattened robot state is not finite")
    return state
