"""Canonical LIBERO policy features. Hub SmolVLA features are not used as-is."""

from __future__ import annotations

from typing import Any, Mapping

POLICY_MAIN_IMAGE = "observation.images.image"
POLICY_WRIST_IMAGE = "observation.images.wrist_image"
POLICY_STATE = "observation.state"
POLICY_ACTION = "action"
POLICY_TASK = "task"

LIBERO_IMAGE_SHAPE = (3, 256, 256)
LIBERO_STATE_DIM = 8
LIBERO_ACTION_DIM = 7

LIBERO_INPUT_FEATURES: dict[str, dict[str, Any]] = {
    POLICY_STATE: {"type": "STATE", "shape": [LIBERO_STATE_DIM]},
    POLICY_MAIN_IMAGE: {"type": "VISUAL", "shape": list(LIBERO_IMAGE_SHAPE)},
    POLICY_WRIST_IMAGE: {"type": "VISUAL", "shape": list(LIBERO_IMAGE_SHAPE)},
}
LIBERO_OUTPUT_FEATURES: dict[str, dict[str, Any]] = {
    POLICY_ACTION: {"type": "ACTION", "shape": [LIBERO_ACTION_DIM]},
}


class FeatureContractError(ValueError):
    """Raised when policy features do not match the LIBERO contract."""


def _shape(value: Mapping[str, Any]) -> list[int]:
    raw = value.get("shape")
    if hasattr(raw, "tolist"):
        raw = raw.tolist()
    if not isinstance(raw, (list, tuple)):
        raise FeatureContractError(f"feature is missing a shape: {value}")
    return [int(item) for item in raw]


def _type_name(value: Mapping[str, Any]) -> str:
    raw = value.get("type")
    if raw is None:
        raise FeatureContractError(f"feature is missing a type: {value}")
    return str(getattr(raw, "name", raw)).upper()


def feature_snapshot(features: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    snapshot: dict[str, dict[str, Any]] = {}
    for key, feature in features.items():
        if hasattr(feature, "__dict__") and not isinstance(feature, dict):
            payload = {
                "type": _type_name({"type": getattr(feature, "type", None)}),
                "shape": _shape({"shape": getattr(feature, "shape", None)}),
            }
        elif isinstance(feature, Mapping):
            payload = {"type": _type_name(feature), "shape": _shape(feature)}
        else:
            raise FeatureContractError(f"unsupported feature value for {key!r}")
        snapshot[str(key)] = payload
    return snapshot


def assert_libero_policy_features(
    *,
    input_features: Mapping[str, Any],
    output_features: Mapping[str, Any],
) -> dict[str, Any]:
    """Fail closed unless LIBERO cameras/state/action replace hub SO100 features."""

    inputs = feature_snapshot(input_features)
    outputs = feature_snapshot(output_features)
    required_inputs = {
        POLICY_MAIN_IMAGE: {"type": "VISUAL", "shape": list(LIBERO_IMAGE_SHAPE)},
        POLICY_WRIST_IMAGE: {"type": "VISUAL", "shape": list(LIBERO_IMAGE_SHAPE)},
        POLICY_STATE: {"type": "STATE", "shape": [LIBERO_STATE_DIM]},
    }
    missing = [key for key in required_inputs if key not in inputs]
    if missing:
        raise FeatureContractError(f"missing LIBERO input features: {missing}")
    mismatches = {
        key: {"expected": expected, "observed": inputs[key]}
        for key, expected in required_inputs.items()
        if inputs[key] != expected
    }
    if mismatches:
        raise FeatureContractError(f"LIBERO input feature mismatch: {mismatches}")
    leftover_visual = sorted(
        key
        for key, feature in inputs.items()
        if feature["type"] == "VISUAL" and key not in required_inputs
    )
    if leftover_visual:
        raise FeatureContractError(
            "hub visual keys remain after LIBERO overlay: "
            f"{leftover_visual}; refusing SO100 camera names"
        )
    action = outputs.get(POLICY_ACTION)
    if action != {"type": "ACTION", "shape": [LIBERO_ACTION_DIM]}:
        raise FeatureContractError(
            f"expected action shape [{LIBERO_ACTION_DIM}], got {action}"
        )
    return {"input_features": inputs, "output_features": outputs}
