"""Pinned LeRobot pre/post processors. Eval must keep the action unnormalize path."""

from __future__ import annotations

from typing import Any


def make_policy_processors(
    policy: Any,
    stats: dict[str, Any],
    *,
    device: str | None = None,
) -> tuple[Any, Any]:
    """Return observation preprocessor and action postprocessor.

    The postprocessor is the MEAN_STD unnormalize + CPU move from pinned
    SmolVLA. Training may ignore it; live eval must apply it before gripper
    conversion. Identity stats remain smoke-only.
    """

    from lerobot.policies.factory import make_pre_post_processors

    try:
        if device is not None:
            preprocessor, postprocessor = make_pre_post_processors(
                policy.config,
                pretrained_path=None,
                dataset_stats=stats,
                preprocessor_overrides={"device_processor": {"device": str(device)}},
            )
        else:
            preprocessor, postprocessor = make_pre_post_processors(
                policy.config,
                pretrained_path=None,
                dataset_stats=stats,
            )
    except TypeError:
        preprocessor, postprocessor = make_pre_post_processors(
            policy.config,
            pretrained_path=None,
            dataset_stats=stats,
        )
    if postprocessor is None:
        raise RuntimeError(
            "SmolVLA action postprocessor is required; refusing to treat "
            "normalized select_action output as dataset-space"
        )
    return preprocessor, postprocessor
