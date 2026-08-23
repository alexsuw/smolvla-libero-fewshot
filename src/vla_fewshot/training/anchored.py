"""Matched frozen-stat full fine-tuning and L2-SP contracts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from vla_fewshot.calibration import assert_train_matches_calibration, load_calibration
from vla_fewshot.config import TrainConfig
from vla_fewshot.training.trainer import TrainError

FROZEN_STATS_METHODS = frozenset({"frozen_stats", "anchored_l2sp"})
FROZEN_STATS_SUITE = "libero_90"
FROZEN_STATS_SHA256 = "b159b6fed3e52edf25bd39b377dd64940221b7a030362daf7f726b1c2ecb30cf"
PREREGISTERED_L2SP_STRENGTH = 1.0e-2


@dataclass(frozen=True)
class L2SPAnchor:
    """FP32 copies of every trainable parameter at the frozen seen origin."""

    parameters: dict[str, Any]
    parameter_count: int


def uses_frozen_seen_stats(config: TrainConfig) -> bool:
    return config.method in FROZEN_STATS_METHODS


def assert_frozen_stats_train_config(config: TrainConfig) -> None:
    """Validate the matched full-FT recipe before model or data loading."""

    if config.stage != "target" or config.method not in FROZEN_STATS_METHODS:
        raise TrainError(
            "frozen-stat FT requires stage=target and a registered frozen-stat method. "
            "no GPU training was started."
        )
    if config.peft is not None:
        raise TrainError("frozen-stat full FT forbids PEFT. no GPU training was started.")
    if config.replay is not None:
        raise TrainError("frozen-stat full FT forbids replay. no GPU training was started.")
    if config.dataset.suite != "libero_goal":
        raise TrainError("frozen-stat full FT trains only selected libero_goal episodes")
    if not config.training.sample_with_replacement:
        raise TrainError("frozen-stat full FT requires sample_with_replacement: true")

    scope = config.trainable_scope
    if (
        not scope.freeze_vlm_backbone
        or not scope.freeze_vision_encoder
        or not scope.train_action_expert
        or not scope.train_state_projection
        or not scope.train_action_projections
        or not scope.strict_allowlist
    ):
        raise TrainError(
            "frozen-stat full FT must train exactly Action Expert plus state/action "
            "projections with VLM and vision frozen"
        )

    normalization = config.normalization
    if (
        normalization is None
        or normalization.source != "libero_90_suite"
        or normalization.suite != FROZEN_STATS_SUITE
        or normalization.expected_sha256 != FROZEN_STATS_SHA256
    ):
        raise TrainError(
            "frozen-stat methods require the preregistered canonical libero_90 stats hash. "
            "no GPU training was started."
        )

    if config.method == "frozen_stats":
        if config.l2sp is not None:
            raise TrainError("Frozen-Stats FT forbids an L2-SP regularizer")
    else:
        l2sp = config.l2sp
        if (
            l2sp is None
            or not l2sp.enabled
            or l2sp.reduction != "sum"
            or l2sp.anchor_dtype != "fp32"
            or abs(l2sp.strength - PREREGISTERED_L2SP_STRENGTH) > 1e-12
        ):
            raise TrainError(
                "Anchored FT requires preregistered FP32 raw-sum L2-SP with lambda=1e-2"
            )

    assert_train_matches_calibration(config, load_calibration())


def capture_l2sp_anchor(policy: Any) -> L2SPAnchor:
    """Capture the exact requires-grad set before any target optimizer update."""

    import torch

    trainable = [
        (name, parameter)
        for name, parameter in policy.named_parameters()
        if parameter.requires_grad
    ]
    if not trainable:
        raise TrainError("cannot create L2-SP anchor with zero trainable parameters")
    references = {
        name: parameter.detach().to(dtype=torch.float32).clone()
        for name, parameter in trainable
    }
    return L2SPAnchor(
        parameters=references,
        parameter_count=sum(int(parameter.numel()) for _, parameter in trainable),
    )


def l2sp_raw_penalty(policy: Any, anchor: L2SPAnchor) -> Any:
    """Return sum(||theta - theta_seen||^2) in FP32, preserving gradients."""

    import torch

    current = {
        name: parameter
        for name, parameter in policy.named_parameters()
        if parameter.requires_grad
    }
    expected_names = set(anchor.parameters)
    current_names = set(current)
    if current_names != expected_names:
        missing = sorted(expected_names - current_names)
        extra = sorted(current_names - expected_names)
        raise TrainError(
            f"L2-SP trainable parameter set changed: missing={missing} extra={extra}"
        )

    first = next(iter(current.values()))
    penalty = torch.zeros((), device=first.device, dtype=torch.float32)
    for name in sorted(current):
        parameter = current[name]
        reference = anchor.parameters[name]
        if tuple(parameter.shape) != tuple(reference.shape):
            raise TrainError(
                f"L2-SP parameter shape changed for {name}: "
                f"{tuple(parameter.shape)} != {tuple(reference.shape)}"
            )
        if reference.device != parameter.device:
            raise TrainError(
                f"L2-SP anchor device changed for {name}: "
                f"{reference.device} != {parameter.device}"
            )
        penalty = penalty + (parameter.float() - reference).square().sum()
    return penalty
