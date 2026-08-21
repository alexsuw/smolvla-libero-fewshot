"""Pinned SmolVLA load, LIBERO feature overlay, and one-step inference smoke."""

from __future__ import annotations

import math
import platform
from pathlib import Path
from typing import Any

from vla_fewshot.config import TrainableScope
from vla_fewshot.env.action_adapter import dataset_action_to_env
from vla_fewshot.model.features import (
    LIBERO_ACTION_DIM,
    LIBERO_INPUT_FEATURES,
    LIBERO_OUTPUT_FEATURES,
    LIBERO_STATE_DIM,
    POLICY_ACTION,
    POLICY_MAIN_IMAGE,
    POLICY_STATE,
    POLICY_TASK,
    POLICY_WRIST_IMAGE,
    assert_libero_policy_features,
    feature_snapshot,
)
from vla_fewshot.model.freezing import (
    assert_module_trainable_scope,
    lerobot_finetune_flags,
)


def require_smolvla_runtime() -> None:
    """Fail closed unless the pinned Linux LeRobot extra can be imported."""

    if platform.system() != "Linux":
        raise RuntimeError(
            f"SmolVLA load requires Linux + gpu extra; current host is {platform.system()}"
        )
    try:
        import lerobot.policies.smolvla  # noqa: F401
        import torch  # noqa: F401
    except ImportError as error:
        raise RuntimeError(
            "SmolVLA load requires `uv sync --frozen --extra gpu` on Linux"
        ) from error


def overlay_libero_features(config: Any) -> dict[str, Any]:
    """Replace hub SO100 camera/action features with the LIBERO contract."""

    from lerobot.configs import FeatureType, PolicyFeature

    hub_inputs = feature_snapshot(getattr(config, "input_features", {}) or {})
    hub_outputs = feature_snapshot(getattr(config, "output_features", {}) or {})
    config.input_features = {
        key: PolicyFeature(
            type=getattr(FeatureType, spec["type"]),
            shape=tuple(spec["shape"]),
        )
        for key, spec in LIBERO_INPUT_FEATURES.items()
    }
    config.output_features = {
        key: PolicyFeature(
            type=getattr(FeatureType, spec["type"]),
            shape=tuple(spec["shape"]),
        )
        for key, spec in LIBERO_OUTPUT_FEATURES.items()
    }
    if getattr(config, "adapt_to_pi_aloha", False):
        raise RuntimeError("adapt_to_pi_aloha must stay false for LIBERO")
    observed = assert_libero_policy_features(
        input_features=config.input_features,
        output_features=config.output_features,
    )
    return {
        "hub_input_features": hub_inputs,
        "hub_output_features": hub_outputs,
        "libero": observed,
    }


def load_pinned_smolvla(
    *,
    repo_id: str,
    revision: str,
    scope: TrainableScope,
    device: str | None = None,
    output_dir: Path | None = None,
) -> dict[str, Any]:
    """Load pinned weights, overlay LIBERO features, then freeze to the allowlist."""

    require_smolvla_runtime()
    import torch
    from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

    resolved_device = device or ("cuda" if torch.cuda.is_available() else None)
    if resolved_device is None or not str(resolved_device).startswith("cuda"):
        raise RuntimeError("full SmolVLA load requires CUDA")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
    flags = lerobot_finetune_flags(scope)
    config = SmolVLAConfig.from_pretrained(repo_id, revision=revision)
    config.freeze_vision_encoder = flags["freeze_vision_encoder"]
    config.train_expert_only = flags["train_expert_only"]
    config.train_state_proj = flags["train_state_proj"]
    config.device = resolved_device
    policy = SmolVLAPolicy.from_pretrained(
        repo_id,
        revision=revision,
        config=config,
    )
    policy.to(resolved_device)
    overlay = overlay_libero_features(policy.config)
    policy.config.device = resolved_device
    allowlist = assert_module_trainable_scope(policy, scope, output_dir=output_dir)
    return {
        "policy": policy,
        "device": resolved_device,
        "feature_overlay": overlay,
        "trainable_scope": allowlist,
        "repo_id": repo_id,
        "revision": revision,
    }


def _identity_stats() -> dict[str, dict[str, Any]]:
    zeros_state = [0.0] * LIBERO_STATE_DIM
    ones_state = [1.0] * LIBERO_STATE_DIM
    zeros_action = [0.0] * LIBERO_ACTION_DIM
    ones_action = [1.0] * LIBERO_ACTION_DIM
    return {
        POLICY_STATE: {"mean": zeros_state, "std": ones_state},
        POLICY_ACTION: {"mean": zeros_action, "std": ones_action},
    }


def run_dummy_inference(
    *,
    policy: Any,
    task_text: str = "put the bowl on the stove",
) -> dict[str, Any]:
    """One finite action chunk through production gripper conversion. No env."""

    import torch
    from lerobot.policies.factory import make_pre_post_processors

    preprocessor, postprocessor = make_pre_post_processors(
        policy.config,
        pretrained_path=None,
        dataset_stats=_identity_stats(),
    )
    device = next(policy.parameters()).device
    batch = {
        POLICY_MAIN_IMAGE: torch.zeros(1, 3, 256, 256, device=device),
        POLICY_WRIST_IMAGE: torch.zeros(1, 3, 256, 256, device=device),
        POLICY_STATE: torch.zeros(1, LIBERO_STATE_DIM, device=device),
        POLICY_TASK: [task_text],
    }
    processed = preprocessor(batch)
    policy.reset()
    action = policy.select_action(processed)
    if postprocessor is not None:
        try:
            action = postprocessor(action)
        except Exception:
            pass
    tensor = action if torch.is_tensor(action) else torch.as_tensor(action)
    if tensor.ndim == 3:
        tensor = tensor[:, 0]
    if tensor.ndim == 2:
        tensor = tensor[0]
    values = [float(item) for item in tensor.detach().cpu().flatten().tolist()]
    if len(values) < LIBERO_ACTION_DIM:
        raise RuntimeError(f"policy action shorter than {LIBERO_ACTION_DIM}: {len(values)}")
    dataset_action = values[:LIBERO_ACTION_DIM]
    if not all(math.isfinite(item) for item in dataset_action):
        raise RuntimeError(f"non-finite policy action: {dataset_action}")
    env_action = dataset_action_to_env(dataset_action, binary=True)
    return {
        "task_text": task_text,
        "policy_action_dataset_space": dataset_action,
        "env_action": env_action,
        "action_dim": LIBERO_ACTION_DIM,
        "finite": True,
        "normalization_stats": "identity_smoke_only",
    }
