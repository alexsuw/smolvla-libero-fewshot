"""Target LoRA wrap and merged-free adapter I/O. Seen-FT LoRA stays refused."""

from __future__ import annotations

import json
import platform
from pathlib import Path
from typing import Any, Mapping

from vla_fewshot.config import PeftConfig, TrainConfig
from vla_fewshot.reproducibility import atomic_write_json
from vla_fewshot.storage.layout import (
    CHECKPOINT_ADAPTER_CONFIG_NAME,
    CHECKPOINT_ADAPTER_DIRNAME,
    CHECKPOINT_ADAPTER_WEIGHTS_NAME,
)
from vla_fewshot.training.trainer import TrainError

# Pinned LeRobot SmolVLAPolicy._get_default_peft_targets expert half only.
# State/action projections stay full-rank via trainable_scope, not LoRA.
SMOLVLA_LORA_TARGET_MODULES = r"model\.vlm_with_expert\.lm_expert\..*\.(q|v)_proj"


def refuse_peft_until_challenger() -> None:
    raise RuntimeError(
        "PEFT/LoRA wrapping is the optional seen challenger, not the M4/M5 primary path"
    )


def is_lora_parameter_name(name: str) -> bool:
    return "lora_A" in name or "lora_B" in name or ".lora_" in name


def require_peft_runtime() -> None:
    if platform.system() != "Linux":
        raise RuntimeError(
            f"PEFT/LoRA wrap requires Linux + gpu extra; current host is {platform.system()}. "
            "no GPU training was started."
        )
    try:
        import peft  # noqa: F401
        import torch  # noqa: F401
    except ImportError as error:
        raise RuntimeError(
            "PEFT/LoRA wrap requires `uv sync --frozen --extra gpu` on Linux. "
            "no GPU training was started."
        ) from error


def adapter_config_payload(peft: PeftConfig) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "method_type": peft.method_type,
        "r": peft.r,
        "lora_alpha": peft.lora_alpha,
        "lora_dropout": peft.lora_dropout,
        "target_modules": SMOLVLA_LORA_TARGET_MODULES,
        "bias": "none",
        "merged": False,
    }


def extract_adapter_state(state_dict: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in state_dict.items() if is_lora_parameter_name(key)}


def assert_adapter_not_merged(policy: Any, *, config_payload: Mapping[str, Any] | None = None) -> None:
    if config_payload is not None and config_payload.get("merged"):
        raise TrainError("implicit LoRA merge is forbidden. no GPU training was started.")
    if bool(getattr(policy, "merged", False)):
        raise TrainError("implicit LoRA merge is forbidden. no GPU training was started.")
    names = [name for name, _parameter in policy.named_parameters() if is_lora_parameter_name(name)]
    if not names:
        raise TrainError(
            "merged-free LoRA load found no adapter parameters. no GPU training was started."
        )


def wrap_policy_lora(policy: Any, config: TrainConfig) -> Any:
    """Inject LoRA after origin load. Never merge. Fail closed before optimizer."""

    if config.stage == "seen":
        refuse_peft_until_challenger()
    if config.stage != "target" or config.method not in {"lora", "replay_lora"}:
        raise TrainError(
            "LoRA wrap is only for target method=lora or replay_lora. "
            "no GPU training was started."
        )
    if config.peft is None:
        raise TrainError("target LoRA requires peft. no GPU training was started.")
    require_peft_runtime()
    from peft import LoraConfig, get_peft_model

    lora = LoraConfig(
        r=config.peft.r,
        lora_alpha=config.peft.lora_alpha,
        lora_dropout=config.peft.lora_dropout,
        target_modules=SMOLVLA_LORA_TARGET_MODULES,
        bias="none",
        inference_mode=False,
    )
    wrapped = get_peft_model(policy, lora)
    adapter = extract_adapter_state(wrapped.state_dict())
    if not adapter:
        raise TrainError(
            "LoRA wrap matched no modules. no GPU training was started."
        )
    return wrapped


def maybe_save_adapter_sidecar(
    checkpoint_dir: Path,
    *,
    policy: Any,
    peft: PeftConfig | None,
) -> Path | None:
    if peft is None:
        return None
    adapter = extract_adapter_state(policy.state_dict())
    if not adapter:
        raise TrainError("LoRA checkpoint is missing adapter weights")
    adapter_dir = checkpoint_dir / CHECKPOINT_ADAPTER_DIRNAME
    adapter_dir.mkdir(parents=True, exist_ok=False)
    import torch

    torch.save(adapter, adapter_dir / CHECKPOINT_ADAPTER_WEIGHTS_NAME)
    atomic_write_json(
        adapter_dir / CHECKPOINT_ADAPTER_CONFIG_NAME,
        adapter_config_payload(peft),
        overwrite=True,
    )
    return adapter_dir


def load_adapter_weights(
    policy: Any,
    adapter_dir: Path,
    *,
    merge: bool = False,
) -> dict[str, Any]:
    if merge:
        raise TrainError("implicit LoRA merge is forbidden. no GPU training was started.")
    config_path = adapter_dir / CHECKPOINT_ADAPTER_CONFIG_NAME
    weights_path = adapter_dir / CHECKPOINT_ADAPTER_WEIGHTS_NAME
    if not config_path.is_file() or not weights_path.is_file():
        raise TrainError(f"missing merged-free adapter files under {adapter_dir}")
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    if payload.get("merged"):
        raise TrainError("implicit LoRA merge is forbidden. no GPU training was started.")
    import torch

    adapter = torch.load(weights_path, map_location=next(policy.parameters()).device, weights_only=True)
    current = policy.state_dict()
    missing = [key for key in adapter if key not in current]
    if missing:
        raise TrainError(f"adapter keys are not on the wrapped policy: {missing[:8]}")
    current.update(adapter)
    policy.load_state_dict(current)
    assert_adapter_not_merged(policy, config_payload=payload)
    return {"adapter_keys": sorted(adapter), "merged": False}


def load_lora_policy_weights(policy: Any, checkpoint_dir: Path, *, peft: PeftConfig) -> None:
    """Load a LoRA checkpoint onto an already-wrapped policy. Never merge."""

    import torch

    from vla_fewshot.storage.layout import CHECKPOINT_WEIGHTS_PT_NAME

    weights = torch.load(
        checkpoint_dir / CHECKPOINT_WEIGHTS_PT_NAME,
        map_location=next(policy.parameters()).device,
        weights_only=True,
    )
    policy.load_state_dict(weights, strict=False)
    if not extract_adapter_state(weights):
        load_adapter_weights(
            policy, checkpoint_dir / CHECKPOINT_ADAPTER_DIRNAME, merge=False
        )
    assert_adapter_not_merged(policy, config_payload=adapter_config_payload(peft))
