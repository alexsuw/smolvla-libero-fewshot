"""Fail-closed trainable-scope allowlist. Must run before optimizer creation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Literal

from vla_fewshot.config import TrainableScope
from vla_fewshot.reproducibility import atomic_write_json, atomic_write_text

ParameterBucket = Literal[
    "vision",
    "vlm",
    "action_expert",
    "state_proj",
    "action_proj",
    "lora",
    "unused_head",
    "unknown",
]


class AllowlistError(ValueError):
    """Raised when trainable parameters are wider or narrower than the scope."""


@dataclass(frozen=True)
class ParameterRecord:
    name: str
    requires_grad: bool
    numel: int = 1


# Substrings matched against pinned LeRobot SmolVLAPolicy.named_parameters().
_VISION = ("vision_model",)
_ACTION_EXPERT = ("vlm_with_expert.lm_expert",)
_VLM = ("vlm_with_expert.vlm",)
_STATE = ("state_proj",)
_ACTION_PROJ = ("action_in_proj", "action_out_proj", "action_time_mlp")
_UNUSED_HEAD = ("lm_head",)


def classify_parameter(name: str) -> ParameterBucket:
    """Map a pinned SmolVLA parameter name onto one allowlist bucket."""

    normalized = name[7:] if name.startswith("module.") else name
    if any(token in normalized for token in _UNUSED_HEAD):
        return "unused_head"
    if _is_lora_parameter_name(normalized):
        return "lora"
    if any(token in normalized for token in _VISION):
        return "vision"
    if any(token in normalized for token in _ACTION_EXPERT):
        return "action_expert"
    if any(token in normalized for token in _VLM):
        return "vlm"
    if any(token in normalized for token in _STATE):
        return "state_proj"
    if any(token in normalized for token in _ACTION_PROJ):
        return "action_proj"
    return "unknown"


def _is_lora_parameter_name(name: str) -> bool:
    return "lora_A" in name or "lora_B" in name or ".lora_" in name


def should_train(
    bucket: ParameterBucket,
    scope: TrainableScope,
    *,
    peft_enabled: bool = False,
) -> bool:
    if bucket in {"unused_head", "unknown"}:
        return False
    if bucket == "lora":
        return peft_enabled
    if bucket == "vision":
        return (not scope.freeze_vision_encoder) and (not scope.freeze_vlm_backbone)
    if bucket == "vlm":
        return not scope.freeze_vlm_backbone
    if bucket == "action_expert":
        return scope.train_action_expert
    if bucket == "state_proj":
        return scope.train_state_projection
    if bucket == "action_proj":
        return scope.train_action_projections
    return False


def lerobot_finetune_flags(scope: TrainableScope) -> dict[str, bool]:
    """Project scope → pinned SmolVLAConfig flags.

    LeRobot has no `train_action_expert` or `train_action_projections` flags.
    Those are applied after load by `apply_trainable_scope`.
    """

    return {
        "freeze_vision_encoder": scope.freeze_vision_encoder,
        "train_expert_only": scope.freeze_vlm_backbone,
        "train_state_proj": scope.train_state_projection,
    }


def apply_scope_to_records(
    records: Iterable[ParameterRecord],
    scope: TrainableScope,
    *,
    peft_enabled: bool = False,
) -> list[ParameterRecord]:
    applied: list[ParameterRecord] = []
    for record in records:
        bucket = classify_parameter(record.name)
        applied.append(
            ParameterRecord(
                name=record.name,
                requires_grad=should_train(bucket, scope, peft_enabled=peft_enabled),
                numel=record.numel,
            )
        )
    return applied


def inspect_trainable_scope(
    records: Iterable[ParameterRecord],
    scope: TrainableScope,
    *,
    peft_enabled: bool = False,
) -> dict[str, Any]:
    if not scope.strict_allowlist:
        raise AllowlistError("strict_allowlist must stay true")
    rows = list(records)
    total = sum(item.numel for item in rows)
    trainable_rows = [item for item in rows if item.requires_grad]
    trainable = sum(item.numel for item in trainable_rows)
    buckets: dict[str, list[str]] = {}
    illegal: list[str] = []
    for item in trainable_rows:
        bucket = classify_parameter(item.name)
        buckets.setdefault(bucket, []).append(item.name)
        if not should_train(bucket, scope, peft_enabled=peft_enabled):
            illegal.append(item.name)
    missing_required: list[str] = []
    required = {
        "action_expert": scope.train_action_expert,
        "state_proj": scope.train_state_projection,
        "action_proj": scope.train_action_projections,
        "vlm": not scope.freeze_vlm_backbone,
        "vision": (not scope.freeze_vision_encoder) and (not scope.freeze_vlm_backbone),
        "lora": peft_enabled,
    }
    present = {classify_parameter(item.name) for item in rows}
    for bucket, needed in required.items():
        if needed and bucket in present and bucket not in buckets:
            missing_required.append(bucket)
        if needed and bucket not in present:
            missing_required.append(f"{bucket}:absent")
    frozen_violations = []
    if scope.freeze_vlm_backbone and buckets.get("vlm"):
        frozen_violations.extend(buckets["vlm"])
    if scope.freeze_vision_encoder and buckets.get("vision"):
        frozen_violations.extend(buckets["vision"])
    if peft_enabled:
        for name in buckets.get("lora", []):
            if "vlm_with_expert.vlm" in name:
                frozen_violations.append(name)
    matches = not illegal and not missing_required and not frozen_violations
    return {
        "total_parameters": total,
        "trainable_parameters": trainable,
        "trainable_percent": (100.0 * trainable / total) if total else 0.0,
        "trainable_names": [item.name for item in trainable_rows],
        "trainable_buckets": {key: sorted(value) for key, value in sorted(buckets.items())},
        "illegal_trainable": illegal,
        "missing_required_buckets": missing_required,
        "frozen_violations": frozen_violations,
        "matches_allowlist": matches,
        "lerobot_flags": lerobot_finetune_flags(scope),
    }


def assert_trainable_allowlist(
    records: Iterable[ParameterRecord],
    scope: TrainableScope,
    *,
    output_dir: Path | None = None,
    peft_enabled: bool = False,
) -> dict[str, Any]:
    report = inspect_trainable_scope(records, scope, peft_enabled=peft_enabled)
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        names = report["trainable_names"]
        atomic_write_text(
            output_dir / "trainable_parameters.txt",
            "\n".join(names) + ("\n" if names else ""),
            overwrite=True,
        )
        atomic_write_json(output_dir / "trainable_scope.json", report, overwrite=True)
    if not report["matches_allowlist"]:
        raise AllowlistError(
            "trainable scope does not match the allowlist: "
            f"illegal={report['illegal_trainable']} "
            f"missing={report['missing_required_buckets']} "
            f"frozen_violations={report['frozen_violations']}"
        )
    return report


def records_from_module(module: Any) -> list[ParameterRecord]:
    rows: list[ParameterRecord] = []
    for name, parameter in module.named_parameters():
        numel = int(parameter.numel()) if hasattr(parameter, "numel") else 1
        rows.append(
            ParameterRecord(
                name=name,
                requires_grad=bool(parameter.requires_grad),
                numel=numel,
            )
        )
    return rows


def apply_trainable_scope(
    module: Any,
    scope: TrainableScope,
    *,
    peft_enabled: bool = False,
) -> None:
    """Overwrite requires_grad from the project allowlist. Call before optimizer."""

    for name, parameter in module.named_parameters():
        parameter.requires_grad = should_train(
            classify_parameter(name), scope, peft_enabled=peft_enabled
        )


def assert_module_trainable_scope(
    module: Any,
    scope: TrainableScope,
    *,
    output_dir: Path | None = None,
    peft_enabled: bool = False,
) -> dict[str, Any]:
    apply_trainable_scope(module, scope, peft_enabled=peft_enabled)
    return assert_trainable_allowlist(
        records_from_module(module),
        scope,
        output_dir=output_dir,
        peft_enabled=peft_enabled,
    )
