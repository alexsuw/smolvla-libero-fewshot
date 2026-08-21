from pathlib import Path

import pytest

from vla_fewshot.config import TrainableScope
from vla_fewshot.model.freezing import (
    AllowlistError,
    ParameterRecord,
    apply_scope_to_records,
    assert_trainable_allowlist,
    classify_parameter,
    lerobot_finetune_flags,
)

SEEN_SCOPE = TrainableScope(
    freeze_vision_encoder=True,
    freeze_vlm_backbone=True,
    train_action_expert=True,
    train_state_projection=True,
    train_action_projections=True,
    strict_allowlist=True,
)


def _fake_smolvla_records() -> list[ParameterRecord]:
    names = [
        "model.vlm_with_expert.vlm.model.vision_model.embeddings.weight",
        "model.vlm_with_expert.vlm.model.text_model.layers.0.self_attn.q_proj.weight",
        "model.vlm_with_expert.vlm.lm_head.weight",
        "model.vlm_with_expert.lm_expert.layers.0.self_attn.q_proj.weight",
        "model.vlm_with_expert.lm_expert.lm_head.weight",
        "model.state_proj.weight",
        "model.action_in_proj.weight",
        "model.action_out_proj.weight",
        "model.action_time_mlp_in.weight",
        "model.action_time_mlp_out.weight",
        "unexpected.mystery.weight",
    ]
    return [ParameterRecord(name=name, requires_grad=True, numel=8) for name in names]


def test_parameter_buckets_match_pinned_smolvla_names() -> None:
    assert classify_parameter("model.vlm_with_expert.vlm.model.vision_model.x") == "vision"
    assert classify_parameter("model.vlm_with_expert.lm_expert.layers.0.x") == "action_expert"
    assert classify_parameter("model.state_proj.bias") == "state_proj"
    assert classify_parameter("model.action_time_mlp_out.weight") == "action_proj"
    assert classify_parameter("model.vlm_with_expert.lm_expert.lm_head.weight") == "unused_head"


def test_seen_pretrain_scope_maps_to_lerobot_flags() -> None:
    flags = lerobot_finetune_flags(SEEN_SCOPE)
    assert flags == {
        "freeze_vision_encoder": True,
        "train_expert_only": True,
        "train_state_proj": True,
    }


def test_seen_pretrain_allowlist_keeps_expert_and_projections() -> None:
    applied = apply_scope_to_records(_fake_smolvla_records(), SEEN_SCOPE)
    report = assert_trainable_allowlist(applied, SEEN_SCOPE)
    names = set(report["trainable_names"])
    assert "model.vlm_with_expert.lm_expert.layers.0.self_attn.q_proj.weight" in names
    assert "model.state_proj.weight" in names
    assert "model.action_out_proj.weight" in names
    assert (
        "model.vlm_with_expert.vlm.model.text_model.layers.0.self_attn.q_proj.weight"
        not in names
    )
    assert "model.vlm_with_expert.vlm.model.vision_model.embeddings.weight" not in names
    assert "unexpected.mystery.weight" not in names
    assert "model.vlm_with_expert.lm_expert.lm_head.weight" not in names


def test_unintended_vlm_trainable_fails_closed() -> None:
    records = apply_scope_to_records(_fake_smolvla_records(), SEEN_SCOPE)
    leaked = ParameterRecord(
        name="model.vlm_with_expert.vlm.model.text_model.layers.0.self_attn.q_proj.weight",
        requires_grad=True,
        numel=8,
    )
    with pytest.raises(AllowlistError, match="illegal"):
        assert_trainable_allowlist([leaked, *records], SEEN_SCOPE)


def test_missing_action_expert_is_too_narrow() -> None:
    records = [
        item
        for item in apply_scope_to_records(_fake_smolvla_records(), SEEN_SCOPE)
        if classify_parameter(item.name) != "action_expert"
    ]
    with pytest.raises(AllowlistError, match="missing"):
        assert_trainable_allowlist(records, SEEN_SCOPE)


def test_trainable_parameters_file_is_written(tmp_path: Path) -> None:
    applied = apply_scope_to_records(_fake_smolvla_records(), SEEN_SCOPE)
    assert_trainable_allowlist(applied, SEEN_SCOPE, output_dir=tmp_path)
    text = (tmp_path / "trainable_parameters.txt").read_text(encoding="utf-8")
    assert "model.state_proj.weight" in text
    assert "vision_model" not in text
