from pathlib import Path
import subprocess
import sys

import pytest
import torch

from vla_fewshot.config import TrainConfig, load_config
from vla_fewshot.evaluation.normalization import normalization_stats_suite
from vla_fewshot.training.anchored import (
    FROZEN_STATS_SHA256,
    PREREGISTERED_L2SP_STRENGTH,
    assert_frozen_stats_train_config,
    capture_l2sp_anchor,
    l2sp_raw_penalty,
)
from vla_fewshot.training.resume import frozen_training_contract
from vla_fewshot.training.target_lora import assert_target_train_config
from vla_fewshot.training.trainer import TrainError


ROOT = Path(__file__).resolve().parents[2]


def _config(name: str) -> TrainConfig:
    loaded = load_config(ROOT / "configs" / "train" / name)
    assert isinstance(loaded, TrainConfig)
    return loaded


@pytest.mark.parametrize(
    ("filename", "method"),
    [
        ("target_frozen_stats.yaml", "frozen_stats"),
        ("target_anchored_l2sp.yaml", "anchored_l2sp"),
    ],
)
def test_matched_frozen_stats_configs_are_registered(
    filename: str, method: str
) -> None:
    config = _config(filename)
    baseline = _config("target_baseline.yaml")

    assert config.method == method
    assert config.optimizer == baseline.optimizer
    assert config.scheduler == baseline.scheduler
    assert config.training == baseline.training
    assert config.trainable_scope == baseline.trainable_scope
    assert config.peft is None
    assert config.replay is None
    assert config.normalization is not None
    assert config.normalization.suite == "libero_90"
    assert config.normalization.expected_sha256 == FROZEN_STATS_SHA256
    assert_frozen_stats_train_config(config)
    assert_target_train_config(config)


def test_l2sp_strength_and_reduction_are_preregistered() -> None:
    anchored = _config("target_anchored_l2sp.yaml")
    frozen = _config("target_frozen_stats.yaml")
    assert anchored.l2sp is not None
    assert anchored.l2sp.strength == PREREGISTERED_L2SP_STRENGTH
    assert anchored.l2sp.reduction == "sum"
    assert anchored.l2sp.anchor_dtype == "fp32"
    assert frozen.l2sp is None

    drifted = anchored.model_copy(
        update={
            "l2sp": anchored.l2sp.model_copy(update={"strength": 2.0e-2})
        }
    )
    with pytest.raises(TrainError, match="lambda=1e-2"):
        assert_frozen_stats_train_config(drifted)


def test_method_specific_config_fields_fail_closed() -> None:
    frozen = _config("target_frozen_stats.yaml")
    without_stats = frozen.model_dump(mode="json")
    without_stats.pop("normalization")
    with pytest.raises(ValueError, match="requires frozen normalization"):
        TrainConfig.model_validate(without_stats)

    baseline = _config("target_baseline.yaml").model_dump(mode="json")
    baseline["normalization"] = frozen.normalization.model_dump(mode="json")
    with pytest.raises(ValueError, match="only valid"):
        TrainConfig.model_validate(baseline)


class _ToyPolicy(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor([1.0, 2.0]))
        self.frozen = torch.nn.Parameter(
            torch.tensor([4.0]), requires_grad=False
        )


def test_l2sp_anchor_tracks_exact_trainable_parameters_in_fp32() -> None:
    policy = _ToyPolicy()
    anchor = capture_l2sp_anchor(policy)
    assert anchor.parameter_count == 2
    assert anchor.parameters["weight"].dtype == torch.float32
    assert l2sp_raw_penalty(policy, anchor).item() == pytest.approx(0.0)

    with torch.no_grad():
        policy.weight.add_(2.0)
        policy.frozen.add_(100.0)
    penalty = l2sp_raw_penalty(policy, anchor)
    assert penalty.item() == pytest.approx(8.0)
    penalty.backward()
    assert policy.weight.grad is not None
    assert policy.weight.grad.tolist() == pytest.approx([4.0, 4.0])

    policy.frozen.requires_grad_(True)
    with pytest.raises(TrainError, match="parameter set changed"):
        l2sp_raw_penalty(policy, anchor)


def test_frozen_stats_eval_uses_libero90_and_resume_contract_pins_method() -> None:
    final = load_config(ROOT / "configs" / "eval" / "final.yaml")
    frozen = _config("target_frozen_stats.yaml")
    anchored = _config("target_anchored_l2sp.yaml")

    assert normalization_stats_suite(final, frozen) == "libero_90"
    assert normalization_stats_suite(final, anchored) == "libero_90"
    frozen_contract = frozen_training_contract(frozen)
    anchored_contract = frozen_training_contract(anchored)
    assert frozen_contract["method"] == "frozen_stats"
    assert anchored_contract["method"] == "anchored_l2sp"
    assert frozen_contract["normalization"]["expected_sha256"] == FROZEN_STATS_SHA256
    assert anchored_contract["l2sp"]["strength"] == PREREGISTERED_L2SP_STRENGTH


def test_matched_grid_prints_exactly_twelve_n1_commands() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "run_matched_n1_grid.py"),
            "--print-grid",
        ],
        cwd=ROOT,
        capture_output=True,
        check=False,
        text=True,
        timeout=20,
    )
    assert completed.returncode == 0, completed.stderr
    lines = [line for line in completed.stdout.splitlines() if line.strip()]
    assert len(lines) == 12
    assert len(set(lines)) == 12
    assert sum("target_frozen_stats.yaml" in line for line in lines) == 6
    assert sum("target_anchored_l2sp.yaml" in line for line in lines) == 6
    assert all("--n-demos 1" in line for line in lines)
