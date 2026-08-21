from pathlib import Path
import os
import subprocess
import sys

import pytest

from vla_fewshot.config import load_config
from vla_fewshot.model.peft import (
    SMOLVLA_LORA_TARGET_MODULES,
    adapter_config_payload,
    extract_adapter_state,
    load_adapter_weights,
    refuse_peft_until_challenger,
    require_peft_runtime,
    wrap_policy_lora,
)
from vla_fewshot.training.baseline import assert_baseline_train_config, episode_ids_for_cell
from vla_fewshot.training.target_lora import (
    assert_target_lora_train_config,
    assert_target_train_config,
    lora_command,
)
from vla_fewshot.training.trainer import TrainError
from vla_fewshot.data.splits import load_target_splits


ROOT = Path(__file__).resolve().parents[2]


def test_target_lora_yaml_matches_frozen_calibration() -> None:
    config = load_config(ROOT / "configs" / "train" / "target_lora.yaml")
    assert_target_lora_train_config(config)
    assert_target_train_config(config)
    assert config.peft is not None
    payload = adapter_config_payload(config.peft)
    assert payload["merged"] is False
    assert payload["r"] == 64
    assert payload["target_modules"] == SMOLVLA_LORA_TARGET_MODULES
    with pytest.raises(TrainError, match="baseline"):
        assert_baseline_train_config(config)


def test_lora_uses_the_same_nested_episode_ids_as_baseline() -> None:
    splits = load_target_splits(ROOT / "configs" / "splits" / "target_splits.json")
    assert episode_ids_for_cell(splits, task_slug="bowl_stove", n_demos=5) == [
        13,
        15,
        16,
        22,
        36,
    ]
    command = lora_command(task="bowl_stove", n_demos=5, seed=42)
    assert "--config" in command
    assert "target_lora.yaml" in command[command.index("--config") + 1]


def test_extract_adapter_state_is_lora_keys_only() -> None:
    state = {
        "model.state_proj.weight": 1,
        "model.vlm_with_expert.lm_expert.layers.0.self_attn.q_proj.lora_A.default.weight": 2,
        "model.vlm_with_expert.lm_expert.layers.0.self_attn.q_proj.base_layer.weight": 3,
    }
    assert extract_adapter_state(state) == {
        "model.vlm_with_expert.lm_expert.layers.0.self_attn.q_proj.lora_A.default.weight": 2
    }


def test_load_adapter_refuses_implicit_merge(tmp_path: Path) -> None:
    with pytest.raises(TrainError, match="merge"):
        load_adapter_weights(object(), tmp_path, merge=True)


def test_seen_peft_wrap_stays_refused() -> None:
    with pytest.raises(RuntimeError, match="seen challenger"):
        refuse_peft_until_challenger()
    with pytest.raises(RuntimeError, match="no GPU training was started"):
        require_peft_runtime()
    seen = load_config(ROOT / "configs" / "train" / "seen_lora.yaml")
    with pytest.raises(RuntimeError, match="seen challenger"):
        wrap_policy_lora(object(), seen)


def test_print_grid_lora_lists_eighteen_commands() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "train_target.py"),
            "--config",
            str(ROOT / "configs" / "train" / "target_lora.yaml"),
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
    assert len(lines) == 18
    assert "target_lora.yaml" in lines[0]
    assert "--task drawer_middle --n-demos 5 --seed 42" in lines[0]


def test_eval_print_grid_lora_passes_train_config() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "eval_target.py"),
            "--train-config",
            str(ROOT / "configs" / "train" / "target_lora.yaml"),
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
    assert len(lines) == 18
    assert "--train-config" in lines[0]
    assert "target_lora.yaml" in lines[0]


def test_train_target_lora_waits_for_frozen_origin(tmp_path: Path) -> None:
    env = os.environ.copy()
    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "train_target.py"),
            "--config",
            str(ROOT / "configs" / "train" / "target_lora.yaml"),
            "--task",
            "drawer_middle",
            "--n-demos",
            "5",
            "--seed",
            "42",
            "--output-dir",
            str(tmp_path / "lora"),
        ],
        cwd=ROOT,
        capture_output=True,
        check=False,
        text=True,
        timeout=20,
        env=env,
    )
    assert completed.returncode == 1
    combined = completed.stdout + completed.stderr
    assert "no GPU training was started" in combined
    assert "frozen" in combined
    assert "baseline forbids" not in combined
