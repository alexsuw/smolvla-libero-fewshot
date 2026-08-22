from pathlib import Path

from vla_fewshot.config import load_config
from vla_fewshot.evaluation.protocol import FINAL_SEED_VALUES, plan_named_rollouts
from vla_fewshot.training.baseline import apply_throughput_batch, cap_optimizer_steps


ROOT = Path(__file__).resolve().parents[2]


def test_throughput_batch_sets_physical_equal_effective() -> None:
    config = load_config(ROOT / "configs" / "train" / "target_baseline.yaml")
    updated = apply_throughput_batch(config, 64)
    assert updated.training.effective_batch_size == 64
    assert updated.training.physical_batch_size == 64
    assert updated.training.gradient_accumulation == 1
    assert config.training.effective_batch_size == 32


def test_epoch_cap_drops_steps_when_batch_grows() -> None:
    at_32 = cap_optimizer_steps(
        max_steps=12000, epochs=100, n_samples=516, effective_batch_size=32
    )
    at_64 = cap_optimizer_steps(
        max_steps=12000, epochs=100, n_samples=516, effective_batch_size=64
    )
    assert at_32 == 1700
    assert at_64 == 900


def test_eval_seed_shard_keeps_official_init_state_id() -> None:
    config = load_config(ROOT / "configs" / "eval" / "final.yaml")
    planned = plan_named_rollouts(
        config,
        task_slug="drawer_middle",
        task_text="open the middle drawer of the cabinet",
        suite="libero_goal",
        task_index=9,
        n_demos=5,
        train_seed=42,
        seeds=[1010, 1011],
    )
    assert planned[0].eval_seed == 1010
    assert planned[0].rollout_index == FINAL_SEED_VALUES.index(1010) == 10
    assert planned[1].rollout_index == 11
