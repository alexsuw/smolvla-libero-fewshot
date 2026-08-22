import random
import sys
import types
from pathlib import Path

import numpy as np
import pytest
import torch

from vla_fewshot.env.action_adapter import (
    dataset_action_to_env,
    flatten_policy_action,
    policy_action_to_dataset,
)
from vla_fewshot.evaluation.live import LiveRolloutAdapter, seed_live_inference


class _MeanStdActionPost:
    """Stand-in for LeRobot UnnormalizerProcessorStep on the gripper channel."""

    def __init__(self, mean: float = 0.529087, std: float = 0.499153) -> None:
        self.mean = mean
        self.std = std

    def __call__(self, action: list[float]) -> list[float]:
        values = list(action)
        values[6] = values[6] * self.std + self.mean
        return values


def test_live_inference_seed_replays_policy_noise() -> None:
    def sample() -> tuple[float, float, float]:
        return random.random(), float(np.random.random()), float(torch.rand(()))

    seed_live_inference(1000)
    first = sample()
    seed_live_inference(1000)
    assert sample() == first


def test_missing_postprocessor_is_refused() -> None:
    with pytest.raises(ValueError, match="postprocessor"):
        policy_action_to_dataset(
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5], postprocessor=None
        )


def test_live_adapter_requires_postprocessor() -> None:
    with pytest.raises(ValueError, match="postprocessor"):
        LiveRolloutAdapter(
            policy=object(),
            preprocessor=object(),
            postprocessor=None,
            device="cpu",
        )


def test_normalized_gripper_is_unnormalized_before_env() -> None:
    # libero_90 gripper z=-1.0 -> 0.529087 - 0.499153, still dataset-space.
    dataset = policy_action_to_dataset(
        [0.1, -0.2, 0.0, 0.0, 0.0, 0.0, -1.0],
        postprocessor=_MeanStdActionPost(),
    )
    assert 0.0 <= dataset[6] <= 1.0
    assert dataset[6] == pytest.approx(0.029934, abs=1e-6)
    env = dataset_action_to_env(dataset, binary=True)
    assert env[6] == 1.0


def test_unnormalized_overshoot_is_clipped_not_treated_as_env_space() -> None:
    dataset = policy_action_to_dataset(
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.2],
        postprocessor=lambda action: action,
    )
    assert dataset[6] == 1.0
    env = dataset_action_to_env(dataset, binary=True)
    assert env[6] == -1.0


def test_env_space_gripper_without_unnormalize_is_still_refused() -> None:
    with pytest.raises(ValueError, match="double conversion"):
        dataset_action_to_env([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0])


def test_flatten_takes_unpadded_prefix() -> None:
    padded = [[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 9.0, 9.0]]
    assert flatten_policy_action(padded) == [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]


def test_load_checkpoint_weights_swaps_state_dict(tmp_path: Path, monkeypatch) -> None:
    class Policy:
        def __init__(self) -> None:
            self.state = None
            self.reset_calls = 0

        def load_state_dict(self, weights):
            self.state = weights

        def eval(self) -> None:
            return None

        def reset(self) -> None:
            self.reset_calls += 1

    policy = Policy()
    adapter = LiveRolloutAdapter(
        policy=policy,
        preprocessor=object(),
        postprocessor=object(),
        device="cpu",
    )
    checkpoint = tmp_path / "ckpt"
    checkpoint.mkdir()
    (checkpoint / "weights.pt").write_bytes(b"x")
    monkeypatch.setattr(
        "vla_fewshot.evaluation.live.is_complete_checkpoint", lambda _path: True
    )
    fake_torch = types.ModuleType("torch")
    fake_torch.load = lambda *args, **kwargs: {"expert": 1}
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    adapter.load_checkpoint_weights(checkpoint)
    assert policy.state == {"expert": 1}
    assert policy.reset_calls == 1
