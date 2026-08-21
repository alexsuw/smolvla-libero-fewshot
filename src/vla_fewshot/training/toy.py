"""CPU toy policy with SmolVLA-like parameter names for allowlist tests."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass

TOY_PARAMETER_NAMES = (
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
)


@dataclass
class ToyParameter:
    data: list[float]
    requires_grad: bool = True
    grad: list[float] | None = None

    def numel(self) -> int:
        return len(self.data)

    def zero_grad(self) -> None:
        self.grad = None


def _init_vector(rng: random.Random, size: int = 2) -> list[float]:
    return [rng.random() - 0.5 for _ in range(size)]


class ToyPolicy:
    """Finite MSE toy used to prove checkpoint/resume without CUDA weights."""

    def __init__(self, seed: int) -> None:
        rng = random.Random(seed)
        self._params = {
            name: ToyParameter(_init_vector(rng), requires_grad=True)
            for name in TOY_PARAMETER_NAMES
        }

    def named_parameters(self):
        return self._params.items()

    def parameters(self):
        return self._params.values()

    def state_dict(self) -> dict[str, list[float]]:
        return {name: list(param.data) for name, param in self._params.items()}

    def load_state_dict(self, state: dict[str, list[float]]) -> None:
        missing = set(self._params) - set(state)
        extra = set(state) - set(self._params)
        if missing or extra:
            raise ValueError(f"weight key mismatch missing={missing} extra={extra}")
        for name, values in state.items():
            if len(values) != len(self._params[name].data):
                raise ValueError(f"weight length mismatch for {name}")
            self._params[name].data = list(values)

    def _get(self, name: str) -> ToyParameter:
        return self._params[name]

    def forward_loss(self, x: float, y: float) -> float:
        """Scalar MSE with contributions from every intended trainable bucket."""

        action_out = self._get("model.action_out_proj.weight").data[0]
        action_in = self._get("model.action_in_proj.weight").data[0]
        state = self._get("model.state_proj.weight").data[0]
        expert = self._get(
            "model.vlm_with_expert.lm_expert.layers.0.self_attn.q_proj.weight"
        ).data[0]
        time_mlp = self._get("model.action_time_mlp_in.weight").data[0]
        pred = (
            action_out * x
            + 0.5 * action_in * x
            + 0.25 * state
            + 0.125 * expert * x
            + 0.0625 * time_mlp
        )
        if not math.isfinite(pred):
            raise FloatingPointError(f"non-finite prediction: {pred}")
        loss = (pred - y) ** 2
        if not math.isfinite(loss):
            raise FloatingPointError(f"non-finite loss: {loss}")
        dpred = 2.0 * (pred - y)
        self._accum_grad("model.action_out_proj.weight", 0, dpred * x)
        self._accum_grad("model.action_in_proj.weight", 0, dpred * 0.5 * x)
        self._accum_grad("model.state_proj.weight", 0, dpred * 0.25)
        self._accum_grad(
            "model.vlm_with_expert.lm_expert.layers.0.self_attn.q_proj.weight",
            0,
            dpred * 0.125 * x,
        )
        self._accum_grad("model.action_time_mlp_in.weight", 0, dpred * 0.0625)
        self._accum_grad("model.action_time_mlp_out.weight", 0, 0.0)
        return loss

    def zero_grad(self) -> None:
        for param in self._params.values():
            param.zero_grad()

    def _accum_grad(self, name: str, index: int, value: float) -> None:
        param = self._params[name]
        if not param.requires_grad:
            return
        if param.grad is None:
            param.grad = [0.0] * len(param.data)
        param.grad[index] += value

    def scale_grads(self, scale: float) -> None:
        for param in self._params.values():
            if param.grad is None:
                continue
            param.grad = [item * scale for item in param.grad]

    def add_grads(self, other: dict[str, list[float]]) -> None:
        for name, values in other.items():
            param = self._params[name]
            if param.grad is None:
                param.grad = [0.0] * len(param.data)
            param.grad = [left + right for left, right in zip(param.grad, values, strict=True)]

    def snapshot_grads(self) -> dict[str, list[float]]:
        snapped: dict[str, list[float]] = {}
        for name, param in self._params.items():
            if param.grad is None:
                continue
            snapped[name] = list(param.grad)
        return snapped
