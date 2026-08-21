"""Deterministic AdamW and cosine schedule without PyTorch."""

from __future__ import annotations

import math
from dataclasses import dataclass

from vla_fewshot.config import OptimizerConfig, SchedulerConfig, TrainingConfig
from vla_fewshot.storage.checksums import decode_floats, encode_floats
from vla_fewshot.training.toy import ToyParameter, ToyPolicy


def resolve_gradient_accumulation(training: TrainingConfig) -> int:
    if training.physical_batch_size == "auto_fit":
        raise RuntimeError(
            "physical_batch_size=auto_fit must be resolved before creating a run"
        )
    physical = int(training.physical_batch_size)
    if training.gradient_accumulation == "auto":
        if training.effective_batch_size % physical != 0:
            raise RuntimeError("effective_batch_size must divide physical_batch_size")
        return training.effective_batch_size // physical
    accumulation = int(training.gradient_accumulation)
    if physical * accumulation != training.effective_batch_size:
        raise RuntimeError(
            "effective_batch_size must equal physical_batch_size * gradient_accumulation"
        )
    return accumulation


def cosine_with_warmup_lr(
    step: int,
    *,
    base_lr: float,
    min_lr: float,
    warmup_steps: int,
    max_steps: int,
) -> float:
    if step < 0:
        raise ValueError("step must be non-negative")
    if warmup_steps > 0 and step < warmup_steps:
        return base_lr * float(step + 1) / float(warmup_steps)
    denom = max(1, max_steps - warmup_steps)
    progress = min(1.0, float(step - warmup_steps) / float(denom))
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_lr + (base_lr - min_lr) * cosine


@dataclass
class AdamWSlot:
    name: str
    param: ToyParameter
    m: list[float]
    v: list[float]


class ToyAdamW:
    """Decoupled AdamW over ToyParameter tensors. Created only after allowlist."""

    def __init__(
        self,
        policy: ToyPolicy,
        config: OptimizerConfig,
    ) -> None:
        self.config = config
        self.step_count = 0
        self.slots: list[AdamWSlot] = []
        for name, param in policy.named_parameters():
            if not param.requires_grad:
                continue
            self.slots.append(
                AdamWSlot(
                    name=name,
                    param=param,
                    m=[0.0] * len(param.data),
                    v=[0.0] * len(param.data),
                )
            )
        if not self.slots:
            raise RuntimeError("optimizer has no trainable parameters")

    def clip_grad_norm(self, max_norm: float) -> float:
        total_sq = 0.0
        for slot in self.slots:
            grads = slot.param.grad or [0.0] * len(slot.param.data)
            total_sq += sum(item * item for item in grads)
        total = math.sqrt(total_sq)
        if not math.isfinite(total):
            raise FloatingPointError(f"non-finite grad norm: {total}")
        if total > max_norm and total > 0.0:
            scale = max_norm / total
            for slot in self.slots:
                if slot.param.grad is None:
                    continue
                slot.param.grad = [item * scale for item in slot.param.grad]
            return max_norm
        return total

    def step(self, learning_rate: float) -> None:
        beta1, beta2 = self.config.betas
        self.step_count += 1
        bias1 = 1.0 - beta1**self.step_count
        bias2 = 1.0 - beta2**self.step_count
        for slot in self.slots:
            grads = slot.param.grad or [0.0] * len(slot.param.data)
            new_data: list[float] = []
            new_m: list[float] = []
            new_v: list[float] = []
            for value, grad, m_prev, v_prev in zip(
                slot.param.data, grads, slot.m, slot.v, strict=True
            ):
                m = beta1 * m_prev + (1.0 - beta1) * grad
                v = beta2 * v_prev + (1.0 - beta2) * grad * grad
                m_hat = m / bias1
                v_hat = v / bias2
                decayed = value * (1.0 - learning_rate * self.config.weight_decay)
                updated = decayed - learning_rate * m_hat / (math.sqrt(v_hat) + self.config.eps)
                if not math.isfinite(updated):
                    raise FloatingPointError(f"non-finite weight update for {slot.name}")
                new_data.append(updated)
                new_m.append(m)
                new_v.append(v)
            slot.param.data = new_data
            slot.m = new_m
            slot.v = new_v

    def state_dict(self) -> dict[str, object]:
        return {
            "name": "adamw",
            "step_count": self.step_count,
            "config": self.config.model_dump(mode="json"),
            "slots": [
                {
                    "name": slot.name,
                    "m": encode_floats(slot.m),
                    "v": encode_floats(slot.v),
                }
                for slot in self.slots
            ],
        }

    def load_state_dict(self, state: dict[str, object]) -> None:
        if state.get("name") != "adamw":
            raise ValueError("optimizer state is not adamw")
        self.step_count = int(state["step_count"])
        saved_slots = {item["name"]: item for item in state["slots"]}  # type: ignore[index]
        if set(saved_slots) != {slot.name for slot in self.slots}:
            raise ValueError("optimizer parameter set mismatch")
        for slot in self.slots:
            payload = saved_slots[slot.name]
            slot.m = decode_floats(payload["m"])  # type: ignore[arg-type]
            slot.v = decode_floats(payload["v"])  # type: ignore[arg-type]


def current_lr(config: SchedulerConfig, optimizer_config: OptimizerConfig, step: int, max_steps: int) -> float:
    if config.name != "cosine":
        raise ValueError(f"unsupported scheduler {config.name}")
    return cosine_with_warmup_lr(
        step,
        base_lr=optimizer_config.lr,
        min_lr=config.min_lr,
        warmup_steps=config.warmup_steps,
        max_steps=max_steps,
    )
