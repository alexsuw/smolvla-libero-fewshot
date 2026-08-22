"""Resolve physical batch / accumulation before a final run ID exists."""

from __future__ import annotations

from vla_fewshot.config import TrainConfig
from vla_fewshot.training.optim import resolve_gradient_accumulation

AUTO_FIT_PHYSICAL = (32, 16, 8, 4, 2, 1)


def with_resolved_batch(config: TrainConfig, physical: int) -> TrainConfig:
    if physical < 1:
        raise ValueError("physical_batch_size must be positive")
    if config.training.effective_batch_size % physical != 0:
        raise RuntimeError(
            f"effective_batch_size={config.training.effective_batch_size} "
            f"is not divisible by physical_batch_size={physical}"
        )
    accumulation = config.training.effective_batch_size // physical
    return config.model_copy(
        update={
            "training": config.training.model_copy(
                update={
                    "physical_batch_size": physical,
                    "gradient_accumulation": accumulation,
                }
            )
        }
    )


def auto_fit_candidates(effective_batch_size: int) -> tuple[int, ...]:
    return tuple(
        size for size in AUTO_FIT_PHYSICAL if effective_batch_size % size == 0
    )


def is_cuda_oom(error: BaseException) -> bool:
    text = f"{type(error).__name__} {error}".lower()
    return "out of memory" in text or "outofmemory" in text.replace(" ", "")


def resolve_training_batch(config: TrainConfig) -> TrainConfig:
    """Return a config whose physical batch is an int. auto_fit is not resolved here."""

    if config.training.physical_batch_size == "auto_fit":
        raise RuntimeError(
            "physical_batch_size=auto_fit must be resolved before creating a run"
        )
    resolve_gradient_accumulation(config.training)
    return with_resolved_batch(config, int(config.training.physical_batch_size))
