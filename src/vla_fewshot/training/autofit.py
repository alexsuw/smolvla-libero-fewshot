"""CUDA auto-fit. Must finish before the final run directory is created."""

from __future__ import annotations

from typing import Any, Callable

from vla_fewshot.config import TrainConfig
from vla_fewshot.training.batching import (
    auto_fit_candidates,
    is_cuda_oom,
    with_resolved_batch,
)


def fit_physical_batch(
    config: TrainConfig,
    *,
    try_batch: Callable[[int], None],
) -> TrainConfig:
    """Try physical sizes 4, 2, 1 that divide the effective batch.

    ``try_batch`` should run one forward+backward at that physical size and
    raise CUDA OOM on failure. Non-OOM errors propagate. After success the
    resolved integers are frozen into a copy of ``config``.
    """

    if config.training.physical_batch_size != "auto_fit":
        return with_resolved_batch(config, int(config.training.physical_batch_size))

    last_error: BaseException | None = None
    for physical in auto_fit_candidates(config.training.effective_batch_size):
        try:
            try_batch(physical)
        except Exception as error:
            if is_cuda_oom(error):
                last_error = error
                _empty_cuda_cache()
                continue
            raise
        return with_resolved_batch(config, physical)
    raise RuntimeError(
        "auto_fit could not find a physical batch that fits in VRAM "
        f"(tried {auto_fit_candidates(config.training.effective_batch_size)}). "
        "no GPU training run directory was created."
    ) from last_error


def _empty_cuda_cache() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        return


def try_smolvla_minibatch(
    *,
    policy: Any,
    optimizer: Any,
    batch: dict[str, Any],
    precision: str,
) -> None:
    """One training micro-step used only during auto-fit."""

    import torch

    from vla_fewshot.training.precision import autocast_cm

    policy.train()
    optimizer.zero_grad(set_to_none=True)
    with autocast_cm(precision):  # type: ignore[arg-type]
        loss, _details = policy.forward(batch)
        if not torch.isfinite(loss):
            raise FloatingPointError("non-finite auto-fit loss")
        loss.backward()
    optimizer.zero_grad(set_to_none=True)
    _empty_cuda_cache()
