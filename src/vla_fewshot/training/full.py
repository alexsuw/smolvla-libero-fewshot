"""Linux CUDA runtime gate for project-owned SmolVLA training.

Pinned LeRobot `lerobot-train` is never called: it constructs WandBLogger.
"""

from __future__ import annotations

import platform

from vla_fewshot.model.smolvla import require_smolvla_runtime


def require_full_training_runtime() -> None:
    """Fail closed unless Linux, the gpu extra, and CUDA are available."""

    if platform.system() != "Linux":
        raise RuntimeError(
            "full SmolVLA training requires Linux + CUDA + gpu extra; "
            f"current host is {platform.system()}. "
            "no GPU training was started. Use --profile static for the CPU "
            "checkpoint/resume smoke."
        )
    try:
        require_smolvla_runtime()
    except RuntimeError as error:
        raise RuntimeError(f"{error}. no GPU training was started.") from error
    try:
        import torch
    except ImportError as error:
        raise RuntimeError(
            "full SmolVLA training requires `uv sync --frozen --extra gpu`; "
            "no GPU training was started."
        ) from error
    if not torch.cuda.is_available():
        raise RuntimeError(
            "full SmolVLA training requires CUDA; no GPU training was started. "
            "Use --profile static for the CPU checkpoint/resume smoke."
        )


def refuse_lerobot_train_cli() -> None:
    """Documented guard: do not shell out to WandB-backed lerobot-train."""

    return None


def refuse_full_smolvla_training() -> None:
    """Backward-compatible alias used by older tests and docs."""

    require_full_training_runtime()
