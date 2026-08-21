"""Refuse LeRobot's WandB-backed trainer; full SmolVLA smoke waits on hardware."""

from __future__ import annotations

import platform

from vla_fewshot.model.smolvla import require_smolvla_runtime


def refuse_full_smolvla_training() -> None:
    """Fail closed before GPU allocation or any WandB-backed LeRobot train CLI."""

    if platform.system() != "Linux":
        raise RuntimeError(
            "full SmolVLA training requires Linux + CUDA + gpu extra; "
            f"current host is {platform.system()}. "
            "no GPU training was started. Use --profile static for the CPU "
            "checkpoint/resume smoke."
        )
    require_smolvla_runtime()
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
    raise RuntimeError(
        "Pinned LeRobot `lerobot-train` uses WandBLogger and is not called. "
        "Full 200-step SmolVLA smoke waits until M1/M3/M4 hardware gates pass. "
        "no GPU training was started. Use --profile static."
    )
