"""Refuse live LIBERO/SmolVLA evaluation until hardware gates pass."""

from __future__ import annotations

import platform


def refuse_full_evaluation() -> None:
    """Fail closed before env/policy allocation."""

    host = platform.system()
    if host != "Linux":
        raise RuntimeError(
            "full LIBERO evaluation requires Linux + CUDA + gpu extra; "
            f"current host is {host}. no GPU evaluation was started. "
            "Use --profile static for the CPU protocol smoke."
        )
    try:
        import torch
        import lerobot.envs.libero  # noqa: F401
        import lerobot.policies.smolvla  # noqa: F401
    except ImportError as error:
        raise RuntimeError(
            "full evaluation requires `uv sync --frozen --extra gpu`; "
            "no GPU evaluation was started."
        ) from error
    if not torch.cuda.is_available():
        raise RuntimeError(
            "full evaluation requires CUDA; no GPU evaluation was started. "
            "Use --profile static for the CPU protocol smoke."
        )
    raise RuntimeError(
        "Live SmolVLA/LIBERO rollouts wait until M1/M3/M4 hardware gates pass. "
        "no GPU evaluation was started. Use --profile static."
    )
