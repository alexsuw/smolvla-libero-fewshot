"""Linux CUDA runtime gate for live LIBERO/SmolVLA evaluation."""

from __future__ import annotations

import platform

from vla_fewshot.calibration import load_selected_checkpoint


def require_full_evaluation_runtime() -> None:
    """Fail closed unless Linux, the gpu extra, and CUDA are available."""

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


def assert_seen_checkpoint_frozen() -> None:
    """Target/language eval must not run before the seen checkpoint is frozen."""

    selected = load_selected_checkpoint()
    if selected.status != "frozen" or not selected.sha256:
        raise RuntimeError(
            "target evaluation waits until configs/selected_seen_checkpoint.yaml "
            "is frozen from seen probes. no GPU evaluation was started."
        )


def refuse_full_evaluation() -> None:
    """Backward-compatible alias: runtime gate only, no extra hardware-wait raise."""

    require_full_evaluation_runtime()
