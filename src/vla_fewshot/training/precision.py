"""Resolve mixed precision without a silent unrecorded fallback."""

from __future__ import annotations

from typing import Any, Literal

PrecisionName = Literal["bf16", "fp16", "fp32"]


def resolve_precision(requested: str, *, cuda_bf16: bool | None = None) -> PrecisionName:
    """Record the dtype that will actually be used.

    ``auto`` may choose bf16, then fp16, then fp32. An explicit ``bf16`` request
    fails if the GPU cannot do it.
    """

    if requested == "fp32":
        return "fp32"
    if requested == "fp16":
        return "fp16"
    if requested == "bf16":
        if cuda_bf16 is False:
            raise RuntimeError("mixed_precision=bf16 was requested but the GPU has no BF16")
        return "bf16"
    if requested != "auto":
        raise ValueError(f"unsupported mixed_precision {requested!r}")
    if cuda_bf16:
        return "bf16"
    return "fp16"


def cuda_bf16_supported() -> bool:
    try:
        import torch
    except ImportError:
        return False
    return bool(torch.cuda.is_available() and torch.cuda.is_bf16_supported())


def autocast_cm(precision: PrecisionName) -> Any:
    from contextlib import nullcontext

    if precision == "fp32":
        return nullcontext()
    import torch

    dtype = torch.bfloat16 if precision == "bf16" else torch.float16
    return torch.autocast("cuda", dtype=dtype)
