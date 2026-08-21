"""Optional LoRA wrapping. Not used by the primary seen-pretrain recipe."""

from __future__ import annotations


def refuse_peft_until_challenger() -> None:
    raise RuntimeError(
        "PEFT/LoRA wrapping is the optional seen challenger, not the M4/M5 primary path"
    )
