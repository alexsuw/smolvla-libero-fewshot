"""Paired correct/wrong instruction mapping and trajectory divergence."""

from __future__ import annotations

import math
from typing import Mapping, Sequence

from vla_fewshot.config import EvalConfig
from vla_fewshot.data.expected import TARGET_TASKS
from vla_fewshot.data.task_text import normalize_task_text, task_text_matches
from vla_fewshot.evaluation.protocol import ProtocolError

DEFAULT_WRONG_INSTRUCTION_MAP = {
    "drawer_middle": "put the bowl on the stove",
    "bowl_stove": "put the wine bottle on top of the cabinet",
    "wine_cabinet": "open the middle drawer of the cabinet",
}


def wrong_instruction_map(config: EvalConfig) -> dict[str, str]:
    mapping = config.wrong_instruction_map or DEFAULT_WRONG_INSTRUCTION_MAP
    required = set(TARGET_TASKS)
    if set(mapping) != required:
        raise ProtocolError(
            f"wrong_instruction_map must cover exactly {sorted(required)}"
        )
    normalized = {
        slug: normalize_task_text(text) for slug, text in mapping.items()
    }
    for slug, text in normalized.items():
        own = str(TARGET_TASKS[slug]["task_text"])
        if task_text_matches(text, own):
            raise ProtocolError(f"wrong instruction for {slug} matches the correct text")
        if text not in {str(item["task_text"]) for item in TARGET_TASKS.values()}:
            raise ProtocolError(
                f"wrong instruction for {slug} is not a tracked target text: {text!r}"
            )
    return normalized


def instruction_for(
    *,
    task_slug: str,
    condition: str,
    mapping: Mapping[str, str],
) -> str:
    if task_slug not in TARGET_TASKS:
        raise ProtocolError(f"unknown target task {task_slug!r}")
    if condition == "correct":
        return str(TARGET_TASKS[task_slug]["task_text"])
    if condition == "wrong":
        return mapping[task_slug]
    raise ProtocolError(f"instruction_condition must be correct|wrong, got {condition!r}")


def _align(left: Sequence[Sequence[float]], right: Sequence[Sequence[float]]) -> int:
    return min(len(left), len(right))


def action_l2_divergence(
    left: Sequence[Sequence[float]],
    right: Sequence[Sequence[float]],
) -> float:
    n = _align(left, right)
    if n == 0:
        raise ProtocolError("cannot compare empty action traces")
    total = 0.0
    for index in range(n):
        if len(left[index]) != len(right[index]):
            raise ProtocolError("action dimension mismatch in paired traces")
        total += math.sqrt(
            sum((a - b) ** 2 for a, b in zip(left[index], right[index], strict=True))
        )
    return total / n


def action_cosine_divergence(
    left: Sequence[Sequence[float]],
    right: Sequence[Sequence[float]],
) -> float:
    n = _align(left, right)
    if n == 0:
        raise ProtocolError("cannot compare empty action traces")
    total = 0.0
    for index in range(n):
        a = left[index]
        b = right[index]
        if len(a) != len(b):
            raise ProtocolError("action dimension mismatch in paired traces")
        dot = sum(x * y for x, y in zip(a, b, strict=True))
        na = math.sqrt(sum(x * x for x in a))
        nb = math.sqrt(sum(y * y for y in b))
        if na == 0.0 or nb == 0.0:
            cosine = 1.0 if na == nb else 0.0
        else:
            cosine = max(-1.0, min(1.0, dot / (na * nb)))
        gap = 1.0 - cosine
        if abs(gap) < 1e-12:
            gap = 0.0
        total += gap
    return total / n
