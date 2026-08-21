"""Paired correct/wrong instruction mapping and trajectory divergence."""

from __future__ import annotations

import math
from pathlib import Path
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


LANGUAGE_CONTROL_SLUGS = tuple(TARGET_TASKS)
MIN_ROLLOUTS = 20


def assert_language_control_config(config: EvalConfig, *, profile: str) -> None:
    if config.stage != "language_control":
        raise ProtocolError(
            "language-control evaluation requires configs/eval/language_control.yaml"
        )
    if not config.protocol.hard_reset:
        raise ProtocolError("language control requires hard_reset: true")
    wrong_instruction_map(config)
    if profile == "full":
        if config.protocol.protocol_id != "final_language_control_v1":
            raise ProtocolError(
                "language-control full eval must use protocol_id=final_language_control_v1"
            )
        if config.protocol.rollouts_per_cell < MIN_ROLLOUTS:
            raise ProtocolError(
                f"language control requires ≥{MIN_ROLLOUTS} paired seeds per task"
            )


def assert_language_control_cell(
    *,
    n_demos: int | None,
    train_seed: int | None,
    episode_ids: list[int],
) -> None:
    if n_demos not in (None, 0):
        raise ProtocolError("language control uses 0 target demonstrations")
    if train_seed is not None:
        raise ProtocolError("language control has no adaptation train seed")
    if episode_ids:
        raise ProtocolError("language control training episode list must be empty")


def language_control_commands(
    *,
    config: Path = Path("configs/eval/language_control.yaml"),
) -> list[list[str]]:
    return [
        [
            "python",
            "scripts/eval_language_control.py",
            "--config",
            str(config),
            "--task",
            task,
            "--profile",
            "full",
        ]
        for task in LANGUAGE_CONTROL_SLUGS
    ]
