"""Fixed evaluation seeds, unique keys, and hard-reset protocol checks."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Sequence

from vla_fewshot.config import EvalConfig
from vla_fewshot.data.expected import TARGET_TASKS
from vla_fewshot.data.splits import TargetSplits
from vla_fewshot.data.task_text import normalize_task_text

InstructionCondition = Literal["correct", "wrong"]

FINAL_SEED_VALUES = list(range(1000, 1020))
UNIQUE_KEY_FIELDS = (
    "checkpoint_sha256",
    "task_slug",
    "n_demos",
    "train_seed",
    "eval_seed",
    "instruction_condition",
    "protocol_id",
)


class ProtocolError(ValueError):
    """Raised when an evaluation contract is violated."""


def load_eval_seeds(path: Path) -> list[int]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload != FINAL_SEED_VALUES:
        raise ProtocolError(f"{path} must contain the fixed seeds 1000..1019")
    return list(payload)


def seeds_for_config(config: EvalConfig, *, project_root: Path) -> list[int]:
    seeds_path = Path(config.protocol.seeds_file)
    if not seeds_path.is_absolute():
        seeds_path = project_root / seeds_path
    seeds = load_eval_seeds(seeds_path)
    count = config.protocol.rollouts_per_cell
    if count > len(seeds):
        raise ProtocolError(
            f"rollouts_per_cell={count} exceeds the {len(seeds)} tracked eval seeds"
        )
    return seeds[:count]


def assert_hard_reset(config: EvalConfig) -> None:
    if config.protocol.hard_reset:
        return
    if config.protocol.protocol_id == "dev_soft_reset":
        return
    raise ProtocolError(
        "hard_reset must stay true for final evaluation; "
        "soft reset is only allowed as protocol_id=dev_soft_reset"
    )


def assert_eval_tracking(config: EvalConfig) -> None:
    if config.tracking.wandb_enabled:
        raise ProtocolError("evaluation tracking.wandb_enabled must stay false")
    if not config.protocol.deterministic:
        raise ProtocolError("final evaluation must request deterministic inference")


def n_demos_token(n_demos: int | None) -> str:
    if n_demos in (None, 0):
        return "n00"
    return f"n{int(n_demos):02d}"


def train_seed_token(train_seed: int | None) -> str:
    if train_seed is None:
        return "sna"
    return f"s{train_seed}"


def rollout_key(
    *,
    checkpoint_sha256: str,
    task_slug: str,
    n_demos: int | None,
    train_seed: int | None,
    eval_seed: int,
    instruction_condition: str,
    protocol_id: str,
) -> tuple[Any, ...]:
    return (
        checkpoint_sha256,
        task_slug,
        0 if n_demos is None else int(n_demos),
        train_seed,
        int(eval_seed),
        instruction_condition,
        protocol_id,
    )


def rollout_key_from_record(record: dict[str, Any]) -> tuple[Any, ...]:
    missing = [field for field in UNIQUE_KEY_FIELDS if field not in record]
    if missing:
        raise ProtocolError(f"rollout record missing unique-key fields: {missing}")
    return rollout_key(
        checkpoint_sha256=str(record["checkpoint_sha256"]),
        task_slug=str(record["task_slug"]),
        n_demos=record["n_demos"],
        train_seed=record["train_seed"],
        eval_seed=int(record["eval_seed"]),
        instruction_condition=str(record["instruction_condition"]),
        protocol_id=str(record["protocol_id"]),
    )


def key_slug(key: tuple[Any, ...]) -> str:
    parts = []
    for item in key:
        if item is None:
            parts.append("none")
        else:
            parts.append(str(item).replace("/", "_"))
    return "__".join(parts)


def target_task(slug: str) -> dict[str, object]:
    if slug not in TARGET_TASKS:
        raise ProtocolError(f"unknown target task {slug!r}")
    return TARGET_TASKS[slug]


def training_episode_ids(
    splits: TargetSplits | None,
    *,
    task_slug: str,
    n_demos: int | None,
) -> list[int]:
    if n_demos in (None, 0):
        return []
    if splits is None:
        raise ProtocolError("target splits are required when n_demos > 0")
    return list(splits.tasks[task_slug].ids_for_budget(int(n_demos)))


@dataclass(frozen=True)
class PlannedRollout:
    task_slug: str
    task_text: str
    suite: str
    task_index: int
    n_demos: int | None
    train_seed: int | None
    eval_seed: int
    rollout_index: int
    instruction_condition: InstructionCondition
    instruction_text_used: str
    protocol_id: str

    def key(self, checkpoint_sha256: str) -> tuple[Any, ...]:
        return rollout_key(
            checkpoint_sha256=checkpoint_sha256,
            task_slug=self.task_slug,
            n_demos=self.n_demos,
            train_seed=self.train_seed,
            eval_seed=self.eval_seed,
            instruction_condition=self.instruction_condition,
            protocol_id=self.protocol_id,
        )


def plan_named_rollouts(
    config: EvalConfig,
    *,
    task_slug: str,
    task_text: str,
    suite: str,
    task_index: int,
    n_demos: int | None,
    train_seed: int | None,
    seeds: Sequence[int],
    instruction_condition: InstructionCondition = "correct",
    instruction_text: str | None = None,
) -> list[PlannedRollout]:
    assert_hard_reset(config)
    assert_eval_tracking(config)
    text = normalize_task_text(instruction_text or task_text)
    return [
        PlannedRollout(
            task_slug=task_slug,
            task_text=normalize_task_text(task_text),
            suite=suite,
            task_index=task_index,
            n_demos=n_demos,
            train_seed=train_seed,
            eval_seed=int(seed),
            rollout_index=index,
            instruction_condition=instruction_condition,
            instruction_text_used=text,
            protocol_id=config.protocol.protocol_id,
        )
        for index, seed in enumerate(seeds)
    ]


def plan_target_rollouts(
    config: EvalConfig,
    *,
    task_slug: str,
    n_demos: int | None,
    train_seed: int | None,
    seeds: Sequence[int],
    instruction_condition: InstructionCondition = "correct",
    instruction_text: str | None = None,
) -> list[PlannedRollout]:
    spec = target_task(task_slug)
    return plan_named_rollouts(
        config,
        task_slug=task_slug,
        task_text=str(spec["task_text"]),
        suite=config.dataset.suite_target,
        task_index=int(spec["task_index"]),
        n_demos=n_demos,
        train_seed=train_seed,
        seeds=seeds,
        instruction_condition=instruction_condition,
        instruction_text=instruction_text,
    )
