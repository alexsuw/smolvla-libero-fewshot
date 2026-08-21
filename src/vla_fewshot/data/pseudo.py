"""Frozen pseudo-target tasks chosen only from libero_90."""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, model_validator

from vla_fewshot.data.expected import SEEN_SUITE, TARGET_TASKS
from vla_fewshot.data.task_text import normalize_task_text, task_text_matches


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class PseudoTargetTask(_StrictModel):
    task_text: str
    task_index: int = Field(ge=0)

    @model_validator(mode="after")
    def normalized_text(self) -> "PseudoTargetTask":
        if self.task_text != normalize_task_text(self.task_text):
            raise ValueError("task_text must already be normalized")
        return self


class PseudoTargetSplits(_StrictModel):
    schema_version: int = Field(ge=1)
    status: str
    dataset_repo_id: str
    dataset_revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    suite: str
    selection_rule: str = Field(min_length=20)
    tasks: dict[str, PseudoTargetTask]

    @model_validator(mode="after")
    def validate_contract(self) -> "PseudoTargetSplits":
        if self.status != "frozen":
            raise ValueError("pseudo-target splits must be frozen before the real grid")
        if self.suite != SEEN_SUITE:
            raise ValueError("pseudo-target suite must be libero_90")
        if len(self.tasks) != 3:
            raise ValueError("pseudo-target calibration uses exactly three tasks")
        target_texts = [str(spec["task_text"]) for spec in TARGET_TASKS.values()]
        for slug, task in self.tasks.items():
            if any(task_text_matches(task.task_text, text) for text in target_texts):
                raise ValueError(
                    f"pseudo-target {slug} copies a held-out target instruction"
                )
        return self

    @property
    def slugs(self) -> tuple[str, ...]:
        return tuple(self.tasks)


def load_pseudo_target_splits(path: str | Path) -> PseudoTargetSplits:
    with Path(path).open("r", encoding="utf-8") as handle:
        return PseudoTargetSplits.model_validate(json.load(handle))
