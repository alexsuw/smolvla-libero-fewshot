"""Tracked target-split structural validation.

Metadata provenance is checked against the pinned dataset only in M2.
"""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, model_validator

from vla_fewshot.data.task_text import normalize_task_text


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class TargetSplitTask(_StrictModel):
    task_text: str
    task_index: int = Field(ge=0)
    available_count: int = Field(ge=25)
    episode_ids_first_25: list[int]

    @model_validator(mode="after")
    def validate_episode_ids(self) -> "TargetSplitTask":
        if len(self.episode_ids_first_25) != 25:
            raise ValueError("episode_ids_first_25 must contain exactly 25 IDs")
        if len(set(self.episode_ids_first_25)) != 25:
            raise ValueError("episode IDs must be unique")
        if any(episode_id < 0 for episode_id in self.episode_ids_first_25):
            raise ValueError("episode IDs must be non-negative")
        if self.task_text != normalize_task_text(self.task_text):
            raise ValueError("task_text must already be normalized")
        return self

    def ids_for_budget(self, n_demos: int) -> list[int]:
        if n_demos not in {5, 10, 25}:
            raise ValueError("n_demos must be one of 5, 10, 25")
        return self.episode_ids_first_25[:n_demos]


class TargetSplits(_StrictModel):
    schema_version: int = Field(ge=1)
    dataset_repo_id: str
    dataset_revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    suite: str
    selection_rule: str
    tasks: dict[str, TargetSplitTask]

    @model_validator(mode="after")
    def validate_contract(self) -> "TargetSplits":
        if self.suite != "libero_goal":
            raise ValueError("target suite must be libero_goal")
        required = {"drawer_middle", "bowl_stove", "wine_cabinet"}
        if set(self.tasks) != required:
            raise ValueError(f"tasks must be exactly {sorted(required)}")
        for task in self.tasks.values():
            ids_5 = task.ids_for_budget(5)
            ids_10 = task.ids_for_budget(10)
            ids_25 = task.ids_for_budget(25)
            if ids_5 != ids_10[:5] or ids_10 != ids_25[:10]:
                raise ValueError("few-shot episode prefixes are not nested")
        return self


def load_target_splits(path: str | Path) -> TargetSplits:
    with Path(path).open("r", encoding="utf-8") as handle:
        return TargetSplits.model_validate(json.load(handle))
