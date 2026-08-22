"""Tracked expert-replay gate and production replay loop."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from pydantic import BaseModel, ConfigDict, Field, model_validator

from vla_fewshot.env.action_adapter import (
    ACTION_DIM,
    assert_env_action_stepable,
    dataset_action_to_env,
    dual_space_trace_record,
)
from vla_fewshot.env.gripper import action_is_finite
from vla_fewshot.reproducibility import atomic_write_json, atomic_write_text


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ReplayGateEpisode(_StrictModel):
    slug: str
    suite: str
    task_text: str
    task_index: int = Field(ge=0)
    episode_id: int = Field(ge=0)
    task_local_index: int = Field(ge=0)
    env_task_id: int | None = Field(default=None, ge=0)
    env_init_state_id: int = Field(default=0, ge=0)


class ReplayGate(_StrictModel):
    schema_version: int = Field(ge=1)
    dataset_repo_id: str
    dataset_revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    selection_rule: str
    episodes: list[ReplayGateEpisode]

    @model_validator(mode="after")
    def require_six_diverse_episodes(self) -> "ReplayGate":
        if len(self.episodes) != 6:
            raise ValueError("replay gate must list exactly six episodes")
        slugs = [item.slug for item in self.episodes]
        if len(set(slugs)) != 6:
            raise ValueError("replay gate slugs must be unique")
        targets = {item.slug for item in self.episodes if item.suite == "libero_goal"}
        seen = {item.slug for item in self.episodes if item.suite == "libero_90"}
        if targets != {"drawer_middle", "bowl_stove", "wine_cabinet"}:
            raise ValueError("replay gate must include one episode per target task")
        if len(seen) != 3:
            raise ValueError("replay gate must include three diverse seen episodes")
        return self


def load_replay_gate(path: str | Path) -> ReplayGate:
    with Path(path).open("r", encoding="utf-8") as handle:
        return ReplayGate.model_validate(json.load(handle))
