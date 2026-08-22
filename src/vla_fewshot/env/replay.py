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


@dataclass(frozen=True)
class ReplayResult:
    success: bool
    steps: int
    terminated: bool
    output_dir: Path
    manifest: dict[str, Any]


def replay_actions_through_env(
    *,
    env: Any,
    dataset_actions: Sequence[Sequence[float]],
    output_dir: Path,
    task_text: str,
    suite: str,
    episode_id: int,
    seed: int,
    save_video: bool,
    binary_gripper: bool = True,
    threshold: float = 0.5,
    save_frame,
) -> ReplayResult:
    """Step production gripper conversion. `env` must implement reset/step/close."""

    if not dataset_actions:
        raise ValueError("dataset_actions is empty")
    output_dir.mkdir(parents=True, exist_ok=True)
    observation, info = env.reset(seed=seed)
    traces: list[dict[str, Any]] = []
    frames: list[Any] = []
    terminated = False
    success = False
    last_info = info if isinstance(info, dict) else {}

    for step, dataset_action in enumerate(dataset_actions):
        if not action_is_finite(dataset_action) or len(dataset_action) != ACTION_DIM:
            raise ValueError(f"illegal dataset action at step {step}")
        env_action = dataset_action_to_env(
            dataset_action, binary=binary_gripper, threshold=threshold
        )
        assert_env_action_stepable(env_action)
        record = dual_space_trace_record(
            step=step,
            dataset_action=dataset_action,
            env_action=env_action,
        )
        observation, _reward, terminated, _truncated, last_info = env.step(env_action)
        success = bool(_info_success(last_info))
        record["is_success"] = success
        record["terminated"] = bool(terminated)
        traces.append(record)
        if save_video and save_frame is not None:
            frames.append(save_frame(observation))
        if terminated or success:
            break

    for record in traces:
        if record.get("out_of_range"):
            raise ValueError(
                f"env action out of range at step {record['step']}: {record['env_action']}"
            )

    manifest = {
        "schema_version": 1,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "suite": suite,
        "task_text": task_text,
        "episode_id": episode_id,
        "seed": seed,
        "steps": len(traces),
        "dataset_action_count": len(dataset_actions),
        "success": success,
        "terminated": bool(terminated),
        "expected_success": True,
        "gripper_postprocessor": "binary_dataset_gripper_to_env" if binary_gripper else "dataset_gripper_to_env",
        "orientation": {
            "lerobot_libero_processor_rot180": True,
            "project_transform": "identity",
        },
        "camera_map": {
            "main": "observation.images.image",
            "wrist": "observation.images.wrist_image",
            "env_wrist": "image2",
        },
        "save_video": save_video,
        "frame_count": len(frames),
    }
    atomic_write_json(output_dir / "manifest.json", manifest, overwrite=True)
    atomic_write_text(
        output_dir / "trace.jsonl",
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in traces),
        overwrite=True,
    )
    if save_video:
        _write_ppm_contact_sheet(output_dir / "frames", frames)
    return ReplayResult(
        success=success,
        steps=len(traces),
        terminated=bool(terminated),
        output_dir=output_dir,
        manifest=manifest,
    )


def _info_success(info: Any) -> bool:
    if not isinstance(info, dict):
        return False
    value = info.get("is_success", False)
    if isinstance(value, (list, tuple)):
        value = value[0] if value else False
    if hasattr(value, "item"):
        value = value.item()
    return bool(value)


def _write_ppm_contact_sheet(frame_dir: Path, frames: Iterable[Any]) -> None:
    """Write HWC uint8-like frames as PPM. MP4 encoding is hardware/ffmpeg gated."""

    frame_dir.mkdir(parents=True, exist_ok=True)
    for index, frame in enumerate(frames):
        if frame is None:
            continue
        path = frame_dir / f"frame-{index:05d}.ppm"
        _write_ppm(path, frame)


def _write_ppm(path: Path, frame: Any) -> None:
    rows = frame
    if hasattr(frame, "tolist"):
        rows = frame.tolist()
    height = len(rows)
    width = len(rows[0]) if height else 0
    header = f"P6\n{width} {height}\n255\n".encode("ascii")
    body = bytearray()
    for row in rows:
        for pixel in row:
            channels = list(pixel[:3])
            body.extend(int(max(0, min(255, channel))) for channel in channels)
    path.write_bytes(header + body)


def load_episode_actions_from_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    start_index: int,
    length: int,
) -> list[list[float]]:
    by_index = {}
    for row in rows:
        if "index" not in row or "action" not in row:
            continue
        by_index[int(row["index"])] = [float(item) for item in row["action"]]
    actions = []
    for offset in range(length):
        key = start_index + offset
        if key not in by_index:
            raise KeyError(f"missing action index {key}")
        action = by_index[key]
        if len(action) != ACTION_DIM or not action_is_finite(action):
            raise ValueError(f"illegal action at index {key}")
        actions.append(action)
    return actions


def load_episode_actions(
    revision_root: Path,
    suite: str,
    episode_id: int,
) -> list[list[float]]:
    from vla_fewshot.data.layout import suite_root
    from vla_fewshot.data.metadata import load_suite_metadata, read_parquet_rows

    metadata = load_suite_metadata(revision_root, suite)
    match = next(
        (
            row
            for row in metadata.episodes
            if int(row["episode_index"]) == episode_id
        ),
        None,
    )
    if match is None:
        raise KeyError(f"episode {episode_id} not in {suite} metadata")
    start = int(match.get("dataset_from_index", 0))
    length = int(match["length"])
    data_files = sorted((suite_root(revision_root, suite) / "data").rglob("*.parquet"))
    if not data_files:
        raise FileNotFoundError(
            f"missing {suite}/data parquet; download with --include-actions"
        )
    return load_episode_actions_from_rows(
        read_parquet_rows(data_files),
        start_index=start,
        length=length,
    )
