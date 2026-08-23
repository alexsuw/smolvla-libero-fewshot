"""Evaluate every baseline checkpoint: ≥20 rollouts, all failures have video."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from vla_fewshot.calibration import load_selected_checkpoint
from vla_fewshot.data.splits import TargetSplits
from vla_fewshot.evaluation.protocol import FINAL_SEED_VALUES
from vla_fewshot.evaluation.select import list_complete_checkpoints
from vla_fewshot.evaluation.store import RolloutStore
from vla_fewshot.logging.manifest import json_load
from vla_fewshot.storage.layout import MANIFEST_NAME, step_directory_name
from vla_fewshot.training.baseline import TARGET_SLUGS, TRAIN_SEEDS, baseline_grid, episode_ids_for_cell

MIN_ROLLOUTS = 20
FINAL_PROTOCOL = "final_v1"


class BaselineEvalError(RuntimeError):
    """Raised when a baseline eval cell is incomplete or leaks protocol."""


def baseline_eval_command(
    *,
    task: str,
    n_demos: int,
    seed: int,
    run_dir: str = "RUN_DIR",
    output_dir: str = "EVAL_DIR",
    config: Path = Path("configs/eval/final.yaml"),
    train_config: Path = Path("configs/train/target_baseline.yaml"),
) -> list[str]:
    command = [
        "python",
        "scripts/eval_target.py",
        "--config",
        str(config),
        "--profile",
        "full",
        "--task",
        task,
        "--n-demos",
        str(n_demos),
        "--seed",
        str(seed),
        "--run-dir",
        run_dir,
        "--output-dir",
        output_dir,
    ]
    if Path(train_config) != Path("configs/train/target_baseline.yaml"):
        command.extend(["--train-config", str(train_config)])
    return command


def baseline_eval_commands(
    *,
    config: Path = Path("configs/eval/final.yaml"),
    train_config: Path = Path("configs/train/target_baseline.yaml"),
) -> list[list[str]]:
    return target_eval_commands(config=config, train_config=train_config)


def target_eval_commands(
    *,
    config: Path = Path("configs/eval/final.yaml"),
    train_config: Path = Path("configs/train/target_baseline.yaml"),
) -> list[list[str]]:
    return [
        baseline_eval_command(
            task=task,
            n_demos=n_demos,
            seed=seed,
            config=config,
            train_config=train_config,
        )
        for task, n_demos, seed in baseline_grid()
    ]


def eval_rollouts_path(eval_dir: Path, *, step: int, task_slug: str) -> Path | None:
    label = step_directory_name(step)
    candidates = (
        eval_dir / label / task_slug / "rollouts.jsonl",
        eval_dir / label / "rollouts.jsonl",
        eval_dir / task_slug / "rollouts.jsonl",
        eval_dir / "rollouts.jsonl",
    )
    for path in candidates:
        if path.is_file():
            return path
    return None


def check_baseline_eval_records(
    records: list[dict[str, Any]],
    *,
    task_slug: str,
    n_demos: int,
    train_seed: int,
    episode_ids: list[int],
    min_rollouts: int = MIN_ROLLOUTS,
    method: str = "baseline",
    require_videos: bool = True,
    require_traces: bool = True,
) -> dict[str, Any]:
    if not records:
        raise BaselineEvalError(
            f"{task_slug} n={n_demos} s={train_seed}: no rollout records"
        )
    seeds = sorted({int(record["eval_seed"]) for record in records})
    if len(records) < min_rollouts:
        raise BaselineEvalError(
            f"{task_slug} n={n_demos} s={train_seed}: {len(records)} rollouts < {min_rollouts}"
        )
    if min_rollouts >= MIN_ROLLOUTS and seeds[:min_rollouts] != FINAL_SEED_VALUES[:min_rollouts]:
        raise BaselineEvalError(
            f"{task_slug} n={n_demos} s={train_seed}: eval seeds must start at 1000"
        )
    failures_missing_video = 0
    missing_traces = 0
    for record in records:
        if str(record.get("protocol_id", "")).startswith("static_"):
            raise BaselineEvalError("static eval rows cannot close the baseline grid")
        if record.get("protocol_id") != FINAL_PROTOCOL:
            raise BaselineEvalError(
                f"baseline eval requires protocol_id={FINAL_PROTOCOL}"
            )
        if record.get("method") != method:
            raise BaselineEvalError(f"eval records must have method={method}")
        if record.get("stage") != "target_eval":
            raise BaselineEvalError("baseline eval records must have stage=target_eval")
        if int(record.get("n_demos") or 0) != n_demos:
            raise BaselineEvalError("n_demos mismatch in baseline eval records")
        if int(record.get("train_seed") or -1) != train_seed:
            raise BaselineEvalError("train_seed mismatch in baseline eval records")
        if record.get("task_slug") != task_slug:
            raise BaselineEvalError("task_slug mismatch in baseline eval records")
        if list(record.get("training_episode_ids") or []) != episode_ids:
            raise BaselineEvalError("training_episode_ids are not the nested prefix")
        if require_traces and not record.get("trace_uri"):
            missing_traces += 1
        if (
            require_videos
            and int(record.get("success") or 0) == 0
            and not record.get("video_uri")
        ):
            failures_missing_video += 1
    if require_traces and missing_traces:
        raise BaselineEvalError(
            f"{task_slug} n={n_demos} s={train_seed}: {missing_traces} rollouts missing traces"
        )
    if require_videos and failures_missing_video:
        raise BaselineEvalError(
            f"{task_slug} n={n_demos} s={train_seed}: "
            f"{failures_missing_video} failures missing video"
        )
    return {
        "task_slug": task_slug,
        "n_demos": n_demos,
        "train_seed": train_seed,
        "n_rollouts": len(records),
        "n_failures": sum(1 for record in records if int(record.get("success") or 0) == 0),
        "n_checkpoints_checked": 1,
    }


def _origin_hash_ok(train_dir: Path) -> None:
    selected = load_selected_checkpoint()
    if selected.status != "frozen" or not selected.sha256:
        return
    manifest_path = train_dir / MANIFEST_NAME
    if not manifest_path.is_file():
        raise BaselineEvalError(f"training manifest missing: {manifest_path}")
    manifest = json_load(manifest_path)
    observed = manifest.get("base_checkpoint_sha256") or manifest.get("origin_checkpoint_sha256")
    if observed != selected.sha256:
        raise BaselineEvalError(
            f"{train_dir} origin hash {observed} != frozen seen {selected.sha256}"
        )


def verify_baseline_run_eval(
    train_dir: Path,
    eval_dir: Path,
    *,
    task_slug: str,
    n_demos: int,
    train_seed: int,
    splits: TargetSplits,
    min_rollouts: int = MIN_ROLLOUTS,
    method: str = "baseline",
    final_only: bool = False,
    require_videos: bool = True,
    require_traces: bool = True,
) -> dict[str, Any]:
    if task_slug not in TARGET_SLUGS:
        raise BaselineEvalError(f"unknown target task {task_slug!r}")
    if n_demos not in {1, 2, 5, 10, 25} or train_seed not in TRAIN_SEEDS:
        raise BaselineEvalError("n_demos must be 1/2/5/10/25 and seed 42/123")
    _origin_hash_ok(train_dir)
    episode_ids = episode_ids_for_cell(splits, task_slug=task_slug, n_demos=n_demos)
    checkpoints = list_complete_checkpoints(train_dir)
    if not checkpoints:
        raise BaselineEvalError(f"no complete checkpoints under {train_dir}")
    if final_only:
        checkpoints = [checkpoints[-1]]
    checked: list[dict[str, Any]] = []
    for step, _ckpt in checkpoints:
        jsonl = eval_rollouts_path(eval_dir, step=step, task_slug=task_slug)
        if jsonl is None:
            raise BaselineEvalError(
                f"missing eval rollouts for {train_dir} step {step}"
            )
        records = list(RolloutStore(jsonl).records())
        checked.append(
            check_baseline_eval_records(
                records,
                task_slug=task_slug,
                n_demos=n_demos,
                train_seed=train_seed,
                episode_ids=episode_ids,
                min_rollouts=min_rollouts,
                method=method,
                require_videos=require_videos,
                require_traces=require_traces,
            )
        )
    return {
        "train_dir": str(train_dir),
        "eval_dir": str(eval_dir),
        "n_checkpoints": len(checkpoints),
        "cells": checked,
        "complete": True,
    }


def discover_baseline_runs(train_root: Path, *, method: str = "baseline") -> list[dict[str, Any]]:
    found: list[dict[str, Any]] = []
    if not train_root.exists():
        return found
    for manifest_path in sorted(train_root.rglob(MANIFEST_NAME)):
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        if payload.get("stage") != "target" or payload.get("method") != method:
            continue
        found.append(
            {
                "train_dir": manifest_path.parent,
                "task_slug": payload.get("task_slug"),
                "n_demos": payload.get("n_demos"),
                "train_seed": payload.get("train_seed"),
            }
        )
    return found


def default_eval_dir(eval_root: Path, *, task_slug: str, n_demos: int, train_seed: int, train_dir: Path) -> Path:
    named = eval_root / f"{task_slug}_n{int(n_demos):02d}_s{train_seed}"
    if named.exists():
        return named
    sibling = eval_root / train_dir.name
    if sibling.exists():
        return sibling
    return named
