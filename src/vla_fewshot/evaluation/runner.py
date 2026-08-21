"""Fixed-seed evaluation runner with JSONL resume, traces, and videos."""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, Sequence

from vla_fewshot.config import EvalConfig
from vla_fewshot.data.splits import TargetSplits
from vla_fewshot.env.action_adapter import dataset_action_to_env
from vla_fewshot.evaluation.language_control import (
    action_cosine_divergence,
    action_l2_divergence,
    instruction_for,
    wrong_instruction_map,
)
from vla_fewshot.evaluation.metrics import cell_summary
from vla_fewshot.evaluation.protocol import (
    PlannedRollout,
    ProtocolError,
    assert_eval_tracking,
    assert_hard_reset,
    n_demos_token,
    plan_named_rollouts,
    plan_target_rollouts,
    rollout_key_from_record,
    seeds_for_config,
    train_seed_token,
    training_episode_ids,
)
from vla_fewshot.evaluation.store import RolloutStore, remaining_keys
from vla_fewshot.evaluation.toy import ToyEvalEnv, ToyEvalPolicy, fingerprint_observation
from vla_fewshot.evaluation.trace import load_actions, write_trace
from vla_fewshot.evaluation.video import (
    cell_id,
    should_persist_video,
    success_cells_from_records,
    write_ppm_video,
)
from vla_fewshot.logging.manifest import git_sha7, utc_timestamp
from vla_fewshot.reproducibility import _git_state, atomic_write_json, redact_text
from vla_fewshot.storage.checksums import sha256_file
from vla_fewshot.training.checkpoint import CheckpointError, verify_checkpoint_dir

EvalMethod = Literal["baseline", "lora", "replay_lora", "seen"]
EvalStage = Literal["zero_shot", "target_eval", "language_control", "seen_probe"]


@dataclass(frozen=True)
class EvalResult:
    output_dir: Path
    eval_run_id: str
    planned: int
    written: int
    skipped: int
    complete: bool
    summary: dict[str, Any] | None


def static_smoke_config(base: EvalConfig) -> EvalConfig:
    """Shrink horizon/count and retag protocol so smoke never looks like final_v1."""

    data = base.model_dump(mode="json")
    if base.stage == "language_control":
        data["protocol"]["protocol_id"] = "static_language_control_v1"
    elif base.stage == "seen_probe":
        data["protocol"]["protocol_id"] = "static_seen_probe_v1"
    else:
        data["protocol"]["protocol_id"] = "static_eval_v1"
    data["protocol"]["rollouts_per_cell"] = 3
    data["protocol"]["max_horizon"] = 8
    return EvalConfig.model_validate(data)


def checkpoint_sha256(path: Path) -> str:
    if path.is_file():
        return sha256_file(path)
    if not path.is_dir():
        raise FileNotFoundError(f"checkpoint does not exist: {path}")
    try:
        report = verify_checkpoint_dir(path)
        return str(report["weights_sha256"])
    except (CheckpointError, FileNotFoundError, ValueError):
        marker = path / "COMPLETED.json"
        if marker.is_file():
            return sha256_file(marker)
        raise


def ensure_checkpoint(path: Path) -> Path:
    if path.exists():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("static-eval-placeholder-checkpoint\n", encoding="utf-8")
    return path


def build_eval_run_id(
    *,
    stage: str,
    task_slug: str,
    n_demos: int | None,
    train_seed: int | None,
    project_root: Path,
) -> str:
    return "__".join(
        [
            "eval",
            stage,
            task_slug,
            n_demos_token(n_demos),
            train_seed_token(train_seed),
            utc_timestamp(),
            f"g{git_sha7(project_root)}",
        ]
    )


def plan_eval_rollouts(
    config: EvalConfig,
    *,
    task_slug: str,
    n_demos: int | None,
    train_seed: int | None,
    project_root: Path,
    language_control: bool,
) -> list[PlannedRollout]:
    seeds = seeds_for_config(config, project_root=project_root)
    if config.stage == "seen_probe" and task_slug not in {
        "drawer_middle",
        "bowl_stove",
        "wine_cabinet",
    }:
        task_text = "pick up the synthetic block"
        task_index = 0
        pseudo_path = project_root / "configs" / "splits" / "pseudo_target_splits.json"
        if task_slug != "synthetic_seen" and pseudo_path.is_file():
            from vla_fewshot.data.pseudo import load_pseudo_target_splits

            probes = load_pseudo_target_splits(pseudo_path)
            if task_slug in probes.tasks:
                spec = probes.tasks[task_slug]
                task_text = spec.task_text
                task_index = spec.task_index
        return plan_named_rollouts(
            config,
            task_slug=task_slug,
            task_text=task_text,
            suite=config.dataset.suite_seen,
            task_index=task_index,
            n_demos=n_demos,
            train_seed=train_seed,
            seeds=seeds,
        )
    if language_control:
        mapping = wrong_instruction_map(config)
        planned: list[PlannedRollout] = []
        for condition in ("correct", "wrong"):
            planned.extend(
                plan_target_rollouts(
                    config,
                    task_slug=task_slug,
                    n_demos=n_demos,
                    train_seed=train_seed,
                    seeds=seeds,
                    instruction_condition=condition,
                    instruction_text=instruction_for(
                        task_slug=task_slug,
                        condition=condition,
                        mapping=mapping,
                    ),
                )
            )
        return planned
    return plan_target_rollouts(
        config,
        task_slug=task_slug,
        n_demos=n_demos,
        train_seed=train_seed,
        seeds=seeds,
    )


def run_static_evaluation(
    *,
    config: EvalConfig,
    output_dir: Path,
    checkpoint: Path,
    task_slug: str,
    n_demos: int | None,
    train_seed: int | None,
    method: EvalMethod,
    stage: EvalStage,
    project_root: Path,
    splits: TargetSplits | None = None,
    command: Sequence[str] | None = None,
    language_control: bool = False,
    eval_run_id: str | None = None,
    max_new_rollouts: int | None = None,
    execute_rollout: Any | None = None,
) -> EvalResult:
    os.environ["WANDB_MODE"] = "disabled"
    os.environ["WANDB_DISABLED"] = "true"
    assert_hard_reset(config)
    assert_eval_tracking(config)
    if config.protocol.hard_reset is not True and config.protocol.protocol_id != "dev_soft_reset":
        raise ProtocolError("hard_reset must stay true")

    checkpoint = ensure_checkpoint(checkpoint)
    digest = checkpoint_sha256(checkpoint)
    planned = plan_eval_rollouts(
        config,
        task_slug=task_slug,
        n_demos=n_demos,
        train_seed=train_seed,
        project_root=project_root,
        language_control=language_control,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    run_id = eval_run_id or build_eval_run_id(
        stage=stage,
        task_slug=task_slug,
        n_demos=n_demos,
        train_seed=train_seed,
        project_root=project_root,
    )
    git = _git_state(project_root)
    store = RolloutStore(output_dir / "rollouts.jsonl")
    success_videos = success_cells_from_records(store.records())
    expected_keys = [item.key(digest) for item in planned]
    written = 0
    skipped = 0
    episode_ids = training_episode_ids(splits, task_slug=task_slug, n_demos=n_demos)
    if stage == "zero_shot":
        from vla_fewshot.evaluation.zero_shot import assert_zero_shot_cell

        assert_zero_shot_cell(
            n_demos=n_demos, train_seed=train_seed, episode_ids=episode_ids
        )

    for spec in planned:
        key = spec.key(digest)
        if key in store.completed_keys():
            skipped += 1
            continue
        record, traces, frames = (execute_rollout or _rollout_once)(
            config=config,
            spec=spec,
            checkpoint_uri=str(checkpoint),
            checkpoint_sha256=digest,
            eval_run_id=run_id,
            method=method,
            stage=stage,
            episode_ids=episode_ids,
            git_commit=git.get("commit"),
        )
        persist = should_persist_video(
            success=bool(record["success"]),
            cell=cell_id(
                method=method,
                task_slug=spec.task_slug,
                n_demos=n_demos,
                train_seed=train_seed,
                instruction_condition=spec.instruction_condition,
            ),
            success_cells_with_video=success_videos,
            save_every_failure=config.protocol.save_every_failure_video,
            save_first_success=config.protocol.save_first_success_video,
        )
        record["trace_uri"] = write_trace(output_dir, key, traces)
        if persist:
            record["video_uri"] = write_ppm_video(output_dir, key, frames)
            if record["success"]:
                success_videos.add(
                    cell_id(
                        method=method,
                        task_slug=spec.task_slug,
                        n_demos=n_demos,
                        train_seed=train_seed,
                        instruction_condition=spec.instruction_condition,
                    )
                )
        else:
            record["video_uri"] = None
        status = store.append(record)
        if status == "written":
            written += 1
        else:
            skipped += 1
        if max_new_rollouts is not None and written >= max_new_rollouts:
            break

    leftover = remaining_keys(expected_keys, store.completed_keys())
    complete = not leftover
    summary = None
    if complete:
        cell_records = [
            record
            for record in store.records()
            if rollout_key_from_record(record) in set(expected_keys)
        ]
        if language_control:
            correct = [item for item in cell_records if item["instruction_condition"] == "correct"]
            summary = cell_summary(
                method=method,
                task_slug=task_slug,
                n_demos=n_demos,
                train_seed=train_seed,
                records=correct,
                checkpoint_sha256=digest,
                protocol_id=config.protocol.protocol_id,
            )
            _write_language_pairs(output_dir, cell_records)
        else:
            summary = cell_summary(
                method=method,
                task_slug=task_slug,
                n_demos=n_demos,
                train_seed=train_seed,
                records=cell_records,
                checkpoint_sha256=digest,
                protocol_id=config.protocol.protocol_id,
            )
        atomic_write_json(output_dir / "summary.json", summary, overwrite=True)

    manifest = {
        "schema_version": 1,
        "eval_run_id": run_id,
        "stage": stage,
        "method": method,
        "status": "completed" if complete else "running",
        "protocol_id": config.protocol.protocol_id,
        "hard_reset": config.protocol.hard_reset,
        "task_slug": task_slug,
        "n_demos": n_demos,
        "train_seed": train_seed,
        "checkpoint_uri": str(checkpoint),
        "checkpoint_sha256": digest,
        "planned": len(planned),
        "completed": len(store),
        "written": written,
        "skipped": skipped,
        "remaining": len(leftover),
        "command": [redact_text(part) for part in (command or [])],
        "created_at_utc": datetime.now(UTC).isoformat(),
        "wandb_enabled": False,
    }
    atomic_write_json(output_dir / "manifest.json", manifest, overwrite=True)
    return EvalResult(
        output_dir=output_dir,
        eval_run_id=run_id,
        planned=len(planned),
        written=written,
        skipped=skipped,
        complete=complete,
        summary=summary,
    )


def _rollout_once(
    *,
    config: EvalConfig,
    spec: PlannedRollout,
    checkpoint_uri: str,
    checkpoint_sha256: str,
    eval_run_id: str,
    method: str,
    stage: str,
    episode_ids: list[int],
    git_commit: str | None,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[Any]]:
    env = ToyEvalEnv(horizon=config.protocol.max_horizon)
    observation, _info = env.reset(seed=spec.eval_seed, instruction=spec.instruction_text_used)
    fingerprint = fingerprint_observation(observation)
    policy = ToyEvalPolicy()
    traces: list[dict[str, Any]] = []
    frames: list[Any] = [env.render_frame()]
    terminated = False
    truncated = False
    success = False
    started = time.perf_counter()
    steps = 0
    while steps < config.protocol.max_horizon:
        chunk = policy.act(observation, chunk_size=config.protocol.action_chunk_horizon)
        stop = False
        for dataset_action in chunk:
            env_action = dataset_action_to_env(dataset_action)
            observation, _reward, terminated, truncated, info = env.step(list(env_action))
            success = bool(info.get("is_success"))
            traces.append(
                {
                    "step": steps,
                    "action": list(env_action),
                    "dataset_action": list(dataset_action),
                    "state": list(observation["observation.state"]),
                    "is_success": success,
                    "terminated": bool(terminated),
                    "truncated": bool(truncated),
                }
            )
            frames.append(env.render_frame())
            steps += 1
            if terminated or truncated or success or steps >= config.protocol.max_horizon:
                stop = True
                break
        if stop:
            break
    env.close()
    record = {
        "schema_version": 1,
        "eval_run_id": eval_run_id,
        "train_run_id": None,
        "stage": stage,
        "method": method,
        "task_slug": spec.task_slug,
        "task_text": spec.task_text,
        "suite": spec.suite,
        "task_index": spec.task_index,
        "n_demos": 0 if spec.n_demos is None else spec.n_demos,
        "train_seed": spec.train_seed,
        "eval_seed": spec.eval_seed,
        "rollout_index": spec.rollout_index,
        "protocol_id": spec.protocol_id,
        "instruction_condition": spec.instruction_condition,
        "instruction_text_used": spec.instruction_text_used,
        "checkpoint_uri": checkpoint_uri,
        "checkpoint_sha256": checkpoint_sha256,
        "dataset_revision": config.dataset.revision,
        "training_episode_ids": episode_ids,
        "success": int(success),
        "terminated": bool(terminated),
        "truncated": bool(truncated),
        "episode_length": steps,
        "wall_time_seconds": time.perf_counter() - started,
        "initial_state_fingerprint": fingerprint,
        "video_uri": None,
        "trace_uri": None,
        "failure_category": None if success else "unknown",
        "notes": "static toy rollout; not a LIBERO result",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "git_commit": git_commit,
    }
    return record, traces, frames


def _write_language_pairs(output_dir: Path, records: list[dict[str, Any]]) -> None:
    by_seed: dict[int, dict[str, dict[str, Any]]] = {}
    for record in records:
        by_seed.setdefault(int(record["eval_seed"]), {})[record["instruction_condition"]] = record
    rows = []
    for seed, pair in sorted(by_seed.items()):
        if set(pair) != {"correct", "wrong"}:
            raise ProtocolError(f"language pair incomplete for seed {seed}")
        if pair["correct"]["initial_state_fingerprint"] != pair["wrong"]["initial_state_fingerprint"]:
            raise ProtocolError(
                f"paired language control fingerprints drifted at seed {seed}"
            )
        left = load_actions(Path(pair["correct"]["trace_uri"]))
        right = load_actions(Path(pair["wrong"]["trace_uri"]))
        rows.append(
            {
                "eval_seed": seed,
                "task_slug": pair["correct"]["task_slug"],
                "fingerprint": pair["correct"]["initial_state_fingerprint"],
                "correct_success": pair["correct"]["success"],
                "wrong_success": pair["wrong"]["success"],
                "action_l2_divergence": action_l2_divergence(left, right),
                "action_cosine_divergence": action_cosine_divergence(left, right),
                "correct_video_uri": pair["correct"].get("video_uri"),
                "wrong_video_uri": pair["wrong"].get("video_uri"),
            }
        )
    atomic_write_json(output_dir / "language_pairs.json", {"pairs": rows}, overwrite=True)
