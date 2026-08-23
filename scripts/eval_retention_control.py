"""Discriminating weights × stats control on frozen seen probes.

The only knobs are which weights.pt to load and which MEAN_STD/processor
to attach. Environment, seeds, horizon, gripper, and success are unchanged.
Does not retrain and does not write into the 900-rollout retention tree.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from vla_fewshot.calibration import load_calibration
from vla_fewshot.data.layout import resolve_datasets_dir
from vla_fewshot.data.splits import load_target_splits
from vla_fewshot.evaluation.cli import _load_train_config, load_eval_config
from vla_fewshot.evaluation.full import require_full_evaluation_runtime
from vla_fewshot.evaluation.normalization import resolve_live_normalization
from vla_fewshot.evaluation.retention_control import CONTROL_SEEDS
from vla_fewshot.evaluation.runner import checkpoint_sha256, run_static_evaluation
from vla_fewshot.evaluation.seen_retention import (
    FROZEN_SEEN_SHA256,
    require_final_checkpoint,
)
from vla_fewshot.evaluation.zero_shot import (
    assert_frozen_checkpoint_hash,
    resolve_frozen_eval_checkpoint,
)
from vla_fewshot.storage.layout import step_directory_name
from vla_fewshot.training.baseline import TARGET_SLUGS, TRAIN_SEEDS, episode_ids_for_cell


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", required=True, choices=("frozen_seen", "target_adapted"))
    parser.add_argument("--stats", required=True, choices=("libero_90", "target_overlay"))
    parser.add_argument("--task", required=True, choices=TARGET_SLUGS)
    parser.add_argument("--n-demos", type=int, required=True, choices=(1, 2, 5, 10, 25))
    parser.add_argument("--seed", type=int, required=True, choices=TRAIN_SEEDS)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split", type=Path, default=Path("configs/splits/target_splits.json"))
    parser.add_argument("--probe-config", type=Path, default=Path("configs/eval/seen_probe.yaml"))
    parser.add_argument("--seen-train-config", type=Path, default=Path("configs/train/seen_expert.yaml"))
    parser.add_argument("--target-train-config", type=Path, default=Path("configs/train/target_baseline.yaml"))
    parser.add_argument("--stats-config", type=Path, default=Path("configs/eval/final.yaml"))
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--probe", action="append", dest="probes")
    parser.add_argument("--keep-videos", action="store_true")
    parser.add_argument("--keep-traces", action="store_true")
    parser.add_argument("--skip-videos", action="store_true", default=True)
    parser.add_argument("--skip-traces", action="store_true", default=True)
    args = parser.parse_args()
    os.environ.setdefault("WANDB_MODE", "disabled")
    os.environ.setdefault("WANDB_DISABLED", "true")

    forbidden = {
        Path("/mnt/vla/eval/seen_retention"),
        Path("/mnt/vla/eval/seen_probes__gd4b8fb8"),
        Path("/mnt/vla/eval/target_baseline"),
        Path("/mnt/vla/eval/target_baseline_n12"),
    }
    if args.output_dir.resolve() in {path.resolve() for path in forbidden}:
        print("refusing to write control eval into a frozen result root", file=sys.stderr)
        return 1
    if "/seen_retention/" in str(args.output_dir.resolve()) or str(
        args.output_dir.resolve()
    ).startswith("/mnt/vla/eval/seen_retention/"):
        print("refusing to write control eval under the 900-rollout tree", file=sys.stderr)
        return 1

    skip_videos = not args.keep_videos
    skip_traces = not args.keep_traces
    probes = tuple(args.probes) if args.probes else tuple(load_calibration().seen_probe_slugs)
    allowed = set(load_calibration().seen_probe_slugs)
    unknown = [item for item in probes if item not in allowed]
    if unknown:
        print(f"--probe must be one of {sorted(allowed)}; got {unknown}", file=sys.stderr)
        return 1

    try:
        require_full_evaluation_runtime()
        probe_config = load_eval_config(args.probe_config)
        if probe_config.protocol.protocol_id != "seen_probe_v1":
            raise RuntimeError("control must use seen_probe_v1")
        if probe_config.protocol.max_horizon != 300:
            raise RuntimeError("control must keep the 300-step horizon")
        if list(CONTROL_SEEDS) != list(range(1000, 1005)):
            raise RuntimeError("control seeds must be the first five probe seeds")
        splits = load_target_splits(args.split)
        episode_ids = episode_ids_for_cell(
            splits, task_slug=args.task, n_demos=args.n_demos
        )
        adapted_step, adapted_ckpt = require_final_checkpoint(args.run_dir)
        seen_ckpt, seen_sha = resolve_frozen_eval_checkpoint(None, purpose="retention control")
        assert_frozen_checkpoint_hash(seen_ckpt, seen_sha)
        if seen_sha != FROZEN_SEEN_SHA256:
            raise RuntimeError("frozen seen hash drifted")
    except (RuntimeError, FileNotFoundError, TypeError, ValueError) as error:
        print(str(error), file=sys.stderr)
        return 1

    if args.weights == "target_adapted":
        checkpoint = adapted_ckpt
        weight_train = _load_train_config(args.target_train_config)
        method = "baseline"
        n_demos = args.n_demos
        train_seed = args.seed
        step_label = step_directory_name(adapted_step)
    else:
        checkpoint = seen_ckpt
        weight_train = _load_train_config(args.seen_train_config)
        method = "seen"
        n_demos = 0
        train_seed = None
        step_label = step_directory_name(100000)

    datasets_dir = resolve_datasets_dir(args.output_root)
    if args.stats == "libero_90":
        stats_eval = probe_config
        stats_train = _load_train_config(args.seen_train_config)
        stats, stats_suite, digest, source = resolve_live_normalization(
            eval_config=stats_eval,
            train_config=stats_train,
            checkpoint=seen_ckpt,
            datasets_dir=datasets_dir,
        )
        if source != "suite" or stats_suite != "libero_90":
            print(
                f"libero_90 control requires suite stats, got {source}/{stats_suite}",
                file=sys.stderr,
            )
            return 1
    else:
        stats_eval = load_eval_config(args.stats_config)
        stats_train = _load_train_config(args.target_train_config)
        stats, stats_suite, digest, source = resolve_live_normalization(
            eval_config=stats_eval,
            train_config=stats_train,
            checkpoint=adapted_ckpt,
            datasets_dir=datasets_dir,
            run_dir=args.run_dir,
            task_slug=args.task,
            n_demos=args.n_demos,
            split_path=args.split,
        )
        if source not in {"sidecar", "subset", "sidecar+subset"} or stats_suite != "libero_goal":
            print(
                f"overlay control requires target sidecar, got {source}/{stats_suite}",
                file=sys.stderr,
            )
            return 1

    from vla_fewshot.evaluation.live import LiveRolloutAdapter, load_eval_policy

    loaded = load_eval_policy(
        checkpoint=checkpoint,
        repo_id=weight_train.model.repo_id,
        revision=weight_train.model.revision,
        scope=weight_train.trainable_scope,
        stats=stats,
        action_chunk_horizon=probe_config.protocol.action_chunk_horizon,
        train=weight_train,
    )
    adapter = LiveRolloutAdapter(
        policy=loaded["policy"],
        preprocessor=loaded["preprocessor"],
        postprocessor=loaded["postprocessor"],
        device=loaded["device"],
        hard_reset=probe_config.protocol.hard_reset,
        normalization_suite=stats_suite,
        normalization_stats_digest=digest,
    )
    if skip_videos or skip_traces:
        adapter.record_artifacts = False

    weight_hash = checkpoint_sha256(checkpoint)
    if args.weights == "frozen_seen" and weight_hash != FROZEN_SEEN_SHA256:
        print("frozen seen weights hash mismatch", file=sys.stderr)
        return 1
    if args.weights == "target_adapted" and weight_hash == FROZEN_SEEN_SHA256:
        print("adapted control unexpectedly loaded frozen seen weights", file=sys.stderr)
        return 1

    codes: list[int] = []
    try:
        for probe in probes:
            output = args.output_dir / step_label / probe
            result = run_static_evaluation(
                config=probe_config,
                output_dir=output,
                checkpoint=checkpoint,
                task_slug=probe,
                n_demos=n_demos,
                train_seed=train_seed,
                method=method,
                stage="seen_retention",
                project_root=Path.cwd(),
                splits=None,
                command=["python", "scripts/eval_retention_control.py", *sys.argv[1:]],
                execute_rollout=adapter,
                seed_values=list(CONTROL_SEEDS),
                skip_videos=skip_videos,
                skip_traces=skip_traces,
                episode_ids=episode_ids if args.weights == "target_adapted" else [],
            )
            print(
                f"control weights={args.weights} stats={args.stats} "
                f"overlay={args.task}_n{args.n_demos}_s{args.seed} probe={probe} "
                f"complete={result.complete} planned={result.planned} "
                f"written={result.written} skipped={result.skipped} "
                f"norm={source} suite={stats_suite} digest={digest[:12]} "
                f"wsha={weight_hash[:12]}",
                flush=True,
            )
            codes.append(0 if result.complete else 1)
    except (RuntimeError, FileNotFoundError, FileExistsError, TypeError, ValueError) as error:
        print(str(error), file=sys.stderr)
        return 1
    finally:
        adapter.close()
    return 0 if codes and all(code == 0 for code in codes) else 1


if __name__ == "__main__":
    raise SystemExit(main())
