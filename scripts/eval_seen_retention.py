"""Evaluate one naive target-adapted final on the frozen seen-probe suite.

Loads the checkpoint with the same overlay MEAN_STD as its official target
eval. Plans libero_90 probe rollouts with seen_probe_v1 seeds 1000-1009.
Does not evaluate the frozen seen checkpoint and does not change weights.
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
from vla_fewshot.evaluation.protocol import FINAL_SEED_VALUES
from vla_fewshot.evaluation.runner import run_static_evaluation
from vla_fewshot.evaluation.seen_retention import (
    PROBE_SEEDS,
    require_final_checkpoint,
)
from vla_fewshot.storage.layout import step_directory_name
from vla_fewshot.training.baseline import (
    TARGET_SLUGS,
    TRAIN_SEEDS,
    episode_ids_for_cell,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True, choices=TARGET_SLUGS)
    parser.add_argument("--n-demos", type=int, required=True, choices=(1, 2, 5, 10, 25))
    parser.add_argument("--seed", type=int, required=True, choices=TRAIN_SEEDS)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split", type=Path, default=Path("configs/splits/target_splits.json"))
    parser.add_argument("--probe-config", type=Path, default=Path("configs/eval/seen_probe.yaml"))
    parser.add_argument("--train-config", type=Path, default=Path("configs/train/target_baseline.yaml"))
    parser.add_argument("--stats-config", type=Path, default=Path("configs/eval/final.yaml"))
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--probe", action="append", dest="probes")
    parser.add_argument("--skip-videos", action="store_true", default=True)
    parser.add_argument("--skip-traces", action="store_true", default=True)
    parser.add_argument("--keep-videos", action="store_true")
    parser.add_argument("--keep-traces", action="store_true")
    args = parser.parse_args()
    os.environ.setdefault("WANDB_MODE", "disabled")
    os.environ.setdefault("WANDB_DISABLED", "true")

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
        stats_config = load_eval_config(args.stats_config)
        train = _load_train_config(args.train_config)
        if train.method != "baseline":
            raise RuntimeError("seen retention evaluates naive baseline finals only")
        if probe_config.protocol.protocol_id != "seen_probe_v1":
            raise RuntimeError("retention must use seen_probe_v1")
        if list(FINAL_SEED_VALUES[:10]) != list(PROBE_SEEDS):
            raise RuntimeError("probe seeds drifted from the tracked 1000-1009 list")
        splits = load_target_splits(args.split)
        episode_ids = episode_ids_for_cell(
            splits, task_slug=args.task, n_demos=args.n_demos
        )
        step, checkpoint = require_final_checkpoint(args.run_dir)
    except (RuntimeError, FileNotFoundError, TypeError, ValueError) as error:
        print(str(error), file=sys.stderr)
        return 1

    from vla_fewshot.evaluation.live import LiveRolloutAdapter, load_eval_policy

    datasets_dir = resolve_datasets_dir(args.output_root)
    stats, stats_suite, digest, source = resolve_live_normalization(
        eval_config=stats_config,
        train_config=train,
        checkpoint=checkpoint,
        datasets_dir=datasets_dir,
        run_dir=args.run_dir,
        task_slug=args.task,
        n_demos=args.n_demos,
        split_path=args.split,
    )
    if source not in {"sidecar", "subset", "sidecar+subset"}:
        print(
            f"retention refuses suite-only MEAN_STD ({source}); "
            "adapted finals must use the official target overlay",
            file=sys.stderr,
        )
        return 1
    loaded = load_eval_policy(
        checkpoint=checkpoint,
        repo_id=train.model.repo_id,
        revision=train.model.revision,
        scope=train.trainable_scope,
        stats=stats,
        action_chunk_horizon=probe_config.protocol.action_chunk_horizon,
        train=train,
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

    codes: list[int] = []
    try:
        for probe in probes:
            output = args.output_dir / step_directory_name(step) / probe
            result = run_static_evaluation(
                config=probe_config,
                output_dir=output,
                checkpoint=checkpoint,
                task_slug=probe,
                n_demos=args.n_demos,
                train_seed=args.seed,
                method="baseline",
                stage="seen_retention",
                project_root=Path.cwd(),
                splits=None,
                command=["python", "scripts/eval_seen_retention.py", *sys.argv[1:]],
                execute_rollout=adapter,
                seed_values=list(PROBE_SEEDS),
                skip_videos=skip_videos,
                skip_traces=skip_traces,
                episode_ids=episode_ids,
            )
            print(
                f"retention target={args.task} n={args.n_demos} seed={args.seed} "
                f"probe={probe} complete={result.complete} planned={result.planned} "
                f"written={result.written} skipped={result.skipped} "
                f"norm={source} suite={stats_suite} digest={digest[:12]}",
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
