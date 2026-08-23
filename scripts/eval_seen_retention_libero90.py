"""Corrected seen retention: adapted weights + libero_90 suite stats only.

Isolates weight forgetting. Refuses target-overlay MEAN_STD. Uses the original
seen-probe seed-only reset so initial-state fingerprints match the frozen 24/30.
Does not retrain, does not rerun the frozen seen probes, and does not write
into the overlay 0/900 tree.
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
    FROZEN_SEEN_SHA256,
    PROBE_SEEDS,
)
from vla_fewshot.evaluation.seen_retention_libero90 import (
    INIT_STATE_MODE,
    ORIGINAL_INIT_STATE_IDS_PATH,
    assert_libero90_suite_stats,
    load_original_init_state_ids,
    verify_adapted_final,
)
from vla_fewshot.evaluation.zero_shot import resolve_frozen_eval_checkpoint
from vla_fewshot.storage.layout import step_directory_name
from vla_fewshot.training.baseline import (
    TARGET_SLUGS,
    TRAIN_SEEDS,
    episode_ids_for_cell,
)
from vla_fewshot.training.trainer import TrainError


FORBIDDEN_OUTPUT_ROOTS = (
    Path("/mnt/vla/eval/seen_retention"),
    Path("/mnt/vla/eval/seen_probes__gd4b8fb8"),
    Path("/mnt/vla/eval/target_baseline"),
    Path("/mnt/vla/eval/target_baseline_n12"),
    Path("/mnt/vla/eval/retention_control"),
    Path("/mnt/vla/eval/zero_shot_v2_seen_stats"),
)


def _refuse_frozen_output(output_dir: Path) -> None:
    resolved = output_dir.resolve()
    for root in FORBIDDEN_OUTPUT_ROOTS:
        root_resolved = root.resolve()
        if resolved == root_resolved or root_resolved in resolved.parents:
            raise TrainError(
                f"refusing to write corrected retention into frozen root {root}"
            )
    text = str(resolved)
    if text.rstrip("/") == "/mnt/vla/eval/seen_retention" or "/eval/seen_retention/" in (
        text + "/"
    ):
        raise TrainError("refusing to overwrite the overlay 0/900 retention tree")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True, choices=TARGET_SLUGS)
    parser.add_argument("--n-demos", type=int, required=True, choices=(1, 2, 5, 10, 25))
    parser.add_argument("--seed", type=int, required=True, choices=TRAIN_SEEDS)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split", type=Path, default=Path("configs/splits/target_splits.json"))
    parser.add_argument("--probe-config", type=Path, default=Path("configs/eval/seen_probe.yaml"))
    parser.add_argument(
        "--weight-train-config",
        type=Path,
        default=Path("configs/train/target_baseline.yaml"),
    )
    parser.add_argument(
        "--stats-train-config",
        type=Path,
        default=Path("configs/train/seen_expert.yaml"),
    )
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--probe", action="append", dest="probes")
    parser.add_argument("--seeds", nargs="+", type=int)
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
    seed_values = list(args.seeds) if args.seeds else list(PROBE_SEEDS)
    if any(seed not in PROBE_SEEDS for seed in seed_values):
        print(f"--seeds must be a subset of {list(PROBE_SEEDS)}", file=sys.stderr)
        return 1

    try:
        _refuse_frozen_output(args.output_dir)
        require_full_evaluation_runtime()
        probe_config = load_eval_config(args.probe_config)
        weight_train = _load_train_config(args.weight_train_config)
        stats_train = _load_train_config(args.stats_train_config)
        if weight_train.method != "baseline":
            raise TrainError("corrected retention evaluates naive baseline finals only")
        if probe_config.stage != "seen_probe":
            raise TrainError("corrected retention must use configs/eval/seen_probe.yaml")
        if probe_config.protocol.protocol_id != "seen_probe_v1":
            raise TrainError("corrected retention must use seen_probe_v1")
        if probe_config.protocol.max_horizon != 300:
            raise TrainError("corrected retention must keep the 300-step horizon")
        if list(FINAL_SEED_VALUES[:10]) != list(PROBE_SEEDS):
            raise TrainError("probe seeds drifted from the tracked 1000-1009 list")
        if stats_train.dataset.suite != "libero_90":
            raise TrainError("stats train config must be the seen libero_90 recipe")
        splits = load_target_splits(args.split)
        episode_ids = episode_ids_for_cell(
            splits, task_slug=args.task, n_demos=args.n_demos
        )
        verified = verify_adapted_final(args.run_dir)
        checkpoint = verified["checkpoint"]
        init_state_ids = load_original_init_state_ids(ORIGINAL_INIT_STATE_IDS_PATH)
        seen_ckpt, seen_sha = resolve_frozen_eval_checkpoint(
            None, purpose="corrected seen retention"
        )
        if seen_sha != FROZEN_SEEN_SHA256:
            raise TrainError("frozen seen hash drifted")
    except (RuntimeError, FileNotFoundError, TypeError, ValueError, TrainError) as error:
        print(str(error), file=sys.stderr)
        return 1

    from vla_fewshot.evaluation.live import LiveRolloutAdapter, load_eval_policy

    datasets_dir = resolve_datasets_dir(args.output_root)
    stats, stats_suite, digest, source = resolve_live_normalization(
        eval_config=probe_config,
        train_config=stats_train,
        checkpoint=seen_ckpt,
        datasets_dir=datasets_dir,
    )
    try:
        assert_libero90_suite_stats(source=source, suite=stats_suite, digest=digest)
        if verified["weights_sha256"] == FROZEN_SEEN_SHA256:
            raise TrainError("refusing to evaluate the frozen seen checkpoint here")
    except TrainError as error:
        print(str(error), file=sys.stderr)
        return 1

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
        init_state_mode=INIT_STATE_MODE,
        init_state_ids=init_state_ids,
    )
    if skip_videos or skip_traces:
        adapter.record_artifacts = False

    codes: list[int] = []
    try:
        for probe in probes:
            output = args.output_dir / step_directory_name(verified["step"]) / probe
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
                command=["python", "scripts/eval_seen_retention_libero90.py", *sys.argv[1:]],
                execute_rollout=adapter,
                seed_values=seed_values,
                skip_videos=skip_videos,
                skip_traces=skip_traces,
                episode_ids=episode_ids,
            )
            print(
                f"corrected_retention target={args.task} n={args.n_demos} "
                f"seed={args.seed} probe={probe} complete={result.complete} "
                f"planned={result.planned} written={result.written} "
                f"skipped={result.skipped} norm={source} suite={stats_suite} "
                f"digest={digest} wsha={verified['weights_sha256']} "
                f"init={INIT_STATE_MODE}",
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
