"""Replay expert actions through production observation/gripper adapters."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

from vla_fewshot.config import EnvConfig, load_config
from vla_fewshot.data.cli import load_data_config, revision_root_from_args
from vla_fewshot.env.libero_env import LiberoRuntime, require_libero_runtime, resolve_env_task_id
from vla_fewshot.env.replay import (
    load_episode_actions,
    load_replay_gate,
    replay_actions_through_env,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/platform/gpu_vm.yaml"))
    parser.add_argument("--env-config", type=Path, default=Path("configs/env.yaml"))
    parser.add_argument("--data-config", type=Path, default=Path("configs/data.yaml"))
    parser.add_argument("--output-root", type=Path)
    parser.add_argument(
        "--gate",
        type=Path,
        default=Path("configs/splits/replay_gate.json"),
    )
    parser.add_argument(
        "--task",
        choices=(
            "drawer_middle",
            "bowl_stove",
            "wine_cabinet",
            "seen_black_bowl_plate",
            "seen_drawer_bowl",
            "seen_book_caddy",
        ),
    )
    parser.add_argument("--episode-id", type=int)
    parser.add_argument("--save-video", action="store_true")
    parser.add_argument(
        "--all-gate",
        action="store_true",
        help="Replay every tracked gate episode.",
    )
    parser.add_argument("--output-dir", type=Path)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        return _run(args)
    except (RuntimeError, FileNotFoundError) as error:
        print(error, file=sys.stderr)
        return 1


def _run(args: argparse.Namespace) -> int:
    load_config(args.config)
    env_config = load_config(args.env_config)
    if not isinstance(env_config, EnvConfig):
        raise SystemExit(f"{args.env_config} is not an env config")
    gate = load_replay_gate(args.gate)
    selected = list(gate.episodes)
    if args.task:
        selected = [item for item in selected if item.slug == args.task]
        if not selected:
            raise SystemExit(f"unknown gate task {args.task}")
    elif args.episode_id is not None:
        selected = [item for item in selected if item.episode_id == args.episode_id]
        if not selected:
            raise SystemExit(f"episode {args.episode_id} is not in the replay gate")
    elif not args.all_gate:
        selected = [item for item in gate.episodes if item.slug == "bowl_stove"]

    require_libero_runtime()
    data_config = load_data_config(args.data_config)
    revision_root = revision_root_from_args(
        data_config=data_config,
        output_root=args.output_root,
    )
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    root_out = args.output_dir or Path("artifacts/expert_replay") / stamp
    summaries = []
    for spec in selected:
        actions = load_episode_actions(revision_root, spec.suite, spec.episode_id)
        runtime = LiberoRuntime(
            suite=spec.suite,
            task_id=resolve_env_task_id(
                suite=spec.suite,
                task_text=spec.task_text,
                configured=spec.env_task_id,
            ),
            seed=0,
            control_mode=env_config.control_mode,
            hard_reset=env_config.hard_reset,
        )
        output_dir = root_out / spec.slug / str(spec.episode_id)
        try:
            result = replay_actions_through_env(
                env=runtime,
                dataset_actions=actions,
                output_dir=output_dir,
                task_text=spec.task_text,
                suite=spec.suite,
                episode_id=spec.episode_id,
                seed=0,
                save_video=args.save_video,
                binary_gripper=True,
                threshold=env_config.gripper.binary_threshold,
                save_frame=runtime.extract_main_hwc,
            )
        finally:
            runtime.close()
        if not result.success:
            print(json.dumps(result.manifest, indent=2, sort_keys=True))
            return 1
        summaries.append(result.manifest)
    print(json.dumps({"output_dir": str(root_out), "replays": summaries}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
