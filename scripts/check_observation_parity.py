"""Save dataset/environment camera parity evidence without decoding the corpus."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

from vla_fewshot.config import EnvConfig, load_config
from vla_fewshot.env.observation_adapter import apply_hwc_transform, candidate_transforms
from vla_fewshot.env.parity import frozen_orientation_contract, write_parity_bundle


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/env.yaml"))
    parser.add_argument(
        "--task",
        choices=("drawer_middle", "bowl_stove", "wine_cabinet"),
        default="bowl_stove",
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--with-env",
        action="store_true",
        help="Reset the pinned LIBERO env. Requires Linux gpu extra.",
    )
    return parser


def _synthetic_frame(seed: int) -> list[list[list[int]]]:
    return [
        [[(seed + row + col) % 256, row * 8, col * 8] for col in range(8)]
        for row in range(8)
    ]


def main() -> int:
    args = build_parser().parse_args()
    try:
        return _run(args)
    except RuntimeError as error:
        print(error, file=sys.stderr)
        return 1


def _run(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    if not isinstance(config, EnvConfig):
        raise SystemExit(f"{args.config} is not an env config")
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    output_dir = args.output_dir or Path("artifacts/observation_parity") / stamp
    dataset_main = _synthetic_frame(1)
    dataset_wrist = _synthetic_frame(2)
    env_main = env_wrist = env_main_p = env_wrist_p = None
    extra: dict = {
        "task": args.task,
        "control_mode": config.control_mode,
        "candidate_transforms": list(candidate_transforms()),
        "synthetic_dataset_frames": True,
    }
    if args.with_env:
        from vla_fewshot.data.expected import TARGET_TASKS
        from vla_fewshot.env.libero_env import LiberoRuntime, resolve_env_task_id

        task_id = resolve_env_task_id(
            suite="libero_goal",
            task_text=str(TARGET_TASKS[args.task]["task_text"]),
            configured=None,
        )
        runtime = LiberoRuntime(suite="libero_goal", task_id=task_id, seed=0)
        try:
            observation, _ = runtime.reset(seed=0)
            env_main_p = runtime.extract_main_hwc(observation)
            extra["env_reset"] = True
            extra["canonical_keys"] = sorted(observation)
        finally:
            runtime.close()
    report = write_parity_bundle(
        output_dir=output_dir,
        dataset_main=dataset_main,
        dataset_wrist=dataset_wrist,
        env_main=env_main,
        env_wrist=env_wrist,
        env_main_processed=env_main_p,
        env_wrist_processed=env_wrist_p,
        extra=extra,
    )
    contract = frozen_orientation_contract()
    # Touch chosen transform so a missing identity path cannot silently pass.
    apply_hwc_transform(dataset_main, contract["project_transform"])
    print(json.dumps({"output_dir": str(output_dir), **report["orientation"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
