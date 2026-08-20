"""Verify tracked target prefixes against pinned metadata."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from vla_fewshot.data.cli import load_data_config, revision_root_from_args
from vla_fewshot.data.expected import TARGET_TASKS
from vla_fewshot.data.metadata import load_suite_metadata
from vla_fewshot.data.splits import load_target_splits, verify_splits_against_metadata


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/data.yaml"))
    parser.add_argument("--output-root", type=Path)
    parser.add_argument(
        "--split",
        type=Path,
        default=Path("configs/splits/target_splits.json"),
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    config = load_data_config(args.config)
    splits = load_target_splits(args.split)
    target = load_suite_metadata(
        revision_root_from_args(data_config=config, output_root=args.output_root),
        "libero_goal",
    )
    episode_ids = {
        slug: target.episode_ids_for_task(str(spec["task_text"]))
        for slug, spec in TARGET_TASKS.items()
    }
    verify_splits_against_metadata(splits, episode_ids)
    print(json.dumps({"acceptance_complete": True, "episode_ids": episode_ids}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
