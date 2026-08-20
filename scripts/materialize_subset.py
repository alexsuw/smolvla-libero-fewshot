"""Create an immutable logical episode subset without copying videos."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from vla_fewshot.data.cli import load_data_config
from vla_fewshot.data.subset import load_or_create_logical_subset


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/data.yaml"))
    parser.add_argument(
        "--split",
        type=Path,
        default=Path("configs/splits/target_splits.json"),
    )
    parser.add_argument(
        "--task",
        required=True,
        choices=("drawer_middle", "bowl_stove", "wine_cabinet"),
    )
    parser.add_argument("--n-demos", type=int, required=True, choices=(5, 10, 25))
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    config = load_data_config(args.config)
    manifest = load_or_create_logical_subset(
        output_dir=args.output_dir,
        splits_path=args.split,
        repo_id=config.dataset.repo_id,
        revision=config.dataset.revision,
        task_slug=args.task,
        n_demos=args.n_demos,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
