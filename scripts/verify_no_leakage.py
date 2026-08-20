"""Fail on target-data or protocol leakage."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from vla_fewshot.data.cli import load_data_config, revision_root_from_args
from vla_fewshot.data.leakage import LeakageError, assert_no_leakage


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/data.yaml"))
    parser.add_argument("--output-root", type=Path)
    parser.add_argument(
        "--split",
        type=Path,
        default=Path("configs/splits/target_splits.json"),
    )
    parser.add_argument("--stage", choices=("seen", "target", "report"))
    parser.add_argument(
        "--task",
        choices=("drawer_middle", "bowl_stove", "wine_cabinet"),
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    config = load_data_config(args.config)
    try:
        report = assert_no_leakage(
            revision_root=revision_root_from_args(
                data_config=config,
                output_root=args.output_root,
            ),
            data_config=config,
            splits_path=args.split,
            stage=args.stage,
            task_slug=args.task,
        )
    except LeakageError as error:
        print(str(error))
        return 1
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
