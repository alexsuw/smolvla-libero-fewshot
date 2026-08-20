"""Validate and collect completed rollout records."""

from __future__ import annotations

import argparse
from pathlib import Path

from vla_fewshot.cli import refuse_until_milestone
from vla_fewshot.data.gates import maybe_assert_no_leakage
from vla_fewshot.data.leakage import LeakageError


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
    try:
        maybe_assert_no_leakage(
            config_path=args.config,
            splits_path=args.split,
            output_root=args.output_root,
            stage="report",
        )
    except LeakageError as error:
        print(str(error))
        return 1
    return refuse_until_milestone("collect_results")


if __name__ == "__main__":
    raise SystemExit(main())
