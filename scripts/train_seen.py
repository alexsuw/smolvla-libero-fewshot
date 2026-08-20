"""Train the seen-domain policy. Leakage is checked before later allocation."""

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
    parser.add_argument("--resume-from", type=Path)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        maybe_assert_no_leakage(
            config_path=args.config,
            splits_path=args.split,
            output_root=args.output_root,
            stage="seen",
        )
    except LeakageError as error:
        print(str(error))
        return 1
    return refuse_until_milestone("train_seen")


if __name__ == "__main__":
    raise SystemExit(main())
