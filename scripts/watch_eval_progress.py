"""Print eval percent and ETA from JSONL. Does not start GPU work."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from vla_fewshot.evaluation.progress import format_eval_progress, progress_from_eval_root


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-root", type=Path, required=True)
    parser.add_argument("--planned", type=int, help="Override inferred total rollouts.")
    parser.add_argument(
        "--watch",
        action="store_true",
        help="Refresh until complete or interrupted.",
    )
    parser.add_argument("--interval", type=float, default=20.0)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.interval <= 0:
        parser.error("--interval must be > 0")

    def once() -> bool:
        progress = progress_from_eval_root(args.eval_root, planned=args.planned)
        print(format_eval_progress(progress), flush=True)
        return progress.remaining == 0 and progress.planned > 0

    if not args.watch:
        once()
        return 0
    while True:
        if once():
            return 0
        time.sleep(args.interval)


if __name__ == "__main__":
    raise SystemExit(main())
