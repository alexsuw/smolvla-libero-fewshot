"""Write CSV/markdown for every seen-probe JSONL row, including leftover cells."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from vla_fewshot.evaluation.select import STAGED_PROBE_STEPS, parse_checkpoint_steps
from vla_fewshot.reporting.seen_probes import export_seen_probe_report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe-root", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Defaults to <probe-root>/report on the same durable disk.",
    )
    parser.add_argument(
        "--pool-steps",
        default=",".join(str(step) for step in STAGED_PROBE_STEPS),
        help="Comma-separated selection-pool steps. Others are leftover.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    paths = export_seen_probe_report(
        args.probe_root,
        output_dir=args.output_dir,
        pool_steps=tuple(parse_checkpoint_steps(args.pool_steps)),
    )
    print(json.dumps({key: str(value) for key, value in paths.items()}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
