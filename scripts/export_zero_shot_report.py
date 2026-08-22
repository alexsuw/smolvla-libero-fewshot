"""Write CSV/markdown for every zero-shot JSONL row."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from vla_fewshot.reporting.zero_shot import export_zero_shot_report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    paths = export_zero_shot_report(args.eval_root, output_dir=args.output_dir)
    print(json.dumps({key: str(value) for key, value in paths.items()}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
