"""Rebuild registry rows from immutable manifests."""

from __future__ import annotations

import argparse
from pathlib import Path

from vla_fewshot.logging.registry import build_registry, write_registry_csv


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    rows = build_registry(args.runs_root)
    write_registry_csv(args.output, rows)
    print(f"wrote {len(rows)} registry rows to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
