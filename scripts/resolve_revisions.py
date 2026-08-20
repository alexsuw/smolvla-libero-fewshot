"""Validate immutable model, dataset, source, and package revisions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from vla_fewshot.config import RevisionsConfig, load_config
from vla_fewshot.reproducibility import atomic_write_json
from vla_fewshot.revisions import validate_revisions


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/revisions.lock.yaml"),
    )
    parser.add_argument("--lock", type=Path, default=Path("uv.lock"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--require-installed", action="store_true")
    parser.add_argument("--offline", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    config = load_config(args.config)
    if not isinstance(config, RevisionsConfig):
        raise SystemExit(f"{args.config} is not a revisions config")
    report = validate_revisions(
        revisions=config,
        lock_path=args.lock,
        require_installed=args.require_installed,
        check_remote=not args.offline,
    )
    if args.output:
        atomic_write_json(args.output, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["acceptance_complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
