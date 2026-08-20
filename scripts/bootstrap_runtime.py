"""Create a non-overwriting environment evidence bundle after dependency sync."""

from __future__ import annotations

import argparse
from pathlib import Path

from vla_fewshot.bootstrap import bootstrap_environment


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--revisions",
        type=Path,
        default=Path("configs/revisions.lock.yaml"),
    )
    parser.add_argument("--lock", type=Path, default=Path("uv.lock"))
    parser.add_argument("--output-dir", type=Path)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    output_dir, complete = bootstrap_environment(
        platform_config_path=args.config,
        revisions_config_path=args.revisions,
        lock_path=args.lock,
        output_dir=args.output_dir,
    )
    print(f"Bootstrap evidence: {output_dir}")
    print(f"Next: source {output_dir / 'runtime.env'}")
    return 0 if complete else 1


if __name__ == "__main__":
    raise SystemExit(main())
