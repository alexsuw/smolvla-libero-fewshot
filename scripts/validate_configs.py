"""Validate every tracked M0 configuration contract."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from vla_fewshot.config import discover_configs, load_config
from vla_fewshot.data.splits import load_target_splits


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("configs"))
    return parser


def main() -> int:
    args = build_parser().parse_args()
    paths = discover_configs(args.root)
    if not paths:
        raise SystemExit(f"no YAML configs found under {args.root}")
    for path in paths:
        config = load_config(path)
        print(f"OK {path}: {config.kind}")

    split_path = args.root / "splits" / "target_splits.json"
    load_target_splits(split_path)
    print(f"OK {split_path}: nested target prefixes")

    seeds_path = args.root / "eval" / "final_seeds.json"
    with seeds_path.open("r", encoding="utf-8") as handle:
        seeds = json.load(handle)
    if seeds != list(range(1000, 1020)):
        raise SystemExit(f"{seeds_path} must contain the fixed seeds 1000..1019")
    print(f"OK {seeds_path}: 20 fixed final seeds")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
