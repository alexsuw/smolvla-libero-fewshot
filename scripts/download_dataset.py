"""Download pinned LIBERO metadata or selected suite files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from vla_fewshot.data.cli import load_data_config
from vla_fewshot.data.download import download_dataset
from vla_fewshot.data.layout import resolve_datasets_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/data.yaml"))
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--metadata-only", action="store_true", default=True)
    parser.add_argument(
        "--include-actions",
        action="store_true",
        help="Also download data parquet (still without videos).",
    )
    parser.add_argument(
        "--include-videos",
        action="store_true",
        help="Download MP4 files. Requires exactly one --suite.",
    )
    parser.add_argument("--suite", choices=("libero_90", "libero_goal"))
    return parser


def main() -> int:
    args = build_parser().parse_args()
    config = load_data_config(args.config)
    suites = (args.suite,) if args.suite else (config.dataset.suite_seen, config.dataset.suite_target)
    metadata_only = not args.include_videos
    result = download_dataset(
        repo_id=config.dataset.repo_id,
        revision=config.dataset.revision,
        datasets_dir=resolve_datasets_dir(args.output_root),
        suites=suites,
        metadata_only=metadata_only,
        include_videos=args.include_videos,
        include_actions=args.include_actions,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
