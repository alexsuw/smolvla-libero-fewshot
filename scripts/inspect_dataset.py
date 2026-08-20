"""Inspect pinned suite metadata without decoding videos."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

from vla_fewshot.data.cli import load_data_config, revision_root_from_args
from vla_fewshot.data.inspection import inspect_and_write


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/data.yaml"))
    parser.add_argument("--output-root", type=Path)
    parser.add_argument(
        "--split",
        type=Path,
        default=Path("configs/splits/target_splits.json"),
    )
    parser.add_argument("--output-dir", type=Path)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    config = load_data_config(args.config)
    revision_root = revision_root_from_args(
        data_config=config,
        output_root=args.output_root,
    )
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    output_dir = args.output_dir or Path("artifacts/dataset_inspection") / stamp
    report = inspect_and_write(
        revision_root=revision_root,
        repo_id=config.dataset.repo_id,
        revision=config.dataset.revision,
        splits_path=args.split,
        output_dir=output_dir,
    )
    failed = [
        check["name"] for check in report["checks"] if check["status"] != "pass"
    ]
    print(
        json.dumps(
            {
                "acceptance_complete": report["acceptance_complete"],
                "videos_decoded": report["videos_decoded"],
                "local_root": report["local_root"],
                "failed_checks": failed,
                "output_dir": str(output_dir),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if report["acceptance_complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
