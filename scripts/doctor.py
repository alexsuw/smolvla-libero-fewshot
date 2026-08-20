"""Run static or full fail-closed runtime diagnostics."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

from vla_fewshot.config import PlatformConfig, RevisionsConfig, load_config
from vla_fewshot.doctor import run_doctor, write_doctor_report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--revisions",
        type=Path,
        default=Path("configs/revisions.lock.yaml"),
    )
    parser.add_argument("--lock", type=Path, default=Path("uv.lock"))
    parser.add_argument("--profile", choices=("static", "full"), default="full")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--offline", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    platform_config = load_config(args.config)
    revisions = load_config(args.revisions)
    if not isinstance(platform_config, PlatformConfig):
        raise SystemExit(f"{args.config} is not a platform config")
    if not isinstance(revisions, RevisionsConfig):
        raise SystemExit(f"{args.revisions} is not a revisions config")
    report, paths = run_doctor(
        profile=args.profile,
        platform_config=platform_config,
        revisions=revisions,
        lock_path=args.lock,
        remote=not args.offline,
    )
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    output_dir = args.output_dir
    if output_dir is None:
        if args.profile == "full":
            output_dir = paths.data_root / "doctor" / timestamp
        else:
            output_dir = Path("artifacts/validation/M1") / f"doctor-static-{timestamp}"
    write_doctor_report(output_dir, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    print(f"Doctor report: {output_dir}")
    return 0 if report["static_checks_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
