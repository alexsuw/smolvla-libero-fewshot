"""Select and freeze the seen checkpoint from libero_90 probe scores."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from vla_fewshot.calibration import load_calibration, load_selected_checkpoint
from vla_fewshot.evaluation.freeze import FreezeError, freeze_selected_checkpoint
from vla_fewshot.evaluation.select import (
    SelectionError,
    collect_probe_scores,
    select_seen_checkpoint,
)
from vla_fewshot.reproducibility import atomic_write_json


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--probe-root", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("configs/selected_seen_checkpoint.yaml"),
    )
    parser.add_argument(
        "--write",
        action="store_true",
        help="Write the frozen YAML. Default is dry-run.",
    )
    parser.add_argument(
        "--report",
        type=Path,
        help="Optional JSON report path.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    cal = load_calibration()
    selected = load_selected_checkpoint()
    try:
        scores = collect_probe_scores(
            run_dir=args.run_dir,
            probe_root=args.probe_root,
            probe_slugs=cal.seen_probe_slugs,
        )
        result = select_seen_checkpoint(
            scores,
            probe_slugs=cal.seen_probe_slugs,
            tolerance=selected.tolerance_success,
            fallback_step=selected.fallback_step,
        )
        frozen = freeze_selected_checkpoint(
            args.output,
            result,
            run_id=args.run_dir.name,
            write=args.write,
        )
    except (SelectionError, FreezeError, FileNotFoundError, ValueError, TypeError) as error:
        print(str(error), file=sys.stderr)
        return 1
    payload = {
        "dry_run": not args.write,
        "used_fallback": result.used_fallback,
        "best_mean": result.best_mean,
        "band_steps": result.band_steps,
        "selected_step": frozen.step,
        "sha256": frozen.sha256,
        "uri": frozen.uri,
        "status": frozen.status,
        "written": args.write,
    }
    if args.report is not None:
        atomic_write_json(args.report, payload, overwrite=True)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
