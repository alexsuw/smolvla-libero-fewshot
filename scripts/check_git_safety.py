"""Reject probable secrets, large files, and runtime payloads before commit."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from vla_fewshot.git_guard import candidate_paths, scan_paths


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="*", type=Path)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    return parser


def main() -> int:
    args = build_parser().parse_args()
    repo_root = args.repo_root.resolve()
    paths = args.paths or candidate_paths(repo_root)
    violations = scan_paths(paths, repo_root)
    if violations:
        for violation in violations:
            print(f"BLOCKED {violation.path}: {violation.reason}", file=sys.stderr)
        return 1
    print(f"Git safety check passed for {len(paths)} file(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
