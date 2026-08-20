"""Verify that every project CLI provides a non-mutating --help path."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scripts-dir", type=Path, default=Path("scripts"))
    return parser


def main() -> int:
    args = build_parser().parse_args()
    commands: list[list[str]] = [
        [sys.executable, str(path), "--help"]
        for path in sorted(args.scripts_dir.glob("*.py"))
    ]
    commands.extend(
        [["bash", str(path), "--help"] for path in sorted(args.scripts_dir.glob("*.sh"))]
    )
    failures: list[str] = []
    for command in commands:
        completed = subprocess.run(
            command,
            capture_output=True,
            check=False,
            text=True,
            timeout=20,
        )
        label = command[1]
        if completed.returncode != 0 or "usage" not in completed.stdout.lower():
            failures.append(
                f"{label}: exit={completed.returncode}, "
                f"stdout={completed.stdout!r}, stderr={completed.stderr!r}"
            )
        else:
            print(f"OK {label}")
    if failures:
        print("\n".join(failures), file=sys.stderr)
        return 1
    print(f"Verified --help for {len(commands)} commands.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
