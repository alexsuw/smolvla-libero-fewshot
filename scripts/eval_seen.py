"""Evaluate only the fixed seen probe suite."""

from __future__ import annotations

from vla_fewshot.evaluation.cli import run_eval_cli


def main() -> int:
    return run_eval_cli("seen")


if __name__ == "__main__":
    raise SystemExit(main())
