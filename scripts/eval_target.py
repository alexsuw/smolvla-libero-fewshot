"""Run resumable fixed-seed target or zero-shot rollouts."""

from __future__ import annotations

from vla_fewshot.evaluation.cli import run_eval_cli


def main() -> int:
    return run_eval_cli("target")


if __name__ == "__main__":
    raise SystemExit(main())
