"""Run paired correct/wrong instruction rollouts."""

from __future__ import annotations

from vla_fewshot.evaluation.cli import run_eval_cli


def main() -> int:
    return run_eval_cli("language_control")


if __name__ == "__main__":
    raise SystemExit(main())
