"""Run zero-shot final evaluation: 3 target tasks × ≥20, empty train list."""

from __future__ import annotations

from vla_fewshot.evaluation.cli import run_eval_cli


def main() -> int:
    return run_eval_cli("zero_shot")


if __name__ == "__main__":
    raise SystemExit(main())
