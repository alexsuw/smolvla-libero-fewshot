from datetime import UTC, datetime, timedelta
from pathlib import Path

from vla_fewshot.evaluation.progress import (
    format_duration,
    format_eval_progress,
    infer_planned_rollouts,
    progress_from_counts,
    progress_from_eval_root,
)


def test_eta_uses_session_rate() -> None:
    progress = progress_from_counts(
        completed=10,
        planned=100,
        elapsed_seconds=200,
        session_completed=10,
    )
    assert progress.fraction == 0.1
    assert progress.seconds_per_item == 20.0
    assert progress.eta_seconds == 1800.0
    text = format_eval_progress(progress)
    assert "10/100" in text
    assert "10.0%" in text
    assert "eta ~30m00s" in text


def test_eta_unknown_before_first_finished_item() -> None:
    progress = progress_from_counts(
        completed=5,
        planned=40,
        elapsed_seconds=12,
        session_completed=0,
    )
    assert progress.eta_seconds is None
    assert "rate n/a" in format_eval_progress(progress)
    assert format_duration(None) == "n/a"


def test_watch_progress_from_jsonl(tmp_path: Path) -> None:
    cell = tmp_path / "drawer_middle"
    cell.mkdir()
    start = datetime(2026, 8, 22, 19, 45, tzinfo=UTC)
    lines = []
    import json

    for index in range(4):
        lines.append(
            json.dumps(
                {
                    "stage": "language_control",
                    "task_slug": "drawer_middle",
                    "eval_seed": 1000 + index,
                    "instruction_condition": "correct",
                    "success": 0,
                    "checkpoint_sha256": "abc",
                    "n_demos": 0,
                    "train_seed": None,
                    "protocol_id": "final_language_control_v1",
                    "created_at_utc": (start + timedelta(seconds=50 * index)).isoformat(),
                    "wall_time_seconds": 50,
                }
            )
        )
    (cell / "rollouts.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (cell / "manifest.json").write_text(
        json.dumps({"stage": "language_control", "planned": 40}),
        encoding="utf-8",
    )
    now = start + timedelta(seconds=150)
    progress = progress_from_eval_root(tmp_path, now=now)
    assert progress.completed == 4
    assert progress.planned == 120
    assert progress.eta_seconds is not None
    assert infer_planned_rollouts(tmp_path, []) == 120
