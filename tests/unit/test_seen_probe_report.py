import csv
import json
from pathlib import Path

from vla_fewshot.reporting.seen_probes import export_seen_probe_report
from vla_fewshot.storage.layout import step_directory_name


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_export_includes_leftover_and_pool_rows(tmp_path: Path) -> None:
    root = tmp_path / "probes"
    _write_jsonl(
        root / step_directory_name(20000) / "black_bowl_plate" / "rollouts.jsonl",
        [
            {
                "eval_run_id": "e1",
                "method": "seen",
                "task_slug": "black_bowl_plate",
                "n_demos": 0,
                "train_seed": None,
                "eval_seed": 1000,
                "instruction_condition": "correct",
                "protocol_id": "seen_probe_v1",
                "success": 1,
                "checkpoint_sha256": "abc",
                "suite": "libero_90",
                "video_uri": "/tmp/ok.mp4",
                "episode_length": 12,
            }
        ],
    )
    _write_jsonl(
        root / step_directory_name(1739) / "black_bowl_plate" / "rollouts.jsonl",
        [
            {
                "eval_run_id": "e0",
                "method": "seen",
                "task_slug": "black_bowl_plate",
                "n_demos": 0,
                "train_seed": None,
                "eval_seed": 1000,
                "instruction_condition": "correct",
                "protocol_id": "seen_probe_v1",
                "success": 0,
                "checkpoint_sha256": "old",
                "suite": "libero_90",
                "video_uri": "/tmp/fail.mp4",
                "episode_length": 80,
            }
        ],
    )
    paths = export_seen_probe_report(root, pool_steps=(20000, 40000, 60000, 80000, 100000))
    long_text = paths["results_long"].read_text(encoding="utf-8")
    markdown = paths["summary_markdown"].read_text(encoding="utf-8")
    with paths["results_long"].open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert {row["step"] for row in rows} == {"1739", "20000"}
    leftover = next(row for row in rows if row["step"] == "1739")
    pool = next(row for row in rows if row["step"] == "20000")
    assert leftover["in_selection_pool"] == "false"
    assert pool["in_selection_pool"] == "true"
    assert leftover["success"] == "0"
    assert pool["success"] == "1"
    assert "black_bowl_plate" in long_text
    assert "leftover" in markdown.lower()
    assert "step_001739" in markdown
    assert "step_020000" in markdown
    rollouts = paths["rollouts_markdown"].read_text(encoding="utf-8")
    assert "1000" in rollouts
    assert "/tmp/fail.mp4" in rollouts
