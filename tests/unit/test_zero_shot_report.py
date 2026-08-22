import json
from pathlib import Path

from vla_fewshot.reporting.zero_shot import export_zero_shot_report


def test_export_zero_shot_summary(tmp_path: Path) -> None:
    cell = tmp_path / "drawer_middle"
    cell.mkdir()
    (cell / "rollouts.jsonl").write_text(
        json.dumps(
            {
                "stage": "zero_shot",
                "method": "seen",
                "task_slug": "drawer_middle",
                "n_demos": 0,
                "train_seed": None,
                "eval_seed": 1000,
                "instruction_condition": "correct",
                "protocol_id": "final_v1",
                "success": 1,
                "checkpoint_sha256": "abc",
                "suite": "libero_goal",
                "eval_run_id": "z1",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    paths = export_zero_shot_report(tmp_path)
    markdown = paths["summary_markdown"].read_text(encoding="utf-8")
    assert "drawer_middle" in markdown
    assert "1/1" in markdown
