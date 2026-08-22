import json
from pathlib import Path

from vla_fewshot.reporting.language_control import export_language_control_report


def test_export_language_control_summary(tmp_path: Path) -> None:
    cell = tmp_path / "drawer_middle"
    cell.mkdir()
    rows = [
        {
            "stage": "language_control",
            "method": "seen",
            "task_slug": "drawer_middle",
            "n_demos": 0,
            "train_seed": None,
            "eval_seed": 1000,
            "instruction_condition": "correct",
            "protocol_id": "final_language_control_v1",
            "success": 1,
            "checkpoint_sha256": "abc",
            "normalization_suite": "libero_90",
            "suite": "libero_goal",
            "eval_run_id": "lc1",
            "initial_state_fingerprint": "fp-1",
        },
        {
            "stage": "language_control",
            "method": "seen",
            "task_slug": "drawer_middle",
            "n_demos": 0,
            "train_seed": None,
            "eval_seed": 1000,
            "instruction_condition": "wrong",
            "protocol_id": "final_language_control_v1",
            "success": 0,
            "checkpoint_sha256": "abc",
            "normalization_suite": "libero_90",
            "suite": "libero_goal",
            "eval_run_id": "lc1",
            "initial_state_fingerprint": "fp-1",
        },
    ]
    (cell / "rollouts.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )
    (cell / "language_pairs.json").write_text(
        json.dumps(
            {
                "pairs": [
                    {
                        "eval_seed": 1000,
                        "task_slug": "drawer_middle",
                        "checkpoint_sha256": "abc",
                        "fingerprint": "fp-1",
                        "correct_success": 1,
                        "wrong_success": 0,
                        "action_l2_divergence": 0.25,
                        "action_cosine_divergence": 0.1,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    paths = export_language_control_report(tmp_path)
    markdown = paths["summary_markdown"].read_text(encoding="utf-8")
    assert "drawer_middle" in markdown
    assert "1/1" in markdown
    assert "mean L2" in markdown
    assert "0.2500" in markdown
    assert "libero_90" in markdown
