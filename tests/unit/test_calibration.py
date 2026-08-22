from pathlib import Path

from vla_fewshot.calibration import assert_frozen_calibration, load_calibration
from vla_fewshot.data.expected import TARGET_TASKS
from vla_fewshot.data.pseudo import load_pseudo_target_splits
from vla_fewshot.data.task_text import task_text_matches


ROOT = Path(__file__).resolve().parents[2]


def test_pseudo_targets_are_frozen_inside_libero_90() -> None:
    splits = load_pseudo_target_splits(
        ROOT / "configs" / "splits" / "pseudo_target_splits.json"
    )
    assert splits.suite == "libero_90"
    assert splits.status == "frozen"
    assert len(splits.tasks) == 3
    target_texts = [str(spec["task_text"]) for spec in TARGET_TASKS.values()]
    for task in splits.tasks.values():
        assert all(
            not task_text_matches(task.task_text, text) for text in target_texts
        )


def test_calibration_matches_tracked_train_configs() -> None:
    assert_frozen_calibration(root=ROOT)
    cal = load_calibration()
    assert cal.seen_probe_slugs == list(
        load_pseudo_target_splits(ROOT / cal.pseudo_target_splits).slugs
    )
    assert cal.eval_seed_start == 1000
    assert cal.eval_seed_end == 1019


def test_predictions_committed_before_target_results() -> None:
    from vla_fewshot.predictions import require_frozen_predictions

    text = (ROOT / "predictions.md").read_text(encoding="utf-8")
    assert "N=0" in text and "N=5" in text
    assert "Parameter-efficient" in text or "parameter-efficient" in text.lower()
    assert "seen replay" in text
    require_frozen_predictions(root=ROOT)
    report = (ROOT / "report" / "report.md").read_text(encoding="utf-8")
    assert "не содержит численных claims" in report or "numerical claims" in report.lower()
