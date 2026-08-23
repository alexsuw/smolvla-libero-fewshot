from pathlib import Path

import pytest

from vla_fewshot.data.splits import load_target_splits
from vla_fewshot.data.task_text import normalize_task_text, task_text_matches


ROOT = Path(__file__).resolve().parents[2]


def test_task_text_normalizes_unicode_and_whitespace_only() -> None:
    assert normalize_task_text("  put  the bowl\non the stove  ") == (
        "put the bowl on the stove"
    )
    assert task_text_matches("put the bowl on the stove", " put  the bowl on the stove ")
    assert not task_text_matches("put the bowl on stove", "put the bowl on the stove")


def test_target_budgets_are_exact_nested_prefixes() -> None:
    splits = load_target_splits(ROOT / "configs" / "splits" / "target_splits.json")
    bowl = splits.tasks["bowl_stove"]
    assert bowl.ids_for_budget(5) == [13, 15, 16, 22, 36]
    assert bowl.ids_for_budget(5) == bowl.ids_for_budget(10)[:5]
    assert bowl.ids_for_budget(10) == bowl.ids_for_budget(25)[:10]
    assert len(set(bowl.ids_for_budget(25))) == 25


def test_unsupported_demo_budget_is_rejected() -> None:
    splits = load_target_splits(ROOT / "configs" / "splits" / "target_splits.json")
    with pytest.raises(ValueError, match="1, 2, 5, 10, 25"):
        splits.tasks["drawer_middle"].ids_for_budget(6)
