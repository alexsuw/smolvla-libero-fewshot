from pathlib import Path

import pytest

from vla_fewshot.evaluation.store import RolloutStore
from vla_fewshot.reporting.collect import collect_rollouts, is_reportable_protocol
from vla_fewshot.reporting.constants import COST_CURVE_N
from vla_fewshot.reporting.plots import assert_cost_curve_xticks, write_cost_curve_svg
from vla_fewshot.reporting.tables import write_report_tables
from vla_fewshot.reporting.bundle import write_report_bundle


def _record(**overrides: object) -> dict[str, object]:
    base: dict[str, object] = {
        "eval_run_id": "eval1",
        "method": "baseline",
        "task_slug": "bowl_stove",
        "n_demos": 5,
        "train_seed": 42,
        "eval_seed": 1000,
        "instruction_condition": "correct",
        "protocol_id": "final_v1",
        "success": True,
        "checkpoint_sha256": "a" * 64,
        "suite": "libero_goal",
        "terminated": True,
        "truncated": False,
        "episode_length": 4,
        "instruction_text_used": "put the bowl on the stove",
        "initial_state_fingerprint": "f" * 64,
    }
    base.update(overrides)
    return base


def test_static_and_dev_protocols_are_not_reportable() -> None:
    assert not is_reportable_protocol("static_eval_v1")
    assert not is_reportable_protocol("dev_soft_reset")
    assert is_reportable_protocol("final_v1")
    assert is_reportable_protocol("final_language_control_v1")
    assert is_reportable_protocol("language_control_v1")


def test_final_language_control_rows_are_reportable(tmp_path: Path) -> None:
    run = tmp_path / "language"
    store = RolloutStore(run / "rollouts.jsonl")
    store.append(
        _record(
            protocol_id="final_language_control_v1",
            method="seen",
            n_demos=0,
            train_seed=None,
            instruction_condition="correct",
        )
    )
    store.append(
        _record(
            protocol_id="final_language_control_v1",
            method="seen",
            n_demos=0,
            train_seed=None,
            eval_seed=1001,
            instruction_condition="wrong",
            success=False,
        )
    )
    out = tmp_path / "tables"
    report = collect_rollouts(tmp_path, output_dir=out, allow_incomplete=True)
    assert report["n_records"] == 2
    tables = write_report_tables(out / "results_long.csv", out)
    language = tables["language_control"].read_text(encoding="utf-8")
    assert "final_language_control_v1" in language


def test_collect_skips_static_rows_and_builds_cost_curve(tmp_path: Path) -> None:
    run = tmp_path / "run"
    store = RolloutStore(run / "rollouts.jsonl")
    store.append(_record())
    store.append(_record(eval_seed=1001, success=False))
    static = RolloutStore(tmp_path / "static" / "rollouts.jsonl")
    static.append(_record(protocol_id="static_eval_v1", eval_seed=1002))

    out = tmp_path / "tables"
    report = collect_rollouts(tmp_path, output_dir=out, allow_incomplete=True)
    assert report["n_records"] == 2
    long_path = out / "results_long.csv"
    svg = write_cost_curve_svg(long_path, tmp_path / "cost.svg")
    text = svg.read_text(encoding="utf-8")
    assert_cost_curve_xticks(text)
    for n in COST_CURVE_N:
        assert f">{n}</text>" in text
    write_report_tables(long_path, out)
    main = (out / "main_results.csv").read_text(encoding="utf-8")
    assert "bowl_stove" in main
    assert "baseline" in main


def test_require_complete_fails_on_empty_grid(tmp_path: Path) -> None:
    from vla_fewshot.reporting.collect import IncompleteGridError, collect_rollouts

    with pytest.raises(IncompleteGridError):
        collect_rollouts(tmp_path, output_dir=tmp_path / "out", allow_incomplete=False)


def test_report_bundle_checksums_without_deleting(tmp_path: Path) -> None:
    report_dir = tmp_path / "report"
    (report_dir / "tables").mkdir(parents=True)
    (report_dir / "figures").mkdir()
    (report_dir / "report.md").write_text("# report\n", encoding="utf-8")
    (report_dir / "tables" / "main_results.csv").write_text("task_slug\n", encoding="utf-8")
    bundle = write_report_bundle(report_dir)
    assert Path(bundle["archive"]).is_file()
    assert len(bundle["sha256"]) == 64
    assert (report_dir / "tables" / "main_results.csv").is_file()
