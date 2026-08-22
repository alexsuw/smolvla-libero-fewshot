from pathlib import Path
import json

import pytest

from vla_fewshot.calibration import load_calibration, load_selected_checkpoint
from vla_fewshot.config import load_config
from vla_fewshot.evaluation.freeze import FreezeError, freeze_selected_checkpoint
from vla_fewshot.evaluation.runner import checkpoint_sha256
from vla_fewshot.evaluation.select import (
    ProbeScore,
    SelectionError,
    SelectionResult,
    collect_probe_scores,
    select_seen_checkpoint,
)
from vla_fewshot.storage.layout import step_directory, step_directory_name
from vla_fewshot.training.trainer import run_static_training


ROOT = Path(__file__).resolve().parents[2]
SLUGS = ("black_bowl_plate", "drawer_bowl", "book_caddy")


def _score(step: int, rates: dict[str, float], **kwargs: object) -> ProbeScore:
    return ProbeScore(
        step=step,
        sha256="a" * 64,
        uri=f"/ckpt/step_{step:06d}",
        per_task=rates,
        n_rollouts={slug: 10 for slug in rates},
        protocol_ids={slug: "seen_probe_v1" for slug in rates},
        unstable=bool(kwargs.get("unstable", False)),
        reason=kwargs.get("reason"),  # type: ignore[arg-type]
    )


def _uniform(step: int, rate: float, **kwargs: object) -> ProbeScore:
    return _score(step, {slug: rate for slug in SLUGS}, **kwargs)


def test_target_success_is_rejected() -> None:
    leaked = _score(10000, {"drawer_middle": 0.9, "bowl_stove": 0.9, "wine_cabinet": 0.9})
    with pytest.raises(SelectionError, match="target-task"):
        select_seen_checkpoint(
            [leaked],
            probe_slugs=SLUGS,
            tolerance=0.02,
            fallback_step=100000,
        )


def test_static_protocol_cannot_freeze() -> None:
    score = ProbeScore(
        step=10000,
        sha256="a" * 64,
        uri="/ckpt/step_010000",
        per_task={slug: 0.4 for slug in SLUGS},
        n_rollouts={slug: 10 for slug in SLUGS},
        protocol_ids={slug: "static_seen_probe_v1" for slug in SLUGS},
    )
    with pytest.raises(SelectionError, match="static protocol"):
        select_seen_checkpoint([score], probe_slugs=SLUGS, tolerance=0.02, fallback_step=100000)


def test_earliest_within_tolerance_of_best() -> None:
    scores = [
        _uniform(10000, 0.50),
        _uniform(40000, 0.80),
        _uniform(60000, 0.81),
        _uniform(100000, 0.79),
    ]
    result = select_seen_checkpoint(
        scores, probe_slugs=SLUGS, tolerance=0.02, fallback_step=100000
    )
    assert result.score.step == 40000
    assert result.used_fallback is False
    assert result.best_mean == pytest.approx(0.81)


def test_indistinguishable_scores_use_fallback_100k() -> None:
    scores = [_uniform(5000, 0.40), _uniform(50000, 0.41), _uniform(100000, 0.40)]
    result = select_seen_checkpoint(
        scores, probe_slugs=SLUGS, tolerance=0.02, fallback_step=100000
    )
    assert result.score.step == 100000
    assert result.used_fallback is True


def test_staged_finalists_skip_100k_fallback_and_take_earliest_in_band() -> None:
    from vla_fewshot.evaluation.select import rank_probe_scores

    scores = [_uniform(20000, 0.20), _uniform(80000, 0.50), _uniform(100000, 0.51)]
    ranked = rank_probe_scores(scores)
    assert [item.step for item in ranked[:2]] == [100000, 80000]
    result = select_seen_checkpoint(
        ranked[:2],
        probe_slugs=SLUGS,
        tolerance=0.02,
        fallback_step=100000,
        indistinguishable_fallback=False,
    )
    assert result.score.step == 80000
    assert result.used_fallback is False


def test_parse_checkpoint_steps_and_overlay_rollouts() -> None:
    from vla_fewshot.evaluation.protocol import seeds_for_config
    from vla_fewshot.evaluation.runner import overlay_eval_rollouts
    from vla_fewshot.evaluation.select import parse_checkpoint_steps

    assert parse_checkpoint_steps("20000,40000,100000") == [20000, 40000, 100000]
    with pytest.raises(SelectionError, match="duplicate"):
        parse_checkpoint_steps("20000,20000")
    config = load_config(ROOT / "configs" / "eval" / "seen_probe.yaml")
    five = overlay_eval_rollouts(config, 5)
    assert five.protocol.rollouts_per_cell == 5
    assert five.protocol.protocol_id == "seen_probe_v1"
    assert seeds_for_config(five, project_root=ROOT) == list(range(1000, 1005))
    ten = overlay_eval_rollouts(config, 10)
    assert seeds_for_config(ten, project_root=ROOT) == list(range(1000, 1010))


def test_nan_checkpoints_are_excluded() -> None:
    scores = [_uniform(20000, 0.9, unstable=True), _uniform(100000, 0.2)]
    result = select_seen_checkpoint(
        scores, probe_slugs=SLUGS, tolerance=0.02, fallback_step=100000
    )
    assert result.score.step == 100000


def test_freeze_is_dry_run_until_write(tmp_path: Path) -> None:
    result = select_seen_checkpoint(
        [_uniform(100000, 0.3)],
        probe_slugs=SLUGS,
        tolerance=0.02,
        fallback_step=100000,
    )
    output = tmp_path / "selected.yaml"
    frozen = freeze_selected_checkpoint(output, result, run_id="run_a", write=False)
    assert frozen.status == "frozen"
    assert frozen.step == 100000
    assert not output.exists()
    written = freeze_selected_checkpoint(output, result, run_id="run_a", write=True)
    assert output.exists()
    assert written.sha256 == "a" * 64
    assert written.uri == "checkpoints/step_100000"
    freeze_selected_checkpoint(output, result, run_id="run_a", write=True)
    other = ProbeScore(
        step=40000,
        sha256="b" * 64,
        uri="/ckpt/step_040000",
        per_task={slug: 0.9 for slug in SLUGS},
        n_rollouts={slug: 10 for slug in SLUGS},
        protocol_ids={slug: "seen_probe_v1" for slug in SLUGS},
    )
    with pytest.raises(FreezeError, match="already frozen"):
        freeze_selected_checkpoint(
            output,
            SelectionResult(
                score=other,
                best_mean=0.9,
                band_steps=[40000],
                used_fallback=False,
                rule="x" * 20,
            ),
            run_id="run_b",
            write=True,
        )


def test_collect_probe_scores_matches_checkpoint_hash(tmp_path: Path) -> None:
    config = load_config(ROOT / "configs" / "train" / "smoke.yaml")
    run_dir = tmp_path / "run"
    run_static_training(
        config=config,
        run_dir=run_dir,
        command=["python", "scripts/train_seen.py"],
        config_path=ROOT / "configs" / "train" / "smoke.yaml",
        project_root=ROOT,
        stop_after=100,
        log_freq=100,
        run_id="run_probe",
    )
    ckpt = step_directory(run_dir, 100)
    digest = checkpoint_sha256(ckpt)
    probe_root = tmp_path / "probes"
    for slug in SLUGS:
        cell = probe_root / step_directory_name(100) / slug
        cell.mkdir(parents=True)
        (cell / "summary.json").write_text(
            json.dumps(
                {
                    "task_slug": slug,
                    "success_rate": 0.4,
                    "n_rollouts": 10,
                    "checkpoint_sha256": digest,
                    "protocol_id": "seen_probe_v1",
                }
            )
            + "\n",
            encoding="utf-8",
        )
    scores = collect_probe_scores(run_dir=run_dir, probe_root=probe_root, probe_slugs=SLUGS)
    assert len(scores) == 1
    assert scores[0].step == 100
    assert scores[0].sha256 == digest
    selected = select_seen_checkpoint(
        scores, probe_slugs=SLUGS, tolerance=0.02, fallback_step=100
    )
    assert selected.score.step == 100


def test_selected_yaml_is_frozen_from_libero_90_probes() -> None:
    selected = load_selected_checkpoint()
    assert selected.status == "frozen"
    assert selected.step == 100000
    assert selected.sha256 is not None and len(selected.sha256) == 64
    assert selected.uri is not None
    cal = load_calibration()
    assert list(cal.seen_probe_slugs) == list(SLUGS)
