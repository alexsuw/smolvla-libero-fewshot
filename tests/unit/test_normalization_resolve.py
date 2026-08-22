from pathlib import Path

import pytest

from vla_fewshot.config import load_config
from vla_fewshot.evaluation.normalization import (
    NormalizationError,
    choose_normalization_stats,
    normalization_stats_suite,
    uses_suite_stats_only,
)
from vla_fewshot.storage.layout import NORMALIZATION_STATS_NAME
from vla_fewshot.training.stats import (
    find_normalization_sidecar,
    load_normalization_stats,
    overlay_dataset_state_action_stats,
    stats_digest,
    write_normalization_stats,
)


ROOT = Path(__file__).resolve().parents[2]


def _suite() -> dict:
    return {
        "observation.state": {"mean": [0.0, 0.0], "std": [1.0, 1.0], "min": [-1.0, -1.0]},
        "action": {"mean": [0.0] * 7, "std": [1.0] * 7},
        "observation.images.image": {"mean": [0.5], "std": [0.2]},
    }


def _dataset(states: list[list[float]], actions: list[list[list[float]]]) -> list[dict]:
    return [
        {"observation.state": state, "action": action}
        for state, action in zip(states, actions, strict=True)
    ]


def test_overlay_replaces_state_action_and_keeps_image_stats() -> None:
    dataset = _dataset(
        [[0.0, 2.0], [2.0, 2.0]],
        [[[0.0] * 7, [2.0] * 7], [[4.0] * 7, [4.0] * 7]],
    )
    overlay = overlay_dataset_state_action_stats(_suite(), dataset)
    assert overlay["observation.state"]["mean"] == [1.0, 2.0]
    assert overlay["action"]["mean"][0] == pytest.approx(2.5)
    assert overlay["observation.images.image"] == _suite()["observation.images.image"]
    assert stats_digest(overlay) != stats_digest(_suite())


def test_stats_digest_accepts_numpy_like_arrays() -> None:
    class _Array:
        def __init__(self, values: list[float]) -> None:
            self._values = values

        def tolist(self) -> list[float]:
            return list(self._values)

    lists = _suite()
    arrays = {
        "observation.state": {
            "mean": _Array([0.0, 0.0]),
            "std": _Array([1.0, 1.0]),
            "min": [-1.0, -1.0],
        },
        "action": {"mean": _Array([0.0] * 7), "std": _Array([1.0] * 7)},
        "observation.images.image": {"mean": _Array([0.5]), "std": _Array([0.2])},
    }
    assert stats_digest(arrays) == stats_digest(lists)


def test_sidecar_roundtrip_matches_overlay_digest(tmp_path: Path) -> None:
    overlay = overlay_dataset_state_action_stats(
        _suite(),
        _dataset([[1.0, 0.0]], [[[1.0] * 7]]),
    )
    path = tmp_path / NORMALIZATION_STATS_NAME
    digest = write_normalization_stats(path, overlay)
    loaded = load_normalization_stats(path)
    assert stats_digest(loaded) == digest


def test_choose_prefers_matching_sidecar_and_subset() -> None:
    suite = _suite()
    overlay = overlay_dataset_state_action_stats(
        suite, _dataset([[3.0, 1.0]], [[[1.0] * 7]])
    )
    stats, source = choose_normalization_stats(
        use_suite_only=False,
        suite=suite,
        sidecar=overlay,
        subset=dict(overlay),
    )
    assert source == "sidecar+subset"
    assert stats_digest(stats) == stats_digest(overlay)


def test_choose_rejects_mismatched_sidecar_and_subset() -> None:
    suite = _suite()
    overlay = overlay_dataset_state_action_stats(
        suite, _dataset([[3.0, 1.0]], [[[1.0] * 7]])
    )
    with pytest.raises(NormalizationError, match="does not match"):
        choose_normalization_stats(
            use_suite_only=False,
            suite=suite,
            sidecar=overlay,
            subset=suite,
        )


def test_target_eval_refuses_suite_only_without_sidecar_or_subset() -> None:
    with pytest.raises(NormalizationError, match="suite-wide"):
        choose_normalization_stats(
            use_suite_only=False,
            suite=_suite(),
            sidecar=None,
            subset=None,
        )


def test_zero_shot_keeps_suite_stats() -> None:
    suite = _suite()
    stats, source = choose_normalization_stats(
        use_suite_only=True,
        suite=suite,
        sidecar={"action": {"mean": [9.0]}},
        subset=None,
    )
    assert source == "suite"
    assert stats is suite


def test_find_sidecar_on_run_dir_when_checkpoint_lacks_it(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    ckpt = run_dir / "checkpoints" / "step_000100"
    ckpt.mkdir(parents=True)
    write_normalization_stats(run_dir / NORMALIZATION_STATS_NAME, _suite())
    found = find_normalization_sidecar(ckpt, run_dir)
    assert found == run_dir / NORMALIZATION_STATS_NAME
    assert find_normalization_sidecar(ckpt, None) == run_dir / NORMALIZATION_STATS_NAME


def test_final_stage_is_not_suite_only() -> None:
    final = load_config(ROOT / "configs" / "eval" / "final.yaml")
    zero = load_config(ROOT / "configs" / "eval" / "zero_shot.yaml")
    seen = load_config(ROOT / "configs" / "train" / "seen_expert.yaml")
    target = load_config(ROOT / "configs" / "train" / "target_baseline.yaml")
    assert uses_suite_stats_only(zero)
    assert not uses_suite_stats_only(final)
    assert normalization_stats_suite(zero, seen) == "libero_90"
    assert normalization_stats_suite(final, target) == "libero_goal"
