from pathlib import Path

import pytest

pytest.importorskip("pyarrow")

from vla_fewshot.config import load_config
from vla_fewshot.training.autofit import fit_physical_batch
from vla_fewshot.training.batching import auto_fit_candidates, with_resolved_batch
from vla_fewshot.training.cursor import FrameCursor
from vla_fewshot.training.data import action_delta_timestamps, select_episode_ids
from vla_fewshot.training.full import require_full_training_runtime
from vla_fewshot.training.full_loop import _fetch_samples
from vla_fewshot.training.precision import resolve_precision
from tests.helpers.libero_fixture import build_pinned_metadata_fixture
from vla_fewshot.data.metadata import load_suite_metadata


ROOT = Path(__file__).resolve().parents[2]


def test_action_delta_timestamps_match_smolvla_chunk() -> None:
    deltas = action_delta_timestamps(fps=20, chunk_size=50)
    assert deltas["action"][0] == 0.0
    assert deltas["action"][1] == 0.05
    assert len(deltas["action"]) == 50
    assert deltas["action"][-1] == pytest.approx(49 / 20)


def test_auto_fit_candidates_divide_effective_batch() -> None:
    assert auto_fit_candidates(32) == (32, 16, 8, 4, 2, 1)
    assert auto_fit_candidates(64) == (64, 32, 16, 8, 4, 2, 1)
    assert auto_fit_candidates(6) == (2, 1)


def test_resolve_precision_auto_and_explicit_bf16() -> None:
    assert resolve_precision("fp32") == "fp32"
    assert resolve_precision("auto", cuda_bf16=True) == "bf16"
    assert resolve_precision("auto", cuda_bf16=False) == "fp16"
    with pytest.raises(RuntimeError, match="no BF16"):
        resolve_precision("bf16", cuda_bf16=False)


def test_with_resolved_batch_freezes_accumulation() -> None:
    config = load_config(ROOT / "configs" / "train" / "seen_expert.yaml")
    resolved = with_resolved_batch(config, 2)
    assert resolved.training.physical_batch_size == 2
    assert resolved.training.gradient_accumulation == 16


def test_frame_cursor_is_deterministic_and_resumable() -> None:
    first = FrameCursor.create(10, seed=42, with_replacement=False)
    second = FrameCursor.create(10, seed=42, with_replacement=False)
    a = first.next_indices(4)
    b = second.next_indices(4)
    assert a == b
    mid = FrameCursor.create(10, seed=42, with_replacement=False)
    mid.load_state_dict(first.state_dict())
    assert mid.next_indices(3) == second.next_indices(3)


def test_select_episode_ids_honors_max_tasks(tmp_path: Path) -> None:
    root = build_pinned_metadata_fixture(tmp_path, ROOT / "configs" / "splits" / "target_splits.json")
    meta = load_suite_metadata(root, "libero_90")
    config = load_config(ROOT / "configs" / "train" / "smoke.yaml")
    ids = select_episode_ids(meta, config.dataset)
    assert ids is not None
    assert len(ids) == 10


def test_fit_physical_batch_skips_oom_then_freezes() -> None:
    config = load_config(ROOT / "configs" / "train" / "seen_expert.yaml")
    tried: list[int] = []

    def try_batch(physical: int) -> None:
        tried.append(physical)
        if physical > 2:
            raise RuntimeError("CUDA out of memory")

    resolved = fit_physical_batch(config, try_batch=try_batch)
    assert tried == [32, 16, 8, 4, 2]
    assert resolved.training.physical_batch_size == 2
    assert resolved.training.gradient_accumulation == 16


def test_fetch_samples_preserves_requested_order() -> None:
    dataset = [{"index": index} for index in range(5)]
    assert _fetch_samples(dataset, [4, 1, 3]) == [
        {"index": 4},
        {"index": 1},
        {"index": 3},
    ]


def test_full_runtime_gate_fails_closed_without_cuda() -> None:
    import platform

    try:
        require_full_training_runtime()
    except RuntimeError as error:
        assert "no GPU training was started" in str(error)
        return
    if platform.system() != "Linux":
        raise AssertionError("non-Linux hosts must refuse full training")
    pytest.skip("Linux CUDA runtime is available on this host")
