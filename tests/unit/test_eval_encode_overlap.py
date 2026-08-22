import threading
import time
from pathlib import Path

from vla_fewshot.config import load_config
from vla_fewshot.data.splits import load_target_splits
from vla_fewshot.evaluation.runner import run_static_evaluation, static_smoke_config
from vla_fewshot.evaluation.store import RolloutStore


ROOT = Path(__file__).resolve().parents[2]


def test_jsonl_waits_until_video_encode_finishes(tmp_path: Path, monkeypatch) -> None:
    started = threading.Event()
    release = threading.Event()

    def slow_write(output_dir, key, frames, **kwargs):
        started.set()
        assert release.wait(timeout=5)
        path = Path(output_dir) / "videos" / "forced.mp4"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"mp4")
        return str(path)

    monkeypatch.setattr("vla_fewshot.evaluation.runner.write_rollout_video", slow_write)
    monkeypatch.setattr(
        "vla_fewshot.evaluation.runner.should_persist_video",
        lambda **_kwargs: True,
    )
    config = static_smoke_config(load_config(ROOT / "configs" / "eval" / "final.yaml"))
    splits = load_target_splits(ROOT / "configs" / "splits" / "target_splits.json")
    output = tmp_path / "eval"
    jsonl = output / "rollouts.jsonl"
    error: list[BaseException] = []

    def run() -> None:
        try:
            run_static_evaluation(
                config=config,
                output_dir=output,
                checkpoint=tmp_path / "ckpt.txt",
                task_slug="bowl_stove",
                n_demos=5,
                train_seed=42,
                method="baseline",
                stage="target_eval",
                project_root=ROOT,
                splits=splits,
                max_new_rollouts=1,
            )
        except BaseException as exc:  # pragma: no cover - surfaced below
            error.append(exc)

    thread = threading.Thread(target=run)
    thread.start()
    assert started.wait(timeout=5)
    assert not jsonl.exists() or jsonl.read_text(encoding="utf-8").strip() == ""
    release.set()
    thread.join(timeout=10)
    assert not thread.is_alive()
    assert error == []
    store = RolloutStore(jsonl)
    assert len(store) == 1
    assert store.records()[0]["video_uri"]


def test_next_rollout_starts_before_previous_encode_finishes(tmp_path: Path, monkeypatch) -> None:
    events: list[str] = []
    lock = threading.Lock()

    def slow_write(output_dir, key, frames, **kwargs):
        with lock:
            events.append("encode_start")
        time.sleep(0.15)
        with lock:
            events.append("encode_end")
        path = Path(output_dir) / "videos" / f"{key[-3]}.mp4"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"mp4")
        return str(path)

    original = None

    def gpu_then_original(**kwargs):
        with lock:
            events.append("gpu")
        assert original is not None
        return original(**kwargs)

    monkeypatch.setattr("vla_fewshot.evaluation.runner.write_rollout_video", slow_write)
    monkeypatch.setattr(
        "vla_fewshot.evaluation.runner.should_persist_video",
        lambda **_kwargs: True,
    )
    import vla_fewshot.evaluation.runner as runner

    original = runner._rollout_once
    monkeypatch.setattr(runner, "_rollout_once", gpu_then_original)

    config = static_smoke_config(load_config(ROOT / "configs" / "eval" / "final.yaml"))
    splits = load_target_splits(ROOT / "configs" / "splits" / "target_splits.json")
    result = run_static_evaluation(
        config=config,
        output_dir=tmp_path / "eval",
        checkpoint=tmp_path / "ckpt.txt",
        task_slug="bowl_stove",
        n_demos=5,
        train_seed=42,
        method="baseline",
        stage="target_eval",
        project_root=ROOT,
        splits=splits,
        max_new_rollouts=2,
    )
    assert result.written == 2
    gpu_indexes = [index for index, name in enumerate(events) if name == "gpu"]
    encode_end = events.index("encode_end")
    assert len(gpu_indexes) >= 2
    assert gpu_indexes[1] < encode_end
