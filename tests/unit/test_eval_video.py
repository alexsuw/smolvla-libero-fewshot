from pathlib import Path
from types import SimpleNamespace

from vla_fewshot.evaluation.video import (
    should_persist_video,
    write_ppm_video,
    write_rollout_video,
)


def _frame(seed: int) -> list[list[list[int]]]:
    return [[[(seed + x + y) % 256, 16, 32] for x in range(4)] for y in range(4)]


def test_failures_are_always_persisted() -> None:
    cell = ("seen", "black_bowl_plate", None, None, "correct")
    assert should_persist_video(
        success=False,
        cell=cell,
        success_cells_with_video={cell},
        save_every_failure=True,
        save_first_success=True,
    )


def test_first_success_video_only() -> None:
    cell = ("seen", "black_bowl_plate", None, None, "correct")
    assert should_persist_video(
        success=True,
        cell=cell,
        success_cells_with_video=set(),
        save_every_failure=True,
        save_first_success=True,
    )
    assert not should_persist_video(
        success=True,
        cell=cell,
        success_cells_with_video={cell},
        save_every_failure=True,
        save_first_success=True,
    )


def test_ppm_fallback_never_drops_frames(tmp_path: Path) -> None:
    uri = write_ppm_video(tmp_path, ("ckpt", "task", 0, None, "correct"), [_frame(1), _frame(2)])
    directory = Path(uri)
    assert (directory / "frame-00000.ppm").is_file()
    assert (directory / "frame-00001.ppm").is_file()
    manifest = (directory / "video_manifest.json").read_text(encoding="utf-8")
    assert '"encoding": "ppm"' in manifest


def test_rollout_video_persists_mp4_or_ppm(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr("vla_fewshot.evaluation.video.shutil.which", lambda _name: None)
    uri = write_rollout_video(tmp_path, ("ckpt", "task", 0, None, "correct"), [_frame(3)])
    path = Path(uri)
    assert path.exists()
    if path.is_file():
        assert path.suffix == ".mp4"
    else:
        assert (path / "frame-00000.ppm").is_file()


def test_av1_encode_enables_row_mt(tmp_path: Path, monkeypatch) -> None:
    captured: dict[str, list[str]] = {}

    def fake_run(command, **_kwargs):
        captured["command"] = list(command)
        output = Path(command[-1])
        output.write_bytes(b"mp4")
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr("vla_fewshot.evaluation.video.shutil.which", lambda _name: "ffmpeg")
    monkeypatch.setattr("vla_fewshot.evaluation.video.subprocess.run", fake_run)
    write_rollout_video(tmp_path, ("ckpt", "task", 0, None, "correct"), [_frame(3)])
    command = captured["command"]
    assert "-cpu-used" in command
    assert command[command.index("-cpu-used") + 1] == "8"
    assert "-row-mt" in command
    assert command[command.index("-row-mt") + 1] == "1"
    assert "-threads" in command
