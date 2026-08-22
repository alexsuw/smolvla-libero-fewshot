"""Scratch-buffered rollout video policy. Failure videos are never dropped."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from vla_fewshot.evaluation.protocol import key_slug
from vla_fewshot.reproducibility import atomic_write_json

DEFAULT_FPS = 20


def cell_id(
    *,
    method: str,
    task_slug: str,
    n_demos: int | None,
    train_seed: int | None,
    instruction_condition: str,
) -> tuple[Any, ...]:
    return (method, task_slug, n_demos, train_seed, instruction_condition)


def should_persist_video(
    *,
    success: bool,
    cell: tuple[Any, ...],
    success_cells_with_video: set[tuple[Any, ...]],
    save_every_failure: bool,
    save_first_success: bool,
) -> bool:
    if not success:
        return save_every_failure
    if not save_first_success:
        return False
    if cell in success_cells_with_video:
        return False
    return True


def success_cells_from_records(records: Iterable[Mapping[str, Any]]) -> set[tuple[Any, ...]]:
    seen: set[tuple[Any, ...]] = set()
    for record in records:
        if int(record.get("success", 0)) != 1:
            continue
        if not record.get("video_uri"):
            continue
        seen.add(
            cell_id(
                method=str(record.get("method", "")),
                task_slug=str(record["task_slug"]),
                n_demos=record.get("n_demos"),
                train_seed=record.get("train_seed"),
                instruction_condition=str(record["instruction_condition"]),
            )
        )
    return seen


def write_rollout_video(
    output_dir: Path,
    key: tuple[Any, ...],
    frames: Sequence[Any],
    *,
    fps: int = DEFAULT_FPS,
) -> str:
    """Prefer pinned-PATH FFmpeg AV1/MP4; fall back to PPM without dropping failures."""

    directory = output_dir / "videos" / key_slug(key)
    directory.mkdir(parents=True, exist_ok=True)
    rgb_frames = [_normalize_hwc(frame) for frame in frames if frame is not None]
    mp4_path = directory / "rollout.mp4"
    if rgb_frames and _try_encode_av1_mp4(rgb_frames, mp4_path, fps=fps):
        atomic_write_json(
            directory / "video_manifest.json",
            {
                "schema_version": 1,
                "encoding": "av1_mp4",
                "codec": "libaom-av1",
                "frame_count": len(rgb_frames),
                "fps": fps,
                "path": str(mp4_path),
            },
            overwrite=True,
        )
        return str(mp4_path)
    return write_ppm_video(output_dir, key, frames)


def write_ppm_video(
    output_dir: Path,
    key: tuple[Any, ...],
    frames: Sequence[Any],
) -> str:
    directory = output_dir / "videos" / key_slug(key)
    directory.mkdir(parents=True, exist_ok=True)
    written = 0
    for index, frame in enumerate(frames):
        if frame is None:
            continue
        _write_ppm_frame(directory / f"frame-{index:05d}.ppm", frame)
        written += 1
    atomic_write_json(
        directory / "video_manifest.json",
        {
            "schema_version": 1,
            "encoding": "ppm",
            "frame_count": written,
            "note": "PPM fallback when FFmpeg libaom-av1 is unavailable",
        },
        overwrite=True,
    )
    return str(directory)


def _normalize_hwc(frame: Any) -> list[list[list[int]]]:
    rows = frame
    if hasattr(frame, "tolist"):
        rows = frame.tolist()
    out: list[list[list[int]]] = []
    for row in rows:
        out.append(
            [
                [int(max(0, min(255, channel))) for channel in list(pixel)[:3]]
                for pixel in row
            ]
        )
    return out


def _try_encode_av1_mp4(
    frames: Sequence[Sequence[Sequence[Sequence[int]]]],
    output: Path,
    *,
    fps: int,
) -> bool:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None or not frames:
        return False
    height = len(frames[0])
    width = len(frames[0][0]) if height else 0
    if height < 2 or width < 2 or height % 2 or width % 2:
        return False
    command = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        f"{width}x{height}",
        "-r",
        str(fps),
        "-i",
        "-",
        "-an",
        "-c:v",
        "libaom-av1",
        "-crf",
        "35",
        "-cpu-used",
        "8",
        "-b:v",
        "0",
        "-pix_fmt",
        "yuv420p",
        str(output),
    ]
    payload = bytearray()
    for frame in frames:
        if len(frame) != height or any(len(row) != width for row in frame):
            return False
        for row in frame:
            for pixel in row:
                payload.extend(int(channel) for channel in pixel[:3])
    try:
        completed = subprocess.run(
            command,
            input=bytes(payload),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=120,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        if output.exists():
            output.unlink()
        return False
    if completed.returncode != 0 or not output.is_file() or output.stat().st_size == 0:
        if output.exists():
            output.unlink()
        return False
    return True


def _write_ppm_frame(path: Path, frame: Any) -> None:
    rows = _normalize_hwc(frame)
    height = len(rows)
    width = len(rows[0]) if height else 0
    header = f"P6\n{width} {height}\n255\n".encode("ascii")
    body = bytearray()
    for row in rows:
        for pixel in row:
            body.extend(pixel[:3])
    path.write_bytes(header + body)
