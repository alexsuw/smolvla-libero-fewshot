"""Scratch-buffered rollout video policy. Failure videos are never dropped."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from vla_fewshot.evaluation.protocol import key_slug
from vla_fewshot.reproducibility import atomic_write_json


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
            "note": "AV1/MP4 encoding stays on the M1 FFmpeg system pin",
        },
        overwrite=True,
    )
    return str(directory)


def _write_ppm_frame(path: Path, frame: Any) -> None:
    rows = frame
    if hasattr(frame, "tolist"):
        rows = frame.tolist()
    height = len(rows)
    width = len(rows[0]) if height else 0
    header = f"P6\n{width} {height}\n255\n".encode("ascii")
    body = bytearray()
    for row in rows:
        for pixel in row:
            channels = list(pixel[:3])
            body.extend(int(max(0, min(255, channel))) for channel in channels)
    path.write_bytes(header + body)
