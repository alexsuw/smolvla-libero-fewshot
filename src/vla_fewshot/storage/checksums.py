"""SHA-256 helpers and exact float round-trip encoding for checkpoints."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def encode_floats(values: Iterable[float]) -> list[str]:
    """Store IEEE-754 values so JSON load restores the exact bits."""

    return [float(value).hex() for value in values]


def decode_floats(values: Iterable[str | float]) -> list[float]:
    decoded: list[float] = []
    for value in values:
        if isinstance(value, str):
            decoded.append(float.fromhex(value))
        else:
            decoded.append(float(value))
    return decoded


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode(
        "utf-8"
    )


def sha256_json(value: Any) -> str:
    return sha256_bytes(canonical_json_bytes(value))


def file_checksums(directory: Path, *, exclude: Iterable[str] = ()) -> dict[str, str]:
    """SHA-256 every regular file under directory, relative POSIX paths."""

    skipped = set(exclude)
    records: dict[str, str] = {}
    for path in sorted(directory.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(directory).as_posix()
        if relative in skipped or path.name in skipped:
            continue
        records[relative] = sha256_file(path)
    return records


def verify_file_checksums(directory: Path, expected: Mapping[str, str]) -> None:
    observed = {name: sha256_file(directory / name) for name in expected}
    if observed != dict(expected):
        raise ValueError(
            "checkpoint checksum mismatch: "
            f"expected={dict(expected)!r} observed={observed!r}"
        )
