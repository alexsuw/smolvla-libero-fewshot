"""Bounded local/Drive durability probes with exact-file cleanup only."""

from __future__ import annotations

import hashlib
import os
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class RoundTripResult:
    root: str
    bytes_written: int
    sha256: str
    verified: bool

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def filesystem_roundtrip(root: Path, payload_size: int = 4096) -> RoundTripResult:
    """Write, fsync, read and remove one uniquely named probe file."""

    if payload_size < 1:
        raise ValueError("payload_size must be positive")
    probe_dir = root / ".vla-doctor"
    probe_dir.mkdir(parents=True, exist_ok=True)
    probe_path = probe_dir / f"roundtrip-{uuid.uuid4().hex}.tmp"
    payload = os.urandom(payload_size)
    expected = hashlib.sha256(payload).hexdigest()
    try:
        with probe_path.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        observed_payload = probe_path.read_bytes()
        observed = hashlib.sha256(observed_payload).hexdigest()
        if observed != expected:
            raise OSError(
                f"round-trip checksum mismatch: expected {expected}, got {observed}"
            )
        return RoundTripResult(
            root=str(root),
            bytes_written=payload_size,
            sha256=expected,
            verified=True,
        )
    finally:
        if probe_path.exists():
            probe_path.unlink()
