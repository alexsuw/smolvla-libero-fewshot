"""Checksum a small report bundle. Never includes videos or weights."""

from __future__ import annotations

import tarfile
from datetime import UTC, datetime
from pathlib import Path

from vla_fewshot.reproducibility import atomic_write_json, atomic_write_text
from vla_fewshot.storage.checksums import sha256_file

BUNDLE_MEMBERS = (
    "figures",
    "tables",
    "failure_cases.md",
    "reproducibility.md",
    "report.md",
)


def write_report_bundle(report_dir: Path, *, output: Path | None = None) -> dict[str, str]:
    report_dir = report_dir.resolve()
    archive = output or (report_dir / "report_bundle.tar.gz")
    archive.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive, "w:gz") as tar:
        for name in BUNDLE_MEMBERS:
            path = report_dir / name
            if path.exists():
                tar.add(path, arcname=name)
    digest = sha256_file(archive)
    atomic_write_text(archive.with_suffix(archive.suffix + ".sha256"), digest + "\n", overwrite=True)
    manifest = {
        "schema_version": 1,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "archive": str(archive),
        "sha256": digest,
        "deleted": 0,
    }
    atomic_write_json(report_dir / "bundle_manifest.json", manifest, overwrite=True)
    return {"archive": str(archive), "sha256": digest}
