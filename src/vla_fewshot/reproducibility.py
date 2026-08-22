"""Atomic, redacted environment evidence for reproducible runs."""

from __future__ import annotations

import importlib.metadata
import json
import os
import platform
import re
import subprocess
import sys
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from vla_fewshot.config import PlatformConfig, RevisionsConfig
from vla_fewshot.paths import ProjectPaths, runtime_environment


_SECRET_ASSIGNMENT = re.compile(
    r"(?i)(token|secret|password|credential|private[_-]?key)"
)
_TOKEN_VALUE = re.compile(
    r"\b(?:gh[opusr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{40,}|"
    r"hf_[A-Za-z0-9]{30,}|(?:AKIA|ASIA)[0-9A-Z]{16})\b"
)
_TRIVIAL_ENV_VALUES = frozenset(
    {"true", "false", "1", "0", "yes", "no", "none", "null", "on", "off"}
)
_SAFE_TOKEN_ENV_NAMES = frozenset({"TOKENIZERS_PARALLELISM"})


def redact_text(value: str) -> str:
    """Redact recognizable credentials without hiding ordinary diagnostics."""

    redacted = _TOKEN_VALUE.sub("[REDACTED]", value)
    for name, secret in os.environ.items():
        if name in _SAFE_TOKEN_ENV_NAMES or not secret:
            continue
        if secret.lower() in _TRIVIAL_ENV_VALUES or len(secret) < 8:
            continue
        if _SECRET_ASSIGNMENT.search(name):
            redacted = redacted.replace(secret, "[REDACTED]")
    return redacted


def atomic_write_text(path: Path, content: str, *, overwrite: bool = False) -> None:
    """Write one file through a sibling temporary path and atomic replace."""

    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not overwrite:
        raise FileExistsError(f"refusing to overwrite {path}")
    temporary = path.with_name(f".{path.name}.tmp-{uuid.uuid4().hex}")
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def atomic_write_json(
    path: Path,
    value: Any,
    *,
    overwrite: bool = False,
) -> None:
    atomic_write_text(
        path,
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        overwrite=overwrite,
    )


def capture_command(command: list[str], timeout: int = 30) -> dict[str, Any]:
    """Capture a diagnostic command without shell expansion."""

    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            check=False,
            text=True,
            timeout=timeout,
        )
        return {
            "command": command,
            "returncode": completed.returncode,
            "stdout": redact_text(completed.stdout),
            "stderr": redact_text(completed.stderr),
        }
    except (FileNotFoundError, subprocess.TimeoutExpired) as error:
        return {
            "command": command,
            "returncode": None,
            "stdout": "",
            "stderr": redact_text(str(error)),
        }


def installed_distributions() -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for distribution in importlib.metadata.distributions():
        name = distribution.metadata.get("Name")
        if not name:
            continue
        record: dict[str, Any] = {"name": name, "version": distribution.version}
        direct_url = distribution.read_text("direct_url.json")
        if direct_url:
            try:
                record["direct_url"] = json.loads(direct_url)
            except json.JSONDecodeError:
                record["direct_url"] = {"invalid": True}
        records.append(record)
    return sorted(records, key=lambda item: item["name"].lower())


def _git_state(project_root: Path) -> dict[str, Any]:
    commit = capture_command(["git", "rev-parse", "HEAD"])
    status = capture_command(["git", "status", "--porcelain"])
    return {
        "commit": commit["stdout"].strip() if commit["returncode"] == 0 else None,
        "dirty": bool(status["stdout"].strip()) if status["returncode"] == 0 else None,
    }


def environment_manifest(
    *,
    paths: ProjectPaths,
    platform_config: PlatformConfig,
    revisions: RevisionsConfig,
    command: list[str],
) -> dict[str, Any]:
    safe_environment = runtime_environment(paths)
    return {
        "schema_version": 1,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "command": [redact_text(part) for part in command],
        "platform_config": platform_config.model_dump(mode="json"),
        "revision_status": revisions.status,
        "python": {
            "version": platform.python_version(),
            "executable": sys.executable,
        },
        "system": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "processor": platform.processor(),
        },
        "git": _git_state(paths.project_root),
        "storage": {
            "durability_backend": paths.durability_backend,
            "ephemeral": paths.ephemeral,
            "data_root": str(paths.data_root),
            "scratch_dir": str(paths.scratch_dir),
            "object_uri_configured": bool(paths.object_uri),
        },
        "runtime_environment": safe_environment,
        "packages": installed_distributions(),
    }


def write_environment_bundle(
    *,
    output_dir: Path,
    paths: ProjectPaths,
    platform_config: PlatformConfig,
    revisions: RevisionsConfig,
    upstream_revisions: dict[str, Any],
    command: list[str],
) -> dict[str, Path]:
    """Create a new evidence directory with all mandatory M1 outputs."""

    output_dir.mkdir(parents=True, exist_ok=False)
    manifest = environment_manifest(
        paths=paths,
        platform_config=platform_config,
        revisions=revisions,
        command=command,
    )
    nvidia_smi = capture_command(["nvidia-smi"])
    ffmpeg = capture_command(["ffmpeg", "-version"])
    uname = capture_command(["uname", "-a"])
    files = {
        "environment_manifest": output_dir / "environment_manifest.json",
        "pip_freeze": output_dir / "pip_freeze.txt",
        "system_info": output_dir / "system_info.txt",
        "nvidia_smi": output_dir / "nvidia_smi.txt",
        "ffmpeg_version": output_dir / "ffmpeg_version.txt",
        "upstream_revisions": output_dir / "upstream_revisions.json",
    }
    atomic_write_json(files["environment_manifest"], manifest)
    package_lines = [
        f"{item['name']}=={item['version']}" for item in manifest["packages"]
    ]
    atomic_write_text(files["pip_freeze"], "\n".join(package_lines) + "\n")
    atomic_write_text(
        files["system_info"],
        redact_text(uname["stdout"] + uname["stderr"]),
    )
    atomic_write_text(
        files["nvidia_smi"],
        redact_text(nvidia_smi["stdout"] + nvidia_smi["stderr"]),
    )
    atomic_write_text(
        files["ffmpeg_version"],
        redact_text(ffmpeg["stdout"] + ffmpeg["stderr"]),
    )
    atomic_write_json(files["upstream_revisions"], upstream_revisions)
    return files
