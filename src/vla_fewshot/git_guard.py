"""Fail-closed checks that keep runtime artifacts and credentials out of Git."""

from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


MAX_FILE_BYTES = 10 * 1024 * 1024
MAX_SECRET_SCAN_BYTES = 2 * 1024 * 1024

RUNTIME_DIRECTORIES = {
    "artifacts",
    "outputs",
    "runs",
    "checkpoints",
    "videos",
    "datasets",
    "cache",
}

FORBIDDEN_SUFFIXES = {
    ".pt",
    ".pth",
    ".ckpt",
    ".safetensors",
    ".mp4",
    ".avi",
    ".mov",
    ".parquet",
}

SECRET_PATTERNS = {
    "AWS access key": re.compile(r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b"),
    "GitHub token": re.compile(r"\b(?:gh[opusr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{40,})\b"),
    "Hugging Face token": re.compile(r"\bhf_[A-Za-z0-9]{30,}\b"),
    "private key": re.compile(r"-----BEGIN (?:[A-Z ]+ )?PRIVATE KEY-----"),
    "assigned secret": re.compile(
        r"(?im)^[ \t]*(?:AWS_SECRET_ACCESS_KEY|AWS_SESSION_TOKEN|HF_TOKEN)"
        r"[ \t]*[:=][ \t]*['\"]?[A-Za-z0-9+/=_-]{16,}"
    ),
}


@dataclass(frozen=True)
class Violation:
    path: Path
    reason: str


def candidate_paths(repo_root: Path) -> list[Path]:
    """List tracked and non-ignored untracked files when no hook args are given."""

    completed = subprocess.run(
        [
            "git",
            "ls-files",
            "--cached",
            "--others",
            "--exclude-standard",
            "-z",
        ],
        cwd=repo_root,
        capture_output=True,
        check=True,
    )
    return [
        repo_root / raw.decode("utf-8")
        for raw in completed.stdout.split(b"\0")
        if raw
    ]


def inspect_file(path: Path, repo_root: Path) -> list[Violation]:
    violations: list[Violation] = []
    try:
        relative = path.resolve().relative_to(repo_root.resolve())
    except ValueError:
        return [Violation(path, "file is outside repository root")]

    if not path.exists() or not path.is_file():
        return violations
    if set(relative.parts) & RUNTIME_DIRECTORIES:
        violations.append(Violation(relative, "runtime artifact directory is forbidden"))
    if path.suffix.lower() in FORBIDDEN_SUFFIXES:
        violations.append(Violation(relative, f"forbidden payload suffix {path.suffix.lower()}"))

    size = path.stat().st_size
    if size > MAX_FILE_BYTES:
        violations.append(
            Violation(relative, f"file is {size} bytes; maximum is {MAX_FILE_BYTES}")
        )
        return violations

    if size <= MAX_SECRET_SCAN_BYTES:
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            return violations
        for name, pattern in SECRET_PATTERNS.items():
            if pattern.search(text):
                violations.append(Violation(relative, f"probable {name}"))
    return violations


def scan_paths(paths: Iterable[Path], repo_root: Path) -> list[Violation]:
    violations: list[Violation] = []
    for path in paths:
        candidate = path if path.is_absolute() else repo_root / path
        violations.extend(inspect_file(candidate, repo_root))
    return violations
