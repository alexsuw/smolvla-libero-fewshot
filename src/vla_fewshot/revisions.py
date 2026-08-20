"""Immutable upstream and lockfile validation for M1."""

from __future__ import annotations

import importlib.metadata
import json
import ssl
import sys
import tomllib
import urllib.error
import urllib.request
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import certifi

from vla_fewshot.config import RevisionsConfig
from vla_fewshot.reproducibility import redact_text


PACKAGE_PIN_FIELDS = {
    "accelerate": "accelerate",
    "hf-egl-probe": "hf_egl_probe",
    "hf-libero": "hf_libero",
    "lerobot": "lerobot",
    "mujoco": "mujoco",
    "numpy": "numpy",
    "peft": "peft",
    "tensorboard": "tensorboard",
    "torch": "torch",
    "torchcodec": "torchcodec",
    "torchvision": "torchvision",
    "transformers": "transformers",
}


def _result(
    name: str,
    *,
    passed: bool,
    required: bool = True,
    detail: str,
) -> dict[str, Any]:
    return {
        "name": name,
        "status": "pass" if passed else "fail",
        "required": required,
        "detail": redact_text(detail),
    }


def _fetch_json(url: str, timeout: int = 20) -> dict[str, Any]:
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "smolvla-libero-fewshot/0.1"},
    )
    context = ssl.create_default_context(cafile=certifi.where())
    with urllib.request.urlopen(request, timeout=timeout, context=context) as response:
        return json.load(response)


def _remote_check(name: str, url: str) -> dict[str, Any]:
    try:
        payload = _fetch_json(url)
        identifier = payload.get("sha") or payload.get("id") or payload.get("name")
        return _result(name, passed=True, detail=f"resolved {identifier or url}")
    except (OSError, ValueError, urllib.error.HTTPError) as error:
        return _result(name, passed=False, detail=str(error))


def _lock_packages(lock_path: Path) -> list[dict[str, Any]]:
    with lock_path.open("rb") as handle:
        lock = tomllib.load(handle)
    packages = lock.get("package")
    if not isinstance(packages, list):
        raise ValueError(f"{lock_path} has no package records")
    return packages


def validate_lock_pins(
    lock_path: Path,
    revisions: RevisionsConfig,
) -> list[dict[str, Any]]:
    packages = _lock_packages(lock_path)
    by_name = {
        item["name"]: item
        for item in packages
        if isinstance(item, dict) and isinstance(item.get("name"), str)
    }
    checks: list[dict[str, Any]] = []
    for package_name, field_name in PACKAGE_PIN_FIELDS.items():
        expected = getattr(revisions.runtime, field_name)
        record = by_name.get(package_name)
        observed = record.get("version") if record else None
        checks.append(
            _result(
                f"lock:{package_name}",
                passed=observed == expected,
                detail=f"expected={expected}, observed={observed}",
            )
        )

    lerobot = by_name.get("lerobot", {})
    source = lerobot.get("source", {}) if isinstance(lerobot, dict) else {}
    git_source = source.get("git", "") if isinstance(source, dict) else ""
    checks.append(
        _result(
            "lock:lerobot_git",
            passed=revisions.source.lerobot_git in git_source,
            detail=f"expected commit {revisions.source.lerobot_git}",
        )
    )
    return checks


def _installed_lerobot_check(
    revisions: RevisionsConfig,
    *,
    required: bool,
) -> dict[str, Any]:
    try:
        distribution = importlib.metadata.distribution("lerobot")
    except importlib.metadata.PackageNotFoundError:
        return _result(
            "installed:lerobot",
            passed=not required,
            required=required,
            detail="not installed in the current environment",
        )
    direct_url_text = distribution.read_text("direct_url.json")
    commit = None
    if direct_url_text:
        payload = json.loads(direct_url_text)
        commit = payload.get("vcs_info", {}).get("commit_id")
    passed = (
        distribution.version == revisions.source.lerobot_version
        and commit == revisions.source.lerobot_git
    )
    return _result(
        "installed:lerobot",
        passed=passed,
        required=required,
        detail=(
            f"version={distribution.version}, commit={commit}; expected "
            f"{revisions.source.lerobot_version}@{revisions.source.lerobot_git}"
        ),
    )


def validate_revisions(
    *,
    revisions: RevisionsConfig,
    lock_path: Path,
    require_installed: bool = False,
    check_remote: bool = True,
) -> dict[str, Any]:
    checks = validate_lock_pins(lock_path, revisions)
    checks.append(
        _result(
            "python",
            passed=sys.version.split()[0] == revisions.python,
            detail=f"expected={revisions.python}, observed={sys.version.split()[0]}",
        )
    )
    checks.append(
        _installed_lerobot_check(revisions, required=require_installed)
    )
    if check_remote:
        checks.extend(
            [
                _remote_check(
                    "remote:model_revision",
                    "https://huggingface.co/api/models/"
                    f"{revisions.model.repo_id}/revision/{revisions.model.revision}",
                ),
                _remote_check(
                    "remote:dataset_revision",
                    "https://huggingface.co/api/datasets/"
                    f"{revisions.dataset.repo_id}/revision/"
                    f"{revisions.dataset.revision}",
                ),
                _remote_check(
                    "remote:libero_assets_revision",
                    "https://huggingface.co/api/datasets/"
                    f"{revisions.source.libero_assets_repo_id}/revision/"
                    f"{revisions.source.libero_assets_revision}",
                ),
                _remote_check(
                    "remote:lerobot_commit",
                    "https://api.github.com/repos/huggingface/lerobot/commits/"
                    f"{revisions.source.lerobot_git}",
                ),
                _remote_check(
                    "remote:libero_reference",
                    "https://api.github.com/repos/"
                    "Lifelong-Robot-Learning/LIBERO/commits/"
                    f"{revisions.source.libero_upstream_reference}",
                ),
                _remote_check(
                    "remote:hf_libero_release",
                    "https://pypi.org/pypi/hf-libero/"
                    f"{revisions.runtime.hf_libero}/json",
                ),
            ]
        )
    required_failures = [
        check
        for check in checks
        if check["required"] and check["status"] != "pass"
    ]
    return {
        "schema_version": 1,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "revision_status": revisions.status,
        "acceptance_complete": not required_failures,
        "checks": checks,
    }
