"""Fail-closed runtime diagnostics for static CI and full GPU acceptance."""

from __future__ import annotations

import importlib.metadata
import os
import platform
import re
import subprocess
import sys
import time
import uuid
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable, Literal

from vla_fewshot.config import PlatformConfig, RevisionsConfig
from vla_fewshot.env.libero_env import run_libero_doctor_probe
from vla_fewshot.paths import (
    ProjectPaths,
    disk_free_gb,
    ensure_runtime_directories,
    resolve_paths,
)
from vla_fewshot.reproducibility import (
    atomic_write_json,
    atomic_write_text,
    capture_command,
    redact_text,
)
from vla_fewshot.revisions import validate_revisions
from vla_fewshot.storage.roundtrip import filesystem_roundtrip


DoctorProfile = Literal["static", "full"]
DoctorStatus = Literal["pass", "fail", "skip"]


@dataclass(frozen=True)
class CheckResult:
    name: str
    status: DoctorStatus
    required: bool
    duration_seconds: float
    detail: Any


def _run_check(
    name: str,
    *,
    required: bool,
    check: Callable[[], Any],
) -> CheckResult:
    started = time.monotonic()
    try:
        detail = check()
        status: DoctorStatus = "pass"
    except Exception as error:
        detail = redact_text(f"{type(error).__name__}: {error}")
        status = "fail"
    return CheckResult(
        name=name,
        status=status,
        required=required,
        duration_seconds=round(time.monotonic() - started, 6),
        detail=detail,
    )


def _skip(name: str, detail: str) -> CheckResult:
    return CheckResult(
        name=name,
        status="skip",
        required=False,
        duration_seconds=0.0,
        detail=detail,
    )


def _numeric_version(value: str) -> tuple[int, ...]:
    parts = re.findall(r"\d+", value)
    return tuple(int(part) for part in parts)


def _check_os() -> dict[str, str]:
    observed = platform.system().lower()
    if observed != "linux":
        raise RuntimeError(f"expected Linux, observed {platform.system()}")
    return {"system": platform.system(), "release": platform.release()}


def _check_python(revisions: RevisionsConfig) -> dict[str, str]:
    observed = platform.python_version()
    if observed != revisions.python:
        raise RuntimeError(f"expected Python {revisions.python}, observed {observed}")
    return {"version": observed, "executable": sys.executable}


def _check_runtime_environment() -> dict[str, str]:
    expected = {
        "MUJOCO_GL": "egl",
        "TOKENIZERS_PARALLELISM": "false",
        "WANDB_MODE": "disabled",
        "WANDB_DISABLED": "true",
    }
    observed = {name: os.environ.get(name) for name in expected}
    mismatches = {
        name: {"expected": value, "observed": observed[name]}
        for name, value in expected.items()
        if observed[name] != value
    }
    if mismatches:
        raise RuntimeError(f"runtime environment mismatch: {mismatches}")
    return {name: value or "" for name, value in observed.items()}


def _check_gpu(revisions: RevisionsConfig) -> dict[str, Any]:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("torch.cuda.is_available() is false")
    driver_query = capture_command(
        [
            "nvidia-smi",
            "--query-gpu=driver_version",
            "--format=csv,noheader",
        ]
    )
    if driver_query["returncode"] != 0:
        raise RuntimeError(driver_query["stderr"] or "nvidia-smi failed")
    driver = driver_query["stdout"].strip().splitlines()[0]
    if _numeric_version(driver) < _numeric_version(
        revisions.runtime.min_nvidia_driver
    ):
        raise RuntimeError(
            f"driver {driver} is below {revisions.runtime.min_nvidia_driver}"
        )
    index = torch.cuda.current_device()
    properties = torch.cuda.get_device_properties(index)
    capability = torch.cuda.get_device_capability(index)
    architecture = f"sm_{capability[0]}{capability[1]}"
    architectures = torch.cuda.get_arch_list()
    if architecture not in architectures:
        raise RuntimeError(
            f"installed torch lacks {architecture}; available={architectures}"
        )
    free_bytes, total_bytes = torch.cuda.mem_get_info(index)
    return {
        "name": properties.name,
        "driver": driver,
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "capability": list(capability),
        "architecture": architecture,
        "architectures": architectures,
        "bf16_supported": bool(torch.cuda.is_bf16_supported()),
        "free_memory_mb": round(free_bytes / 1024**2),
        "total_memory_mb": round(total_bytes / 1024**2),
    }


def _check_mujoco_egl() -> dict[str, Any]:
    import mujoco
    import numpy as np

    probe = capture_command(
        [sys.executable, "-m", "egl_probe.get_available_devices"],
        timeout=60,
    )
    if probe["returncode"] != 0:
        raise RuntimeError(probe["stderr"] or "hf-egl-probe failed")
    model = mujoco.MjModel.from_xml_string(
        "<mujoco><worldbody><geom type='box' size='.1 .1 .1'/></worldbody></mujoco>"
    )
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    renderer = mujoco.Renderer(model, height=32, width=32)
    try:
        renderer.update_scene(data)
        pixels = renderer.render()
    finally:
        renderer.close()
    if pixels.shape != (32, 32, 3) or pixels.dtype != np.uint8:
        raise RuntimeError(
            f"unexpected EGL render output shape={pixels.shape}, dtype={pixels.dtype}"
        )
    return {
        "mujoco": importlib.metadata.version("mujoco"),
        "egl_probe": probe["stdout"].strip(),
        "render_shape": list(pixels.shape),
        "render_dtype": str(pixels.dtype),
    }


def _check_ffmpeg(revisions: RevisionsConfig, scratch_dir: Path) -> dict[str, Any]:
    version = capture_command(["ffmpeg", "-version"])
    if version["returncode"] != 0:
        raise RuntimeError(version["stderr"] or "ffmpeg is unavailable")
    first_line = version["stdout"].splitlines()[0] if version["stdout"] else ""
    match = re.search(r"ffmpeg version\s+([^\s]+)", first_line)
    observed = match.group(1).lstrip("n") if match else ""
    if not observed.startswith(revisions.runtime.ffmpeg):
        raise RuntimeError(
            f"expected FFmpeg {revisions.runtime.ffmpeg}, observed {observed or first_line}"
        )
    scratch_dir.mkdir(parents=True, exist_ok=True)
    sample = scratch_dir / f"doctor-av1-{uuid.uuid4().hex}.mp4"
    try:
        encode = capture_command(
            [
                "ffmpeg",
                "-v",
                "error",
                "-f",
                "lavfi",
                "-i",
                "color=c=black:s=32x32:d=0.2",
                "-c:v",
                "libaom-av1",
                "-y",
                str(sample),
            ],
            timeout=60,
        )
        if encode["returncode"] != 0:
            raise RuntimeError(encode["stderr"] or "AV1 encode probe failed")
        decode = capture_command(
            [
                "ffmpeg",
                "-v",
                "error",
                "-i",
                str(sample),
                "-frames:v",
                "1",
                "-f",
                "null",
                "-",
            ],
            timeout=60,
        )
        if decode["returncode"] != 0:
            raise RuntimeError(decode["stderr"] or "AV1 decode probe failed")
        return {
            "version": observed,
            "av1_roundtrip": True,
            "sample_bytes": sample.stat().st_size,
        }
    finally:
        if sample.exists():
            sample.unlink()


def _check_storage(
    paths: ProjectPaths,
    platform_config: PlatformConfig,
) -> dict[str, Any]:
    ensure_runtime_directories(paths)
    free_gb = disk_free_gb(paths.data_root)
    if free_gb < platform_config.storage.reserve_gb:
        raise RuntimeError(
            f"free space {free_gb:.2f} GB is below "
            f"{platform_config.storage.reserve_gb} GB reserve"
        )
    if platform_config.storage.require_verified_backup and paths.ephemeral:
        raise RuntimeError(
            f"{paths.durability_backend} storage is not mounted as durable"
        )
    roundtrip = filesystem_roundtrip(paths.data_root)
    return {
        "data_root": str(paths.data_root),
        "scratch_dir": str(paths.scratch_dir),
        "free_gb": round(free_gb, 3),
        "reserve_gb": platform_config.storage.reserve_gb,
        "durability_backend": paths.durability_backend,
        "ephemeral": paths.ephemeral,
        "roundtrip": roundtrip.as_dict(),
    }


def _revision_results(
    revisions: RevisionsConfig,
    lock_path: Path,
    *,
    profile: DoctorProfile,
    remote: bool,
) -> list[CheckResult]:
    report = validate_revisions(
        revisions=revisions,
        lock_path=lock_path,
        require_installed=profile == "full",
        check_remote=remote,
    )
    results: list[CheckResult] = []
    for item in report["checks"]:
        required = bool(item["required"])
        status: DoctorStatus = item["status"]
        results.append(
            CheckResult(
                name=f"revisions:{item['name']}",
                status=status,
                required=required,
                duration_seconds=0.0,
                detail=item["detail"],
            )
        )
    return results


def run_doctor(
    *,
    profile: DoctorProfile,
    platform_config: PlatformConfig,
    revisions: RevisionsConfig,
    lock_path: Path,
    remote: bool,
) -> tuple[dict[str, Any], ProjectPaths]:
    paths = resolve_paths(platform_config)
    checks = [
        _run_check("python", required=True, check=lambda: _check_python(revisions)),
        _run_check(
            "operating_system",
            required=profile == "full",
            check=_check_os,
        ),
    ]
    checks.extend(
        _revision_results(
            revisions,
            lock_path,
            profile=profile,
            remote=remote,
        )
    )
    if profile == "full":
        checks.extend(
            [
                _run_check(
                    "runtime_environment",
                    required=True,
                    check=_check_runtime_environment,
                ),
                _run_check(
                    "gpu",
                    required=True,
                    check=lambda: _check_gpu(revisions),
                ),
                _run_check(
                    "mujoco_egl",
                    required=True,
                    check=_check_mujoco_egl,
                ),
                _run_check(
                    "storage",
                    required=True,
                    check=lambda: _check_storage(paths, platform_config),
                ),
                _run_check(
                    "ffmpeg_av1",
                    required=True,
                    check=lambda: _check_ffmpeg(revisions, paths.scratch_dir),
                ),
                _run_check(
                    "libero_two_camera",
                    required=True,
                    check=run_libero_doctor_probe,
                ),
            ]
        )
    else:
        checks.extend(
            [
                _skip("runtime_environment", "requires full profile"),
                _skip("gpu", "requires full profile"),
                _skip("mujoco_egl", "requires full profile"),
                _skip("storage", "requires full profile"),
                _skip("ffmpeg_av1", "requires full profile"),
                _skip("libero_two_camera", "requires full profile"),
            ]
        )
    required_failures = [
        check for check in checks if check.required and check.status != "pass"
    ]
    report = {
        "schema_version": 1,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "profile": profile,
        "platform": platform_config.name,
        "revision_status": revisions.status,
        "static_checks_passed": not required_failures,
        "acceptance_complete": (
            profile == "full"
            and revisions.status == "validated_m1"
            and not required_failures
        ),
        "hardware_validation_pending": revisions.status != "validated_m1",
        "checks": [asdict(check) for check in checks],
    }
    return report, paths


def doctor_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Doctor report",
        "",
        f"- Profile: `{report['profile']}`",
        f"- Platform: `{report['platform']}`",
        f"- Revision status: `{report['revision_status']}`",
        f"- Static checks passed: `{report['static_checks_passed']}`",
        f"- Acceptance complete: `{report['acceptance_complete']}`",
        "",
        "## Checks",
        "",
    ]
    for check in report["checks"]:
        lines.append(
            f"- `{check['status']}` **{check['name']}** "
            f"(required={check['required']}): {check['detail']}"
        )
    return "\n".join(lines) + "\n"


def write_doctor_report(output_dir: Path, report: dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=False)
    atomic_write_json(output_dir / "doctor.json", report)
    atomic_write_text(output_dir / "doctor.md", doctor_markdown(report))
