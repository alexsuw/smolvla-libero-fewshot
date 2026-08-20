from pathlib import Path

import pytest

import vla_fewshot.doctor as doctor_module
from vla_fewshot.config import PlatformConfig, RevisionsConfig, load_config
from vla_fewshot.doctor import run_doctor


ROOT = Path(__file__).resolve().parents[2]


def _configs() -> tuple[PlatformConfig, RevisionsConfig]:
    platform = load_config(ROOT / "configs" / "platform" / "gpu_vm.yaml")
    revisions = load_config(ROOT / "configs" / "revisions.lock.yaml")
    assert isinstance(platform, PlatformConfig)
    assert isinstance(revisions, RevisionsConfig)
    return platform, revisions


def test_static_profile_never_claims_hardware_acceptance() -> None:
    platform, revisions = _configs()
    report, _ = run_doctor(
        profile="static",
        platform_config=platform,
        revisions=revisions,
        lock_path=ROOT / "uv.lock",
        remote=False,
    )
    assert report["static_checks_passed"]
    assert not report["acceptance_complete"]
    assert report["hardware_validation_pending"]
    assert all(
        check["status"] == "skip"
        for check in report["checks"]
        if check["name"] in {"gpu", "mujoco_egl", "libero_two_camera"}
    )


def test_full_profile_stays_pending_until_lock_status_is_validated(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    platform, revisions = _configs()
    monkeypatch.setattr(doctor_module, "_revision_results", lambda *args, **kwargs: [])
    monkeypatch.setattr(doctor_module, "_check_os", lambda: {"system": "Linux"})
    monkeypatch.setattr(
        doctor_module,
        "_check_runtime_environment",
        lambda: {"MUJOCO_GL": "egl"},
    )
    monkeypatch.setattr(doctor_module, "_check_gpu", lambda value: {"cuda": True})
    monkeypatch.setattr(doctor_module, "_check_mujoco_egl", lambda: {"egl": True})
    monkeypatch.setattr(
        doctor_module,
        "_check_storage",
        lambda paths, config: {"durable": True},
    )
    monkeypatch.setattr(
        doctor_module,
        "_check_ffmpeg",
        lambda value, path: {"av1": True},
    )
    monkeypatch.setattr(
        doctor_module,
        "run_libero_doctor_probe",
        lambda: {"cameras": ["image", "image2"]},
    )

    report, _ = run_doctor(
        profile="full",
        platform_config=platform,
        revisions=revisions,
        lock_path=ROOT / "uv.lock",
        remote=False,
    )

    assert report["static_checks_passed"]
    assert not report["acceptance_complete"]
    assert report["hardware_validation_pending"]
