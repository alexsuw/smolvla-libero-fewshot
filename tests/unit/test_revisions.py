from pathlib import Path

from vla_fewshot.config import RevisionsConfig, load_config
from vla_fewshot.revisions import validate_lock_pins, validate_revisions


ROOT = Path(__file__).resolve().parents[2]


def _revisions() -> RevisionsConfig:
    config = load_config(ROOT / "configs" / "revisions.lock.yaml")
    assert isinstance(config, RevisionsConfig)
    return config


def test_universal_lock_contains_every_exact_runtime_pin() -> None:
    checks = validate_lock_pins(ROOT / "uv.lock", _revisions())
    failures = [check for check in checks if check["status"] != "pass"]
    assert failures == []


def test_offline_revision_validation_does_not_claim_hardware_acceptance() -> None:
    report = validate_revisions(
        revisions=_revisions(),
        lock_path=ROOT / "uv.lock",
        require_installed=False,
        check_remote=False,
    )
    assert report["acceptance_complete"]
    assert report["revision_status"] == "resolved_m1_pending_hardware"
