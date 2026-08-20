from pathlib import Path
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("script", sorted((ROOT / "scripts").glob("*.py")))
def test_python_cli_help_is_non_mutating(script: Path) -> None:
    completed = subprocess.run(
        [sys.executable, str(script), "--help"],
        cwd=ROOT,
        capture_output=True,
        check=False,
        text=True,
        timeout=20,
    )
    assert completed.returncode == 0, completed.stderr
    assert "usage" in completed.stdout.lower()


@pytest.mark.parametrize("script", sorted((ROOT / "scripts").glob("*.sh")))
def test_shell_cli_help_is_non_mutating(script: Path) -> None:
    completed = subprocess.run(
        ["bash", str(script), "--help"],
        cwd=ROOT,
        capture_output=True,
        check=False,
        text=True,
        timeout=20,
    )
    assert completed.returncode == 0, completed.stderr
    assert "usage" in completed.stdout.lower()


def test_training_stub_fails_before_compute() -> None:
    completed = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "train_seen.py")],
        cwd=ROOT,
        capture_output=True,
        check=False,
        text=True,
        timeout=20,
    )
    assert completed.returncode == 2
    assert "no compute or external write was started" in completed.stderr
