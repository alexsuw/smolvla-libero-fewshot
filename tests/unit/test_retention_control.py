from pathlib import Path

from vla_fewshot.config import load_config
from vla_fewshot.evaluation.normalization import uses_suite_stats_only
from vla_fewshot.evaluation.retention_control import (
    CONTROL_ADAPTED,
    CONTROL_SEEDS,
    control_command,
    control_jobs,
)
from vla_fewshot.evaluation.seen_retention import FROZEN_SEEN_SHA256


ROOT = Path(__file__).resolve().parents[2]


def test_control_is_paired_2x2_and_uses_five_probe_seeds() -> None:
    jobs = control_jobs()
    assert len(jobs) == 8
    assert CONTROL_SEEDS == list(range(1000, 1005))
    assert CONTROL_ADAPTED == (
        ("drawer_middle", 1, 42),
        ("drawer_middle", 25, 42),
        ("wine_cabinet", 1, 123),
        ("wine_cabinet", 25, 123),
    )
    kinds = {(job["weights"], job["stats"]) for job in jobs}
    assert kinds == {("target_adapted", "libero_90"), ("frozen_seen", "target_overlay")}
    assert all(job["weights"] != "frozen_seen" or job["stats"] != "libero_90" for job in jobs)


def test_control_command_does_not_rerun_900_or_change_success_path() -> None:
    command = control_command(
        weights="target_adapted",
        stats="libero_90",
        task="drawer_middle",
        n_demos=1,
        seed=42,
        run_dir=Path("/tmp/run"),
        output_dir=Path("/tmp/control"),
    )
    joined = " ".join(command)
    assert "--skip-videos" in command
    assert "--skip-traces" in command
    assert "eval_seen_retention.py" not in joined
    assert FROZEN_SEEN_SHA256 not in joined
    probe = load_config(ROOT / "configs" / "eval" / "seen_probe.yaml")
    assert uses_suite_stats_only(probe) is True
    assert probe.protocol.max_horizon == 300
