import json
from pathlib import Path

import pytest

from vla_fewshot.config import load_config
from vla_fewshot.evaluation.live import LiveRolloutAdapter
from vla_fewshot.evaluation.normalization import uses_suite_stats_only
from vla_fewshot.evaluation.seen_retention import (
    FROZEN_SEEN_RATE,
    FROZEN_SEEN_SHA256,
    PROBE_SEEDS,
    seen_probe_slugs,
)
from vla_fewshot.evaluation.seen_retention_libero90 import (
    INIT_STATE_MODE,
    LIBERO90_SUITE_STATS_SHA256,
    ORIGINAL_INIT_STATE_IDS_PATH,
    SMOKE_CELL,
    SMOKE_PROBE,
    SMOKE_SEED,
    assert_corrected_rollout_record,
    assert_libero90_suite_stats,
    corrected_retention_command,
    load_original_init_state_ids,
    load_original_seen_probe_fingerprints,
    verify_adapted_final,
)
from vla_fewshot.storage.checksums import sha256_file
from vla_fewshot.storage.layout import (
    CHECKPOINT_CHECKSUMS_NAME,
    CHECKPOINT_COMPLETED_NAME,
    CHECKPOINT_WEIGHTS_PT_NAME,
    MANIFEST_NAME,
)
from vla_fewshot.training.trainer import TrainError


ROOT = Path(__file__).resolve().parents[2]


def test_default_live_adapter_still_pins_rollout_index() -> None:
    adapter = LiveRolloutAdapter(
        policy=object(),
        preprocessor=object(),
        postprocessor=object(),
        device="cpu",
    )
    assert adapter.init_state_mode == "pin_rollout_index"


def test_seed_only_mode_is_explicit_and_invalid_modes_fail() -> None:
    adapter = LiveRolloutAdapter(
        policy=object(),
        preprocessor=object(),
        postprocessor=object(),
        device="cpu",
        init_state_mode="seed_only",
    )
    assert adapter.init_state_mode == "seed_only"
    with pytest.raises(ValueError, match="original_seen_probe"):
        LiveRolloutAdapter(
            policy=object(),
            preprocessor=object(),
            postprocessor=object(),
            device="cpu",
            init_state_mode="original_seen_probe",
        )
    with pytest.raises(ValueError, match="init_state_mode"):
        LiveRolloutAdapter(
            policy=object(),
            preprocessor=object(),
            postprocessor=object(),
            device="cpu",
            init_state_mode="overlay",
        )


def test_corrected_command_cannot_select_overlay_or_old_tree() -> None:
    command = corrected_retention_command(
        task="drawer_middle",
        n_demos=1,
        seed=42,
        run_dir=Path("/tmp/run"),
        output_dir=Path("/tmp/eval_libero90"),
        probes=("black_bowl_plate",),
        seeds=(1000,),
    )
    joined = " ".join(command)
    assert "eval_seen_retention_libero90.py" in joined
    assert "eval_seen_retention.py" not in joined
    assert "final.yaml" not in joined
    assert "--skip-videos" in command
    assert "--skip-traces" in command
    assert "--probe" in command and "black_bowl_plate" in command
    assert "--seeds" in command and "1000" in command
    probe = load_config(ROOT / "configs" / "eval" / "seen_probe.yaml")
    assert uses_suite_stats_only(probe) is True
    assert probe.stage == "seen_probe"
    assert INIT_STATE_MODE == "original_seen_probe"
    ids = load_original_init_state_ids(ROOT / ORIGINAL_INIT_STATE_IDS_PATH)
    assert ids[("black_bowl_plate", 1000)] == 10
    assert ids[("book_caddy", 1000)] == 11
    assert len(ids) == 30
    assert SMOKE_CELL == ("drawer_middle", 1, 42)
    assert SMOKE_PROBE in seen_probe_slugs()
    assert SMOKE_SEED in PROBE_SEEDS


def test_corrected_command_can_select_lora_weight_config() -> None:
    config = Path("configs/train/target_lora.yaml")
    command = corrected_retention_command(
        task="drawer_middle",
        n_demos=1,
        seed=42,
        run_dir=Path("/tmp/run"),
        output_dir=Path("/tmp/eval_libero90"),
        weight_train_config=config,
    )
    assert command[command.index("--weight-train-config") + 1] == str(config)


def test_libero90_stats_gate_refuses_overlay() -> None:
    assert_libero90_suite_stats(
        source="suite",
        suite="libero_90",
        digest=LIBERO90_SUITE_STATS_SHA256,
    )
    with pytest.raises(TrainError, match="non-suite"):
        assert_libero90_suite_stats(
            source="sidecar+subset",
            suite="libero_goal",
            digest="abc",
        )
    with pytest.raises(TrainError, match="libero_90"):
        assert_libero90_suite_stats(
            source="suite",
            suite="libero_goal",
            digest=LIBERO90_SUITE_STATS_SHA256,
        )


def test_verify_adapted_final_matches_manifest_and_completed(tmp_path: Path) -> None:
    run_dir = tmp_path / "drawer_middle_n01_s42"
    checkpoint = run_dir / "checkpoints" / "step_000500"
    checkpoint.mkdir(parents=True)
    weights = checkpoint / CHECKPOINT_WEIGHTS_PT_NAME
    weights.write_bytes(b"adapted-weights")
    digest = sha256_file(weights)
    (checkpoint / CHECKPOINT_COMPLETED_NAME).write_text(
        json.dumps({"weights_sha256": digest}) + "\n", encoding="utf-8"
    )
    (checkpoint / CHECKPOINT_CHECKSUMS_NAME).write_text(
        json.dumps({"files": {CHECKPOINT_WEIGHTS_PT_NAME: digest}}) + "\n",
        encoding="utf-8",
    )
    (checkpoint / "train_state.json").write_text("{}", encoding="utf-8")
    (checkpoint / "rng.json").write_text("{}", encoding="utf-8")
    (checkpoint / "optimizer.pt").write_bytes(b"opt")
    (run_dir / MANIFEST_NAME).write_text(
        json.dumps(
            {
                "status": "completed",
                "final_checkpoint_uri": str(checkpoint),
                "task_slug": "drawer_middle",
                "n_demos": 1,
                "train_seed": 42,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    verified = verify_adapted_final(run_dir)
    assert verified["weights_sha256"] == digest
    assert verified["checkpoint"] == checkpoint
    assert digest != FROZEN_SEEN_SHA256


def test_corrected_record_requires_original_fingerprint() -> None:
    fingerprints = {("black_bowl_plate", 1000): "sha256:original"}
    row = {
        "checkpoint_sha256": "abc123",
        "normalization_suite": "libero_90",
        "normalization_stats_sha256": LIBERO90_SUITE_STATS_SHA256,
        "task_slug": "black_bowl_plate",
        "eval_seed": 1000,
        "initial_state_fingerprint": "sha256:original",
        "init_state_mode": "original_seen_probe",
        "notes": "live SmolVLA/LIBERO rollout",
    }
    passed = assert_corrected_rollout_record(
        row,
        original_fingerprints=fingerprints,
        expected_weights="abc123",
        probe="black_bowl_plate",
        eval_seed=1000,
    )
    assert passed == [
        "loaded_weights_hash",
        "stats_source_libero_90",
        "seen_probe_task",
        "eval_seed_1000_1009",
        "original_fingerprint",
        "no_target_overlay",
    ]
    bad = dict(row)
    bad["initial_state_fingerprint"] = "sha256:pinned-mismatch"
    with pytest.raises(TrainError, match="fingerprint"):
        assert_corrected_rollout_record(
            bad,
            original_fingerprints=fingerprints,
            expected_weights="abc123",
        )
    frozen = dict(row)
    frozen["checkpoint_sha256"] = FROZEN_SEEN_SHA256
    with pytest.raises(TrainError, match="weights"):
        assert_corrected_rollout_record(
            frozen,
            original_fingerprints=fingerprints,
            expected_weights="abc123",
        )


def test_original_fingerprint_loader_reads_frozen_24_30(tmp_path: Path) -> None:
    for probe, seed, digest in (
        ("black_bowl_plate", 1000, "sha256:a"),
        ("drawer_bowl", 1000, "sha256:b"),
        ("book_caddy", 1000, "sha256:c"),
    ):
        directory = tmp_path / "step_100000" / probe
        directory.mkdir(parents=True, exist_ok=True)
        rows = []
        for item in PROBE_SEEDS:
            suffix = digest if item == seed else f"sha256:{probe}-{item}"
            rows.append(
                {
                    "task_slug": probe,
                    "eval_seed": item,
                    "checkpoint_sha256": FROZEN_SEEN_SHA256,
                    "instruction_condition": "correct",
                    "initial_state_fingerprint": suffix,
                }
            )
        (directory / "rollouts.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
        )
    loaded = load_original_seen_probe_fingerprints(tmp_path)
    assert loaded[("black_bowl_plate", 1000)] == "sha256:a"
    assert loaded[("drawer_bowl", 1000)] == "sha256:b"
    assert len(loaded) == 30
    assert FROZEN_SEEN_RATE == 0.8
