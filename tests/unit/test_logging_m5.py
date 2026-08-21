from pathlib import Path

from vla_fewshot.logging.csv_logger import (
    METRICS_COLUMNS,
    CsvMetricsLogger,
    metrics_row,
    repair_trailing_line,
)
from vla_fewshot.logging.manifest import build_run_id, n_demos_token
from vla_fewshot.logging.registry import build_registry, write_registry_csv
from vla_fewshot.logging.tensorboard import TensorBoardLogger
from vla_fewshot.config import load_config
from vla_fewshot.reproducibility import atomic_write_json
from vla_fewshot.storage.layout import MANIFEST_NAME


ROOT = Path(__file__).resolve().parents[2]


def test_csv_repairs_unterminated_trailing_line(tmp_path: Path) -> None:
    path = tmp_path / "metrics.csv"
    logger = CsvMetricsLogger(path)
    logger.append(
        metrics_row(
            elapsed_seconds=1.0,
            global_step=1,
            samples_seen=2,
            epoch_fraction=0.1,
            loss=0.5,
            learning_rate=1e-4,
            grad_norm=0.2,
        )
    )
    path.write_bytes(path.read_bytes() + b"1,2,3,broken")
    assert repair_trailing_line(path) is True
    logger = CsvMetricsLogger(path)
    logger.append(
        metrics_row(
            elapsed_seconds=2.0,
            global_step=2,
            samples_seen=4,
            epoch_fraction=0.2,
            loss=0.4,
            learning_rate=1e-4,
            grad_norm=0.1,
        )
    )
    lines = path.read_text(encoding="utf-8").splitlines()
    assert lines[0] == ",".join(METRICS_COLUMNS)
    assert logger.row_count() == 2


def test_tensorboard_jsonl_fallback_uses_stable_tags(tmp_path: Path) -> None:
    logger = TensorBoardLogger(tmp_path / "tensorboard")
    logger.log_train_step(
        step=1,
        loss=0.25,
        learning_rate=1e-4,
        grad_norm=0.5,
        samples_per_second=2.0,
    )
    logger.close()
    text = (tmp_path / "tensorboard" / "tags.jsonl").read_text(encoding="utf-8")
    assert "train/loss" in text
    assert "train/learning_rate" in text
    assert logger.backend in {"jsonl", "torch_summary_writer"}


def test_run_id_matches_canonical_shape() -> None:
    config = load_config(ROOT / "configs" / "train" / "smoke.yaml")
    run_id = build_run_id(
        config,
        project_root=ROOT,
        created_at="20260821T001800Z",
    )
    assert run_id.startswith("smoke__smoke__libero90__n10__s42__20260821T001800Z__g")
    assert n_demos_token(config) == "n10"


def test_registry_is_built_from_manifests_only(tmp_path: Path) -> None:
    run = tmp_path / "runs" / "example"
    run.mkdir(parents=True)
    atomic_write_json(
        run / MANIFEST_NAME,
        {
            "run_id": "smoke__smoke__libero90__n10__s42__stamp__gabc1234",
            "stage": "smoke",
            "method": "smoke",
            "status": "completed",
            "git_commit": "abc",
            "train_seed": 42,
            "final_checkpoint_uri": "step_000200",
        },
    )
    rows = build_registry(tmp_path / "runs")
    assert len(rows) == 1
    output = tmp_path / "registry.csv"
    write_registry_csv(output, rows)
    text = output.read_text(encoding="utf-8")
    assert text.splitlines()[0].startswith("run_id,stage,method,status,manifest_path")
    assert "smoke__smoke__libero90" in text


def test_training_sources_never_import_wandb() -> None:
    root = ROOT / "src" / "vla_fewshot"
    offenders: list[str] = []
    for path in root.rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        if "import wandb" in text or "from wandb" in text:
            offenders.append(str(path.relative_to(ROOT)))
    assert offenders == []
