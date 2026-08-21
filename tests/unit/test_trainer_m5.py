from pathlib import Path

import pytest

from vla_fewshot.config import load_config
from vla_fewshot.logging.csv_logger import read_metric_column
from vla_fewshot.storage.layout import METRICS_CSV_NAME, step_directory
from vla_fewshot.training.compare import compare_checkpoints
from vla_fewshot.training.optim import ToyAdamW
from vla_fewshot.training.trainer import prepare_static_modules, run_static_training


ROOT = Path(__file__).resolve().parents[2]


def test_allowlist_is_asserted_before_optimizer(monkeypatch, tmp_path: Path) -> None:
    order: list[str] = []
    import vla_fewshot.training.trainer as trainer

    real_assert = trainer.assert_module_trainable_scope
    real_adam = trainer.ToyAdamW

    def wrapped_assert(*args, **kwargs):
        order.append("allowlist")
        return real_assert(*args, **kwargs)

    def wrapped_adam(*args, **kwargs):
        order.append("optimizer")
        return real_adam(*args, **kwargs)

    monkeypatch.setattr(trainer, "assert_module_trainable_scope", wrapped_assert)
    monkeypatch.setattr(trainer, "ToyAdamW", wrapped_adam)
    config = load_config(ROOT / "configs" / "train" / "smoke.yaml")
    prepare_static_modules(config, output_dir=tmp_path)
    assert order == ["allowlist", "optimizer"]
    assert (tmp_path / "trainable_parameters.txt").exists()


def test_in_process_resume_matches_continuous_run(tmp_path: Path) -> None:
    config = load_config(ROOT / "configs" / "train" / "smoke.yaml")
    command = ["python", "scripts/train_seen.py"]
    config_path = ROOT / "configs" / "train" / "smoke.yaml"
    run_a = tmp_path / "run_a"
    run_b = tmp_path / "run_b"
    run_static_training(
        config=config,
        run_dir=run_a,
        command=command,
        config_path=config_path,
        project_root=ROOT,
        log_freq=1,
        run_id="run_a",
    )
    run_static_training(
        config=config,
        run_dir=run_b,
        command=command,
        config_path=config_path,
        project_root=ROOT,
        stop_after=100,
        log_freq=1,
        run_id="run_b",
    )
    run_static_training(
        config=config,
        run_dir=run_b,
        command=command,
        config_path=config_path,
        project_root=ROOT,
        resume_from=step_directory(run_b, 100),
        log_freq=1,
        run_id="run_b",
    )
    report = compare_checkpoints(
        step_directory(run_a, 200),
        step_directory(run_b, 200),
        run_a=run_a,
        run_b=run_b,
    )
    assert report["passed"], report["checks"]
    losses = read_metric_column(run_a / METRICS_CSV_NAME, "loss")
    assert len(losses) == 200
    assert all(item == item and abs(item) < 1e6 for item in losses)


def test_optimizer_rejects_empty_trainable_set() -> None:
    from vla_fewshot.config import OptimizerConfig
    from vla_fewshot.training.toy import ToyPolicy

    policy = ToyPolicy(seed=0)
    for _, param in policy.named_parameters():
        param.requires_grad = False
    with pytest.raises(RuntimeError, match="no trainable parameters"):
        ToyAdamW(
            policy,
            OptimizerConfig(
                name="adamw",
                lr=1e-4,
                weight_decay=0.01,
                betas=(0.9, 0.95),
                eps=1e-8,
            ),
        )
