"""TensorBoard scalars with a JSONL fallback when the gpu extra is absent."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from vla_fewshot.logging.csv_logger import repair_trailing_line

STABLE_TAGS = (
    "train/loss",
    "train/learning_rate",
    "train/grad_norm",
    "train/samples_per_second",
)


class TensorBoardLogger:
    """Always write tensorboard/tags.jsonl. Use SummaryWriter only if present."""

    def __init__(self, logdir: Path) -> None:
        self.logdir = logdir
        self.logdir.mkdir(parents=True, exist_ok=True)
        self._jsonl = self.logdir / "tags.jsonl"
        repair_trailing_line(self._jsonl)
        if not self._jsonl.exists():
            self._jsonl.touch()
        self._writer: Any = None
        self.backend = "jsonl"
        try:
            from torch.utils.tensorboard import SummaryWriter

            self._writer = SummaryWriter(log_dir=str(self.logdir / "events"))
            self.backend = "torch_summary_writer"
        except ImportError:
            self.backend = "jsonl"

    def add_scalar(self, tag: str, value: float, step: int) -> None:
        record = {"tag": tag, "value": float(value), "step": int(step)}
        with self._jsonl.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        if self._writer is not None:
            self._writer.add_scalar(tag, float(value), int(step))

    def log_train_step(
        self,
        *,
        step: int,
        loss: float,
        learning_rate: float,
        grad_norm: float,
        samples_per_second: float,
    ) -> None:
        self.add_scalar("train/loss", loss, step)
        self.add_scalar("train/learning_rate", learning_rate, step)
        self.add_scalar("train/grad_norm", grad_norm, step)
        self.add_scalar("train/samples_per_second", samples_per_second, step)

    def flush(self) -> None:
        if self._writer is not None and hasattr(self._writer, "flush"):
            self._writer.flush()

    def close(self) -> None:
        self.flush()
        if self._writer is not None and hasattr(self._writer, "close"):
            self._writer.close()
            self._writer = None
