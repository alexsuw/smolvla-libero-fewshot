"""Seen-checkpoint selection from probe scores. Never reads target success."""

from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from vla_fewshot.data.expected import TARGET_TASKS
from vla_fewshot.evaluation.runner import checkpoint_sha256
from vla_fewshot.storage.layout import METRICS_CSV_NAME, checkpoints_root, step_directory_name
from vla_fewshot.training.checkpoint import is_complete_checkpoint

TARGET_SLUGS = frozenset(TARGET_TASKS)
STATIC_PROTOCOL_PREFIX = "static_"


class SelectionError(ValueError):
    """Raised when probe scores cannot legally freeze a seen checkpoint."""


@dataclass(frozen=True)
class ProbeScore:
    step: int
    sha256: str
    uri: str
    per_task: dict[str, float]
    n_rollouts: dict[str, int]
    protocol_ids: dict[str, str]
    unstable: bool = False
    reason: str | None = None

    @property
    def mean_success(self) -> float:
        if not self.per_task:
            return float("nan")
        return sum(self.per_task.values()) / float(len(self.per_task))


@dataclass(frozen=True)
class SelectionResult:
    score: ProbeScore
    best_mean: float
    band_steps: list[int]
    used_fallback: bool
    rule: str


def parse_step_directory_name(name: str) -> int | None:
    if not name.startswith("step_"):
        return None
    suffix = name.split(".", 1)[0][5:]
    if not suffix.isdigit():
        return None
    return int(suffix)


STAGED_PROBE_STEPS: tuple[int, ...] = (20000, 40000, 60000, 80000, 100000)
STAGE1_PROBE_ROLLOUTS = 5
STAGE2_PROBE_ROLLOUTS = 10
STAGE2_FINALIST_COUNT = 2


def parse_checkpoint_steps(raw: str) -> list[int]:
    parts = [item.strip() for item in raw.split(",") if item.strip()]
    if not parts:
        raise SelectionError("checkpoint --steps is empty")
    steps: list[int] = []
    for part in parts:
        if not part.isdigit():
            raise SelectionError(f"invalid checkpoint step: {part!r}")
        steps.append(int(part))
    if len(set(steps)) != len(steps):
        raise SelectionError(f"duplicate checkpoint steps: {steps}")
    return steps


def rank_probe_scores(scores: Sequence[ProbeScore]) -> list[ProbeScore]:
    """Higher mean first; earlier step breaks ties. Never reads target tasks."""

    return sorted(scores, key=lambda score: (-score.mean_success, score.step))


def list_complete_checkpoints(run_dir: Path) -> list[tuple[int, Path]]:
    root = checkpoints_root(run_dir)
    found: list[tuple[int, Path]] = []
    if not root.is_dir():
        return found
    for path in sorted(root.iterdir()):
        step = parse_step_directory_name(path.name)
        if step is None or not is_complete_checkpoint(path):
            continue
        found.append((step, path))
    return found


def metrics_nonfinite_at_step(run_dir: Path, step: int) -> bool:
    path = run_dir / METRICS_CSV_NAME
    if not path.is_file():
        return False
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if int(float(row.get("global_step") or -1)) != step:
                continue
            loss = row.get("loss")
            grad = row.get("grad_norm")
            for raw in (loss, grad):
                if raw in {None, ""}:
                    continue
                value = float(raw)
                if not math.isfinite(value):
                    return True
    return False


def _assert_probe_tasks(per_task: Sequence[str], probe_slugs: Sequence[str]) -> None:
    slugs = set(per_task)
    leaked = slugs & TARGET_SLUGS
    if leaked:
        raise SelectionError(
            f"target-task success leaked into seen selection: {sorted(leaked)}"
        )
    missing = set(probe_slugs) - slugs
    extra = slugs - set(probe_slugs)
    if missing or extra:
        raise SelectionError(
            f"seen probes must be exactly {list(probe_slugs)}; "
            f"missing={sorted(missing)} extra={sorted(extra)}"
        )


def select_seen_checkpoint(
    scores: Sequence[ProbeScore],
    *,
    probe_slugs: Sequence[str],
    tolerance: float,
    fallback_step: int,
    indistinguishable_fallback: bool = True,
) -> SelectionResult:
    """Earliest checkpoint within tolerance of best probe mean; else fallback."""

    if tolerance < 0:
        raise SelectionError("tolerance must be non-negative")
    if len(probe_slugs) != 3:
        raise SelectionError("seen probes must be exactly three libero_90 tasks")
    if not scores:
        raise SelectionError("no checkpoint probe scores were provided")

    for score in scores:
        _assert_probe_tasks(list(score.per_task), probe_slugs)
        for protocol_id in score.protocol_ids.values():
            if protocol_id.startswith(STATIC_PROTOCOL_PREFIX):
                raise SelectionError(
                    f"static protocol {protocol_id!r} cannot freeze the seen checkpoint"
                )

    stable = [score for score in scores if not score.unstable]
    if not stable:
        raise SelectionError("every checkpoint is marked unstable or NaN")

    means = {score.step: score.mean_success for score in stable}
    if any(not math.isfinite(value) for value in means.values()):
        raise SelectionError("stable checkpoint has a non-finite probe mean")
    best_mean = max(means.values())
    band = [score for score in stable if means[score.step] + 1e-12 >= best_mean - tolerance]
    band_steps = sorted(score.step for score in band)
    all_indistinguishable = {score.step for score in band} == set(means)
    rule = (
        "Exclude NaN or unstable checkpoints. Score only the three frozen "
        "libero_90 seen-probe tasks. Take the earliest checkpoint within "
        f"{tolerance} mean success of the best probe. If noise prevents a "
        f"distinction, use step {fallback_step}. Never look at target-task success."
    )
    if indistinguishable_fallback and all_indistinguishable:
        fallback = next((score for score in stable if score.step == fallback_step), None)
        if fallback is None:
            raise SelectionError(
                f"probe scores are within {tolerance} of each other but "
                f"fallback step {fallback_step} is missing"
            )
        return SelectionResult(
            score=fallback,
            best_mean=best_mean,
            band_steps=band_steps,
            used_fallback=True,
            rule=rule,
        )
    chosen = min(band, key=lambda score: score.step)
    return SelectionResult(
        score=chosen,
        best_mean=best_mean,
        band_steps=band_steps,
        used_fallback=False,
        rule=rule,
    )


def load_summary(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise SelectionError(f"{path} is not a JSON object")
    return payload


def collect_probe_scores(
    *,
    run_dir: Path,
    probe_root: Path,
    probe_slugs: Sequence[str],
    steps: Sequence[int] | None = None,
    min_rollouts: int = 1,
) -> list[ProbeScore]:
    """Read `{probe_root}/step_XXXXXX/<slug>/summary.json` plus run checkpoints."""

    wanted = set(steps) if steps is not None else None
    scores: list[ProbeScore] = []
    for step, ckpt in list_complete_checkpoints(run_dir):
        if wanted is not None and step not in wanted:
            continue
        cell_root = probe_root / step_directory_name(step)
        if not cell_root.is_dir():
            continue
        per_task: dict[str, float] = {}
        n_rollouts: dict[str, int] = {}
        protocol_ids: dict[str, str] = {}
        sha256 = checkpoint_sha256(ckpt)
        complete = True
        for slug in probe_slugs:
            summary_path = cell_root / slug / "summary.json"
            if not summary_path.is_file():
                complete = False
                break
            summary = load_summary(summary_path)
            if str(summary.get("task_slug")) != slug:
                raise SelectionError(f"{summary_path} task_slug mismatch")
            observed = str(summary.get("checkpoint_sha256") or "")
            if observed and observed != sha256:
                raise SelectionError(
                    f"{summary_path} checkpoint hash {observed} != {sha256}"
                )
            per_task[slug] = float(summary["success_rate"])
            n_rollouts[slug] = int(summary["n_rollouts"])
            protocol_ids[slug] = str(summary["protocol_id"])
            if n_rollouts[slug] < min_rollouts:
                complete = False
                break
        if not complete:
            continue
        unstable = metrics_nonfinite_at_step(run_dir, step)
        scores.append(
            ProbeScore(
                step=step,
                sha256=sha256,
                uri=str(ckpt),
                per_task=per_task,
                n_rollouts=n_rollouts,
                protocol_ids=protocol_ids,
                unstable=unstable,
                reason="non-finite train metrics" if unstable else None,
            )
        )
    return scores
