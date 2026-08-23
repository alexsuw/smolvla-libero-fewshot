"""Summarize corrected seen retention. Does not start GPU work.

The overlay 0/900 tree is read only as a deployment-normalization note.
It is never labeled catastrophic forgetting.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

from vla_fewshot.evaluation.seen_retention import (
    FROZEN_SEEN_RATE,
    FROZEN_SEEN_ROLLOUTS,
    FROZEN_SEEN_SHA256,
    FROZEN_SEEN_SUCCESSES,
    cell_name,
    retention_grid,
    seen_probe_slugs,
)
from vla_fewshot.evaluation.seen_retention_libero90 import (
    LIBERO90_SUITE_STATS_SHA256,
    assert_corrected_rollout_record,
    load_original_seen_probe_fingerprints,
)
from vla_fewshot.training.trainer import TrainError


def _rollouts(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _load_frozen_seen(probe_root: Path) -> dict[str, Any]:
    probes = seen_probe_slugs()
    per_probe = {}
    suc = tot = 0
    for probe in probes:
        path = probe_root / "step_100000" / probe / "rollouts.jsonl"
        rows = [
            row
            for row in _rollouts(path)
            if row.get("instruction_condition") in (None, "correct")
            and row.get("checkpoint_sha256") == FROZEN_SEEN_SHA256
        ]
        rows = sorted(rows, key=lambda row: int(row["eval_seed"]))
        wins = sum(int(row.get("success") or 0) for row in rows)
        per_probe[probe] = [wins, len(rows)]
        suc += wins
        tot += len(rows)
    return {
        "checkpoint_sha256": FROZEN_SEEN_SHA256,
        "successes": suc,
        "n_rollouts": tot,
        "rate": (suc / tot) if tot else None,
        "per_probe": per_probe,
        "source": str(probe_root),
        "reran": False,
        "stats": "libero_90",
    }


def _target_success(eval_root: Path, task: str, n_demos: int, seed: int) -> tuple[int, int] | None:
    named = eval_root / cell_name(task, n_demos, seed)
    files = list(named.rglob("rollouts.jsonl"))
    if not files:
        return None
    rows = []
    for path in files:
        rows.extend(_rollouts(path))
    rows = [
        row
        for row in rows
        if row.get("instruction_condition") in (None, "correct")
        and row.get("task_slug") == task
    ]
    if not rows:
        return None
    return sum(int(row.get("success") or 0) for row in rows), len(rows)


def _zero_shot_success(eval_root: Path) -> dict[str, Any]:
    suc = tot = 0
    per_task: dict[str, list[int]] = {}
    for task in ("drawer_middle", "bowl_stove", "wine_cabinet"):
        files = list((eval_root / task).rglob("rollouts.jsonl"))
        rows = []
        for path in files:
            rows.extend(_rollouts(path))
        rows = [
            row
            for row in rows
            if row.get("instruction_condition") in (None, "correct")
            and row.get("task_slug") == task
        ]
        wins = sum(int(row.get("success") or 0) for row in rows)
        per_task[task] = [wins, len(rows)]
        suc += wins
        tot += len(rows)
    return {
        "successes": suc,
        "n_rollouts": tot,
        "rate": (suc / tot) if tot else None,
        "per_task": per_task,
        "source": str(eval_root),
    }


def _pearson(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) < 2 or len(xs) != len(ys):
        return None
    mean_x = sum(xs) / len(xs)
    mean_y = sum(ys) / len(ys)
    num = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys, strict=True))
    den_x = math.sqrt(sum((x - mean_x) ** 2 for x in xs))
    den_y = math.sqrt(sum((y - mean_y) ** 2 for y in ys))
    if den_x == 0 or den_y == 0:
        return None
    return num / (den_x * den_y)


def collect(
    *,
    eval_root: Path,
    official_target_eval: Path,
    n12_target_eval: Path,
    frozen_probe_root: Path,
    zero_shot_eval: Path,
) -> dict[str, Any]:
    frozen = _load_frozen_seen(frozen_probe_root)
    fingerprints = load_original_seen_probe_fingerprints(frozen_probe_root)
    zero_shot = _zero_shot_success(zero_shot_eval)
    probes = seen_probe_slugs()
    cells = []
    all_rows: list[dict[str, Any]] = []
    for task, n_demos, seed in retention_grid():
        name = cell_name(task, n_demos, seed)
        cell_dir = eval_root / name
        probe_scores = {}
        issues = []
        hashes: set[str] = set()
        uris: set[str] = set()
        stats_suites: set[str] = set()
        stats_hashes: set[str] = set()
        suc = tot = 0
        for probe in probes:
            files = list(cell_dir.rglob(f"{probe}/rollouts.jsonl"))
            rows: list[dict[str, Any]] = []
            for path in files:
                rows.extend(_rollouts(path))
            rows = [
                row
                for row in rows
                if row.get("instruction_condition") in (None, "correct")
            ]
            seeds = sorted(int(row["eval_seed"]) for row in rows if "eval_seed" in row)
            wins = sum(int(row.get("success") or 0) for row in rows)
            probe_scores[probe] = {
                "successes": wins,
                "n_rollouts": len(rows),
                "eval_seeds": seeds,
            }
            suc += wins
            tot += len(rows)
            if len(rows) != 10:
                issues.append(f"{probe} rollouts={len(rows)}")
            if seeds != list(range(1000, 1010)):
                issues.append(f"{probe} seeds={seeds}")
            for row in rows:
                hashes.add(str(row.get("checkpoint_sha256")))
                uris.add(str(row.get("checkpoint_uri")))
                stats_suites.add(str(row.get("normalization_suite")))
                stats_hashes.add(str(row.get("normalization_stats_sha256")))
                all_rows.append(
                    {
                        "target_task": task,
                        "n_demos": n_demos,
                        "train_seed": seed,
                        "seen_probe": probe,
                        **row,
                    }
                )
                try:
                    if len(hashes) == 1:
                        assert_corrected_rollout_record(
                            row,
                            original_fingerprints=fingerprints,
                            expected_weights=next(iter(hashes)),
                            probe=probe,
                        )
                except TrainError as error:
                    issues.append(str(error))
        target_eval = n12_target_eval if n_demos in (1, 2) else official_target_eval
        target = _target_success(target_eval, task, n_demos, seed)
        cells.append(
            {
                "cell": name,
                "target_task": task,
                "n_demos": n_demos,
                "train_seed": seed,
                "probes": probe_scores,
                "retention_successes": suc,
                "retention_rollouts": tot,
                "retention_rate": (suc / tot) if tot else None,
                "retention_delta_vs_frozen": ((suc / tot) - FROZEN_SEEN_RATE) if tot else None,
                "target_successes": None if target is None else target[0],
                "target_rollouts": None if target is None else target[1],
                "target_rate": None if target is None or not target[1] else target[0] / target[1],
                "checkpoint_sha256": next(iter(hashes)) if len(hashes) == 1 else list(hashes),
                "checkpoint_uri": next(iter(uris)) if len(uris) == 1 else list(uris),
                "normalization_suite": next(iter(stats_suites)) if len(stats_suites) == 1 else list(stats_suites),
                "normalization_stats_sha256": (
                    next(iter(stats_hashes)) if len(stats_hashes) == 1 else list(stats_hashes)
                ),
                "ok": not issues
                and tot == 30
                and len(hashes) == 1
                and stats_suites == {"libero_90"}
                and stats_hashes == {LIBERO90_SUITE_STATS_SHA256},
                "issues": issues,
            }
        )

    def _pool(rows: list[dict[str, Any]]) -> dict[str, Any]:
        suc = sum(int(cell["retention_successes"] or 0) for cell in rows)
        tot = sum(int(cell["retention_rollouts"] or 0) for cell in rows)
        t_suc = sum(int(cell["target_successes"] or 0) for cell in rows if cell["target_successes"] is not None)
        t_tot = sum(int(cell["target_rollouts"] or 0) for cell in rows if cell["target_rollouts"] is not None)
        rate = (suc / tot) if tot else None
        return {
            "retention_successes": suc,
            "retention_rollouts": tot,
            "retention_rate": rate,
            "retention_delta_vs_frozen": (rate - FROZEN_SEEN_RATE) if rate is not None else None,
            "target_successes": t_suc if t_tot else None,
            "target_rollouts": t_tot if t_tot else None,
            "target_rate": (t_suc / t_tot) if t_tot else None,
            "n_checkpoints": len(rows),
        }

    by_n = {n: _pool([cell for cell in cells if cell["n_demos"] == n]) for n in (1, 2, 5, 10, 25)}
    by_task_n = {
        f"{task}_n{n:02d}": _pool(
            [cell for cell in cells if cell["target_task"] == task and cell["n_demos"] == n]
        )
        for task in ("drawer_middle", "bowl_stove", "wine_cabinet")
        for n in (1, 2, 5, 10, 25)
    }
    by_seed = {
        str(seed): _pool([cell for cell in cells if cell["train_seed"] == seed])
        for seed in (42, 123)
    }
    pairs = [
        (float(cell["target_rate"]), float(cell["retention_rate"]))
        for cell in cells
        if cell["target_rate"] is not None and cell["retention_rate"] is not None
    ]
    correlation = _pearson([p[0] for p in pairs], [p[1] for p in pairs])
    n_table = [
        {
            "n_demos": 0,
            "naive_target_successes": zero_shot["successes"],
            "naive_target_rollouts": zero_shot["n_rollouts"],
            "naive_target_rate": zero_shot["rate"],
            "corrected_seen_successes": FROZEN_SEEN_SUCCESSES,
            "corrected_seen_rollouts": FROZEN_SEEN_ROLLOUTS,
            "corrected_seen_rate": FROZEN_SEEN_RATE,
            "delta_vs_frozen_seen": 0.0,
            "source": "frozen_seen_24_30_plus_zero_shot",
        }
    ]
    for n in (1, 2, 5, 10, 25):
        pooled = by_n[n]
        n_table.append(
            {
                "n_demos": n,
                "naive_target_successes": pooled["target_successes"],
                "naive_target_rollouts": pooled["target_rollouts"],
                "naive_target_rate": pooled["target_rate"],
                "corrected_seen_successes": pooled["retention_successes"],
                "corrected_seen_rollouts": pooled["retention_rollouts"],
                "corrected_seen_rate": pooled["retention_rate"],
                "delta_vs_frozen_seen": pooled["retention_delta_vs_frozen"],
                "source": "corrected_adapted_plus_libero90",
            }
        )
    return {
        "ok": all(cell["ok"] for cell in cells) and len(cells) == 30,
        "metric": "weight_forgetting_adapted_weights_plus_libero90_stats",
        "interpretation": {
            "weight_forgetting_comparison": (
                "adapted weights + libero_90 stats vs frozen weights + libero_90 stats"
            ),
            "not_forgetting": (
                "old 0/900 adapted+target-overlay is a deployment-normalization result"
            ),
            "frozen_seen_reused": True,
            "target_success_rerun": False,
        },
        "frozen_seen": frozen,
        "frozen_seen_expected": {
            "successes": FROZEN_SEEN_SUCCESSES,
            "n_rollouts": FROZEN_SEEN_ROLLOUTS,
            "rate": FROZEN_SEEN_RATE,
            "sha256": FROZEN_SEEN_SHA256,
        },
        "zero_shot_target": zero_shot,
        "cells": cells,
        "pooled_by_n": {str(n): value for n, value in by_n.items()},
        "pooled_by_task_n": by_task_n,
        "pooled_by_train_seed": by_seed,
        "target_vs_retention_pearson": correlation,
        "n_table": n_table,
        "n_rollouts": len(all_rows),
        "all_rows": all_rows,
    }


def write_outputs(payload: dict[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    slim = {key: value for key, value in payload.items() if key != "all_rows"}
    (output_dir / "retention.json").write_text(
        json.dumps(slim, indent=2) + "\n", encoding="utf-8"
    )
    (output_dir / "rollouts.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in payload["all_rows"]),
        encoding="utf-8",
    )
    fieldnames = [
        "target_task",
        "n_demos",
        "train_seed",
        "seen_probe",
        "eval_seed",
        "success",
        "episode_length",
        "initial_state_fingerprint",
        "init_state_mode",
        "checkpoint_uri",
        "checkpoint_sha256",
        "normalization_suite",
        "normalization_stats_sha256",
    ]
    with (output_dir / "rollouts.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in payload["all_rows"]:
            writer.writerow(row)
    with (output_dir / "retention_long.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "target_task",
                "n_demos",
                "train_seed",
                "seen_probe",
                "successes",
                "n_rollouts",
                "retention_30",
                "retention_rate",
                "retention_delta_vs_0p80",
                "target_successes",
                "target_rollouts",
                "target_rate",
                "checkpoint_sha256",
                "checkpoint_uri",
                "normalization_suite",
                "normalization_stats_sha256",
            ],
        )
        writer.writeheader()
        for cell in payload["cells"]:
            for probe, score in cell["probes"].items():
                writer.writerow(
                    {
                        "target_task": cell["target_task"],
                        "n_demos": cell["n_demos"],
                        "train_seed": cell["train_seed"],
                        "seen_probe": probe,
                        "successes": score["successes"],
                        "n_rollouts": score["n_rollouts"],
                        "retention_30": cell["retention_successes"],
                        "retention_rate": cell["retention_rate"],
                        "retention_delta_vs_0p80": cell["retention_delta_vs_frozen"],
                        "target_successes": cell["target_successes"],
                        "target_rollouts": cell["target_rollouts"],
                        "target_rate": cell["target_rate"],
                        "checkpoint_sha256": cell["checkpoint_sha256"],
                        "checkpoint_uri": cell["checkpoint_uri"],
                        "normalization_suite": cell["normalization_suite"],
                        "normalization_stats_sha256": cell["normalization_stats_sha256"],
                    }
                )
    lines = [
        "# Corrected seen retention (adapted weights + libero_90 stats)",
        "",
        "This is the weight-forgetting comparison:",
        "",
        "- adapted weights + `libero_90` stats",
        "- versus frozen seen weights + `libero_90` stats (reused 24/30 = 0.80)",
        "",
        "The previous overlay 0/900 under `/mnt/vla/eval/seen_retention` is a",
        "separate deployment-normalization result. It is **not** catastrophic",
        "forgetting.",
        "",
        f"Integrity ok: `{payload['ok']}`",
        f"Frozen seen reference (not rerun): "
        f"{payload['frozen_seen']['successes']}/{payload['frozen_seen']['n_rollouts']}",
        f"Target success (not rerun). Pearson(target, retention) = "
        f"{'—' if payload['target_vs_retention_pearson'] is None else f'{payload['target_vs_retention_pearson']:+.3f}'}",
        "",
        "## N | naive target success | corrected seen retention | delta vs frozen seen",
        "",
        "| N | naive target success | corrected seen retention | delta vs frozen seen |",
        "|---|---|---|---|",
    ]
    for row in payload["n_table"]:
        target = (
            "—"
            if row["naive_target_successes"] is None
            else f"{row['naive_target_successes']}/{row['naive_target_rollouts']} "
            f"({row['naive_target_rate']:.3f})"
        )
        seen = (
            "—"
            if row["corrected_seen_successes"] is None
            else f"{row['corrected_seen_successes']}/{row['corrected_seen_rollouts']} "
            f"({row['corrected_seen_rate']:.3f})"
        )
        delta = row["delta_vs_frozen_seen"]
        delta_s = "—" if delta is None else f"{delta:+.3f}"
        lines.append(f"| {row['n_demos']} | {target} | {seen} | {delta_s} |")
    lines += [
        "",
        "## Per checkpoint / probe",
        "",
        "| target | N | seed | black_bowl_plate | drawer_bowl | book_caddy | seen /30 | rate | Δ vs 0.80 | target | weights | stats |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for cell in payload["cells"]:
        probes = cell["probes"]

        def _fmt(probe: str) -> str:
            item = probes[probe]
            return f"{item['successes']}/{item['n_rollouts']}"

        target = (
            "—"
            if cell["target_successes"] is None
            else f"{cell['target_successes']}/{cell['target_rollouts']}"
        )
        delta = cell["retention_delta_vs_frozen"]
        delta_s = "—" if delta is None else f"{delta:+.3f}"
        rate = "—" if cell["retention_rate"] is None else f"{cell['retention_rate']:.3f}"
        weights = cell["checkpoint_sha256"]
        weights_s = weights[:12] if isinstance(weights, str) else str(weights)
        lines.append(
            f"| {cell['target_task']} | {cell['n_demos']} | {cell['train_seed']} | "
            f"{_fmt('black_bowl_plate')} | {_fmt('drawer_bowl')} | {_fmt('book_caddy')} | "
            f"{cell['retention_successes']}/{cell['retention_rollouts']} | {rate} | "
            f"{delta_s} | {target} | `{weights_s}` | "
            f"{cell['normalization_suite']}/`{str(cell['normalization_stats_sha256'])[:12]}` |"
        )
    lines += [
        "",
        "## Pooled by N (6 adapted checkpoints)",
        "",
        "| N | seen retention | Δ vs 0.80 | target success |",
        "|---|---|---|---|",
    ]
    for key, value in payload["pooled_by_n"].items():
        delta = value["retention_delta_vs_frozen"]
        lines.append(
            f"| {key} | {value['retention_successes']}/{value['retention_rollouts']} "
            f"({value['retention_rate']:.3f}) | "
            f"{'—' if delta is None else f'{delta:+.3f}'} | "
            f"{value['target_successes']}/{value['target_rollouts']} |"
        )
    lines += [
        "",
        "## Pooled by target task × N",
        "",
        "| cell | seen retention | target success |",
        "|---|---|---|",
    ]
    for key, value in payload["pooled_by_task_n"].items():
        lines.append(
            f"| {key} | {value['retention_successes']}/{value['retention_rollouts']} | "
            f"{value['target_successes']}/{value['target_rollouts']} |"
        )
    lines += [
        "",
        "## Pooled by train seed",
        "",
        "| seed | seen retention | target success |",
        "|---|---|---|",
    ]
    for key, value in payload["pooled_by_train_seed"].items():
        lines.append(
            f"| {key} | {value['retention_successes']}/{value['retention_rollouts']} | "
            f"{value['target_successes']}/{value['target_rollouts']} |"
        )
    lines += [
        "",
        "## Target success vs corrected seen retention (per checkpoint)",
        "",
        "| target | N | seed | target rate | seen rate |",
        "|---|---|---|---|---|",
    ]
    for cell in payload["cells"]:
        t_rate = "—" if cell["target_rate"] is None else f"{cell['target_rate']:.3f}"
        s_rate = "—" if cell["retention_rate"] is None else f"{cell['retention_rate']:.3f}"
        lines.append(
            f"| {cell['target_task']} | {cell['n_demos']} | {cell['train_seed']} | "
            f"{t_rate} | {s_rate} |"
        )
    lines.append("")
    (output_dir / "retention.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--eval-root",
        type=Path,
        default=Path("/mnt/vla/eval/seen_retention_libero90"),
    )
    parser.add_argument(
        "--official-target-eval",
        type=Path,
        default=Path("/mnt/vla/eval/target_baseline"),
    )
    parser.add_argument(
        "--n12-target-eval",
        type=Path,
        default=Path("/mnt/vla/eval/target_baseline_n12"),
    )
    parser.add_argument(
        "--frozen-probe-root",
        type=Path,
        default=Path("/mnt/vla/eval/seen_probes__gd4b8fb8"),
    )
    parser.add_argument(
        "--zero-shot-eval",
        type=Path,
        default=Path("/mnt/vla/eval/zero_shot_v2_seen_stats"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/mnt/vla/validation/TODO28_retention_libero90"),
    )
    args = parser.parse_args()
    if args.eval_root.resolve() == Path("/mnt/vla/eval/seen_retention").resolve():
        print("refusing to summarize the overlay 0/900 tree as corrected retention")
        return 1
    payload = collect(
        eval_root=args.eval_root,
        official_target_eval=args.official_target_eval,
        n12_target_eval=args.n12_target_eval,
        frozen_probe_root=args.frozen_probe_root,
        zero_shot_eval=args.zero_shot_eval,
    )
    write_outputs(payload, args.output_dir)
    print((args.output_dir / "retention.md").read_text(encoding="utf-8"))
    return 0 if payload["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
