"""Build seen-retention tables from artifacts. Does not start GPU work."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

from vla_fewshot.evaluation.seen_retention import (
    FROZEN_SEEN_RATE,
    FROZEN_SEEN_ROLLOUTS,
    FROZEN_SEEN_SHA256,
    FROZEN_SEEN_SUCCESSES,
    cell_name,
    retention_grid,
    seen_probe_slugs,
)


def _rollouts(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _load_frozen_seen(probe_root: Path) -> dict:
    probes = seen_probe_slugs()
    per_probe = {}
    per_seed = {}
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
        per_seed[probe] = {
            int(row["eval_seed"]): int(row.get("success") or 0) for row in rows
        }
        suc += wins
        tot += len(rows)
    return {
        "checkpoint_sha256": FROZEN_SEEN_SHA256,
        "successes": suc,
        "n_rollouts": tot,
        "rate": (suc / tot) if tot else None,
        "per_probe": per_probe,
        "per_seed": per_seed,
        "source": str(probe_root),
        "reran": False,
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


def collect(
    *,
    eval_root: Path,
    official_target_eval: Path,
    n12_target_eval: Path,
    frozen_probe_root: Path,
) -> dict:
    frozen = _load_frozen_seen(frozen_probe_root)
    probes = seen_probe_slugs()
    cells = []
    for task, n_demos, seed in retention_grid():
        name = cell_name(task, n_demos, seed)
        cell_dir = eval_root / name
        probe_scores = {}
        issues = []
        hashes = set()
        uris = set()
        suc = tot = 0
        for probe in probes:
            files = list((cell_dir).rglob(f"{probe}/rollouts.jsonl"))
            rows = []
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
                hashes.add(row.get("checkpoint_sha256"))
                uris.add(row.get("checkpoint_uri"))
                if row.get("protocol_id") != "seen_probe_v1":
                    issues.append(f"{probe} protocol {row.get('protocol_id')}")
                if int(row.get("n_demos") or 0) != n_demos:
                    issues.append(f"{probe} n_demos mismatch")
                if int(row.get("train_seed") or 0) != seed:
                    issues.append(f"{probe} train_seed mismatch")
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
                "ok": not issues and tot == 30 and len(hashes) == 1,
                "issues": issues,
            }
        )
    by_n: dict[int, list[float]] = defaultdict(list)
    by_task_n: dict[tuple[str, int], list[float]] = defaultdict(list)
    by_n_seed: dict[tuple[int, int], list[float]] = defaultdict(list)
    for cell in cells:
        if cell["retention_rate"] is None:
            continue
        by_n[cell["n_demos"]].append(cell["retention_rate"])
        by_task_n[(cell["target_task"], cell["n_demos"])].append(cell["retention_rate"])
        by_n_seed[(cell["n_demos"], cell["train_seed"])].append(cell["retention_rate"])
    return {
        "ok": all(cell["ok"] for cell in cells) and len(cells) == 30,
        "frozen_seen": frozen,
        "frozen_seen_expected": {
            "successes": FROZEN_SEEN_SUCCESSES,
            "n_rollouts": FROZEN_SEEN_ROLLOUTS,
            "rate": FROZEN_SEEN_RATE,
            "sha256": FROZEN_SEEN_SHA256,
        },
        "cells": cells,
        "mean_retention_by_n": {
            str(n): (sum(vals) / len(vals) if vals else None) for n, vals in sorted(by_n.items())
        },
        "mean_retention_by_task_n": {
            f"{task}_n{n:02d}": (sum(vals) / len(vals) if vals else None)
            for (task, n), vals in sorted(by_task_n.items())
        },
        "mean_retention_by_n_seed": {
            f"n{n:02d}_s{seed}": (sum(vals) / len(vals) if vals else None)
            for (n, seed), vals in sorted(by_n_seed.items())
        },
    }


def write_outputs(payload: dict, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "retention.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    long_path = output_dir / "retention_long.csv"
    with long_path.open("w", encoding="utf-8", newline="") as handle:
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
                    }
                )
    lines = [
        "# Seen retention of naive target-adapted finals",
        "",
        f"Integrity ok: `{payload['ok']}`",
        f"Frozen seen reference (not rerun): "
        f"{payload['frozen_seen']['successes']}/{payload['frozen_seen']['n_rollouts']}",
        "",
        "## Per checkpoint / probe",
        "",
        "| target | N | seed | black_bowl_plate | drawer_bowl | book_caddy | seen /30 | Δ vs 0.80 | target |",
        "|---|---|---|---|---|---|---|---|---|",
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
        lines.append(
            f"| {cell['target_task']} | {cell['n_demos']} | {cell['train_seed']} | "
            f"{_fmt('black_bowl_plate')} | {_fmt('drawer_bowl')} | {_fmt('book_caddy')} | "
            f"{cell['retention_successes']}/{cell['retention_rollouts']} | {delta_s} | {target} |"
        )
    lines += [
        "",
        "## Mean retention by N",
        "",
        "| N | mean seen retention |",
        "|---|---|",
    ]
    for key, value in payload["mean_retention_by_n"].items():
        lines.append(f"| {key} | {'—' if value is None else f'{value:.3f}'} |")
    lines += [
        "",
        "## Mean retention by target task and N",
        "",
        "| cell | mean seen retention |",
        "|---|---|",
    ]
    for key, value in payload["mean_retention_by_task_n"].items():
        lines.append(f"| {key} | {'—' if value is None else f'{value:.3f}'} |")
    lines += [
        "",
        "## Mean retention by N and train seed",
        "",
        "| cell | mean seen retention |",
        "|---|---|",
    ]
    for key, value in payload["mean_retention_by_n_seed"].items():
        lines.append(f"| {key} | {'—' if value is None else f'{value:.3f}'} |")
    lines.append("")
    (output_dir / "retention.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-root", type=Path, default=Path("/mnt/vla/eval/seen_retention"))
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
        "--output-dir",
        type=Path,
        default=Path("/mnt/vla/validation/TODO28_retention"),
    )
    args = parser.parse_args()
    payload = collect(
        eval_root=args.eval_root,
        official_target_eval=args.official_target_eval,
        n12_target_eval=args.n12_target_eval,
        frozen_probe_root=args.frozen_probe_root,
    )
    write_outputs(payload, args.output_dir)
    print((args.output_dir / "retention.md").read_text(encoding="utf-8"))
    return 0 if payload["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
