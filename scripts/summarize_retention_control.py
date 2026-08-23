"""Build the 2×2 weights × stats table. No GPU work."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from vla_fewshot.evaluation.retention_control import CONTROL_ADAPTED, CONTROL_SEEDS
from vla_fewshot.evaluation.seen_retention import (
    FROZEN_SEEN_SHA256,
    cell_name,
    seen_probe_slugs,
)


def _rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _filter(records: list[dict], *, seeds: list[int] | None = None) -> list[dict]:
    out = [
        row
        for row in records
        if row.get("instruction_condition") in (None, "correct")
    ]
    if seeds is not None:
        allowed = set(seeds)
        out = [row for row in out if int(row["eval_seed"]) in allowed]
    return out


def _score(records: list[dict]) -> dict[str, object]:
    suc = sum(int(row.get("success") or 0) for row in records)
    tot = len(records)
    return {"successes": suc, "n_rollouts": tot, "rate": (suc / tot) if tot else None}


def _load_dir(root: Path, seeds: list[int] | None = None) -> list[dict]:
    records: list[dict] = []
    if not root.exists():
        return records
    for path in root.rglob("rollouts.jsonl"):
        records.extend(_filter(_rows(path), seeds=seeds))
    return records


def collect(
    *,
    frozen_probe_root: Path,
    adapted_overlay_root: Path,
    control_root: Path,
) -> dict:
    probes = seen_probe_slugs()
    seeds = list(CONTROL_SEEDS)
    frozen_90 = _filter(
        _load_dir(frozen_probe_root / "step_100000"),
        seeds=seeds,
    )
    frozen_90 = [
        row for row in frozen_90 if row.get("checkpoint_sha256") == FROZEN_SEEN_SHA256
    ]
    adapted_overlay = []
    for task, n_demos, seed in CONTROL_ADAPTED:
        adapted_overlay.extend(
            _filter(
                _load_dir(adapted_overlay_root / cell_name(task, n_demos, seed)),
                seeds=seeds,
            )
        )
    adapted_90 = _load_dir(control_root, seeds=seeds)
    adapted_90 = [
        row
        for row in adapted_90
        if row.get("method") == "baseline"
        and row.get("normalization_suite") == "libero_90"
    ]
    frozen_overlay = _load_dir(control_root, seeds=seeds)
    frozen_overlay = [
        row
        for row in frozen_overlay
        if row.get("checkpoint_sha256") == FROZEN_SEEN_SHA256
        and row.get("normalization_suite") == "libero_goal"
    ]

    table = {
        ("frozen_seen", "libero_90"): {
            **_score(frozen_90),
            "source": "existing_seen_probes_5seed",
            "reran": False,
        },
        ("frozen_seen", "target_overlay"): {
            **_score(frozen_overlay),
            "source": "new_control",
            "reran": True,
        },
        ("target_adapted", "target_overlay"): {
            **_score(adapted_overlay),
            "source": "existing_900_5seed_subset",
            "reran": False,
        },
        ("target_adapted", "libero_90"): {
            **_score(adapted_90),
            "source": "new_control",
            "reran": True,
        },
    }
    per_cell = []
    for task, n_demos, seed in CONTROL_ADAPTED:
        name = cell_name(task, n_demos, seed)
        cell_overlay = [
            row
            for row in adapted_overlay
            if int(row.get("n_demos") or 0) == n_demos
            and int(row.get("train_seed") or 0) == seed
        ]
        cell_90 = [
            row
            for row in adapted_90
            if int(row.get("n_demos") or 0) == n_demos
            and int(row.get("train_seed") or 0) == seed
        ]
        overlay_dir = control_root / f"frozen_overlay__{name}"
        frozen_cell = _load_dir(overlay_dir, seeds=seeds)
        per_cell.append(
            {
                "cell": name,
                "frozen_libero_90": _score(frozen_90),
                "frozen_overlay": _score(frozen_cell),
                "adapted_overlay": _score(cell_overlay),
                "adapted_libero_90": _score(cell_90),
            }
        )
    return {
        "ok": all(
            isinstance(item["n_rollouts"], int) and item["n_rollouts"] > 0
            for item in table.values()
        ),
        "probes": list(probes),
        "eval_seeds": seeds,
        "table": {
            f"{weights}__{stats}": payload for (weights, stats), payload in table.items()
        },
        "per_cell": per_cell,
    }


def markdown(payload: dict) -> str:
    t = payload["table"]
    lines = [
        "# Retention 2×2 control: weights × stats",
        "",
        "Existing 0/900 is **not** labeled catastrophic forgetting here.",
        "Frozen seen × libero_90 and adapted × overlay are reused, not rerun.",
        "",
        "| weights \\ stats | libero_90 | target_overlay |",
        "|---|---|---|",
    ]

    def cell(key: str) -> str:
        item = t[key]
        rate = "—" if item["rate"] is None else f"{item['rate']:.3f}"
        return f"{item['successes']}/{item['n_rollouts']} ({rate})"

    lines.append(
        f"| frozen_seen | {cell('frozen_seen__libero_90')} | {cell('frozen_seen__target_overlay')} |"
    )
    lines.append(
        f"| target_adapted | {cell('target_adapted__libero_90')} | {cell('target_adapted__target_overlay')} |"
    )
    lines += ["", "## Per adapted cell", ""]
    lines.append(
        "| cell | frozen+90 | frozen+overlay | adapted+overlay | adapted+90 |"
    )
    lines.append("|---|---|---|---|---|")
    for row in payload["per_cell"]:

        def fmt(item: dict) -> str:
            if not item["n_rollouts"]:
                return "—"
            return f"{item['successes']}/{item['n_rollouts']}"

        lines.append(
            f"| {row['cell']} | {fmt(row['frozen_libero_90'])} | "
            f"{fmt(row['frozen_overlay'])} | {fmt(row['adapted_overlay'])} | "
            f"{fmt(row['adapted_libero_90'])} |"
        )
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--frozen-probe-root",
        type=Path,
        default=Path("/mnt/vla/eval/seen_probes__gd4b8fb8"),
    )
    parser.add_argument(
        "--adapted-overlay-root",
        type=Path,
        default=Path("/mnt/vla/eval/seen_retention"),
    )
    parser.add_argument(
        "--control-root",
        type=Path,
        default=Path("/mnt/vla/eval/retention_control"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/mnt/vla/validation/TODO28_retention_control"),
    )
    args = parser.parse_args()
    payload = collect(
        frozen_probe_root=args.frozen_probe_root,
        adapted_overlay_root=args.adapted_overlay_root,
        control_root=args.control_root,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "control.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    md = markdown(payload)
    (args.output_dir / "control.md").write_text(md, encoding="utf-8")
    with (args.output_dir / "control_2x2.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["weights", "stats", "successes", "n_rollouts", "rate", "source", "reran"],
        )
        writer.writeheader()
        for key, item in payload["table"].items():
            weights, stats = key.split("__", 1)
            writer.writerow(
                {
                    "weights": weights,
                    "stats": stats,
                    "successes": item["successes"],
                    "n_rollouts": item["n_rollouts"],
                    "rate": item["rate"],
                    "source": item["source"],
                    "reran": item["reran"],
                }
            )
    print(md)
    return 0 if payload["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
