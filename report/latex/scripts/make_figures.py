#!/usr/bin/env python3
"""Rebuild the paper figures from frozen experiment artifacts."""

from __future__ import annotations

import csv
import math
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "imgs"
RUNS = Path("/mnt/vla/runs")

COLORS = {
    "blue": "#276FBF",
    "orange": "#F28E2B",
    "green": "#2A9D8F",
    "red": "#D1495B",
    "purple": "#7A5195",
    "gray": "#667085",
}

mpl.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 8,
        "axes.titlesize": 8.5,
        "axes.labelsize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.22,
        "grid.linewidth": 0.6,
        "figure.dpi": 180,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.025,
    }
)


def wilson(successes: int, total: int, z: float = 1.959963984540054) -> tuple[float, float]:
    p = successes / total
    den = 1 + z * z / total
    centre = (p + z * z / (2 * total)) / den
    radius = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / den
    return centre - radius, centre + radius


def save(fig: plt.Figure, stem: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{stem}.pdf")
    fig.savefig(OUT / f"{stem}.png", dpi=220)
    plt.close(fig)


def plot_cost_curve() -> None:
    ns = np.array([0, 1, 2, 5, 10, 25])
    wins = np.array([1, 109, 100, 107, 116, 114])
    totals = np.array([60, 120, 120, 120, 120, 120])
    rates = wins / totals
    intervals = np.array([wilson(int(k), int(n)) for k, n in zip(wins, totals)])

    fig, ax = plt.subplots(figsize=(3.25, 2.10))
    ax.errorbar(
        ns,
        rates * 100,
        yerr=np.vstack(((rates - intervals[:, 0]) * 100, (intervals[:, 1] - rates) * 100)),
        color=COLORS["blue"],
        marker="o",
        lw=1.8,
        capsize=2.5,
        label="Naive fine-tuning",
    )
    ax.scatter([1], [rates[1] * 100], s=70, facecolor="white", edgecolor=COLORS["orange"], lw=2.0, zorder=4)
    label_offsets = [(0, 5), (0, 8), (0, -13), (0, 5), (0, 5), (0, 5)]
    for x, y, offset in zip(ns, rates * 100, label_offsets):
        ax.annotate(f"{y:.1f}", (x, y), xytext=offset, textcoords="offset points", ha="center", fontsize=6.6)
    ax.set(xticks=ns, xlabel="Target demonstrations, N", ylabel="Target success (%)", ylim=(-5, 108))
    ax.set_title("One demonstration reaches the high-success regime")
    save(fig, "cost_curve")


def plot_retention_curve() -> None:
    ns = np.array([0, 1, 2, 5, 10, 25])
    wins = np.array([24, 37, 21, 0, 0, 0])
    totals = np.array([30, 180, 180, 180, 180, 180])
    rates = wins / totals

    fig, ax = plt.subplots(figsize=(3.25, 2.10))
    ax.plot(ns, rates * 100, color=COLORS["red"], marker="o", lw=1.8)
    ax.fill_between(ns, rates * 100, 0, color=COLORS["red"], alpha=0.08)
    for x, y in zip(ns, rates * 100):
        ax.annotate(f"{y:.1f}", (x, y), xytext=(0, 5), textcoords="offset points", ha="center", fontsize=6.6)
    ax.axhline(80, color=COLORS["gray"], lw=0.9, ls="--", label="Frozen seen policy")
    ax.set(xticks=ns, xlabel="Target demonstrations, N", ylabel="Seen retention (%)", ylim=(-5, 90))
    ax.set_title("Target success hides severe forgetting")
    ax.legend(loc="upper right", frameon=False)
    save(fig, "retention_curve")


def plot_method_frontier() -> None:
    methods = ["Naive", "Target-LoRA", "Replay-LoRA"]
    target = np.array([90.8, 82.5, 55.8])
    retention = np.array([20.6, 10.6, 1.1])
    params = np.array([99.880992, 4.215632, 4.215632])
    colors = [COLORS["blue"], COLORS["green"], COLORS["orange"]]

    fig, ax = plt.subplots(figsize=(3.25, 2.10))
    sizes = 28 + 2.1 * params
    ax.scatter(retention, target, s=sizes, c=colors, alpha=0.9, edgecolor="white", linewidth=0.8)
    offsets = [(5, 3), (5, 3), (5, 3)]
    for name, x, y, off in zip(methods, retention, target, offsets):
        ax.annotate(name, (x, y), xytext=off, textcoords="offset points", fontsize=6.8)
    ax.set(xlabel="Seen retention (%)", ylabel="Target success (%)", xlim=(-2, 27), ylim=(48, 96))
    ax.set_title("N=1 target--retention frontier")
    ax.text(0.02, 0.03, "Marker area $\\propto$ trainable parameters", transform=ax.transAxes, fontsize=6.2, color=COLORS["gray"])
    save(fig, "method_frontier")


def read_metrics(path: Path) -> dict[str, np.ndarray]:
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    return {
        key: np.asarray([float(row[key]) for row in rows], dtype=float)
        for key in ("global_step", "loss", "learning_rate", "grad_norm", "gpu_memory_reserved_mb")
    }


def plot_seen_training() -> None:
    path = RUNS / "seen__expert__libero90__nall__s42__20260822T010019Z__gd4b8fb8" / "metrics.csv"
    data = read_metrics(path)
    step = data["global_step"]
    loss = data["loss"]
    window = 35
    kernel = np.ones(window) / window
    smooth = np.convolve(loss, kernel, mode="valid")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.8, 2.05))
    ax1.plot(step, loss, color=COLORS["blue"], alpha=0.16, lw=0.55, label="logged loss")
    ax1.plot(step[window - 1 :], smooth, color=COLORS["blue"], lw=1.3, label=f"{window}-log moving mean")
    ax1.set(xlabel="Optimizer step", ylabel="Training loss", title="Seen expert: 100k steps")
    ax1.legend(frameon=False)
    ax2.plot(step, data["learning_rate"], color=COLORS["orange"], lw=1.35)
    ax2.set(xlabel="Optimizer step", ylabel="Learning rate", title="Warm-up and cosine schedule")
    save(fig, "seen_training")


def interpolate_traces(paths: list[Path], field: str = "loss", points: int = 160) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    grid = np.linspace(0, 1, points)
    traces = []
    for path in paths:
        data = read_metrics(path)
        x = np.linspace(0, 1, len(data[field]))
        traces.append(np.interp(grid, x, data[field]))
    values = np.asarray(traces)
    return grid, np.median(values, axis=0), np.percentile(values, [25, 75], axis=0)


def plot_baseline_training() -> None:
    groups = {
        1: sorted((RUNS / "target_baseline_n12").glob("*_n01_*/metrics.csv")),
        2: sorted((RUNS / "target_baseline_n12").glob("*_n02_*/metrics.csv")),
        5: sorted((RUNS / "target_baseline").glob("*_n05_*/metrics.csv")),
        10: sorted((RUNS / "target_baseline").glob("*_n10_*/metrics.csv")),
        25: sorted((RUNS / "target_baseline").glob("*_n25_*/metrics.csv")),
    }
    palette = [COLORS["blue"], COLORS["orange"], COLORS["green"], COLORS["red"], COLORS["purple"]]
    fig, ax = plt.subplots(figsize=(6.8, 2.35))
    for (n_demo, paths), color in zip(groups.items(), palette):
        grid, median, spread = interpolate_traces(paths)
        ax.plot(grid * 100, median, color=color, lw=1.25, label=f"N={n_demo}")
        ax.fill_between(grid * 100, spread[0], spread[1], color=color, alpha=0.10)
    ax.set(xlabel="Training progress (%)", ylabel="Loss (median and IQR)", title="Naive target fine-tuning across three tasks and two seeds")
    ax.set_yscale("log")
    ax.legend(ncol=5, frameon=False, loc="upper right")
    save(fig, "baseline_training")


def plot_lora_diagnostics() -> None:
    task_names = [("drawer_middle", "Drawer"), ("bowl_stove", "Bowl"), ("wine_cabinet", "Wine")]
    methods = [
        ("Naive", RUNS / "target_baseline_n12", "*_n01_*/metrics.csv", COLORS["blue"]),
        ("Target-LoRA", RUNS / "task2_n1" / "target_lora", "*_n01_*/metrics.csv", COLORS["green"]),
        ("Replay-LoRA", RUNS / "task2_n1" / "replay_lora", "*_n01_*/metrics.csv", COLORS["orange"]),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(6.8, 2.25), sharex=True)
    for ax, (slug, label) in zip(axes, task_names):
        for method, root, pattern, color in methods:
            paths = sorted(root.glob(f"{slug}_n01_*/metrics.csv"))
            grid, median, spread = interpolate_traces(paths)
            ax.plot(grid * 100, median, color=color, lw=1.25, label=method)
            ax.fill_between(grid * 100, spread[0], spread[1], color=color, alpha=0.10)
        ax.set(title=label, xlabel="Progress (%)")
        ax.set_yscale("log")
    axes[0].set_ylabel("Training loss (log scale)")
    axes[-1].legend(frameon=False, fontsize=6.4, loc="best")
    fig.suptitle("N=1 training diagnostics; bands span the two train seeds", y=1.01, fontsize=8.5)
    save(fig, "lora_diagnostics")


def plot_compute() -> None:
    methods = ["Naive", "Target-LoRA", "Replay-LoRA"]
    wall = [162.8, 244.0, 309.3]
    vram = [7540, 6944, 6944]
    params = [99.880992, 4.215632, 4.215632]
    colors = [COLORS["blue"], COLORS["green"], COLORS["orange"]]
    fig, axes = plt.subplots(1, 3, figsize=(6.8, 2.15))
    for ax, values, title, unit in zip(
        axes,
        (wall, vram, params),
        ("Wall time / cell", "Peak reserved VRAM", "Trainable parameters"),
        ("s", "MiB", "million"),
    ):
        bars = ax.bar(methods, values, color=colors, width=0.68)
        ax.set(title=title, ylabel=unit)
        ax.tick_params(axis="x", rotation=25)
        ax.bar_label(bars, fmt="%.1f", fontsize=6.3, padding=2)
    fig.suptitle("Measured N=1 training cost (six cells per method)", y=1.01, fontsize=8.5)
    save(fig, "compute_cost")


def main() -> None:
    plot_cost_curve()
    plot_retention_curve()
    plot_method_frontier()
    plot_seen_training()
    plot_baseline_training()
    plot_lora_diagnostics()
    plot_compute()


if __name__ == "__main__":
    main()
