"""Cost-curve figures without matplotlib. X ticks stay {0,5,10,25}."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable
from xml.sax.saxutils import escape

from vla_fewshot.reporting.constants import COST_CURVE_N
from vla_fewshot.reporting.tables import cost_curve_points, records_from_long

COLORS = (
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
)


def write_cost_curve_svg(
    long_path: Path,
    output_path: Path,
    *,
    title: str = "Cost curve (macro mean across tasks)",
) -> Path:
    records = records_from_long(long_path)
    points = cost_curve_points(records)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(_svg(points, title=title), encoding="utf-8")
    return output_path


def _svg(
    series: dict[str, list[tuple[int, float, float, float]]],
    *,
    title: str,
) -> str:
    width, height = 720, 420
    left, right, top, bottom = 60, 24, 36, 48
    plot_w = width - left - right
    plot_h = height - top - bottom

    def x_pix(n: int) -> float:
        return left + (n / 25.0) * plot_w

    def y_pix(rate: float) -> float:
        return top + (1.0 - rate) * plot_h

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{width / 2:.1f}" y="22" text-anchor="middle" font-size="16">'
        f"{escape(title)}</text>",
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}" '
        'stroke="#222" stroke-width="1"/>',
        f'<line x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" '
        f'y2="{top + plot_h}" stroke="#222" stroke-width="1"/>',
    ]
    for n in COST_CURVE_N:
        x = x_pix(n)
        parts.append(
            f'<line x1="{x:.1f}" y1="{top + plot_h}" x2="{x:.1f}" '
            f'y2="{top + plot_h + 6}" stroke="#222"/>'
        )
        parts.append(
            f'<text x="{x:.1f}" y="{top + plot_h + 22}" text-anchor="middle" '
            f'font-size="12">{n}</text>'
        )
    parts.append(
        f'<text x="{left + plot_w / 2:.1f}" y="{height - 8}" text-anchor="middle" '
        'font-size="12">N demonstrations</text>'
    )
    parts.append(
        f'<text x="16" y="{top + plot_h / 2:.1f}" text-anchor="middle" font-size="12" '
        f'transform="rotate(-90 16 {top + plot_h / 2:.1f})">success</text>'
    )
    legend_y = top + 8
    for index, (method, points) in enumerate(series.items()):
        color = COLORS[index % len(COLORS)]
        if points:
            polyline = " ".join(f"{x_pix(n):.1f},{y_pix(rate):.1f}" for n, rate, _lo, _hi in points)
            parts.append(
                f'<polyline fill="none" stroke="{color}" stroke-width="2" points="{polyline}"/>'
            )
            for n, rate, low, high in points:
                x = x_pix(n)
                parts.append(
                    f'<line x1="{x:.1f}" y1="{y_pix(low):.1f}" x2="{x:.1f}" '
                    f'y2="{y_pix(high):.1f}" stroke="{color}" stroke-width="1"/>'
                )
                parts.append(
                    f'<circle cx="{x:.1f}" cy="{y_pix(rate):.1f}" r="3.5" fill="{color}"/>'
                )
        parts.append(
            f'<rect x="{left + 8}" y="{legend_y - 8}" width="10" height="10" fill="{color}"/>'
        )
        parts.append(
            f'<text x="{left + 24}" y="{legend_y}" font-size="12">{escape(method)}</text>'
        )
        legend_y += 16
    parts.append("</svg>\n")
    return "\n".join(parts)


def assert_cost_curve_xticks(svg_text: str) -> None:
    for n in COST_CURVE_N:
        needle = f">{n}</text>"
        if needle not in svg_text:
            raise AssertionError(f"cost curve SVG is missing x tick {n}")


def write_language_control_svg(
    points: Iterable[tuple[str, float, float]],
    output_path: Path,
) -> Path:
    """Tiny paired bar figure: (task, correct_rate, wrong_rate)."""

    rows = list(points)
    width = 640
    height = 80 + 48 * max(len(rows), 1)
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<text x="320" y="24" text-anchor="middle" font-size="16">'
        "Language control (correct vs wrong)</text>",
    ]
    for index, (task, correct, wrong) in enumerate(rows):
        y = 50 + index * 48
        parts.append(f'<text x="12" y="{y + 14}" font-size="12">{escape(task)}</text>')
        parts.append(
            f'<rect x="180" y="{y}" width="{correct * 400:.1f}" height="14" fill="#1f77b4"/>'
        )
        parts.append(
            f'<rect x="180" y="{y + 18}" width="{wrong * 400:.1f}" height="14" fill="#d62728"/>'
        )
    parts.append("</svg>\n")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(parts), encoding="utf-8")
    return output_path
