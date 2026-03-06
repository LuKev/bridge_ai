"""Lightweight SVG plotting helpers for experiment metrics."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence


def write_accuracy_svg(
    history: Sequence[dict],
    path: str | Path,
    *,
    bid_key: str = "holdout_bid_accuracy",
    belief_key: str = "holdout_belief_accuracy",
    play_belief_key: str | None = "holdout_play_belief_accuracy",
) -> str:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)

    points = [
        row
        for row in history
        if row.get("epoch") is not None and row.get(bid_key) is not None and row.get(belief_key) is not None
    ]
    if not points:
        destination.write_text("<svg xmlns='http://www.w3.org/2000/svg' width='800' height='400'></svg>", encoding="utf-8")
        return str(destination)

    width = 800
    height = 420
    margin_left = 70
    margin_right = 30
    margin_top = 30
    margin_bottom = 55
    plot_width = width - margin_left - margin_right
    plot_height = height - margin_top - margin_bottom

    xs = [float(row["epoch"]) for row in points]
    x_min = min(xs)
    x_max = max(xs) if max(xs) > min(xs) else min(xs) + 1.0

    def _x(value: float) -> float:
        return margin_left + ((value - x_min) / (x_max - x_min)) * plot_width

    def _y(value: float) -> float:
        return margin_top + (1.0 - value) * plot_height

    def _polyline(metric_key: str) -> str:
        coords = []
        for row in points:
            raw = row.get(metric_key)
            if raw is None:
                continue
            value = float(raw)
            coords.append(f"{_x(float(row['epoch'])):.2f},{_y(value):.2f}")
        return " ".join(coords)

    ticks = [0.0, 0.25, 0.5, 0.75, 1.0]
    x_ticks = sorted(set(int(row["epoch"]) for row in points))

    svg = [
        f"<svg xmlns='http://www.w3.org/2000/svg' width='{width}' height='{height}' viewBox='0 0 {width} {height}'>",
        "<style>",
        "text { font-family: Helvetica, Arial, sans-serif; fill: #222; }",
        ".grid { stroke: #d8d8d8; stroke-width: 1; }",
        ".axis { stroke: #444; stroke-width: 1.2; }",
        ".bid { fill: none; stroke: #1565c0; stroke-width: 3; }",
        ".belief { fill: none; stroke: #ef6c00; stroke-width: 3; }",
        ".playbelief { fill: none; stroke: #2e7d32; stroke-width: 3; }",
        "</style>",
        f"<rect x='0' y='0' width='{width}' height='{height}' fill='#fffdf8' />",
        f"<text x='{margin_left}' y='20' font-size='18' font-weight='700'>Holdout Accuracy Over Training</text>",
    ]

    for tick in ticks:
        y = _y(tick)
        svg.append(f"<line class='grid' x1='{margin_left}' y1='{y:.2f}' x2='{width - margin_right}' y2='{y:.2f}' />")
        svg.append(f"<text x='{margin_left - 10}' y='{y + 4:.2f}' font-size='12' text-anchor='end'>{tick:.2f}</text>")

    for tick in x_ticks:
        x = _x(float(tick))
        svg.append(f"<line class='grid' x1='{x:.2f}' y1='{margin_top}' x2='{x:.2f}' y2='{height - margin_bottom}' />")
        svg.append(f"<text x='{x:.2f}' y='{height - margin_bottom + 22}' font-size='12' text-anchor='middle'>{tick}</text>")

    svg.append(
        f"<line class='axis' x1='{margin_left}' y1='{height - margin_bottom}' "
        f"x2='{width - margin_right}' y2='{height - margin_bottom}' />"
    )
    svg.append(
        f"<line class='axis' x1='{margin_left}' y1='{margin_top}' "
        f"x2='{margin_left}' y2='{height - margin_bottom}' />"
    )

    svg.append(f"<polyline class='bid' points='{_polyline(bid_key)}' />")
    svg.append(f"<polyline class='belief' points='{_polyline(belief_key)}' />")
    if play_belief_key and any(row.get(play_belief_key) is not None for row in points):
        svg.append(f"<polyline class='playbelief' points='{_polyline(play_belief_key)}' />")

    legend_x = width - 220
    legend_y = 26
    svg.append(f"<line class='bid' x1='{legend_x}' y1='{legend_y}' x2='{legend_x + 28}' y2='{legend_y}' />")
    svg.append(f"<text x='{legend_x + 36}' y='{legend_y + 4}' font-size='12'>Bid accuracy</text>")
    svg.append(f"<line class='belief' x1='{legend_x}' y1='{legend_y + 22}' x2='{legend_x + 28}' y2='{legend_y + 22}' />")
    svg.append(f"<text x='{legend_x + 36}' y='{legend_y + 26}' font-size='12'>Belief accuracy</text>")
    if play_belief_key and any(row.get(play_belief_key) is not None for row in points):
        svg.append(f"<line class='playbelief' x1='{legend_x}' y1='{legend_y + 44}' x2='{legend_x + 28}' y2='{legend_y + 44}' />")
        svg.append(f"<text x='{legend_x + 36}' y='{legend_y + 48}' font-size='12'>Play belief accuracy</text>")

    svg.append(f"<text x='{width / 2:.2f}' y='{height - 12}' font-size='12' text-anchor='middle'>Epoch</text>")
    svg.append(
        f"<text x='18' y='{height / 2:.2f}' font-size='12' text-anchor='middle' "
        f"transform='rotate(-90 18,{height / 2:.2f})'>Accuracy</text>"
    )
    svg.append("</svg>")
    destination.write_text("\n".join(svg), encoding="utf-8")
    return str(destination)
