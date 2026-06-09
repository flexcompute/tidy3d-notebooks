#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, Rectangle
from matplotlib.patches import Polygon as MplPolygon


def make_mask() -> np.ndarray:
    return np.array(
        [
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 1, 1, 1, 0, 0, 0, 0],
            [0, 1, 1, 1, 1, 1, 1, 0, 0, 0],
            [0, 1, 1, 1, 1, 1, 1, 1, 0, 0],
            [0, 1, 1, 1, 1, 1, 1, 1, 1, 0],
            [0, 1, 1, 1, 1, 1, 1, 1, 0, 0],
            [0, 1, 1, 1, 1, 1, 1, 0, 0, 0],
            [0, 0, 1, 1, 1, 1, 0, 0, 0, 0],
            [0, 0, 0, 1, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        ],
        dtype=int,
    )


def boundary_polygon(mask: np.ndarray) -> np.ndarray:
    nrows, ncols = mask.shape
    edges: set[tuple[tuple[float, float], tuple[float, float]]] = set()

    def add_edge(p0: tuple[float, float], p1: tuple[float, float]) -> None:
        edge = (p0, p1) if p0 <= p1 else (p1, p0)
        edges.add(edge)

    for row in range(nrows):
        for col in range(ncols):
            if not mask[row, col]:
                continue

            x0 = float(col)
            x1 = float(col + 1)
            y0 = float(nrows - row - 1)
            y1 = float(nrows - row)

            if row == 0 or not mask[row - 1, col]:
                add_edge((x0, y1), (x1, y1))
            if row == nrows - 1 or not mask[row + 1, col]:
                add_edge((x0, y0), (x1, y0))
            if col == 0 or not mask[row, col - 1]:
                add_edge((x0, y0), (x0, y1))
            if col == ncols - 1 or not mask[row, col + 1]:
                add_edge((x1, y0), (x1, y1))

    adjacency: dict[tuple[float, float], list[tuple[float, float]]] = {}
    for p0, p1 in edges:
        adjacency.setdefault(p0, []).append(p1)
        adjacency.setdefault(p1, []).append(p0)

    start = min(adjacency, key=lambda p: (p[1], p[0]))
    candidates = adjacency[start]
    next_point = min(candidates, key=lambda p: (abs(p[1] - start[1]) > 0, p[0], p[1]))

    polygon = [start, next_point]
    previous = start
    current = next_point

    while True:
        neighbors = adjacency[current]
        following = neighbors[0] if neighbors[0] != previous else neighbors[1]
        if following == start:
            break
        polygon.append(following)
        previous, current = current, following

    return np.asarray(polygon, dtype=float)


def laplacian_smooth(points: np.ndarray, weight: float = 0.2, steps: int = 10) -> np.ndarray:
    smoothed = points.copy()
    for _ in range(steps):
        prev_pts = np.roll(smoothed, 1, axis=0)
        next_pts = np.roll(smoothed, -1, axis=0)
        smoothed = (1.0 - weight) * smoothed + 0.5 * weight * (prev_pts + next_pts)
    return smoothed


def perturb_polygon(points: np.ndarray, scale: float = 3.0) -> np.ndarray:
    prev_pts = np.roll(points, 1, axis=0)
    next_pts = np.roll(points, -1, axis=0)
    tangents = next_pts - prev_pts
    tangents /= np.linalg.norm(tangents, axis=1, keepdims=True)
    normals = np.column_stack((-tangents[:, 1], tangents[:, 0]))

    phase = np.linspace(0.0, 2.0 * np.pi, len(points), endpoint=False)
    raw = 0.18 * np.sin(2.0 * phase + 0.4) + 0.08 * np.sin(5.0 * phase - 0.7)
    kernel = np.array([0.2, 0.6, 0.2])
    smooth_noise = np.convolve(np.r_[raw[-1], raw, raw[0]], kernel, mode="same")[1:-1]
    return points + scale * smooth_noise[:, None] * normals


def style_panel(ax: plt.Axes, title: str) -> None:
    ax.set_aspect("equal")
    ax.set_xlim(0.0, 10.0)
    ax.set_ylim(0.0, 10.0)
    ax.axis("off")
    ax.text(0.5, 1.01, title, transform=ax.transAxes, ha="center", va="bottom", fontsize=12)


def draw_pixel(ax: plt.Axes, mask: np.ndarray) -> None:
    nrows, ncols = mask.shape
    for row in range(nrows):
        for col in range(ncols):
            if not mask[row, col]:
                continue
            rect = Rectangle((col, nrows - row - 1), 1, 1, facecolor="black", edgecolor="none")
            ax.add_patch(rect)


def draw_polygon(
    ax: plt.Axes, points: np.ndarray, color: str = "black", alpha: float = 1.0
) -> None:
    patch = MplPolygon(points, closed=True, fill=False, edgecolor=color, linewidth=2.4, alpha=alpha)
    ax.add_patch(patch)


def draw_markers(
    ax: plt.Axes, points: np.ndarray, color: str = "black", alpha: float = 1.0
) -> None:
    ax.plot(
        points[:, 0],
        points[:, 1],
        linestyle="none",
        marker="o",
        markersize=3.6,
        markerfacecolor="white",
        markeredgecolor=color,
        markeredgewidth=1.1,
        alpha=alpha,
    )


def add_panel_arrow(
    fig: plt.Figure,
    start: tuple[float, float],
    end: tuple[float, float],
    label: str,
    *,
    label_offset: tuple[float, float] = (0.0, 0.0),
    label_ha: str = "center",
    label_va: str = "center",
) -> None:
    arrow = FancyArrowPatch(
        start,
        end,
        transform=fig.transFigure,
        arrowstyle="-|>",
        mutation_scale=12,
        linewidth=1.8,
        color="0.25",
    )
    fig.add_artist(arrow)
    label_pos = (
        (start[0] + end[0]) * 0.5 + label_offset[0],
        (start[1] + end[1]) * 0.5 + label_offset[1],
    )
    fig.text(label_pos[0], label_pos[1], label, ha=label_ha, va=label_va, fontsize=10, color="0.25")


def build_figure(output_path: Path) -> None:
    mask = make_mask()
    polygon = boundary_polygon(mask)
    smoothed = laplacian_smooth(polygon, weight=0.22, steps=14)
    fine_tuned = perturb_polygon(smoothed, scale=3.0)

    fig = plt.figure(figsize=(8.4, 6.4), facecolor="white")
    gs = fig.add_gridspec(2, 2, wspace=0.12, hspace=0.34)

    ax_pixel = fig.add_subplot(gs[0, 0])
    ax_fine = fig.add_subplot(gs[0, 1])
    ax_polygon = fig.add_subplot(gs[1, 0])
    ax_smoothed = fig.add_subplot(gs[1, 1])

    style_panel(ax_pixel, "pixel design")
    style_panel(ax_polygon, "polygon")
    style_panel(ax_smoothed, "smoothed")
    style_panel(ax_fine, "fine-tuned design")

    draw_pixel(ax_pixel, mask)

    draw_polygon(ax_polygon, polygon)
    draw_markers(ax_polygon, polygon)

    draw_polygon(ax_smoothed, smoothed)
    draw_markers(ax_smoothed, smoothed)

    draw_polygon(ax_fine, smoothed, color="0.65", alpha=0.65)
    draw_markers(ax_fine, smoothed, color="0.65", alpha=0.65)
    draw_polygon(ax_fine, fine_tuned)
    draw_markers(ax_fine, fine_tuned)
    for idx in range(0, len(smoothed), 2):
        ax_fine.annotate(
            "",
            xy=fine_tuned[idx],
            xytext=smoothed[idx],
            arrowprops=dict(arrowstyle="->", color="0.35", lw=0.9, shrinkA=1, shrinkB=1),
        )

    pixel_box = ax_pixel.get_position()
    polygon_box = ax_polygon.get_position()
    smooth_box = ax_smoothed.get_position()
    fine_box = ax_fine.get_position()

    left_arrow_x = pixel_box.x0 + pixel_box.width * 0.5
    right_arrow_x = fine_box.x0 + fine_box.width * 0.5
    bottom_arrow_left = polygon_box.x1 + 0.004
    bottom_arrow_right = smooth_box.x0 - 0.004
    bottom_arrow_y = polygon_box.y0 + polygon_box.height * 0.47

    left_top = pixel_box.y0 - 0.006
    left_bottom_full = polygon_box.y1 + 0.006
    left_bottom = left_top - 0.8 * (left_top - left_bottom_full)
    left_shift = 0.1 * (left_top - left_bottom)
    left_top += left_shift
    left_bottom += left_shift

    add_panel_arrow(
        fig,
        (left_arrow_x, left_top),
        (left_arrow_x, left_bottom),
        "convert",
        label_offset=(0.010, 0.0),
        label_ha="left",
    )
    add_panel_arrow(
        fig,
        (bottom_arrow_left, bottom_arrow_y),
        (bottom_arrow_right, bottom_arrow_y),
        "smooth",
        label_offset=(0.0, 0.024),
        label_va="bottom",
    )
    right_bottom_full = smooth_box.y1 + 0.006
    right_top = fine_box.y0 - 0.006
    right_bottom = right_bottom_full + 0.2 * (right_top - right_bottom_full)
    right_shift = 0.1 * (right_top - right_bottom)
    right_top += right_shift
    right_bottom += right_shift

    add_panel_arrow(
        fig,
        (right_arrow_x, right_bottom),
        (right_arrow_x, right_top),
        "optimize",
        label_offset=(-0.010, 0.0),
        label_ha="right",
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate the Autograd32 thumbnail image.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("img/adjoint_32.png"),
        help="Output image path.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    build_figure(args.output)
    print(args.output)


if __name__ == "__main__":
    main()
