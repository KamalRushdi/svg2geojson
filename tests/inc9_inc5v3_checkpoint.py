"""Inc 9 + Inc 5 v3 checkpoint: full polygonization with multi-leg window splitting.

Same pipeline as inc9_checkpoint.py but inserts the new door/window closing
step (with adaptive multi-leg splitting for L/U/T-shaped windows) before
cleaning and polygonization.

Pipeline:
    parse -> filter hatching -> close openings (with split_l_shaped_windows=True)
          -> clean -> polygonize

Usage:
    .venv/bin/python -m tests.inc9_inc5v3_checkpoint
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import re

import matplotlib.pyplot as plt
import numpy as np
from shapely.geometry import LineString, Polygon


def _parse_rgb(stroke: str | None, fallback: str) -> tuple[float, float, float] | str:
    """Convert 'rgb(r, g, b)' to a matplotlib (r, g, b) tuple in [0, 1]."""
    if not stroke:
        return fallback
    m = re.match(r"rgb\(\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\)", stroke)
    if not m:
        return fallback
    return (int(m.group(1)) / 255, int(m.group(2)) / 255, int(m.group(3)) / 255)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.cleaning import clean_geometry
from src.config import (
    CleaningConfig,
    DoorClosingConfig,
    HatchingFilterConfig,
    PolygonizationConfig,
)
from src.door_closer import close_openings
from src.hatching_filter import filter_hatching
from src.polygonizer import polygonize_lines
from src.svg_parser import parse_svg

logging.basicConfig(level=logging.INFO, format="%(name)s — %(message)s")
logger = logging.getLogger(__name__)

SAMPLES = [
    Path("input/sample/ABLOK_12KAT_parça1.svg"),
    Path("input/sample/HKÜ_HUKUK_3KAT_parça1.svg"),
    Path("input/sample/ARI 8_1.kat_parça1_layered.svg"),
    Path("input/sample/ESKİŞEHİR_ADLİYESİ_1.KAT_parça15.svg"),
    Path("input/sample/GaziUni_2Kat_parça2.svg"),
]
OUTPUT_DIR = Path("output/inc9_inc5v3")


def _compute_bounds(lines):
    max_x = 0.0
    max_y = 0.0
    for ls in lines:
        for c in ls.coords:
            if c[0] > max_x:
                max_x = c[0]
            if c[1] > max_y:
                max_y = c[1]
    return max_x, max_y


def plot_three_panel(parse_result, boundary, noded_lines, poly_result,
                     closing_result, save_path, sample_name,
                     viewbox_height, input_count, boundary_rects=None):
    """3-panel: full building | boundary only | polygonization result."""
    mx, _ = _compute_bounds(noded_lines)
    xlim = mx * 1.05 if mx > 0 else viewbox_height * 1.5
    ylim = viewbox_height

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(36, 14))

    # Panel 1: Everything — all parsed primitives, color-coded by their SVG stroke
    for p in parse_result.boundary:
        if isinstance(p.geometry, LineString):
            xs, ys = p.geometry.xy
            color = _parse_rgb(p.stroke, "#a75c20")
            ax1.plot(xs, ys, color=color, linewidth=0.5, alpha=0.8)
    for p in parse_result.doors:
        if isinstance(p.geometry, LineString):
            xs, ys = p.geometry.xy
            color = _parse_rgb(p.stroke, "#e03e9b")
            ax1.plot(xs, ys, color=color, linewidth=0.5, alpha=0.6)
    for p in parse_result.windows:
        if isinstance(p.geometry, LineString):
            xs, ys = p.geometry.xy
            color = _parse_rgb(p.stroke, "#604ef5")
            ax1.plot(xs, ys, color=color, linewidth=0.5, alpha=0.6)
    for p in parse_result.classification:
        if isinstance(p.geometry, LineString):
            xs, ys = p.geometry.xy
            color = _parse_rgb(p.stroke, "#888888")
            ax1.plot(xs, ys, color=color, linewidth=0.3, alpha=0.4)
    if boundary_rects:
        for rect in boundary_rects:
            xs, ys = rect.xy
            ax1.plot(xs, ys, color="black", linewidth=2.0, linestyle="--")
    ax1.set_title(
        f"{sample_name}\nFull building (all primitives)",
        fontsize=9,
    )
    ax1.set_aspect("equal")
    ax1.set_xlim(0, xlim)
    ax1.set_ylim(0, ylim)

    # Panel 2: Boundary only (after hatching filter)
    for p in boundary:
        if isinstance(p.geometry, LineString):
            xs, ys = p.geometry.xy
            ax2.plot(xs, ys, color="#a75c20", linewidth=0.5, alpha=0.8)
    if boundary_rects:
        for rect in boundary_rects:
            xs, ys = rect.xy
            ax2.plot(xs, ys, color="black", linewidth=2.0, linestyle="--")
    ax2.set_title(
        f"{sample_name}\nBoundary after hatching filter ({len(boundary)})",
        fontsize=9,
    )
    ax2.set_aspect("equal")
    ax2.set_xlim(0, xlim)
    ax2.set_ylim(0, ylim)

    # Panel 3: Polygonization result + door/window overlays
    np.random.seed(42)
    for poly in poly_result.polygons:
        color = np.random.rand(3)
        coords = np.array(poly.exterior.coords)
        patch = plt.Polygon(
            coords, alpha=0.4, facecolor=color, edgecolor="black", linewidth=0.3,
        )
        ax3.add_patch(patch)
    for d in poly_result.dangles:
        xs, ys = d.xy
        ax3.plot(xs, ys, color="#e03e9b", linewidth=1.5, alpha=0.8)
    for ce in poly_result.cut_edges:
        xs, ys = ce.xy
        ax3.plot(xs, ys, color="#ed702d", linewidth=1.0, alpha=0.8)
    for dp in closing_result.door_polygons:
        if isinstance(dp.geometry, Polygon):
            xs, ys = dp.geometry.exterior.xy
            color = _parse_rgb(dp.stroke, "#e03e9b")
            ax3.fill(xs, ys, alpha=0.6, color=color)
            ax3.plot(xs, ys, color=color, linewidth=1.5)
    for wp in closing_result.window_polygons:
        if isinstance(wp.geometry, Polygon):
            xs, ys = wp.geometry.exterior.xy
            color = _parse_rgb(wp.stroke, "#604ef5")
            ax3.fill(xs, ys, alpha=0.6, color=color)
            ax3.plot(xs, ys, color=color, linewidth=1.5)
    if boundary_rects:
        for rect in boundary_rects:
            xs, ys = rect.xy
            ax3.plot(xs, ys, color="black", linewidth=2.0, linestyle="--")

    ax3.plot([], [], color="#e03e9b", linewidth=1.5,
             label=f"Dangles ({poly_result.stats.dangle_count})")
    if poly_result.cut_edges:
        ax3.plot([], [], color="#ed702d", linewidth=1.0,
                 label=f"Cut edges ({poly_result.stats.cut_edge_count})")
    ax3.legend(loc="upper right", fontsize=8)

    ax3.set_title(
        f"{sample_name}\nPolygons ({poly_result.stats.filtered_polygon_count})"
        f" + dangles ({poly_result.stats.dangle_count})",
        fontsize=9,
    )
    ax3.set_aspect("equal")
    ax3.set_xlim(0, xlim)
    ax3.set_ylim(0, ylim)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    logger.info("Saved %s", save_path)


def print_summary(sample_name, stats, input_count, closing_result):
    print(f"\n{'='*60}")
    print(f"{sample_name}")
    print(f"{'='*60}")
    print(f"  Closing: {closing_result.door_count}D / "
          f"{closing_result.window_count}W "
          f"({len(closing_result.door_edges) + len(closing_result.window_edges)} edges)")
    print(f"  Input lines:       {input_count}")
    print(f"  Raw polygons:      {stats.raw_polygon_count}")
    print(f"  After area filter: {stats.filtered_polygon_count}"
          f" ({stats.removed_small_count} removed)")
    print(f"  Dangles:           {stats.dangle_count}"
          f" (total length: {stats.total_dangle_length:.1f})")
    print(f"  Cut edges:         {stats.cut_edge_count}")
    print(f"  Invalid rings:     {stats.invalid_ring_count}")
    if stats.filtered_polygon_count > 0:
        print(f"  Total area:        {stats.total_area:.1f}")
        print(f"  Area range:        [{stats.min_area:.1f}, {stats.max_area:.1f}]")
    if stats.filtered_polygon_count == 0:
        print("  WARNING: no polygons formed")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    door_cfg = DoorClosingConfig(split_l_shaped_windows=True)

    for svg_path in SAMPLES:
        if not svg_path.exists():
            logger.warning("Sample not found: %s", svg_path)
            continue

        sample_name = svg_path.stem

        # Parse
        result = parse_svg(svg_path)
        boundary_rect_prims = [
            p for p in result.all_primitives if p.primitive_id >= 1000000
        ]
        boundary_rects = [p.geometry for p in boundary_rect_prims]

        # Filter hatching
        filtered = filter_hatching(result.boundary, HatchingFilterConfig())
        boundary = filtered.kept

        # Door/window closing (with multi-leg splitting)
        closing = close_openings(result.doors, result.windows, boundary, door_cfg)

        # Combine + clean + polygonize.
        # Include the SVG viewBox boundary rectangles so polygonize_full
        # can close polygons against them (otherwise wall segments at the
        # SVG edge create dangles that prevent room formation).
        combined = (
            boundary
            + closing.door_edges
            + closing.window_edges
            + boundary_rect_prims
        )
        cleaning = clean_geometry(combined, CleaningConfig())
        poly = polygonize_lines(cleaning.lines, PolygonizationConfig())

        input_count = len(cleaning.lines)
        print_summary(sample_name, poly.stats, input_count, closing)

        plot_three_panel(
            result, boundary, cleaning.lines, poly, closing,
            OUTPUT_DIR / f"{sample_name}_polygonize.png",
            sample_name, result.viewbox_height, input_count,
            boundary_rects=boundary_rects,
        )

    print("\nDone. Outputs in", OUTPUT_DIR)


if __name__ == "__main__":
    main()
