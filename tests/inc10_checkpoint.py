"""Inc 10 checkpoint: wall detection and merging.

Pipeline:
    parse -> filter hatching -> close openings -> clean -> polygonize -> SEPARATE

Generates a 3-panel comparison plot per sample:
  Panel 1 - Original SVG primitives (walls in stroke colors, windows in blue)
  Panel 2 - Polygonization result (random colors per polygon)
  Panel 3 - Walls only (merged walls in gray)

Usage:
    .venv/bin/python -m tests.inc10_checkpoint
"""

from __future__ import annotations

import logging
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from shapely.geometry import LineString, Polygon

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
from src.separator import SeparatorResult, separate_polygons
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
OUTPUT_DIR = Path("output/inc10")

WINDOW_BLUE = "#2563eb"
WALL_GRAY = "#6b7280"
WALL_EDGE = "#1f2937"


def _parse_rgb(stroke: str | None, fallback: str):
    if not stroke:
        return fallback
    m = re.match(r"rgb\(\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\)", stroke)
    if not m:
        return fallback
    return (int(m.group(1)) / 255, int(m.group(2)) / 255, int(m.group(3)) / 255)


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


def _fill_polygon(ax, poly: Polygon, *, facecolor, edgecolor="black",
                  linewidth=0.3, alpha=0.5):
    coords = np.array(poly.exterior.coords)
    ax.add_patch(plt.Polygon(
        coords, alpha=alpha, facecolor=facecolor,
        edgecolor=edgecolor, linewidth=linewidth,
    ))
    for hole in poly.interiors:
        hole_coords = np.array(hole.coords)
        ax.add_patch(plt.Polygon(
            hole_coords, alpha=1.0, facecolor="white",
            edgecolor=edgecolor, linewidth=linewidth,
        ))


def plot_three_panel(
    parse_result, noded_lines, poly_result, sep: SeparatorResult,
    save_path, sample_name, viewbox_height, boundary_rects=None,
):
    mx, _ = _compute_bounds(noded_lines)
    xlim = mx * 1.05 if mx > 0 else viewbox_height * 1.5
    ylim = viewbox_height

    fig, axes = plt.subplots(1, 3, figsize=(30, 12))
    ax1, ax2, ax3 = axes

    # ---- Panel 1: original SVG primitives ----
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
    # Windows always blue regardless of original stroke.
    for p in parse_result.windows:
        if isinstance(p.geometry, LineString):
            xs, ys = p.geometry.xy
            ax1.plot(xs, ys, color=WINDOW_BLUE, linewidth=0.7, alpha=0.85)
    for p in parse_result.classification:
        if isinstance(p.geometry, LineString):
            xs, ys = p.geometry.xy
            color = _parse_rgb(p.stroke, "#888888")
            ax1.plot(xs, ys, color=color, linewidth=0.3, alpha=0.4)
    ax1.set_title(f"{sample_name}\nOriginal SVG (windows in blue)", fontsize=10)

    # Build a canonical color map for door/window polygons. Polygon objects
    # in poly_result.polygons are the same instances the separator wraps in
    # sep.doors / sep.windows, so id()-based lookup is reliable. Wherever a
    # door/window polygon gets drawn (Panel 2 raw, Panel 3 classified), it
    # uses its semantic color — random/role colors only fill the rest.
    semantic_color: dict[int, tuple] = {}
    for d in sep.doors:
        if isinstance(d.geometry, Polygon):
            semantic_color[id(d.geometry)] = _parse_rgb(d.stroke, "#e03e9b")
    for w in sep.windows:
        if isinstance(w.geometry, Polygon):
            semantic_color[id(w.geometry)] = WINDOW_BLUE

    # ---- Panel 2: polygonization output (semantic colors for openings) ----
    np.random.seed(42)
    for poly in poly_result.polygons:
        sem = semantic_color.get(id(poly))
        if sem is not None:
            _fill_polygon(ax2, poly, facecolor=sem, alpha=0.85,
                          edgecolor=sem, linewidth=0.6)
        else:
            _fill_polygon(ax2, poly, facecolor=np.random.rand(3), alpha=0.45)
    ax2.set_title(
        f"{sample_name}\nPolygonized ({len(poly_result.polygons)} polygons; "
        f"doors pink, windows blue)",
        fontsize=10,
    )

    # ---- Panel 3: walls (with columns folded in) + door/window overlay ----
    for w in sep.merged_walls:
        _fill_polygon(ax3, w, facecolor=WALL_GRAY, alpha=0.85,
                      edgecolor=WALL_EDGE, linewidth=0.6)
    for d in sep.doors:
        if isinstance(d.geometry, Polygon):
            color = semantic_color[id(d.geometry)]
            _fill_polygon(ax3, d.geometry, facecolor=color, alpha=0.85,
                          edgecolor=color, linewidth=0.6)
    for w in sep.windows:
        if isinstance(w.geometry, Polygon):
            color = semantic_color[id(w.geometry)]
            _fill_polygon(ax3, w.geometry, facecolor=color, alpha=0.85,
                          edgecolor=color, linewidth=0.6)
    ax3.set_title(
        f"{sample_name}\nWalls ({sep.stats.merged_wall_count} merged "
        f"from {sep.stats.wall_count}, incl. {sep.stats.column_count} columns "
        f"+ {sep.stats.synth_wall_count} synthesized) "
        f"+ doors ({sep.stats.door_count}) + windows ({sep.stats.window_count})",
        fontsize=10,
    )

    for ax in axes:
        ax.set_aspect("equal")
        ax.set_xlim(0, xlim)
        ax.set_ylim(0, ylim)
        if boundary_rects:
            for rect in boundary_rects:
                xs, ys = rect.xy
                ax.plot(xs, ys, color="black", linewidth=1.0,
                        linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    logger.info("Saved %s", save_path)


def print_summary(sample_name, sep: SeparatorResult, raw_count: int):
    print(f"\n{'='*60}")
    print(f"{sample_name}")
    print(f"{'='*60}")
    print(f"  Raw polygons:           {raw_count}")
    print(f"  Threshold method:       {sep.stats.threshold_method}")
    print(f"  Thickness threshold:    {sep.stats.thickness_threshold:.3f}")
    print(f"  Largest log10-gap:      {sep.stats.largest_gap_size:.3f}")
    print(f"  Thickness percentiles:  "
          f"p10={sep.stats.thickness_p10:.2f}  "
          f"p50={sep.stats.thickness_p50:.2f}  "
          f"p90={sep.stats.thickness_p90:.2f}")
    print(f"  Walls (raw):            {sep.stats.wall_count}"
          f"  (small/dropped: {sep.stats.small_wall_count}, "
          f"columns folded in: {sep.stats.column_count})")
    print(f"  Walls (merged):         {sep.stats.merged_wall_count}")
    print(f"  Walls (synthesized):    {sep.stats.synth_wall_count}"
          f"  (single-line walls buffered)")
    print(f"  Hatched polygons:       {sep.stats.hatched_wall_count}"
          f"  (calibrated threshold: {sep.stats.hatching_calibrated})")
    print(f"  Doors:                  {sep.stats.door_count}")
    print(f"  Windows:                {sep.stats.window_count}")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    door_cfg = DoorClosingConfig(split_l_shaped_windows=True)
    poly_cfg = PolygonizationConfig()

    for svg_path in SAMPLES:
        if not svg_path.exists():
            logger.warning("Sample not found: %s", svg_path)
            continue

        sample_name = svg_path.stem

        result = parse_svg(svg_path)
        boundary_rect_prims = [
            p for p in result.all_primitives if p.primitive_id >= 1000000
        ]
        boundary_rects = [p.geometry for p in boundary_rect_prims]

        filtered = filter_hatching(result.boundary, HatchingFilterConfig())
        boundary = filtered.kept
        # Capture wall LineStrings BEFORE they get mixed with door/window
        # edges; Stage 5.5 needs original wall provenance to synthesize
        # single-line walls.
        wall_lines = [
            p.geometry for p in boundary
            if isinstance(p.geometry, LineString)
        ]
        # Hatching strokes that the filter removed — used by the separator
        # as ground truth to calibrate the wall/room thickness threshold.
        hatching_lines = [
            p.geometry for p in filtered.removed
            if isinstance(p.geometry, LineString)
        ]

        # Add door/window edges so wall loops actually close into rooms; the
        # separator IoU-matches the resulting AABB polygons against
        # closing.door_polygons / closing.window_polygons so they're excluded
        # from the wall classification.
        closing = close_openings(result.doors, result.windows, boundary, door_cfg)
        combined = (
            boundary
            + closing.door_edges
            + closing.window_edges
            + boundary_rect_prims
        )
        cleaning = clean_geometry(combined, CleaningConfig())
        poly = polygonize_lines(cleaning.lines, poly_cfg)

        sep = separate_polygons(
            poly.polygons, poly_cfg,
            door_polygons=closing.door_polygons,
            window_polygons=closing.window_polygons,
            wall_lines=wall_lines,
            hatching_lines=hatching_lines,
        )

        print_summary(sample_name, sep, len(poly.polygons))

        plot_three_panel(
            result, cleaning.lines, poly, sep,
            OUTPUT_DIR / f"{sample_name}_separated.png",
            sample_name, result.viewbox_height,
            boundary_rects=boundary_rects,
        )

    print("\nDone. Outputs in", OUTPUT_DIR)


if __name__ == "__main__":
    main()
