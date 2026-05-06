"""Pipeline overview: 6-panel image per working-plan SVG.

Generates one PNG per file in `input/working_plans/` with stages laid out
side-by-side so each phase can be compared at the same coordinates:

    Original SVG          — every parsed primitive in its stroke color
    Inc 4   Boundary       — kept walls after hatching filter, doors/windows
    Inc 5   Closing AABBs  — door/window bbox rectangles
    Inc 9   Polygonize     — polygonize_full output (random colors)
    Inc 10  Wall detect    — merged walls + doors + windows in semantic colors
    Inc 13  Room classify  — locked + threshold rooms with facil_type colors

Output:
    output/inc13_working/<sample>_pipeline.png

Usage:
    .venv/bin/python -m tests.pipeline_overview_checkpoint
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

from src.classifier import (
    TYPE_COLORS,
    classification_summary,
    classify_by_geometry,
    classify_rooms,
    reclassify_kitchen_near_wc,
)
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
from src.separator import _instance_centroids, separate_polygons
from src.svg_parser import parse_svg

logging.basicConfig(level=logging.INFO, format="%(name)s — %(message)s")
logger = logging.getLogger(__name__)

WORKING_DIR = Path("input/working_plans")
OUTPUT_DIR = Path("output/inc13_working")

WINDOW_BLUE = "#2563eb"
DOOR_PINK = "#e03e9b"
WALL_GRAY = "#6b7280"
WALL_EDGE = "#1f2937"
UNCLASSIFIED_FILL = "#ffffff"
UNCLASSIFIED_EDGE = "#9ca3af"


def _parse_rgb(stroke: str | None, fallback: str):
    if not stroke:
        return fallback
    m = re.match(r"rgb\(\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\)", stroke)
    if not m:
        return fallback
    return (int(m.group(1)) / 255, int(m.group(2)) / 255, int(m.group(3)) / 255)


def _compute_bounds(geoms_iter):
    """Return (min_x, min_y, max_x, max_y) over any iterable of shapely geoms."""
    min_x = min_y = float("inf")
    max_x = max_y = float("-inf")
    for g in geoms_iter:
        if g is None or g.is_empty:
            continue
        bx0, by0, bx1, by1 = g.bounds
        if bx0 < min_x:
            min_x = bx0
        if bx1 > max_x:
            max_x = bx1
        if by0 < min_y:
            min_y = by0
        if by1 > max_y:
            max_y = by1
    if min_x == float("inf"):
        return None
    return min_x, min_y, max_x, max_y


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


def _plot_lines(ax, prims, *, fallback, linewidth, alpha):
    for p in prims:
        g = p.geometry
        if isinstance(g, LineString):
            xs, ys = g.xy
            ax.plot(xs, ys, color=_parse_rgb(p.stroke, fallback),
                    linewidth=linewidth, alpha=alpha)


def _draw_panel_original(ax, parse_result):
    """Original SVG: every parsed primitive in its source stroke color."""
    # Walls + curtain walls (boundary group): brown / purple from stroke.
    _plot_lines(ax, parse_result.boundary, fallback="#a75c20",
                linewidth=0.5, alpha=0.85)
    # Doors (DOOR_IDS): pink / magenta family.
    _plot_lines(ax, parse_result.doors, fallback=DOOR_PINK,
                linewidth=0.5, alpha=0.7)
    # Windows: blue regardless of original stroke for visual consistency.
    for p in parse_result.windows:
        if isinstance(p.geometry, LineString):
            xs, ys = p.geometry.xy
            ax.plot(xs, ys, color=WINDOW_BLUE, linewidth=0.7, alpha=0.85)
    # Classification primitives (furniture, fixtures, stairs, etc.) faintly.
    _plot_lines(ax, parse_result.classification, fallback="#888888",
                linewidth=0.3, alpha=0.45)
    n = (len(parse_result.boundary) + len(parse_result.doors)
         + len(parse_result.windows) + len(parse_result.classification))
    ax.set_title(
        f"Original SVG\n{n} primitives parsed",
        fontsize=10,
    )


def _draw_panel_4_boundary(ax, parse_result, kept_boundary, removed_hatch):
    # Walls (kept boundary) in their stroke color.
    _plot_lines(ax, kept_boundary, fallback="#a75c20",
                linewidth=0.5, alpha=0.85)
    # Hatching that the filter removed — drawn faintly to show what was dropped.
    _plot_lines(ax, removed_hatch, fallback="#fbbf24",
                linewidth=0.3, alpha=0.35)
    # Doors and windows from parse_result so it's visible what's NOT a wall.
    _plot_lines(ax, parse_result.doors, fallback=DOOR_PINK,
                linewidth=0.5, alpha=0.6)
    for p in parse_result.windows:
        if isinstance(p.geometry, LineString):
            xs, ys = p.geometry.xy
            ax.plot(xs, ys, color=WINDOW_BLUE, linewidth=0.7, alpha=0.85)
    n_kept = len(kept_boundary)
    n_dropped = len(removed_hatch)
    ax.set_title(
        f"Inc 4 — Boundary\n"
        f"walls kept: {n_kept}  |  hatching removed: {n_dropped}",
        fontsize=10,
    )


def _draw_panel_5_closing(ax, kept_boundary, closing):
    # Light wall context behind the AABBs.
    _plot_lines(ax, kept_boundary, fallback="#d1d5db",
                linewidth=0.3, alpha=0.6)
    # Door bbox rectangles in pink.
    for d in closing.door_polygons:
        if isinstance(d.geometry, Polygon):
            color = _parse_rgb(d.stroke, DOOR_PINK)
            _fill_polygon(ax, d.geometry, facecolor=color, alpha=0.6,
                          edgecolor=color, linewidth=0.6)
    # Window bbox rectangles in blue.
    for w in closing.window_polygons:
        if isinstance(w.geometry, Polygon):
            _fill_polygon(ax, w.geometry, facecolor=WINDOW_BLUE, alpha=0.6,
                          edgecolor=WINDOW_BLUE, linewidth=0.6)
    ax.set_title(
        f"Inc 5 — Door/Window AABBs\n"
        f"doors: {len(closing.door_polygons)}  |  "
        f"windows: {len(closing.window_polygons)}",
        fontsize=10,
    )


def _draw_panel_9_polygonize(ax, poly_result, sep):
    # Map opening polygons to semantic colors so they don't get random fills.
    semantic_color: dict[int, tuple] = {}
    for d in sep.doors:
        if isinstance(d.geometry, Polygon):
            semantic_color[id(d.geometry)] = _parse_rgb(d.stroke, DOOR_PINK)
    for w in sep.windows:
        if isinstance(w.geometry, Polygon):
            semantic_color[id(w.geometry)] = WINDOW_BLUE
    np.random.seed(42)
    for p in poly_result.polygons:
        sem = semantic_color.get(id(p))
        if sem is not None:
            _fill_polygon(ax, p, facecolor=sem, alpha=0.85,
                          edgecolor=sem, linewidth=0.4)
        else:
            _fill_polygon(ax, p, facecolor=np.random.rand(3), alpha=0.45,
                          linewidth=0.3)
    ax.set_title(
        f"Inc 9 — Polygonize\n"
        f"{len(poly_result.polygons)} polygons (doors pink, windows blue)",
        fontsize=10,
    )


def _draw_panel_10_walls(ax, sep):
    for w in sep.merged_walls:
        if isinstance(w, Polygon):
            _fill_polygon(ax, w, facecolor=WALL_GRAY, alpha=0.85,
                          edgecolor=WALL_EDGE, linewidth=0.5)
    for d in sep.doors:
        if isinstance(d.geometry, Polygon):
            color = _parse_rgb(d.stroke, DOOR_PINK)
            _fill_polygon(ax, d.geometry, facecolor=color, alpha=0.85,
                          edgecolor=color, linewidth=0.5)
    for w in sep.windows:
        if isinstance(w.geometry, Polygon):
            _fill_polygon(ax, w.geometry, facecolor=WINDOW_BLUE, alpha=0.85,
                          edgecolor=WINDOW_BLUE, linewidth=0.5)
    ax.set_title(
        f"Inc 10 — Walls\n"
        f"{sep.stats.merged_wall_count} merged | "
        f"{sep.stats.column_count} columns | "
        f"{sep.stats.synth_wall_count} synth",
        fontsize=10,
    )


def _draw_panel_13_classified(ax, sep, viewbox_height,
                              classification_primitives):
    bbox_span = max(viewbox_height, 1e-6)

    def is_svg_exterior(g: Polygon) -> bool:
        b = g.bounds
        return (b[2] - b[0]) >= 0.85 * bbox_span and \
               (b[3] - b[1]) >= 0.85 * bbox_span

    rooms_z_ordered = sorted(
        sep.rooms,
        key=lambda r: (1 if r.facil_type and r.facil_type in TYPE_COLORS else 0),
    )
    for r in rooms_z_ordered:
        if not isinstance(r.geometry, Polygon):
            continue
        if is_svg_exterior(r.geometry) and not r.contained_semantics:
            continue
        if r.facil_type and r.facil_type in TYPE_COLORS:
            color = TYPE_COLORS[r.facil_type]
            _fill_polygon(ax, r.geometry, facecolor=color, alpha=0.65,
                          edgecolor=color, linewidth=0.4)
        else:
            _fill_polygon(ax, r.geometry, facecolor=UNCLASSIFIED_FILL,
                          alpha=0.5, edgecolor=UNCLASSIFIED_EDGE,
                          linewidth=0.4)
    for w in sep.merged_walls:
        if isinstance(w, Polygon):
            xs, ys = w.exterior.xy
            ax.plot(xs, ys, color=WALL_EDGE, linewidth=0.5, alpha=0.85)
    for d in sep.doors:
        if isinstance(d.geometry, Polygon):
            color = _parse_rgb(d.stroke, DOOR_PINK)
            _fill_polygon(ax, d.geometry, facecolor=color, alpha=0.85,
                          edgecolor=color, linewidth=0.4)
    for w in sep.windows:
        if isinstance(w.geometry, Polygon):
            _fill_polygon(ax, w.geometry, facecolor=WINDOW_BLUE, alpha=0.85,
                          edgecolor=WINDOW_BLUE, linewidth=0.4)
    if classification_primitives:
        for sem, pt in _instance_centroids(classification_primitives):
            if pt.is_empty:
                continue
            ax.plot(pt.x, pt.y, marker="o", markersize=3.5,
                    markerfacecolor="black", markeredgecolor="white",
                    markeredgewidth=0.4, linestyle="", alpha=0.9)
    summary = classification_summary(sep.rooms)
    summary_str = ", ".join(f"{k}={v}" for k, v in sorted(summary.items())) \
        or "none"
    ax.set_title(
        f"Inc 13 — Rooms\n{summary_str}",
        fontsize=9,
    )


def plot_pipeline(parse_result, kept_boundary, removed_hatch, closing,
                  poly_result, sep, save_path, sample_name,
                  classification_primitives):
    """Render the 5-panel pipeline overview at consistent bounds."""
    # Use bounds across all displayed geometry so every panel shares axes.
    geom_iter: list = []
    for p in parse_result.boundary:
        geom_iter.append(p.geometry)
    for d in closing.door_polygons:
        geom_iter.append(d.geometry)
    for w in closing.window_polygons:
        geom_iter.append(w.geometry)
    geom_iter.extend(poly_result.polygons)
    bounds = _compute_bounds(geom_iter)
    if bounds is None:
        bounds = (0, 0, parse_result.viewbox_height,
                  parse_result.viewbox_height)
    bx0, by0, bx1, by1 = bounds
    pad_x = max((bx1 - bx0) * 0.025, 1e-3)
    pad_y = max((by1 - by0) * 0.025, 1e-3)
    xlim = (bx0 - pad_x, bx1 + pad_x)
    ylim = (by0 - pad_y, by1 + pad_y)

    fig, axes = plt.subplots(1, 6, figsize=(60, 12))
    ax_orig, ax4, ax5, ax9, ax10, ax13 = axes
    fig.suptitle(sample_name, fontsize=14, y=0.995)

    _draw_panel_original(ax_orig, parse_result)
    _draw_panel_4_boundary(ax4, parse_result, kept_boundary, removed_hatch)
    _draw_panel_5_closing(ax5, kept_boundary, closing)
    _draw_panel_9_polygonize(ax9, poly_result, sep)
    _draw_panel_10_walls(ax10, sep)
    _draw_panel_13_classified(ax13, sep, parse_result.viewbox_height,
                              classification_primitives)

    for ax in axes:
        ax.set_aspect("equal")
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.tick_params(labelsize=7)

    plt.tight_layout(rect=(0, 0, 1, 0.985))
    plt.savefig(save_path, dpi=120)
    plt.close()
    logger.info("Saved %s", save_path)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    door_cfg = DoorClosingConfig(split_l_shaped_windows=True)
    poly_cfg = PolygonizationConfig()

    samples = sorted(WORKING_DIR.glob("*.svg"))
    if not samples:
        raise SystemExit(f"No SVGs found in {WORKING_DIR}")

    for svg_path in samples:
        sample_name = svg_path.stem
        parse_result = parse_svg(svg_path)
        boundary_rect_prims = [
            p for p in parse_result.all_primitives if p.primitive_id >= 1000000
        ]
        filtered = filter_hatching(parse_result.boundary, HatchingFilterConfig())
        kept_boundary = filtered.kept
        removed_hatch = filtered.removed
        wall_lines = [
            p.geometry for p in kept_boundary
            if isinstance(p.geometry, LineString)
        ]
        hatching_lines = [
            p.geometry for p in removed_hatch
            if isinstance(p.geometry, LineString)
        ]
        closing = close_openings(
            parse_result.doors, parse_result.windows, kept_boundary, door_cfg,
        )
        combined = (
            kept_boundary
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
            classification_primitives=parse_result.classification,
        )
        classify_rooms(sep.rooms)
        reclassify_kitchen_near_wc(sep.rooms)
        classify_by_geometry(sep.rooms)

        plot_pipeline(
            parse_result, kept_boundary, removed_hatch, closing,
            poly, sep,
            OUTPUT_DIR / f"{sample_name}_pipeline.png",
            sample_name,
            parse_result.classification,
        )

    print("\nDone. Outputs in", OUTPUT_DIR)


if __name__ == "__main__":
    main()
