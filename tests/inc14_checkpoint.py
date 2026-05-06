"""Inc 5 resize checkpoint: shrink arc-doors to fit inside their walls.

Pipeline:
    parse -> filter hatching -> close openings -> clean -> polygonize
        -> separator (Inc 10) -> classify (Inc 13/14) -> RESIZE doors

Generates a 3-panel comparison plot per sample:
  Panel 1 - Pre-resize: classified rooms + walls + door AABBs as they came
            out of polygonize. Shows the over-large arc-door rectangles
            eating into adjacent rooms.
  Panel 2 - Post-resize: same scene with door rectangles shrunk to
            target_thickness on the perpendicular-to-wall axis. Slivers
            handed to neighbouring rooms.
  Panel 3 - Hinges + ray-cast directions: the (cx, cy) the resizer used,
            with cyan dots and crosshairs showing the +x/-x/+y/-y rays.

Usage:
    .venv/bin/python -m tests.inc5_resize_checkpoint
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
from src.door_resizer import resize_arc_doors
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
OUTPUT_DIR = Path("output/inc5_resize")

WINDOW_BLUE = "#2563eb"
WALL_GRAY = "#6b7280"
WALL_EDGE = "#1f2937"
HINGE_COLOR = "#06b6d4"     # cyan dot for the arc center
RAY_COLOR = "#f43f5e"       # rose ray segments


def _parse_rgb(stroke: str | None, fallback: str):
    if not stroke:
        return fallback
    m = re.match(r"rgb\(\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\)", stroke)
    if not m:
        return fallback
    return (int(m.group(1)) / 255, int(m.group(2)) / 255, int(m.group(3)) / 255)


def _fill_polygon(ax, poly, *, facecolor, edgecolor="black",
                  linewidth=0.3, alpha=0.5):
    if not isinstance(poly, Polygon):
        return
    coords = np.array(poly.exterior.coords)
    ax.add_patch(plt.Polygon(
        coords, alpha=alpha, facecolor=facecolor,
        edgecolor=edgecolor, linewidth=linewidth,
    ))


def _draw_rooms_and_walls(ax, sep: SeparatorResult, doors):
    """Common base layer: classified rooms + walls (outline only) +
    windows (semantic colors). Doors are drawn separately so we can
    swap pre/post-resize geometry."""
    bbox_span = max(sep.outline.bounds[2], sep.outline.bounds[3]) \
        if sep.outline else 0.0
    for r in sorted(sep.rooms, key=lambda r: 1 if (r.facil_type and r.facil_type in TYPE_COLORS) else 0):
        if not isinstance(r.geometry, Polygon):
            continue
        # Skip the SVG-exterior leftover (canvas-spanning, no semantics).
        b = r.geometry.bounds
        if (
            (b[2] - b[0]) >= 0.85 * bbox_span
            and (b[3] - b[1]) >= 0.85 * bbox_span
            and not r.contained_semantics
        ):
            continue
        if r.facil_type and r.facil_type in TYPE_COLORS:
            color = TYPE_COLORS[r.facil_type]
            _fill_polygon(ax, r.geometry, facecolor=color, alpha=0.65,
                          edgecolor=color, linewidth=0.4)
        else:
            _fill_polygon(ax, r.geometry, facecolor="#ffffff", alpha=0.5,
                          edgecolor="#9ca3af", linewidth=0.4)
    for w in sep.merged_walls:
        if isinstance(w, Polygon):
            xs, ys = w.exterior.xy
            ax.plot(xs, ys, color=WALL_EDGE, linewidth=0.7, alpha=0.85)
    for d in doors:
        if isinstance(d.geometry, Polygon):
            color = _parse_rgb(d.stroke, "#e03e9b")
            _fill_polygon(ax, d.geometry, facecolor=color, alpha=0.85,
                          edgecolor=color, linewidth=0.6)
    for w in sep.windows:
        if isinstance(w.geometry, Polygon):
            _fill_polygon(ax, w.geometry, facecolor=WINDOW_BLUE, alpha=0.85,
                          edgecolor=WINDOW_BLUE, linewidth=0.6)


def plot_three_panel(
    sep_pre: SeparatorResult,
    doors_post,
    rooms_post,
    save_path,
    sample_name,
    viewbox_height,
    target_thickness,
):
    fig, axes = plt.subplots(1, 3, figsize=(30, 11))
    ax1, ax2, ax3 = axes
    xlim = viewbox_height * 1.05
    ylim = viewbox_height

    # Panel 1: pre-resize
    _draw_rooms_and_walls(ax1, sep_pre, sep_pre.doors)
    ax1.set_title(f"{sample_name}\nPre-resize — door AABBs as polygonize emitted them",
                  fontsize=10)

    # Panel 2: post-resize. Need to splice in the new rooms list too.
    sep_post = SeparatorResult(
        rooms=rooms_post,
        walls=sep_pre.walls,
        merged_walls=sep_pre.merged_walls,
        wall_components=sep_pre.wall_components,
        doors=doors_post,
        windows=sep_pre.windows,
        outline=sep_pre.outline,
        stats=sep_pre.stats,
    )
    _draw_rooms_and_walls(ax2, sep_post, doors_post)
    n_arc = sum(1 for d in sep_pre.doors if d.arc_hinge is not None)
    n_resized = sum(
        1 for pre, post in zip(sep_pre.doors, doors_post)
        if pre.geometry is not post.geometry
    )
    ax2.set_title(
        f"{sample_name}\nPost-resize — arc-doors shrunk to "
        f"{target_thickness:.2f} on perpendicular axis "
        f"({n_resized}/{n_arc} resized)",
        fontsize=10,
    )

    # Panel 3: hinges + ray crosshairs (over the post-resize layout).
    _draw_rooms_and_walls(ax3, sep_post, doors_post)
    ray_len = 0.06 * viewbox_height
    for d in sep_pre.doors:
        if d.arc_hinge is None:
            continue
        cx, cy = d.arc_hinge
        ax3.plot([cx - ray_len, cx + ray_len], [cy, cy],
                 color=RAY_COLOR, linewidth=0.7, alpha=0.7)
        ax3.plot([cx, cx], [cy - ray_len, cy + ray_len],
                 color=RAY_COLOR, linewidth=0.7, alpha=0.7)
        ax3.plot(cx, cy, marker="o", markersize=5,
                 markerfacecolor=HINGE_COLOR,
                 markeredgecolor="black", markeredgewidth=0.5)
    ax3.set_title(f"{sample_name}\nHinges (cyan) + ray-cast directions (rose)",
                  fontsize=10)

    for ax in axes:
        ax.set_aspect("equal")
        ax.set_xlim(0, xlim)
        ax.set_ylim(0, ylim)

    plt.tight_layout()
    plt.savefig(save_path, dpi=140)
    plt.close()
    logger.info("Saved %s", save_path)


def print_summary(sample_name, sep, doors_pre, resize_result, target_thickness):
    n_arc = sum(1 for d in doors_pre if d.arc_hinge is not None)
    summary = classification_summary(sep.rooms)
    print(f"\n{'='*60}")
    print(f"{sample_name}")
    print(f"{'='*60}")
    print(f"  Doors total / with arc hinge:   "
          f"{len(doors_pre)} / {n_arc}")
    print(f"  Resize target thickness:        {target_thickness:.3f}")
    print(f"  Resized:                        "
          f"{resize_result.stats.resized}")
    print(f"  Skipped (no clear axis):        "
          f"{resize_result.stats.skipped_no_axis}")
    print(f"  Skipped (not rectangle):        "
          f"{resize_result.stats.skipped_not_rect}")
    print(f"  Freed slivers handed to rooms:  "
          f"{resize_result.stats.freed_slivers}")
    print(f"  Rooms that grew:                "
          f"{resize_result.stats.rooms_grew}")
    print(f"  Pre-resize room breakdown: "
          f"{', '.join(f'{k}={v}' for k, v in sorted(summary.items()))}")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    door_cfg = DoorClosingConfig(split_l_shaped_windows=True)
    poly_cfg = PolygonizationConfig()
    target_thickness = poly_cfg.arc_door_target_thickness

    for svg_path in SAMPLES:
        if not svg_path.exists():
            logger.warning("Sample not found: %s", svg_path)
            continue
        sample_name = svg_path.stem

        result = parse_svg(svg_path)
        boundary_rect_prims = [
            p for p in result.all_primitives if p.primitive_id >= 1000000
        ]
        filtered = filter_hatching(result.boundary, HatchingFilterConfig())
        boundary = filtered.kept
        wall_lines = [
            p.geometry for p in boundary
            if isinstance(p.geometry, LineString)
        ]
        hatching_lines = [
            p.geometry for p in filtered.removed
            if isinstance(p.geometry, LineString)
        ]
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
            classification_primitives=result.classification,
        )
        classify_rooms(sep.rooms)
        reclassify_kitchen_near_wc(sep.rooms)
        classify_by_geometry(sep.rooms)

        # Snapshot pre-resize doors before mutating.
        doors_pre = list(sep.doors)

        resize_result = resize_arc_doors(
            doors=sep.doors,
            walls=sep.merged_walls,
            rooms=sep.rooms,
            target_thickness=target_thickness,
            max_ray_factor=poly_cfg.arc_door_max_ray_factor,
        )

        print_summary(sample_name, sep, doors_pre, resize_result, target_thickness)
        plot_three_panel(
            sep_pre=sep,
            doors_post=resize_result.doors,
            rooms_post=resize_result.rooms,
            save_path=OUTPUT_DIR / f"{sample_name}_resize.png",
            sample_name=sample_name,
            viewbox_height=result.viewbox_height,
            target_thickness=target_thickness,
        )

    print("\nDone. Outputs in", OUTPUT_DIR)


if __name__ == "__main__":
    main()
