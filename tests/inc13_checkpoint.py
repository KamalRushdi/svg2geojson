"""Inc 13 checkpoint: rule-based room type classification.

Pipeline:
    parse -> filter hatching -> close openings -> clean -> polygonize
        -> SEPARATE (with Inc 12 object-room locking)
        -> CLASSIFY (Inc 13: assign facil_type from contained_semantics)

Generates a 3-panel comparison plot per sample:
  Panel 1 - Original SVG primitives (walls in stroke colors, windows in blue)
  Panel 2 - Polygonization result (random colors per polygon)
  Panel 3 - Classified rooms colored by facil_type, walls in gray,
            doors/windows in semantic color

Usage:
    .venv/bin/python -m tests.inc13_checkpoint
"""

from __future__ import annotations

import logging
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import PathPatch
from matplotlib.path import Path as MplPath
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
from src.models import (
    ELEVATOR_IDS,
    ESCALATOR_IDS,
    FURNITURE_IDS,
    KITCHEN_IDS,
    PARKING_IDS,
    RAILING_IDS,
    ROW_CHAIR_IDS,
    STAIR_IDS,
    TOILET_IDS,
)
from src.polygonizer import polygonize_lines
from src.separator import SeparatorResult, _instance_centroids, separate_polygons
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
OUTPUT_DIR = Path("output/inc13")

WINDOW_BLUE = "#2563eb"
WALL_GRAY = "#6b7280"
WALL_EDGE = "#1f2937"
UNCLASSIFIED_FILL = "#ffffff"
UNCLASSIFIED_EDGE = "#9ca3af"

# Centroid markers — each classification primitive's centroid plotted as a
# small dot in Panel 3 so we can see WHERE the lock is probing. Colors
# match the room-type palette so a stair primitive's centroid is yellow,
# a toilet's is pink, etc.
CENTROID_MARKERS = [
    (STAIR_IDS,      "#f7ce4b", "Stairs"),
    (ESCALATOR_IDS,  "#f7ce4b", "Escalator"),
    (ELEVATOR_IDS,   "#ed702d", "Elevator"),
    (TOILET_IDS,     "#ee7ca2", "WC"),
    (KITCHEN_IDS,    "#719752", "Kitchen"),
    (ROW_CHAIR_IDS,  "#426b51", "Auditorium"),
    (FURNITURE_IDS,  "#7ab591", "Office"),
    (PARKING_IDS,    "#66433e", "Parking"),
    (RAILING_IDS,    "#403469", "Railing"),
]


def _centroid_color(sem_id: int | None) -> str | None:
    if sem_id is None:
        return None
    for ids, color, _ in CENTROID_MARKERS:
        if sem_id in ids:
            return color
    return None


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
    """Render a Shapely polygon (with holes) as a single PathPatch.

    Uses matplotlib's even-odd fill rule: the exterior is one closed
    sub-path, each hole is another closed sub-path. Holes are rendered
    truly transparent — anything painted underneath shows through.
    """
    if poly.is_empty:
        return
    verts: list[tuple[float, float]] = []
    codes: list[int] = []
    for ring in [poly.exterior, *poly.interiors]:
        ring_coords = list(ring.coords)
        if len(ring_coords) < 3:
            continue
        verts.extend(ring_coords)
        codes.append(MplPath.MOVETO)
        codes.extend([MplPath.LINETO] * (len(ring_coords) - 2))
        codes.append(MplPath.CLOSEPOLY)
    if not verts:
        return
    ax.add_patch(PathPatch(
        MplPath(verts, codes),
        alpha=alpha, facecolor=facecolor,
        edgecolor=edgecolor, linewidth=linewidth,
    ))


def plot_three_panel(
    parse_result, noded_lines, poly_result, sep: SeparatorResult,
    save_path, sample_name, viewbox_height, boundary_rects=None,
    classification_primitives=None,
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

    # ---- Panel 2: polygonization output (semantic colors for openings) ----
    semantic_color: dict[int, tuple] = {}
    for d in sep.doors:
        if isinstance(d.geometry, Polygon):
            semantic_color[id(d.geometry)] = _parse_rgb(d.stroke, "#e03e9b")
    for w in sep.windows:
        if isinstance(w.geometry, Polygon):
            semantic_color[id(w.geometry)] = WINDOW_BLUE

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

    # ---- Panel 3: classified rooms + walls + door/window overlay ----
    # Sort largest polygons first so smaller real rooms paint on top —
    # matters because the SVG-exterior leftover (full-canvas polygon) is
    # always huge and would otherwise cover real rooms.
    rooms_in_z_order = sorted(sep.rooms, key=lambda r: -r.geometry.area)
    for r in rooms_in_z_order:
        if not isinstance(r.geometry, Polygon):
            continue
        if r.facil_type and r.facil_type in TYPE_COLORS:
            color = TYPE_COLORS[r.facil_type]
            _fill_polygon(ax3, r.geometry, facecolor=color, alpha=0.65,
                          edgecolor=color, linewidth=0.4)
        else:
            _fill_polygon(ax3, r.geometry, facecolor=UNCLASSIFIED_FILL,
                          alpha=0.5, edgecolor=UNCLASSIFIED_EDGE,
                          linewidth=0.4)
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

    # Overlay one centroid per OBJECT INSTANCE (not per primitive) — the
    # exact same points the locker tests. A primitive-level centroid
    # cloud scatters across drawing strokes and many fall on walls; the
    # instance centroid is the average and lands near the object's true
    # center. If a dot lands in a wall (or outside any polygon), the
    # corresponding object failed to lock a room.
    if classification_primitives:
        for sem, pt in _instance_centroids(classification_primitives):
            color = _centroid_color(sem)
            if color is None or pt.is_empty:
                continue
            cx, cy = pt.coords[0]
            ax3.plot(cx, cy, marker="o", markersize=6.0,
                     markerfacecolor=color,
                     markeredgecolor="black", markeredgewidth=0.6,
                     linestyle="", alpha=0.95)

    summary = classification_summary(sep.rooms)
    summary_str = ", ".join(f"{k}={v}" for k, v in sorted(summary.items())) \
        or "none"
    legend_str = "  ".join(
        f"[{lab}]" for lab in TYPE_COLORS if summary.get(lab)
    )
    ax3.set_title(
        f"{sample_name}\nClassified rooms ({sum(summary.values())} total): "
        f"{summary_str}\n{legend_str}",
        fontsize=9,
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
    summary = classification_summary(sep.rooms)
    print(f"\n{'='*60}")
    print(f"{sample_name}")
    print(f"{'='*60}")
    print(f"  Raw polygons:           {raw_count}")
    print(f"  Threshold method:       {sep.stats.threshold_method}")
    print(f"  Locked rooms (Inc 12):  {sep.stats.locked_room_count}"
          f"  (overruled by hatching: "
          f"{sep.stats.locked_overruled_by_hatching_count})")
    print(f"  Total rooms:            {sep.stats.room_count}")
    print(f"  Doors / Windows:        {sep.stats.door_count} "
          f"/ {sep.stats.window_count}")
    print(f"  facil_type breakdown:")
    for label in TYPE_COLORS:
        if summary.get(label):
            print(f"    {label:12s}        {summary[label]}")
    if summary.get("Unclassified"):
        print(f"    Unclassified         {summary['Unclassified']}"
              f"  (no objects inside; Inc 14 will assign)")


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

        # Inc 13: assign facil_type from contained_semantics.
        classify_rooms(sep.rooms)
        # Inc 13 refinement: a sink adjacent to a WC is a handwashing sink,
        # not a kitchen sink — reclassify those Kitchen rooms as WC.
        reclassify_kitchen_near_wc(sep.rooms)
        # Inc 14: classify the rest by aspect ratio of the OBB.
        classify_by_geometry(sep.rooms)

        print_summary(sample_name, sep, len(poly.polygons))

        plot_three_panel(
            result, cleaning.lines, poly, sep,
            OUTPUT_DIR / f"{sample_name}_classified.png",
            sample_name, result.viewbox_height,
            boundary_rects=boundary_rects,
            classification_primitives=result.classification,
        )

    print("\nDone. Outputs in", OUTPUT_DIR)


if __name__ == "__main__":
    main()
