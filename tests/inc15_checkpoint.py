"""Inc 15 checkpoint: absorb arc-door polygons into adjacent rooms.

Pipeline:
    parse -> filter hatching -> close openings -> clean -> polygonize
        -> separate (Inc 12 lock) -> classify (Inc 13/14)
        -> ABSORB ARC DOORS  (this increment)

Generates a 2-panel comparison plot per sample:
  Panel 1 - Inc 13 output: classified rooms + walls + door/window overlay
  Panel 2 - Inc 15 output: arc doors absorbed into the room they open into
            (absorbed area drawn with diagonal hatching so it's visible)

Usage:
    .venv/bin/python -m tests.inc15_checkpoint
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import matplotlib.pyplot as plt
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
from src.door_absorber import absorb_arc_doors
from src.door_closer import close_openings
from src.hatching_filter import filter_hatching
from src.polygonizer import polygonize_lines
from src.separator import separate_polygons
from src.svg_parser import parse_svg
from tests.inc13_checkpoint import (
    UNCLASSIFIED_EDGE,
    UNCLASSIFIED_FILL,
    WALL_EDGE,
    WALL_GRAY,
    WINDOW_BLUE,
    _compute_bounds,
    _fill_polygon,
    _parse_rgb,
)

logging.basicConfig(level=logging.INFO, format="%(name)s — %(message)s")
logger = logging.getLogger(__name__)

SAMPLES = [
    Path("input/sample/ABLOK_12KAT_parça1.svg"),
    Path("input/sample/HKÜ_HUKUK_3KAT_parça1.svg"),
    Path("input/sample/ARI 8_1.kat_parça1_layered.svg"),
    Path("input/sample/ESKİŞEHİR_ADLİYESİ_1.KAT_parça15.svg"),
    Path("input/sample/GaziUni_2Kat_parça2.svg"),
]
OUTPUT_DIR = Path("output/inc15")


def _draw_classified(ax, rooms, walls, doors, windows, semantic_color,
                     absorbed_geoms=None):
    rooms_in_z = sorted(rooms, key=lambda r: -r.geometry.area)
    for r in rooms_in_z:
        if not isinstance(r.geometry, Polygon):
            continue
        if r.facil_type and r.facil_type in TYPE_COLORS:
            color = TYPE_COLORS[r.facil_type]
            _fill_polygon(ax, r.geometry, facecolor=color, alpha=0.65,
                          edgecolor=color, linewidth=0.4)
        else:
            _fill_polygon(ax, r.geometry, facecolor=UNCLASSIFIED_FILL,
                          alpha=0.5, edgecolor=UNCLASSIFIED_EDGE,
                          linewidth=0.4)
    for w in walls:
        _fill_polygon(ax, w, facecolor=WALL_GRAY, alpha=0.85,
                      edgecolor=WALL_EDGE, linewidth=0.6)
    for d in doors:
        if isinstance(d.geometry, Polygon):
            color = semantic_color.get(id(d.geometry), "#e03e9b")
            _fill_polygon(ax, d.geometry, facecolor=color, alpha=0.85,
                          edgecolor=color, linewidth=0.6)
    for w in windows:
        if isinstance(w.geometry, Polygon):
            color = semantic_color.get(id(w.geometry), WINDOW_BLUE)
            _fill_polygon(ax, w.geometry, facecolor=color, alpha=0.85,
                          edgecolor=color, linewidth=0.6)
    # Outline of absorbed door footprint so it's visible against the room
    if absorbed_geoms:
        for g in absorbed_geoms:
            if not isinstance(g, Polygon):
                continue
            xs, ys = g.exterior.xy
            ax.plot(xs, ys, color="#e03e9b", linewidth=0.8,
                    linestyle="--", alpha=0.9)


def plot_two_panel(parse_result, noded_lines, sep, absorbed,
                   save_path, sample_name, viewbox_height):
    bounds = _compute_bounds(noded_lines)
    if bounds is None:
        xlim_lo, xlim_hi = 0.0, viewbox_height * 1.5
        ylim_lo, ylim_hi = 0.0, viewbox_height
    else:
        bx0, by0, bx1, by1 = bounds
        pad_x = max((bx1 - bx0) * 0.025, 1e-3)
        pad_y = max((by1 - by0) * 0.025, 1e-3)
        xlim_lo, xlim_hi = bx0 - pad_x, bx1 + pad_x
        ylim_lo, ylim_hi = by0 - pad_y, by1 + pad_y

    semantic_color: dict[int, tuple] = {}
    for d in sep.doors:
        if isinstance(d.geometry, Polygon):
            semantic_color[id(d.geometry)] = _parse_rgb(d.stroke, "#e03e9b")
    for w in sep.windows:
        if isinstance(w.geometry, Polygon):
            semantic_color[id(w.geometry)] = WINDOW_BLUE

    fig, axes = plt.subplots(1, 2, figsize=(24, 12))
    ax1, ax2 = axes

    # Panel 1: Inc 13 output (all doors visible, no absorption)
    _draw_classified(ax1, sep.rooms, sep.merged_walls, sep.doors, sep.windows,
                     semantic_color)
    summary_before = classification_summary(sep.rooms)
    ax1.set_title(
        f"{sample_name}\nInc 13: {sum(summary_before.values())} rooms, "
        f"{len(sep.doors)} doors visible",
        fontsize=10,
    )

    # Panel 2: Inc 15 output (arc doors absorbed, walls clipped)
    _draw_classified(ax2, absorbed.rooms, absorbed.walls,
                     absorbed.doors, sep.windows, semantic_color,
                     absorbed_geoms=absorbed.absorbed_door_geoms)
    summary_after = classification_summary(absorbed.rooms)
    ax2.set_title(
        f"{sample_name}\nInc 15: {absorbed.absorbed_count} arc doors absorbed, "
        f"{absorbed.orphan_count} orphan, "
        f"{len(absorbed.doors)} doors remain (dashed = absorbed footprint)",
        fontsize=10,
    )

    for ax in axes:
        ax.set_aspect("equal")
        ax.set_xlim(xlim_lo, xlim_hi)
        ax.set_ylim(ylim_lo, ylim_hi)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    logger.info("Saved %s", save_path)


def print_summary(sample_name, sep, absorbed):
    print(f"\n{'='*60}")
    print(f"{sample_name}")
    print(f"{'='*60}")
    arc_total = sum(1 for d in sep.doors if d.has_arc)
    print(f"  Total doors:              {len(sep.doors)}")
    print(f"  Arc-bearing doors:        {arc_total}")
    print(f"  Absorbed into a room:     {absorbed.absorbed_count}")
    print(f"  Orphan arc doors:         {absorbed.orphan_count}")
    print(f"  Doors remaining:          {len(absorbed.doors)}")


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

        absorbed = absorb_arc_doors(sep.rooms, sep.doors, sep.merged_walls)

        print_summary(sample_name, sep, absorbed)

        plot_two_panel(
            result, cleaning.lines, sep, absorbed,
            OUTPUT_DIR / f"{sample_name}_absorbed.png",
            sample_name, result.viewbox_height,
        )

    print("\nDone. Outputs in", OUTPUT_DIR)


if __name__ == "__main__":
    main()
