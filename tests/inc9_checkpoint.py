"""Inc 9 checkpoint: First polygonize attempt visual validation.

Runs the full pipeline (parse -> filter hatching -> clean -> polygonize)
on 5 samples and generates 2-panel comparison plots.

Usage:
    .venv/bin/python -m tests.inc9_checkpoint
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from shapely.geometry import LineString

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.cleaning import clean_geometry
from src.config import CleaningConfig, HatchingFilterConfig, PolygonizationConfig
from src.polygonizer import PolygonizeResult, polygonize_lines
from src.svg_parser import parse_svg
from src.hatching_filter import filter_hatching

logging.basicConfig(level=logging.INFO, format="%(name)s — %(message)s")
logger = logging.getLogger(__name__)

SAMPLES = [
    Path("input/sample/ABLOK_12KAT_parça1.svg"),
    Path("input/sample/HKÜ_HUKUK_3KAT_parça1.svg"),
    Path("input/sample/ARI 8_1.kat_parça1_layered.svg"),
    Path("input/sample/ESKİŞEHİR_ADLİYESİ_1.KAT_parça15.svg"),
    Path("input/sample/GaziUni_2Kat_parça2.svg"),
]
OUTPUT_DIR = Path("output/inc9")


def _compute_bounds(lines):
    """Compute (max_x, max_y) from LineStrings."""
    max_x = 0.0
    max_y = 0.0
    for ls in lines:
        for c in ls.coords:
            if c[0] > max_x:
                max_x = c[0]
            if c[1] > max_y:
                max_y = c[1]
    return max_x, max_y


def plot_two_panel(noded_lines, poly_result, save_path, sample_name,
                   viewbox_height, input_count, boundary_rects=None):
    """2-panel: noded lines (left) | polygons + dangles (right)."""
    mx, _ = _compute_bounds(noded_lines)
    xlim = mx * 1.05 if mx > 0 else viewbox_height * 1.5
    ylim = viewbox_height

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(28, 14))

    # Left: noded lines
    for ls in noded_lines:
        xs, ys = ls.xy
        ax1.plot(xs, ys, color="#a75c20", linewidth=0.5, alpha=0.8)

    # Overlay boundary rectangle
    if boundary_rects:
        for rect in boundary_rects:
            xs, ys = rect.xy
            ax1.plot(xs, ys, color="black", linewidth=2.0, linestyle="--", label="Boundary")
    ax1.set_title(
        f"{sample_name}\nNoded lines ({input_count})",
        fontsize=9,
    )
    ax1.set_aspect("equal")
    ax1.set_xlim(0, xlim)
    ax1.set_ylim(0, ylim)

    # Right: polygons with random colors + dangle overlay
    np.random.seed(42)
    for poly in poly_result.polygons:
        color = np.random.rand(3)
        coords = np.array(poly.exterior.coords)
        patch = plt.Polygon(
            coords, alpha=0.4, facecolor=color, edgecolor="black",
            linewidth=0.3,
        )
        ax2.add_patch(patch)

    # Overlay dangles in red
    for d in poly_result.dangles:
        xs, ys = d.xy
        ax2.plot(xs, ys, color="#e03e9b", linewidth=1.5, alpha=0.8)

    # Overlay cut edges in orange
    for ce in poly_result.cut_edges:
        xs, ys = ce.xy
        ax2.plot(xs, ys, color="#ed702d", linewidth=1.0, alpha=0.8)

    # Overlay boundary rectangle
    if boundary_rects:
        for rect in boundary_rects:
            xs, ys = rect.xy
            ax2.plot(xs, ys, color="black", linewidth=2.0, linestyle="--")

    # Legend
    ax2.plot([], [], color="#e03e9b", linewidth=1.5,
             label=f"Dangles ({poly_result.stats.dangle_count})")
    if poly_result.cut_edges:
        ax2.plot([], [], color="#ed702d", linewidth=1.0,
                 label=f"Cut edges ({poly_result.stats.cut_edge_count})")
    ax2.legend(loc="upper right", fontsize=8)

    ax2.set_title(
        f"{sample_name}\nPolygons ({poly_result.stats.filtered_polygon_count})"
        f" + dangles ({poly_result.stats.dangle_count})",
        fontsize=9,
    )
    ax2.set_aspect("equal")
    ax2.set_xlim(0, xlim)
    ax2.set_ylim(0, ylim)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    logger.info("Saved %s", save_path)


def print_summary(sample_name, stats, input_count):
    """Print polygonization summary."""
    print(f"\n{'='*60}")
    print(f"{sample_name}")
    print(f"{'='*60}")
    print(f"  Input lines:       {input_count}")
    print(f"  Raw polygons:      {stats.raw_polygon_count}")
    print(f"  After area filter: {stats.filtered_polygon_count}"
          f" ({stats.removed_small_count} removed,"
          f" min_room_area={PolygonizationConfig().min_room_area})")
    print(f"  Dangles:           {stats.dangle_count}"
          f" (total length: {stats.total_dangle_length:.1f})")
    print(f"  Cut edges:         {stats.cut_edge_count}")
    print(f"  Invalid rings:     {stats.invalid_ring_count}")
    if stats.filtered_polygon_count > 0:
        print(f"  Total area:        {stats.total_area:.1f}")
        print(f"  Area range:        [{stats.min_area:.1f}, {stats.max_area:.1f}]")

    if stats.filtered_polygon_count == 0:
        print("  WARNING: no polygons formed — gap closing likely needed")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for svg_path in SAMPLES:
        if not svg_path.exists():
            logger.warning("Sample not found: %s", svg_path)
            continue

        sample_name = svg_path.stem

        # Parse -> filter hatching -> clean -> polygonize
        result = parse_svg(svg_path)

        # Extract boundary rectangles (primitive_id >= 1000000)
        boundary_rect_prims = [
            p for p in result.all_primitives if p.primitive_id >= 1000000
        ]
        boundary_rects = [p.geometry for p in boundary_rect_prims]

        filtered = filter_hatching(result.boundary, HatchingFilterConfig())
        # Include the SVG viewBox boundary rectangles in the polygonize
        # input so wall segments at the SVG edge can close against them.
        cleaning = clean_geometry(filtered.kept + boundary_rect_prims, CleaningConfig())
        poly = polygonize_lines(cleaning.lines, PolygonizationConfig())

        input_count = len(cleaning.lines)

        print_summary(sample_name, poly.stats, input_count)

        plot_two_panel(
            cleaning.lines, poly,
            OUTPUT_DIR / f"{sample_name}_polygonize.png",
            sample_name, result.viewbox_height, input_count,
            boundary_rects=boundary_rects,
        )

    print("\nDone. Outputs in", OUTPUT_DIR)


if __name__ == "__main__":
    main()
