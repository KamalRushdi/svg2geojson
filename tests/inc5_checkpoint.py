"""Inc 5 checkpoint: Door/window closing visual validation.

Runs the unified adaptive + wall-constraining approach on 5 samples
and generates 3-panel comparison plots.

Usage:
    .venv/bin/python -m tests.inc5_checkpoint
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from shapely.geometry import LineString, Polygon

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import DoorClosingConfig, HatchingFilterConfig
from src.door_closer import close_openings
from src.hatching_filter import filter_hatching
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
OUTPUT_DIR = Path("output/inc5")


def _estimate_width(parse_result):
    """Estimate viewbox width from primitive coordinates."""
    max_x = 0.0
    for p in parse_result.boundary:
        if isinstance(p.geometry, LineString):
            for c in p.geometry.coords:
                if c[0] > max_x:
                    max_x = c[0]
    return max_x * 1.1 if max_x > 0 else parse_result.viewbox_height * 1.5


def plot_all_primitives(ax, parse_result, title, viewbox_height):
    """Plot full building: all parsed primitives color-coded by type."""
    vw = _estimate_width(parse_result)
    for p in parse_result.boundary:
        if isinstance(p.geometry, LineString):
            xs, ys = p.geometry.xy
            ax.plot(xs, ys, color="#a75c20", linewidth=0.5, alpha=0.7)
    for p in parse_result.doors:
        if isinstance(p.geometry, LineString):
            xs, ys = p.geometry.xy
            ax.plot(xs, ys, color="#e03e9b", linewidth=0.5, alpha=0.5)
    for p in parse_result.windows:
        if isinstance(p.geometry, LineString):
            xs, ys = p.geometry.xy
            ax.plot(xs, ys, color="#604ef5", linewidth=0.5, alpha=0.5)
    for p in parse_result.classification:
        if isinstance(p.geometry, LineString):
            xs, ys = p.geometry.xy
            ax.plot(xs, ys, color="#888888", linewidth=0.3, alpha=0.3)
    ax.set_title(title, fontsize=9)
    ax.set_aspect("equal")
    ax.set_xlim(0, vw)
    ax.set_ylim(0, viewbox_height)


def plot_boundary_lines(ax, primitives, title, viewbox_height, viewbox_width):
    """Plot boundary LineStrings on an axes."""
    for p in primitives:
        if isinstance(p.geometry, LineString):
            xs, ys = p.geometry.xy
            ax.plot(xs, ys, color="#a75c20", linewidth=0.5, alpha=0.8)
    ax.set_title(title, fontsize=9)
    ax.set_aspect("equal")
    ax.set_xlim(0, viewbox_width)
    ax.set_ylim(0, viewbox_height)


def plot_three_panel(parse_result, boundary, closing_result, save_path, sample_name):
    """3-panel: full building | boundary only | boundary + closing result."""
    vh = parse_result.viewbox_height
    vw = _estimate_width(parse_result)
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(36, 14))

    # Left: full building (all primitives)
    plot_all_primitives(
        ax1, parse_result,
        f"{sample_name}\nFull building (all primitives)",
        vh,
    )

    # Middle: boundary only (after hatching filter)
    plot_boundary_lines(
        ax2, boundary,
        f"{sample_name}\nBoundary after hatching filter ({len(boundary)})",
        vh, vw,
    )

    # Right: boundary + closing edges + door/window polygons
    combined = boundary + closing_result.door_edges + closing_result.window_edges
    plot_boundary_lines(
        ax3, combined,
        f"{sample_name}\nWith closing ({len(combined)} edges, "
        f"{closing_result.door_count}D/{closing_result.window_count}W)",
        vh, vw,
    )

    # Overlay door rectangles in pink
    for dp in closing_result.door_polygons:
        if isinstance(dp.geometry, Polygon):
            xs, ys = dp.geometry.exterior.xy
            ax3.fill(xs, ys, alpha=0.3, color="#e03e9b")
            ax3.plot(xs, ys, color="#e03e9b", linewidth=1.5)

    # Overlay window rectangles in blue
    for wp in closing_result.window_polygons:
        if isinstance(wp.geometry, Polygon):
            xs, ys = wp.geometry.exterior.xy
            ax3.fill(xs, ys, alpha=0.3, color="#604ef5")
            ax3.plot(xs, ys, color="#604ef5", linewidth=1.5)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    logger.info("Saved %s", save_path)


def print_summary(sample_name, result):
    """Print closing summary."""
    total_edges = len(result.door_edges) + len(result.window_edges)
    print(f"\n{'='*60}")
    print(f"{sample_name}")
    print(f"{'='*60}")
    print(f"  Doors:   {result.door_count} instances ({len(result.door_edges)} closing edges)")
    print(f"  Windows: {result.window_count} instances ({len(result.window_edges)} closing edges)")
    print(f"  Skipped: {len(result.skipped_instances)}")
    if result.skipped_instances:
        for iid, reason in result.skipped_instances[:5]:
            print(f"    instance {iid}: {reason}")
    print(f"  Total closing edges: {total_edges}")

    # Print door rectangle dimensions
    if result.door_polygons:
        print(f"\n  Door rectangle dimensions:")
        for i, dp in enumerate(result.door_polygons[:10]):
            coords = list(dp.geometry.exterior.coords[:-1])
            edges = [
                LineString([coords[j], coords[(j + 1) % 4]]).length
                for j in range(4)
            ]
            short = min(edges)
            long = max(edges)
            print(f"    Door {i}: {short:.2f} x {long:.2f}")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    config = DoorClosingConfig()

    for svg_path in SAMPLES:
        if not svg_path.exists():
            logger.warning("Sample not found: %s", svg_path)
            continue

        sample_name = svg_path.stem

        # Parse and filter hatching
        result = parse_svg(svg_path)
        filtered = filter_hatching(result.boundary, HatchingFilterConfig())
        boundary = filtered.kept

        print(f"\n{'#'*60}")
        print(f"# {sample_name}")
        print(f"{'#'*60}")
        print(f"  Boundary (after hatching filter): {len(boundary)}")
        print(f"  Door primitives: {len(result.doors)}")
        print(f"  Window primitives: {len(result.windows)}")

        # Run unified closing
        closing = close_openings(result.doors, result.windows, boundary, config)
        print_summary(sample_name, closing)

        # 3-panel plot
        plot_three_panel(
            result, boundary, closing,
            OUTPUT_DIR / f"{sample_name}_3panel.png",
            sample_name,
        )

    print("\nDone. Outputs in", OUTPUT_DIR)


if __name__ == "__main__":
    main()
