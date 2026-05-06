"""Inc 6/7/8 checkpoint: Geometry cleaning visual validation.

Runs coordinate rounding, endpoint snapping, and noding on 5 samples
and generates 3-panel comparison plots.

Usage:
    .venv/bin/python -m tests.inc678_checkpoint
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from shapely.geometry import LineString

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.cleaning import clean_geometry
from src.config import CleaningConfig, HatchingFilterConfig
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
OUTPUT_DIR = Path("output/inc678")


def _compute_bounds(primitives=None, lines=None):
    """Compute (max_x, max_y) from primitives or raw LineStrings."""
    max_x = 0.0
    max_y = 0.0
    if primitives is not None:
        for p in primitives:
            if isinstance(p.geometry, LineString):
                for c in p.geometry.coords:
                    if c[0] > max_x:
                        max_x = c[0]
                    if c[1] > max_y:
                        max_y = c[1]
    if lines is not None:
        for ls in lines:
            for c in ls.coords:
                if c[0] > max_x:
                    max_x = c[0]
                if c[1] > max_y:
                    max_y = c[1]
    return max_x, max_y


def plot_primitives(ax, primitives, title, xlim, ylim):
    """Plot FloorplanPrimitive LineStrings."""
    for p in primitives:
        if isinstance(p.geometry, LineString):
            xs, ys = p.geometry.xy
            ax.plot(xs, ys, color="#a75c20", linewidth=0.5, alpha=0.8)
    ax.set_title(title, fontsize=9)
    ax.set_aspect("equal")
    ax.set_xlim(0, xlim)
    ax.set_ylim(0, ylim)


def plot_lines(ax, lines, title, xlim, ylim):
    """Plot raw LineStrings."""
    for ls in lines:
        xs, ys = ls.xy
        ax.plot(xs, ys, color="#a75c20", linewidth=0.5, alpha=0.8)
    ax.set_title(title, fontsize=9)
    ax.set_aspect("equal")
    ax.set_xlim(0, xlim)
    ax.set_ylim(0, ylim)


def plot_three_panel(boundary, snapped_lines, noded_lines, save_path,
                     sample_name, stats, viewbox_height):
    """3-panel: before | after rounding+snapping | after noding."""
    mx, _ = _compute_bounds(primitives=boundary, lines=noded_lines)
    xlim = mx * 1.05 if mx > 0 else viewbox_height * 1.5
    ylim = viewbox_height

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(43, 14))

    plot_primitives(
        ax1, boundary,
        f"{sample_name}\nBefore cleaning ({stats.input_count} lines)",
        xlim, ylim,
    )
    plot_lines(
        ax2, snapped_lines,
        f"{sample_name}\nAfter round + snap ({len(snapped_lines)} lines, "
        f"{stats.snap_groups} groups)",
        xlim, ylim,
    )
    plot_lines(
        ax3, noded_lines,
        f"{sample_name}\nAfter noding ({stats.after_noding} lines)",
        xlim, ylim,
    )

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    logger.info("Saved %s", save_path)


def print_summary(sample_name, stats):
    """Print cleaning summary."""
    print(f"\n{'='*60}")
    print(f"{sample_name}")
    print(f"{'='*60}")
    print(f"  Input:              {stats.input_count} lines"
          f" ({stats.non_linestring_skipped} non-LineString skipped)")
    print(f"  After rounding:     {stats.after_rounding}"
          f" ({stats.degenerates_removed} degenerates removed)")
    print(f"  After snapping:     {stats.after_snap}"
          f" ({stats.snap_groups} groups, {stats.endpoints_snapped} endpoints moved)")
    print(f"  After noding:       {stats.after_noding}")
    ratio = stats.after_noding / stats.input_count if stats.input_count > 0 else 0
    print(f"  Noding expansion:   {ratio:.2f}x")

    if stats.after_noding == 0:
        print("  WARNING: no lines after cleaning!")
    if stats.input_count > 0 and stats.degenerates_removed > stats.input_count * 0.5:
        print("  WARNING: >50% degenerates removed")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    config = CleaningConfig()

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

        # Run cleaning pipeline
        cleaning = clean_geometry(boundary, config)

        # For the middle panel, we need the intermediate state (after snap, before node).
        # Re-run stages 1-2 only for visualization.
        from src.cleaning import _round_and_remove_degenerates, _snap_endpoints
        lines = [p.geometry for p in boundary if isinstance(p.geometry, LineString)]
        lines, _ = _round_and_remove_degenerates(
            lines, config.round_precision, config.min_line_length
        )
        if config.snap_tolerance > 0 and lines:
            lines, _, _ = _snap_endpoints(
                lines, config.snap_tolerance, config.round_precision
            )
            lines, _ = _round_and_remove_degenerates(
                lines, config.round_precision, config.min_line_length
            )
        snapped_lines = lines

        print_summary(sample_name, cleaning.stats)

        # 3-panel plot
        plot_three_panel(
            boundary, snapped_lines, cleaning.lines,
            OUTPUT_DIR / f"{sample_name}_cleaning.png",
            sample_name, cleaning.stats, result.viewbox_height,
        )

    print("\nDone. Outputs in", OUTPUT_DIR)


if __name__ == "__main__":
    main()
