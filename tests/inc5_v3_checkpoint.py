"""Inc 5 v3 checkpoint: bbox closing with adaptive multi-leg window splitting.

Same as v2 but enables `split_l_shaped_windows`. Each window instance whose
segment angles cluster into 2+ angular groups (separated by gaps >=
l_shape_angle_threshold_deg) is split into N rectangles, one per leg.
Doors are NOT split (their arcs would confuse angle clustering).

Usage:
    .venv/bin/python -m tests.inc5_v3_checkpoint
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import re

import matplotlib.pyplot as plt
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
OUTPUT_DIR = Path("output/inc5_v3")


def _estimate_width(parse_result):
    max_x = 0.0
    for p in parse_result.boundary:
        if isinstance(p.geometry, LineString):
            for c in p.geometry.coords:
                if c[0] > max_x:
                    max_x = c[0]
    return max_x * 1.1 if max_x > 0 else parse_result.viewbox_height * 1.5


def plot_two_panel(
    parse_result, boundary, closing_result, save_path, sample_name,
):
    """2-panel: boundary only | boundary + closing rects (multi-leg windows split)."""
    vh = parse_result.viewbox_height
    vw = _estimate_width(parse_result)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(28, 14))

    # Panel 1: Boundary after hatching filter
    for p in boundary:
        if isinstance(p.geometry, LineString):
            xs, ys = p.geometry.xy
            ax1.plot(xs, ys, color="#a75c20", linewidth=0.5, alpha=0.8)
    ax1.set_title(
        f"{sample_name}\nBoundary after hatching filter ({len(boundary)})",
        fontsize=9,
    )
    ax1.set_aspect("equal")
    ax1.set_xlim(0, vw)
    ax1.set_ylim(0, vh)

    # Panel 2: Boundary + closing edges + door/window polygons
    combined = boundary + closing_result.door_edges + closing_result.window_edges
    for p in combined:
        if isinstance(p.geometry, LineString):
            xs, ys = p.geometry.xy
            ax2.plot(xs, ys, color="#a75c20", linewidth=0.5, alpha=0.8)
    for dp in closing_result.door_polygons:
        if isinstance(dp.geometry, Polygon):
            xs, ys = dp.geometry.exterior.xy
            color = _parse_rgb(dp.stroke, "#e03e9b")
            ax2.fill(xs, ys, alpha=0.3, color=color)
            ax2.plot(xs, ys, color=color, linewidth=1.5)
    for wp in closing_result.window_polygons:
        if isinstance(wp.geometry, Polygon):
            xs, ys = wp.geometry.exterior.xy
            color = _parse_rgb(wp.stroke, "#604ef5")
            ax2.fill(xs, ys, alpha=0.3, color=color)
            ax2.plot(xs, ys, color=color, linewidth=1.5)
    ax2.set_title(
        f"{sample_name}\nWith bbox closing + multi-leg split "
        f"({closing_result.door_count}D / {closing_result.window_count}W)",
        fontsize=9,
    )
    ax2.set_aspect("equal")
    ax2.set_xlim(0, vw)
    ax2.set_ylim(0, vh)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    logger.info("Saved %s", save_path)


def print_summary(sample_name, closing_result, raw_window_instance_count):
    total_edges = len(closing_result.door_edges) + len(closing_result.window_edges)
    extra_legs = closing_result.window_count - raw_window_instance_count
    print(f"\n{'='*60}")
    print(f"{sample_name}")
    print(f"{'='*60}")
    print(f"  Doors:   {closing_result.door_count} instances "
          f"({len(closing_result.door_edges)} closing edges)")
    print(f"  Windows: {closing_result.window_count} rectangles "
          f"from {raw_window_instance_count} raw instances "
          f"({extra_legs:+d} from multi-leg splits, "
          f"{len(closing_result.window_edges)} closing edges)")
    print(f"  Skipped: {len(closing_result.skipped_instances)}")
    print(f"  Total closing edges: {total_edges}")


def _count_window_instances(window_prims):
    iids = set()
    for p in window_prims:
        if p.instance_id is not None and p.instance_id != -1:
            iids.add(p.instance_id)
    return len(iids)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    # v3: spatial multi-leg splitting ENABLED for windows
    door_cfg = DoorClosingConfig(split_l_shaped_windows=True)
    hatch_cfg = HatchingFilterConfig()

    for svg_path in SAMPLES:
        if not svg_path.exists():
            logger.warning("Sample not found: %s", svg_path)
            continue

        sample_name = svg_path.stem
        print(f"\n{'#'*60}")
        print(f"# {sample_name}")
        print(f"{'#'*60}")

        result = parse_svg(svg_path)
        filtered = filter_hatching(result.boundary, hatch_cfg)
        boundary = filtered.kept

        raw_w_count = _count_window_instances(result.windows)
        closing = close_openings(result.doors, result.windows, boundary, door_cfg)
        print_summary(sample_name, closing, raw_w_count)

        plot_two_panel(
            result, boundary, closing,
            OUTPUT_DIR / f"{sample_name}_2panel.png",
            sample_name,
        )

    print("\nDone. Outputs in", OUTPUT_DIR)


if __name__ == "__main__":
    main()
