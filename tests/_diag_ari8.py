"""Diagnostic: dump full thickness/area distribution for ARI 8.

Run: .venv/bin/python -m tests._diag_ari8

Prints the candidate polygons that reach the thickness stage, sorted by
thickness, with their classification (wall / room) so we can see where
the threshold lands and which polygons are close to the boundary.
"""
from __future__ import annotations

import logging
import sys
from math import log10
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.cleaning import clean_geometry
from src.config import (
    CleaningConfig, DoorClosingConfig, HatchingFilterConfig,
    PolygonizationConfig,
)
from src.door_closer import close_openings
from src.hatching_filter import filter_hatching
from src.polygonizer import polygonize_lines
from src.separator import (
    _classify_columns, _compute_thickness, _resolve_threshold,
    _split_off_openings,
)
from src.svg_parser import parse_svg

logging.basicConfig(level=logging.WARNING, format="%(name)s — %(message)s")

SAMPLE = Path("input/sample/ARI 8_1.kat_parça1_layered.svg")


def main():
    cfg = PolygonizationConfig()
    parsed = parse_svg(SAMPLE)
    boundary_rect_prims = [
        p for p in parsed.all_primitives if p.primitive_id >= 1000000
    ]
    boundary = filter_hatching(parsed.boundary, HatchingFilterConfig()).kept
    closing = close_openings(parsed.doors, parsed.windows, boundary,
                             DoorClosingConfig(split_l_shaped_windows=True))
    combined = (boundary + closing.door_edges + closing.window_edges
                + boundary_rect_prims)
    cleaning = clean_geometry(combined, CleaningConfig())
    poly = polygonize_lines(cleaning.lines, cfg)

    print(f"Raw polygons: {len(poly.polygons)}")
    print(f"Door AABBs: {len(closing.door_polygons)}, "
          f"Window AABBs: {len(closing.window_polygons)}")

    # Replicate the separator's stages 1, 2, 2.5
    candidates = list(poly.polygons)
    doors, windows, candidates = _split_off_openings(
        candidates, closing.door_polygons, closing.window_polygons,
    )
    print(f"After Stage 1 (openings): {len(doors)} doors, {len(windows)} windows, "
          f"{len(candidates)} remain")

    small_walls, sized = [], []
    for p in candidates:
        if p.area < cfg.separator_min_area:
            small_walls.append(p)
        else:
            sized.append(p)
    print(f"After Stage 2 (min-area): {len(small_walls)} dropped, "
          f"{len(sized)} sized")

    columns, sized = _classify_columns(sized, cfg)
    print(f"After Stage 2.5 (columns): {len(columns)} columns, "
          f"{len(sized)} remain")

    # Now the meat: thickness for every remaining candidate
    rows = []
    for i, p in enumerate(sized):
        t = _compute_thickness(p)
        rows.append({"i": i, "area": p.area, "thickness": t})

    # Sort by thickness ascending so the wall/room boundary is visible
    rows.sort(key=lambda r: r["thickness"])

    # Apply min-thickness filter and resolve threshold
    sliver_thickness = cfg.separator_min_thickness
    above = [r["thickness"] for r in rows
             if r["thickness"] >= sliver_thickness]
    threshold, method, gap_size = _resolve_threshold(above, cfg)

    print()
    print(f"Min-thickness filter: {sliver_thickness}")
    print(f"Threshold method: {method}")
    print(f"Threshold: {threshold:.3f}")
    print(f"Largest log10-gap chosen: {gap_size:.3f}")
    print()

    print(f"{'idx':>3} {'area':>10} {'thickness':>10} {'log10':>7}  class")
    for r in rows:
        if r["thickness"] < sliver_thickness:
            cls = "sliver_wall"
        elif r["thickness"] < threshold:
            cls = "WALL"
        else:
            cls = "ROOM"
        log_t = log10(r["thickness"]) if r["thickness"] > 0 else float("-inf")
        print(f"{r['i']:>3} {r['area']:>10.2f} {r['thickness']:>10.3f} "
              f"{log_t:>7.3f}  {cls}")

    # Adjacent log10-gaps for the values that fed the threshold
    print()
    print("Adjacent log10-gaps in thickness distribution (for above-min only):")
    log_vals = sorted(log10(t) for t in above if t > 0)
    n = len(log_vals)
    for i in range(n - 1):
        gap = log_vals[i + 1] - log_vals[i]
        left = i + 1
        right = n - left
        score = gap * min(left, right)
        marker = "  <-- chosen" if abs(
            10 ** ((log_vals[i] + log_vals[i + 1]) / 2) - threshold) < 0.01 else ""
        print(f"  log10 split @ {(log_vals[i] + log_vals[i+1])/2:>6.3f} "
              f"gap={gap:.3f} left={left:>2} right={right:>2} "
              f"score={score:.3f}{marker}")


if __name__ == "__main__":
    main()
