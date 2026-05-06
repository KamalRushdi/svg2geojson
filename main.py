"""End-to-end CLI: SVG floorplan -> GeoJSON FeatureCollection.

Currently runs with a placeholder affine so the OUTPUT FORMAT is correct
even without real control points. To produce real-world coordinates,
replace the placeholder with `compute_affine(svg_pts, geo_pts)` from
`src.georeferencer`.

Usage:
    .venv/bin/python -m main \\
        --input "input/sample/AB2_2KAT_parça18.svg=2" \\
        --input "input/sample/ABLOK_12KAT_parça1.svg=12" \\
        --output output/demo.geojson \\
        --name "demo_export"

Each --input is `path[=level_id]`. If level_id is omitted, it's inferred
from the filename (e.g. `..._12KAT_...` -> "12", `..._1BODRUM_...` -> "-1").
"""

from __future__ import annotations

import argparse
import logging
import re
import sys
from pathlib import Path

from shapely.geometry import LineString

from src.classifier import (
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
from src.georeferencer import (
    build_floor_features,
    export_geojson,
    placeholder_affine_for_viewbox,
)
from src.hatching_filter import filter_hatching
from src.polygonizer import polygonize_lines
from src.separator import separate_polygons
from src.svg_parser import parse_svg

logging.basicConfig(level=logging.INFO, format="%(name)s — %(message)s")
logger = logging.getLogger(__name__)


_KAT_RE = re.compile(r"_(\d+)\s*\.?\s*KAT", re.IGNORECASE)
_BODRUM_NUM_RE = re.compile(r"_(\d+)\s*BODRUM", re.IGNORECASE)
_BODRUM_BARE_RE = re.compile(r"_BODRUM(?:KAT)?[_.]", re.IGNORECASE)
_ZEMIN_RE = re.compile(r"_ZEM[İI]N(?:KAT)?[_.]", re.IGNORECASE)


def infer_level_id(svg_path: Path) -> str:
    """Best-effort extraction of a floor number from the filename.

    Recognizes Turkish CAD conventions in the sample set:
      `_12KAT_`       -> "12"   (12th floor)
      `_1.kat_`       -> "1"    (1st floor)
      `_1BODRUM_`     -> "-1"   (1st basement)
      `_BodrumKat_`   -> "-1"   (basement, no number)
      `_ZeminKat_`    -> "0"    (ground floor, Turkish)
      anything else   -> "0"

    Override per file via `path=level_id` on the CLI when this guesses wrong.
    """
    name = svg_path.name
    m = _BODRUM_NUM_RE.search(name)
    if m:
        return f"-{m.group(1)}"
    if _BODRUM_BARE_RE.search(name):
        return "-1"
    if _ZEMIN_RE.search(name):
        return "0"
    m = _KAT_RE.search(name)
    if m:
        return m.group(1)
    return "0"


def run_floor(svg_path: Path, level_id: str) -> tuple:
    """Run the full pipeline on one SVG. Returns (sep_result, viewbox_size)."""
    parse_result = parse_svg(svg_path)

    boundary_rect_prims = [
        p for p in parse_result.all_primitives if p.primitive_id >= 1000000
    ]

    filtered = filter_hatching(parse_result.boundary, HatchingFilterConfig())
    boundary = filtered.kept
    wall_lines = [
        p.geometry for p in boundary if isinstance(p.geometry, LineString)
    ]
    hatching_lines = [
        p.geometry for p in filtered.removed
        if isinstance(p.geometry, LineString)
    ]

    door_cfg = DoorClosingConfig(split_l_shaped_windows=True)
    closing = close_openings(
        parse_result.doors, parse_result.windows, boundary, door_cfg,
    )
    combined = (
        boundary
        + closing.door_edges
        + closing.window_edges
        + boundary_rect_prims
    )

    cleaning = clean_geometry(combined, CleaningConfig())

    poly_cfg = PolygonizationConfig()
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

    viewbox_size = (parse_result.viewbox_height, parse_result.viewbox_height)
    return sep, viewbox_size, level_id


def parse_input_arg(arg: str) -> tuple[Path, str]:
    """Parse `path[=level_id]` from a single --input argument."""
    if "=" in arg:
        path_str, lvl = arg.rsplit("=", 1)
        path = Path(path_str)
        return path, lvl
    path = Path(arg)
    return path, infer_level_id(path)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="SVG floorplan(s) -> GeoJSON FeatureCollection",
    )
    ap.add_argument(
        "--input", action="append", required=True,
        help="path[=level_id]; pass multiple times for multi-floor output",
    )
    ap.add_argument(
        "--output", required=True, type=Path,
        help="path to write the GeoJSON file",
    )
    ap.add_argument(
        "--name", default="svg2geojson_export",
        help="value for the top-level GeoJSON `name` field",
    )
    ap.add_argument(
        "--window-as", default="Wall", choices=["Wall", "Window"],
        help="emit window features as 'Wall' (matches ARI6_YENI) or 'Window'",
    )
    args = ap.parse_args(argv)

    inputs = [parse_input_arg(a) for a in args.input]

    all_features: list[dict] = []
    for svg_path, level_id in inputs:
        if not svg_path.exists():
            logger.error("Skipping missing file: %s", svg_path)
            continue
        logger.info("Processing %s (level_id=%s)", svg_path.name, level_id)

        sep, (vbw, vbh), lvl = run_floor(svg_path, level_id)
        matrix = placeholder_affine_for_viewbox(vbw, vbh)

        floor_features = build_floor_features(
            sep, lvl, matrix, window_as=args.window_as,
        )
        logger.info("  -> %d features", len(floor_features))
        all_features.extend(floor_features)

    out = export_geojson(all_features, args.output, name=args.name)
    logger.info("Wrote %d features to %s", len(all_features), out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
