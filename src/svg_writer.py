"""Write a clean post-processing SVG of rooms, walls, doors, windows.

Outputs a standalone SVG with one <path> per polygon, colored by
facil_type for rooms, gray for walls, and stroke-derived colors for
doors/windows. The result is meant to be opened directly in a browser
or vector editor — no debug overlays, no axis decorations.

The pipeline keeps coordinates in y-up math space (svg_parser flips the
incoming y axis). On output we wrap everything in a single
`<g transform="scale(1,-1) translate(0,-H)">` so the saved SVG renders
right-side-up under the standard SVG y-down convention.
"""

from __future__ import annotations

import re
from pathlib import Path

from shapely.geometry import LinearRing, MultiPolygon, Polygon

from src.classifier import TYPE_COLORS
from src.models import RoomPolygon

WALL_GRAY = "#6b7280"
WALL_EDGE = "#1f2937"
WINDOW_BLUE = "#2563eb"
DOOR_PINK = "#e03e9b"
UNCLASSIFIED_FILL = "#ffffff"
UNCLASSIFIED_EDGE = "#9ca3af"


def _parse_rgb_to_hex(stroke: str | None, fallback: str) -> str:
    if not stroke:
        return fallback
    m = re.match(r"rgb\(\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\)", stroke)
    if not m:
        return fallback
    r, g, b = (int(m.group(i)) for i in (1, 2, 3))
    return f"#{r:02x}{g:02x}{b:02x}"


def _ring_to_path(ring: LinearRing) -> str:
    coords = list(ring.coords)
    if len(coords) < 3:
        return ""
    head = f"M {coords[0][0]:.3f} {coords[0][1]:.3f}"
    body = " ".join(f"L {x:.3f} {y:.3f}" for x, y in coords[1:])
    return f"{head} {body} Z"


def _polygon_to_path(geom) -> str:
    if isinstance(geom, MultiPolygon):
        return " ".join(_polygon_to_path(p) for p in geom.geoms if not p.is_empty)
    if not isinstance(geom, Polygon) or geom.is_empty:
        return ""
    parts = [_ring_to_path(geom.exterior)]
    for hole in geom.interiors:
        parts.append(_ring_to_path(hole))
    return " ".join(p for p in parts if p)


def _path_element(d: str, fill: str, stroke: str, *,
                  fill_opacity: float = 1.0, stroke_width: float = 0.4) -> str:
    if not d:
        return ""
    return (
        f'  <path d="{d}" fill="{fill}" fill-opacity="{fill_opacity:.2f}" '
        f'stroke="{stroke}" stroke-width="{stroke_width:.2f}"/>'
    )


def write_floorplan_svg(
    path: Path,
    rooms: list[RoomPolygon],
    walls: list,
    doors: list[RoomPolygon],
    windows: list[RoomPolygon],
    bounds: tuple[float, float, float, float],
) -> None:
    """Write a single SVG showing the post-Inc-15 classified floor plan.

    Args:
        path:    Output file path.
        rooms:   Classified room polygons (facil_type set).
        walls:   Wall polygons (e.g. absorbed.walls — already clipped).
        doors:   Door polygons (the wall-side halves remaining after Inc 15).
        windows: Window polygons.
        bounds:  (min_x, min_y, max_x, max_y) of the data extent. Used as
                 the SVG viewBox so the saved file is cropped to the
                 actual drawing.
    """
    min_x, min_y, max_x, max_y = bounds
    width = max_x - min_x
    height = max_y - min_y
    # SVG transforms apply right-to-left mathematically, so the rightmost
    # transform is applied to the point first. Translate first to slide
    # data so its y-axis is centered at 0, then scale(1,-1) flips it,
    # leaving the data back in [min_y, max_y]. Equivalent to
    #     y_svg = (min_y + max_y) - y_data
    flip_y = -(min_y + max_y)

    lines: list[str] = []
    lines.append('<?xml version="1.0" encoding="UTF-8"?>')
    lines.append(
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'viewBox="{min_x:.3f} {min_y:.3f} {width:.3f} {height:.3f}">'
    )
    lines.append(
        f'<g transform="scale(1 -1) translate(0 {flip_y:.3f})" '
        f'fill-rule="evenodd">'
    )

    # Rooms first (largest underneath so smaller rooms paint on top).
    for r in sorted(rooms, key=lambda r: -r.geometry.area):
        if r.facil_type and r.facil_type in TYPE_COLORS:
            color = TYPE_COLORS[r.facil_type]
            edge = color
            fop = 0.65
        else:
            color = UNCLASSIFIED_FILL
            edge = UNCLASSIFIED_EDGE
            fop = 0.5
        elem = _path_element(
            _polygon_to_path(r.geometry), color, edge,
            fill_opacity=fop, stroke_width=0.4,
        )
        if elem:
            lines.append(elem)

    # Walls (gray) on top of rooms.
    for w in walls:
        elem = _path_element(
            _polygon_to_path(w), WALL_GRAY, WALL_EDGE,
            fill_opacity=0.85, stroke_width=0.6,
        )
        if elem:
            lines.append(elem)

    # Doors (pink rectangles or stroke-derived color).
    for d in doors:
        color = _parse_rgb_to_hex(d.stroke, DOOR_PINK)
        elem = _path_element(
            _polygon_to_path(d.geometry), color, color,
            fill_opacity=0.85, stroke_width=0.6,
        )
        if elem:
            lines.append(elem)

    # Windows (blue).
    for w in windows:
        elem = _path_element(
            _polygon_to_path(w.geometry), WINDOW_BLUE, WINDOW_BLUE,
            fill_opacity=0.85, stroke_width=0.6,
        )
        if elem:
            lines.append(elem)

    lines.append('</g>')
    lines.append('</svg>')

    path.write_text("\n".join(lines), encoding="utf-8")
