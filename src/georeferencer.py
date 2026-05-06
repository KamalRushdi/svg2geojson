"""Phase 5: Georeferencing and GeoJSON export.

Converts the room/wall/door/window output of the separator+classifier into a
FeatureCollection matching the schema used by the target file
`output/ARI6_YENI.geojson`:

    {
      "type": "FeatureCollection",
      "name": "...",
      "crs": { "type": "name", "properties": { "name": "urn:ogc:def:crs:OGC:1.3:CRS84" } },
      "features": [
        {
          "type": "Feature",
          "properties": {
            "id":          "<bare uuid4>",
            "isClickable": "TRUE" | "FALSE",
            "clickId":     "{<uuid4 wrapped in braces>}",
            "facil_type":  "Wall" | "Door" | "Office" | ...,
            "level_id":    "<floor as string>",
            "facil_name":  null | "<human label>",
            "floorLevel":  "<floor as string, same as level_id>"
          },
          "geometry": { "type": "MultiPolygon", "coordinates": [[[lon, lat], ...]] }
        }
      ]
    }

The georeferencer does three things:
    1. Compute a 6-parameter affine from N>=3 SVG<->lon/lat point pairs (Inc 15).
    2. Apply that affine to every Shapely Polygon (Inc 16).
    3. Serialize to a GeoJSON FeatureCollection with the schema above (Inc 17).

For demos / dry runs you can pass a synthetic affine via
`placeholder_affine_for_viewbox()` so the output shape is correct even
without real control points.
"""

from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Iterable

import numpy as np
from shapely import to_geojson
from shapely.affinity import affine_transform
from shapely.geometry import MultiPolygon, Polygon
from shapely.geometry.base import BaseGeometry

from src.models import RoomPolygon
from src.separator import SeparatorResult


# facil_types that are NOT clickable in the target file. Anything not in this
# set defaults to clickable. Derived empirically from ARI6_YENI.geojson.
NON_CLICKABLE_TYPES: set[str] = {
    "Wall",
    "Door",
    "Window",
    "Stairs",
    "Escalator",
    "Elevator",
    "Walkways",
    "Gallery",
    "Deadzone",
}

CRS_CRS84 = {
    "type": "name",
    "properties": {"name": "urn:ogc:def:crs:OGC:1.3:CRS84"},
}


# -----------------------------------------------------------------------------
# Inc 15: affine from control points
# -----------------------------------------------------------------------------

def compute_affine(
    svg_points: list[tuple[float, float]],
    geo_points: list[tuple[float, float]],
) -> list[float]:
    """Compute a 6-parameter affine matrix in Shapely's format.

    Solves the least-squares system:

        lon = a*x + b*y + tx
        lat = d*x + e*y + ty

    `geo_points` MUST be (longitude, latitude) order — same as GeoJSON.

    Returns the 6-list expected by `shapely.affinity.affine_transform`:
        [a, b, d, e, tx, ty]

    Three pairs uniquely determine the affine; more give a least-squares fit.
    """
    if len(svg_points) != len(geo_points):
        raise ValueError("svg_points and geo_points must be the same length")
    if len(svg_points) < 3:
        raise ValueError("need at least 3 control point pairs")

    n = len(svg_points)
    A = np.zeros((2 * n, 6))
    b = np.zeros(2 * n)
    for i, ((sx, sy), (gx, gy)) in enumerate(zip(svg_points, geo_points)):
        A[2 * i] = [sx, sy, 1.0, 0.0, 0.0, 0.0]
        A[2 * i + 1] = [0.0, 0.0, 0.0, sx, sy, 1.0]
        b[2 * i] = gx
        b[2 * i + 1] = gy

    params, *_ = np.linalg.lstsq(A, b, rcond=None)
    a, b_val, tx, d, e, ty = params
    return [float(a), float(b_val), float(d), float(e), float(tx), float(ty)]


def placeholder_affine_for_viewbox(
    viewbox_width: float,
    viewbox_height: float,
    anchor_lonlat: tuple[float, float] = (29.0301, 41.1083),
    span_deg: tuple[float, float] = (0.0075, 0.0010),
) -> list[float]:
    """Build a synthetic affine that maps the SVG viewBox onto a small
    rectangle near `anchor_lonlat`.

    Useful for end-to-end smoke tests when real control points aren't
    available yet. The default anchor + span roughly matches the bounding
    box of the reference `ARI6_YENI.geojson` file (Istanbul, ~29.03E /
    41.108N) so demo output renders in a recognizable spot on geojson.io.

    NOT a substitute for real control points — coordinates produced with
    this affine are arbitrary.
    """
    if viewbox_width <= 0 or viewbox_height <= 0:
        raise ValueError("viewbox dimensions must be positive")
    lon0, lat0 = anchor_lonlat
    dlon, dlat = span_deg
    # x -> lon: scale dlon / vw, no rotation/shear
    # y -> lat: scale dlat / vh
    a = dlon / viewbox_width
    e = dlat / viewbox_height
    return [a, 0.0, 0.0, e, lon0, lat0]


# -----------------------------------------------------------------------------
# Inc 16: apply transform
# -----------------------------------------------------------------------------

def to_multipolygon(geom: BaseGeometry) -> MultiPolygon | None:
    """Wrap a Shapely Polygon (or pass through MultiPolygon) as a MultiPolygon.

    Returns None if the input isn't a polygonal geometry — those aren't
    representable in the target schema (which uses MultiPolygon for every
    feature, including walls and doors).
    """
    if geom is None or geom.is_empty:
        return None
    if isinstance(geom, MultiPolygon):
        return geom
    if isinstance(geom, Polygon):
        return MultiPolygon([geom])
    return None


# -----------------------------------------------------------------------------
# Inc 17: build FeatureCollection
# -----------------------------------------------------------------------------

def _make_props(
    facil_type: str,
    level_id: str,
    facil_name: str | None = None,
) -> dict:
    """Build the 7-key properties block matching ARI6_YENI's schema."""
    is_clickable = "FALSE" if facil_type in NON_CLICKABLE_TYPES else "TRUE"
    return {
        "id": str(uuid.uuid4()),
        "isClickable": is_clickable,
        "clickId": "{" + str(uuid.uuid4()) + "}",
        "facil_type": facil_type,
        "level_id": str(level_id),
        "facil_name": facil_name,
        "floorLevel": str(level_id),
    }


def _feature_from_polygon(
    geom: BaseGeometry,
    facil_type: str,
    level_id: str,
    matrix: list[float],
    facil_name: str | None = None,
) -> dict | None:
    """Apply affine, wrap as MultiPolygon, build a Feature dict."""
    mp = to_multipolygon(geom)
    if mp is None:
        return None
    transformed = affine_transform(mp, matrix)
    return {
        "type": "Feature",
        "properties": _make_props(facil_type, level_id, facil_name),
        "geometry": json.loads(to_geojson(transformed)),
    }


def build_floor_features(
    sep: SeparatorResult,
    level_id: str,
    matrix: list[float],
    *,
    window_as: str = "Wall",
) -> list[dict]:
    """Build all features for one floor.

    Emits, in this order:
      - merged_walls       -> facil_type "Wall"
      - sep.doors          -> facil_type from RoomPolygon.facil_type ("Door")
      - sep.windows        -> facil_type = `window_as` (default "Wall" to match
                              the target file, which has no Window features)
      - sep.rooms          -> facil_type from classifier (Office, WC, ...)

    Rooms whose facil_type is None are skipped — every emitted feature has
    a real label.
    """
    features: list[dict] = []

    for w in sep.merged_walls:
        f = _feature_from_polygon(w, "Wall", level_id, matrix)
        if f is not None:
            features.append(f)

    for d in sep.doors:
        ftype = d.facil_type or "Door"
        f = _feature_from_polygon(d.geometry, ftype, level_id, matrix)
        if f is not None:
            features.append(f)

    for w in sep.windows:
        f = _feature_from_polygon(w.geometry, window_as, level_id, matrix)
        if f is not None:
            features.append(f)

    for r in sep.rooms:
        if r.facil_type is None:
            continue
        f = _feature_from_polygon(
            r.geometry, r.facil_type, level_id, matrix, r.facil_name,
        )
        if f is not None:
            features.append(f)

    return features


def build_feature_collection(
    features: Iterable[dict],
    name: str = "svg2geojson_export",
) -> dict:
    """Wrap features in a CRS84 FeatureCollection with `name` + `crs`."""
    return {
        "type": "FeatureCollection",
        "name": name,
        "crs": CRS_CRS84,
        "features": list(features),
    }


def export_geojson(
    features: Iterable[dict],
    output_path: Path,
    name: str = "svg2geojson_export",
) -> Path:
    """Write the FeatureCollection to disk as JSON."""
    fc = build_feature_collection(features, name=name)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(fc, f, indent=2)
    return output_path
