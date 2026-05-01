"""Inc 13 + Inc 14: room type classification.

Inc 13 (rule-based, `classify_rooms`):
    Reads the `contained_semantics` map populated by Inc 12 (in the
    separator) and assigns `facil_type` using a fixed priority list.
    Priority encodes "functional rooms beat generic rooms": a stair
    landing that happens to contain a chair is still a stair landing, a
    WC with a sink stays a WC, etc. v1 uses presence-is-enough (one
    toilet -> WC).

Inc 14 (geometric heuristic, `classify_by_geometry`):
    Picks up rooms still unclassified after Inc 13 — they have no
    detectable objects inside, so they're classified by SHAPE. We use
    the short/long aspect ratio of the minimum_rotated_rectangle (OBB):
    above-threshold (compact / square-ish) -> Office, below-threshold
    (elongated) -> Walkways.
"""

from __future__ import annotations

from shapely.geometry import Polygon
from shapely.ops import unary_union

from src.models import (
    ELEVATOR_IDS,
    ESCALATOR_IDS,
    FURNITURE_IDS,
    KITCHEN_IDS,
    PARKING_IDS,
    ROW_CHAIR_IDS,
    STAIR_IDS,
    TOILET_IDS,
    RoomPolygon,
)
from src.separator import _compute_thickness

# (label, semantic_id_set) — order is the priority. First match wins.
PRIORITY: list[tuple[str, set[int]]] = [
    ("Stairs",     STAIR_IDS),
    ("Escalator",  ESCALATOR_IDS),
    ("Elevator",   ELEVATOR_IDS),
    ("WC",         TOILET_IDS),
    ("Kitchen",    KITCHEN_IDS),
    ("Auditorium", ROW_CHAIR_IDS),
    ("Office",     FURNITURE_IDS),
    ("Parking",    PARKING_IDS),
]

# Display palette aligned with the priority list. Each room-type color is
# derived from the SVG stroke of its dominant semantic primitive (sampled
# across all 5 reference floorplans), so a "Stairs" room renders in the
# same yellow the stair primitive is drawn in. For room types that combine
# multiple semantic IDs (WC, Office), we pick the most recognizable / most
# common single ID as the canonical color.
TYPE_COLORS: dict[str, str] = {
    "Stairs":     "#f7ce4b",  # id 28 stair yellow
    "Escalator":  "#f7ce4b",  # not in data; reuse stair yellow (same family)
    "Elevator":   "#ed702d",  # id 29 elevator orange
    "WC":         "#ee7ca2",  # id 27 toilet pink (canonical for 25/26/27)
    "Kitchen":    "#719752",  # id 19 sink green
    "Auditorium": "#426b51",  # not in data; reuse id 13 chair dark green
    "Office":     "#7ab591",  # id 11 sofa light green (canonical for 11/13/14/17)
    "Parking":    "#66433e",  # id 32 parking brown
    # Non-priority fallbacks reserved for Inc 14:
    "Walkways":   "#e5e7eb",  # smoked white (cool off-white)
    "Deadzone":   "#cbd5e1",  # light slate
}


def classify_rooms(rooms: list[RoomPolygon]) -> None:
    """Assign `facil_type` in place using the priority rules.

    Rooms with no objects inside (empty `contained_semantics`) keep their
    existing `facil_type` (typically None) so Inc 14 can decide later.
    Rooms whose only contained semantics are railings (or other ids not in
    the priority list) get `facil_type="Deadzone"` — they're locked as
    rooms but not classifiable from objects alone.
    """
    for r in rooms:
        if not r.contained_semantics:
            continue
        for label, ids in PRIORITY:
            if any(sem_id in ids for sem_id in r.contained_semantics):
                r.facil_type = label
                break
        else:
            r.facil_type = "Deadzone"


def reclassify_kitchen_near_wc(
    rooms: list[RoomPolygon],
    buffer_distance: float = 5.0,
) -> None:
    """Reclassify Kitchen rooms adjacent to a WC as WC.

    A "Kitchen" assigned by Inc 13 just means the room contains a sink
    primitive. Sinks adjacent to WCs are usually handwashing sinks, not
    kitchen sinks — so any Kitchen-classified polygon within
    `buffer_distance` of a WC polygon gets reclassified to WC.

    Buffer default 5.0 SVG units ≈ one interior wall thickness, so the
    check reaches across a typical wall to the neighboring polygon.
    """
    wc_geoms = [
        r.geometry for r in rooms
        if r.facil_type == "WC" and isinstance(r.geometry, Polygon)
    ]
    if not wc_geoms:
        return
    wc_buffered = unary_union([g.buffer(buffer_distance) for g in wc_geoms])
    for r in rooms:
        if r.facil_type != "Kitchen":
            continue
        if not isinstance(r.geometry, Polygon):
            continue
        if r.geometry.intersects(wc_buffered):
            r.facil_type = "WC"


def classify_by_geometry(
    rooms: list[RoomPolygon],
    walkway_max_thickness: float = 14.0,
    walkway_max_solidity: float = 0.8,
) -> None:
    """Inc 14: assign facil_type to rooms that Inc 13 left unclassified.

    Two signals decide Walkway vs Office:

    1. **Thickness** = 2 * maximum_inscribed_circle radius. Width of the
       narrowest passage through the polygon. Below
       `walkway_max_thickness` (default 14.0 SVG units) -> Walkway.

    2. **Solidity** = polygon area / convex hull area. Catches L/U/T-shaped
       rooms even when their thickness is large: a rectangle has solidity
       ~1.0, a typical L-corridor ~0.7, a U-shape ~0.6. Below
       `walkway_max_solidity` (default 0.8) -> Walkway. The L-shape signal
       is what tells us "this is a corridor that turns a corner".

    Either signal triggers Walkway; otherwise Office. Tune both args if
    the splits look off on a particular sample.
    """
    for r in rooms:
        if r.facil_type is not None:
            continue
        if not isinstance(r.geometry, Polygon) or r.geometry.is_empty:
            continue
        thickness = _compute_thickness(r.geometry)
        if thickness <= 0:
            continue
        hull = r.geometry.convex_hull
        solidity = (
            r.geometry.area / hull.area
            if isinstance(hull, Polygon) and hull.area > 0
            else 1.0
        )
        is_walkway = (
            thickness < walkway_max_thickness
            or solidity < walkway_max_solidity
        )
        r.facil_type = "Walkways" if is_walkway else "Office"


def classification_summary(rooms: list[RoomPolygon]) -> dict[str, int]:
    """Return {facil_type: count} for diagnostic logging."""
    out: dict[str, int] = {}
    for r in rooms:
        key = r.facil_type or "Unclassified"
        out[key] = out.get(key, 0) + 1
    return out
