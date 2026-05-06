"""Inc 15: Absorb arc-door polygons into the rooms they open into.

After Inc 10/13 produce rooms+walls+doors, swing-door polygons (the AABB
around the door slab segments) sit inside the wall corridor and break the
room visualization. For each door whose original primitives included an
arc — i.e. a swing door — find the adjacent room that shares the most
boundary length with the door polygon and merge the door into that room.

Doors without arcs (sliding/folding/etc.) are left untouched. Arc doors
that touch zero room boundary (e.g. between two non-room areas) are
passed through as orphans.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from shapely.geometry import LineString, MultiPolygon, Polygon, box
from shapely.ops import unary_union

from src.models import RoomPolygon

logger = logging.getLogger(__name__)


@dataclass
class AbsorptionResult:
    rooms: list[RoomPolygon]
    doors: list[RoomPolygon]
    absorbed_count: int = 0
    orphan_count: int = 0
    absorbed_door_geoms: list[Polygon] = None  # type: ignore[assignment]

    def __post_init__(self):
        if self.absorbed_door_geoms is None:
            self.absorbed_door_geoms = []


def _split_door_at_room_side(
    door: Polygon | MultiPolygon,
    room: Polygon | MultiPolygon,
    hinge,
) -> tuple[Polygon | MultiPolygon | None, Polygon | MultiPolygon | None]:
    """Split a door AABB into (room_half, wall_half) at its mid-depth.

    The hinge (= arc circle center) sits on the wall side of the door.
    Pick the door long edge that is FARTHEST from the hinge as the
    room-facing side, then cut perpendicular to that edge through the
    door centroid. The half adjacent to the room-facing edge is the
    `room_half`; the other half stays as the wall side.

    Falls back to picking the bbox edge with the longest intersection
    against the room's boundary when no hinge is available.

    Returns (None, door) when nothing absorbable can be inferred.
    """
    minx, miny, maxx, maxy = door.bounds
    cx = (minx + maxx) / 2.0
    cy = (miny + maxy) / 2.0

    # Step 1: pick the edge with the longest intersection against the
    # chosen room's boundary. This gives us an edge that's known to
    # touch a room.
    edges = {
        "top":    LineString([(minx, maxy), (maxx, maxy)]),
        "bottom": LineString([(minx, miny), (maxx, miny)]),
        "left":   LineString([(minx, miny), (minx, maxy)]),
        "right":  LineString([(maxx, miny), (maxx, maxy)]),
    }
    rb = room.boundary
    best_side: str | None = None
    best_len = 0.0
    for side, edge in edges.items():
        try:
            length = edge.intersection(rb).length
        except Exception:
            continue
        if length > best_len:
            best_len = length
            best_side = side
    if best_side is None or best_len <= 0:
        return None, door

    # Step 2: if the chosen edge sits on the hinge side of the door,
    # flip to the opposite side. The hinge (= arc circle center) is on
    # the wall side, so the half adjacent to it is wall, not room.
    if hinge is not None and not hinge.is_empty:
        hx, hy = hinge.x, hinge.y
        opposite = {
            "top": "bottom", "bottom": "top",
            "left": "right", "right": "left",
        }
        on_hinge_side = (
            (best_side == "top"    and hy > cy) or
            (best_side == "bottom" and hy < cy) or
            (best_side == "left"   and hx < cx) or
            (best_side == "right"  and hx > cx)
        )
        if on_hinge_side:
            best_side = opposite[best_side]

    if best_side == "top":
        room_box = box(minx, cy, maxx, maxy)
        wall_box = box(minx, miny, maxx, cy)
    elif best_side == "bottom":
        room_box = box(minx, miny, maxx, cy)
        wall_box = box(minx, cy, maxx, maxy)
    elif best_side == "left":
        room_box = box(minx, miny, cx, maxy)
        wall_box = box(cx, miny, maxx, maxy)
    else:  # right
        room_box = box(cx, miny, maxx, maxy)
        wall_box = box(minx, miny, cx, maxy)

    try:
        room_half = door.intersection(room_box)
        wall_half = door.intersection(wall_box)
    except Exception:
        return None, door
    return room_half, wall_half


def absorb_arc_doors(
    rooms: list[RoomPolygon],
    doors: list[RoomPolygon],
) -> AbsorptionResult:
    """Merge arc-bearing doors into the room with the longest shared boundary.

    Args:
        rooms:  RoomPolygons from the separator (with facil_type already set).
        doors:  RoomPolygons (facil_type="Door"), with `has_arc` populated by
                door_closer.

    Returns:
        AbsorptionResult with possibly-mutated rooms, the remaining doors
        (non-arc + arc orphans), counts, and the geometries that were
        absorbed (for visualization overlay).
    """
    out_rooms = [
        RoomPolygon(
            geometry=r.geometry, facil_type=r.facil_type,
            facil_name=r.facil_name, stroke=r.stroke,
            contained_semantics=dict(r.contained_semantics),
            has_arc=r.has_arc,
        )
        for r in rooms
    ]
    remaining: list[RoomPolygon] = []
    absorbed_geoms: list[Polygon] = []
    absorbed = 0
    orphan = 0

    for door in doors:
        if not door.has_arc or not isinstance(
            door.geometry, (Polygon, MultiPolygon)
        ):
            remaining.append(door)
            continue
        d_boundary = door.geometry.boundary
        if d_boundary.is_empty:
            remaining.append(door)
            continue

        best_idx: int | None = None
        best_len = 0.0
        for i, r in enumerate(out_rooms):
            if not isinstance(r.geometry, (Polygon, MultiPolygon)):
                continue
            if not r.geometry.envelope.intersects(door.geometry):
                continue
            try:
                shared = r.geometry.boundary.intersection(d_boundary)
            except Exception:
                continue
            length = shared.length if not shared.is_empty else 0.0
            if length > best_len:
                best_len = length
                best_idx = i

        if best_idx is None or best_len <= 0:
            orphan += 1
            remaining.append(door)
            continue

        room_half, wall_half = _split_door_at_room_side(
            door.geometry, out_rooms[best_idx].geometry, door.hinge,
        )
        if room_half is None or room_half.is_empty:
            orphan += 1
            remaining.append(door)
            continue

        merged = unary_union([out_rooms[best_idx].geometry, room_half])
        out_rooms[best_idx].geometry = merged
        absorbed_geoms.append(room_half)

        if (
            wall_half is not None
            and not wall_half.is_empty
            and wall_half.area > 1e-3
        ):
            remaining.append(RoomPolygon(
                geometry=wall_half, facil_type="Door",
                stroke=door.stroke, has_arc=False,
            ))
        absorbed += 1

    logger.info(
        "Door absorption: %d arc doors absorbed, %d orphan, %d non-arc passthrough",
        absorbed, orphan, len(remaining) - orphan,
    )

    return AbsorptionResult(
        rooms=out_rooms,
        doors=remaining,
        absorbed_count=absorbed,
        orphan_count=orphan,
        absorbed_door_geoms=absorbed_geoms,
    )
