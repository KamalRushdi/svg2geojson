"""Close door and window gaps in the boundary set.

FloorplanCAD walls stop at door/window openings, leaving gaps that prevent
polygonize() from forming closed room loops. This module constructs a
closing rectangle for each door/window instance, then adds the rectangle
edges to the boundary set.

The closing rectangles also serve as Door/Window polygon features in the
final GeoJSON output.

Algorithm per door/window instance:
  1. Build an axis-aligned bounding rectangle from all primitive endpoints.
  2. Filter mega-instances by area ratio (rect_area / total_area).
  3. For each of the 4 rectangle edges, search for coincident wall
     segments (parallel + close).
  4. Snap matching edges to the wall lines → correct wall thickness.
  5. Edges without matches are the opening sides → keep as-is.

Typical usage:

    from src.door_closer import close_openings

    closing = close_openings(result.doors, result.windows, boundary)
    sealed_boundary = boundary + closing.door_edges + closing.window_edges
"""

from __future__ import annotations

import logging
import math
from collections import defaultdict
from dataclasses import dataclass, field

from shapely.geometry import LineString, Polygon, box

from src.config import DoorClosingConfig
from src.models import FloorplanPrimitive, RoomPolygon

logger = logging.getLogger(__name__)


@dataclass
class ClosingResult:
    """Result of door/window gap closing.

    Attributes:
        door_edges:       Synthetic LineString edges (4 per door rect) to add
                          to the boundary set for polygonization.
        window_edges:     Same for window rectangles.
        door_polygons:    One RoomPolygon per door instance (facil_type="Door").
        window_polygons:  One RoomPolygon per window instance.
        door_count:       Number of door instances successfully processed.
        window_count:     Number of window instances successfully processed.
        skipped_instances: List of (instance_id, reason) for failed instances.
    """

    door_edges: list[FloorplanPrimitive] = field(default_factory=list)
    window_edges: list[FloorplanPrimitive] = field(default_factory=list)
    door_polygons: list[RoomPolygon] = field(default_factory=list)
    window_polygons: list[RoomPolygon] = field(default_factory=list)
    door_count: int = 0
    window_count: int = 0
    skipped_instances: list[tuple[int, str]] = field(default_factory=list)


def close_openings(
    doors: list[FloorplanPrimitive],
    windows: list[FloorplanPrimitive],
    boundary: list[FloorplanPrimitive],
    config: DoorClosingConfig | None = None,
) -> ClosingResult:
    """Close door and window gaps by constructing wall-snapped rectangles.

    For each door/window instance (grouped by instance_id), builds a
    bounding rectangle, snaps its edges to nearby wall segments, and
    extracts the result as synthetic boundary primitives.

    Args:
        doors:    Door primitives from ParseResult.doors.
        windows:  Window primitives from ParseResult.windows.
        boundary: Boundary primitives (walls) for wall-snap.
        config:   Closing thresholds. Uses DoorClosingConfig defaults if None.

    Returns:
        ClosingResult with closing edges, door/window polygons, and counts.
    """
    if config is None:
        config = DoorClosingConfig()

    result = ClosingResult()

    if not config.enabled:
        logger.info("Door/window closing disabled, skipping")
        return result

    # Collect wall LineStrings for wall-snap
    wall_lines = [
        p.geometry for p in boundary if isinstance(p.geometry, LineString)
    ]

    # Compute total building area for mega-instance filtering
    total_area = _compute_total_area(wall_lines)

    # Process doors
    door_instances = _group_by_instance(doors)
    for iid, prims in door_instances.items():
        rect = _build_wall_snapped_rect(prims, wall_lines, total_area, config)
        if rect is None:
            result.skipped_instances.append((iid, "could not build rectangle"))
            continue

        edges = _extract_rect_edges(rect, iid, prims[0].semantic_id)
        result.door_edges.extend(edges)
        result.door_polygons.append(
            RoomPolygon(geometry=rect, facil_type="Door")
        )
        result.door_count += 1

    # Process windows
    window_instances = _group_by_instance(windows)
    for iid, prims in window_instances.items():
        rect = _build_wall_snapped_rect(prims, wall_lines, total_area, config)
        if rect is None:
            result.skipped_instances.append((iid, "could not build rectangle"))
            continue

        edges = _extract_rect_edges(rect, iid, prims[0].semantic_id)
        result.window_edges.extend(edges)
        result.window_polygons.append(
            RoomPolygon(geometry=rect, facil_type="Window")
        )
        result.window_count += 1

    logger.info(
        "Door closing: %d doors (%d edges), %d windows (%d edges), %d skipped",
        result.door_count,
        len(result.door_edges),
        result.window_count,
        len(result.window_edges),
        len(result.skipped_instances),
    )

    return result


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _compute_total_area(wall_lines: list[LineString]) -> float:
    """Compute total bounding area of all wall segments."""
    if not wall_lines:
        return 1.0
    all_coords = []
    for ls in wall_lines:
        all_coords.extend(ls.coords)
    if not all_coords:
        return 1.0
    xs = [c[0] for c in all_coords]
    ys = [c[1] for c in all_coords]
    return (max(xs) - min(xs)) * (max(ys) - min(ys))


def _group_by_instance(
    primitives: list[FloorplanPrimitive],
) -> dict[int, list[FloorplanPrimitive]]:
    """Group primitives by instance_id, skipping None/-1."""
    groups: dict[int, list[FloorplanPrimitive]] = defaultdict(list)
    for p in primitives:
        if p.instance_id is not None and p.instance_id != -1:
            groups[p.instance_id].append(p)
    return dict(groups)


# ---------------------------------------------------------------------------
# Per-instance rectangle construction
# ---------------------------------------------------------------------------


def _build_wall_snapped_rect(
    prims: list[FloorplanPrimitive],
    wall_lines: list[LineString],
    total_area: float,
    config: DoorClosingConfig,
) -> Polygon | None:
    """Build a closing rectangle for one door/window instance.

    Pipeline:
      1. Axis-aligned bounding box from all primitive endpoints.
      2. Mega-instance filter (area ratio).
      3. Search all 4 edges for coincident wall segments.
      4. Snap matching edges to wall lines.
    """
    # Step 1: Collect endpoints from segment primitives only (exclude arcs).
    # Arcs are swing arms that extend far into the room; segments are
    # frame details within/near the wall thickness.
    pts = []
    for p in prims:
        if isinstance(p.geometry, LineString) and p.original_type == "segment":
            pts.extend(list(p.geometry.coords))
    # Fallback to all primitives if no segments
    if len(pts) < 2:
        pts = []
        for p in prims:
            if isinstance(p.geometry, LineString):
                pts.extend(list(p.geometry.coords))

    if len(pts) < 2:
        return None

    xs = [c[0] for c in pts]
    ys = [c[1] for c in pts]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)

    width = max_x - min_x
    height = max_y - min_y

    if width < config.min_rect_opening and height < config.min_rect_opening:
        return None

    # Step 2: Mega-instance filter
    rect_area = width * height
    if total_area > 0 and rect_area / total_area > config.max_area_ratio:
        logger.debug(
            "Skipping mega-instance: area ratio %.3f > %.3f",
            rect_area / total_area,
            config.max_area_ratio,
        )
        return None

    # Build initial axis-aligned rectangle
    rect = box(min_x, min_y, max_x, max_y)

    # Step 3 & 4: Search edges for coincident walls and snap
    rect = _snap_to_walls(rect, wall_lines, config)

    if rect is None or not rect.is_valid or rect.area < 0.01:
        return None

    # Final validation
    short, long = _rect_dimensions(rect)
    if short < config.min_rect_thickness:
        return None

    return rect


def _snap_to_walls(
    rect: Polygon,
    wall_lines: list[LineString],
    config: DoorClosingConfig,
) -> Polygon | None:
    """Snap rectangle to wall face pair with the smallest gap.

    Finds horizontal and vertical wall segments in/near the bounding box,
    collects their positions, and finds the pair of parallel wall lines
    with the smallest gap (= wall thickness). Snaps the corresponding
    axis to this pair.

    Only the across-wall axis is snapped; the along-wall axis (opening
    width) is kept from the original bounding box.
    """
    min_x, min_y, max_x, max_y = rect.bounds
    w = max_x - min_x
    h = max_y - min_y
    angle_tol = math.radians(config.wall_angle_tolerance)

    # Search area: expand box by 50% to catch walls adjacent to opening
    margin = max(w, h) * 0.5
    sx0, sy0 = min_x - margin, min_y - margin
    sx1, sy1 = max_x + margin, max_y + margin

    horiz_ys = []  # y-positions of horizontal walls in/near box
    vert_xs = []   # x-positions of vertical walls in/near box

    for wl in wall_lines:
        wl_c = list(wl.coords)
        p1, p2 = wl_c[0], wl_c[-1]
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        wl_len = math.sqrt(dx * dx + dy * dy)
        if wl_len < 0.01:
            continue

        wl_angle = math.atan2(abs(dy), abs(dx))
        is_horiz = wl_angle < angle_tol
        is_vert = wl_angle > math.pi / 2 - angle_tol

        if is_horiz:
            wl_y = (p1[1] + p2[1]) / 2
            # Wall must be within expanded search area (y)
            if wl_y < sy0 or wl_y > sy1:
                continue
            # Wall must be near the box laterally (x)
            wl_xmin = min(p1[0], p2[0])
            wl_xmax = max(p1[0], p2[0])
            if wl_xmax < sx0 or wl_xmin > sx1:
                continue
            horiz_ys.append(wl_y)

        elif is_vert:
            wl_x = (p1[0] + p2[0]) / 2
            if wl_x < sx0 or wl_x > sx1:
                continue
            wl_ymin = min(p1[1], p2[1])
            wl_ymax = max(p1[1], p2[1])
            if wl_ymax < sy0 or wl_ymin > sy1:
                continue
            vert_xs.append(wl_x)

    # Find wall face pair: two parallel lines with smallest gap
    horiz_pair = _find_wall_face_pair(horiz_ys, min_y, max_y)
    vert_pair = _find_wall_face_pair(vert_xs, min_x, max_x)

    # Apply the pair that gives the thinnest result (= wall thickness)
    if horiz_pair is not None and vert_pair is not None:
        h_gap = horiz_pair[1] - horiz_pair[0]
        v_gap = vert_pair[1] - vert_pair[0]
        if h_gap <= v_gap:
            return box(min_x, horiz_pair[0], max_x, horiz_pair[1])
        else:
            return box(vert_pair[0], min_y, vert_pair[1], max_y)
    elif horiz_pair is not None:
        return box(min_x, horiz_pair[0], max_x, horiz_pair[1])
    elif vert_pair is not None:
        return box(vert_pair[0], min_y, vert_pair[1], max_y)

    return rect


def _find_wall_face_pair(
    positions: list[float],
    box_min: float,
    box_max: float,
) -> tuple[float, float] | None:
    """Find the pair of wall lines with smallest gap within the box range.

    Wall faces form pairs (inner + outer face of a wall). This finds the
    tightest such pair within the bounding box range.

    Returns (low, high) positions or None if no valid pair found.
    """
    if len(positions) < 2:
        return None

    # Deduplicate: round to 0.1 and take unique values
    rounded = sorted(set(round(p, 1) for p in positions))
    if len(rounded) < 2:
        return None

    # Find consecutive pair with smallest gap that's within the box
    best_pair = None
    best_gap = float("inf")
    for i in range(len(rounded) - 1):
        low, high = rounded[i], rounded[i + 1]
        gap = high - low
        # Gap must be reasonable wall thickness (0.3 to 5.0 units)
        if gap < 0.3 or gap > 5.0:
            continue
        # Pair must be within or near the box range
        if high < box_min - 1.0 or low > box_max + 1.0:
            continue
        if gap < best_gap:
            best_gap = gap
            best_pair = (low, high)

    return best_pair


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------


def _angle_diff(a: float, b: float) -> float:
    """Signed angle difference in [-pi, pi]."""
    d = a - b
    while d > math.pi:
        d -= 2 * math.pi
    while d < -math.pi:
        d += 2 * math.pi
    return d


def _rect_dimensions(rect: Polygon) -> tuple[float, float]:
    """Return (short_edge, long_edge) lengths of a rectangle polygon."""
    coords = list(rect.exterior.coords[:-1])
    if len(coords) < 4:
        return 0.0, 0.0
    edge_lengths = [
        LineString([coords[i], coords[(i + 1) % len(coords)]]).length
        for i in range(len(coords))
    ]
    return min(edge_lengths), max(edge_lengths)


# ---------------------------------------------------------------------------
# Edge extraction
# ---------------------------------------------------------------------------


def _extract_rect_edges(
    rect: Polygon,
    instance_id: int,
    semantic_id: int | None,
) -> list[FloorplanPrimitive]:
    """Extract 4 LineString edges from a rectangle Polygon.

    Returns synthetic FloorplanPrimitive objects that can be added
    directly to the boundary set.
    """
    coords = list(rect.exterior.coords)  # 5 points (closed ring)
    edges = []
    for i in range(len(coords) - 1):
        edge_geom = LineString([coords[i], coords[i + 1]])
        edges.append(
            FloorplanPrimitive(
                geometry=edge_geom,
                semantic_id=semantic_id,
                instance_id=instance_id,
                primitive_id=-1,
                original_type="synthetic_closing",
                layer="door_closing",
                stroke="",
            )
        )
    return edges
