"""Inc 10: Separate polygonize_full output into rooms vs walls.

The input SVGs are PARTIAL slices of larger drawings, not whole-building
files. There is no "exterior region" wrapping a building footprint —
every polygon polygonize_full produces is a candidate room, wall, or
opening. The separator assigns each polygon a role and synthesizes a
silhouette of the visible piece.

After Inc 9 the polygon list mixes:
  - Real interior rooms
  - Wall slivers (long thin polygons between two parallel wall edges)
  - Structural columns (small, roughly square pillars — folded into walls)
  - Door/window AABBs (already labeled by the closing stage)
  - Micro-slivers / noise from imperfect noding

Pipeline:
    1.  Split off openings   - door/window AABB matches via IoU
    2.  Min-area filter      - tiny polygons -> walls (noise tail)
    2.5 Classify columns     - small + roughly square -> walls (folded in,
                               so a column doesn't accidentally pass the
                               thickness threshold and become a "room")
    3.  Compute thickness    - 2 * maximum_inscribed_circle radius
    4.  Dynamic threshold    - largest-gap on log10(thickness),
                               Otsu fallback if no clean valley
    5.  Classify             - thickness < threshold -> wall, else -> room
    6.  Merge walls          - connected components by intersection
                               (columns get unioned with adjacent walls here)
    7.  Outline              - unary_union(rooms + walls).exterior_ring
                               (silhouette of the visible slice)

Typical usage:

    from src.separator import separate_polygons
    from src.config import PolygonizationConfig

    result = separate_polygons(poly_result, PolygonizationConfig())
    print(f"{len(result.rooms)} rooms, {len(result.merged_walls)} walls")
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from math import hypot, log10

import numpy as np
from shapely import maximum_inscribed_circle
from shapely.geometry import LineString, MultiPolygon, Polygon
from shapely.geometry.base import BaseGeometry
from shapely.ops import unary_union
from shapely.strtree import STRtree

from src.config import PolygonizationConfig
from src.models import FloorplanPrimitive, RoomPolygon

logger = logging.getLogger(__name__)


@dataclass
class SeparatorStats:
    """Diagnostic information from the separator run."""

    threshold_method: str = "none"  # "largest_gap" | "otsu" | "median" | "manual" | "none"
    thickness_threshold: float = 0.0
    largest_gap_size: float = 0.0  # log10 units
    wall_count: int = 0
    room_count: int = 0
    merged_wall_count: int = 0
    small_wall_count: int = 0  # rejected by min_area filter
    door_count: int = 0
    window_count: int = 0
    column_count: int = 0
    synth_wall_count: int = 0
    hatched_wall_count: int = 0
    hatching_calibrated: bool = False
    locked_room_count: int = 0
    locked_overruled_by_hatching_count: int = 0
    thickness_p10: float = 0.0
    thickness_p50: float = 0.0
    thickness_p90: float = 0.0


@dataclass
class SeparatorResult:
    """Result of the outline / rooms / walls separation.

    Attributes:
        rooms:           RoomPolygons classified as interior rooms. Each
                         carries `contained_semantics` populated by Inc 12
                         (object-bearing rooms locked to room status).
                         Empty dict for threshold-classified rooms.
        walls:           Individual wall slivers (post-classification, pre-merge).
        merged_walls:    Walls unioned by connected component into bigger objects.
        wall_components: For each merged wall, the indices into `walls` it came
                         from. Length matches `merged_walls`.
        doors:           RoomPolygons (facil_type="Door") whose geometry is
                         the cleaned polygon from polygonize_full and whose
                         stroke is inherited from the source door AABB so
                         visualizations can render in the original SVG color.
        windows:         RoomPolygons (facil_type="Window"), same treatment
                         as doors.
        outline:         Silhouette of the visible polygons (None if empty
                         input). For partial-SVG slices this is just the
                         outer boundary of whatever rooms/walls/columns
                         were classified, not a "building footprint".
        stats:           Diagnostic counts and threshold info.
    """

    rooms: list[RoomPolygon] = field(default_factory=list)
    walls: list[Polygon] = field(default_factory=list)
    merged_walls: list[Polygon] = field(default_factory=list)
    wall_components: list[list[int]] = field(default_factory=list)
    doors: list[RoomPolygon] = field(default_factory=list)
    windows: list[RoomPolygon] = field(default_factory=list)
    outline: Polygon | None = None
    stats: SeparatorStats = field(default_factory=SeparatorStats)


def separate_polygons(
    polygons: list[Polygon],
    config: PolygonizationConfig | None = None,
    door_polygons: list[RoomPolygon] | None = None,
    window_polygons: list[RoomPolygon] | None = None,
    wall_lines: list[LineString] | None = None,
    hatching_lines: list[LineString] | None = None,
    classification_primitives: list[FloorplanPrimitive] | None = None,
) -> SeparatorResult:
    """Run the Inc 10 separator on a list of polygons from polygonize_full.

    Args:
        polygons:        Polygons from PolygonizeResult.polygons.
        config:          Polygonization config. Uses defaults if None.
        door_polygons:   Door RoomPolygons from the closing stage (each
                         carries the AABB geometry plus the source primitive's
                         stroke color). A polygon in `polygons` whose IoU
                         with any door AABB exceeds the matching threshold
                         is wrapped as a RoomPolygon with facil_type="Door"
                         and the source's stroke, and added to result.doors.
        window_polygons: Window RoomPolygons, same treatment for result.windows.
        wall_lines:      Original wall LineStrings (semanticId 33/34) BEFORE
                         they were merged with door/window edges in
                         close_openings. If supplied, Stage 5.5 synthesizes
                         thin wall polygons from any LineString that bounds
                         a polygon but isn't already covered by a wall
                         polygon. If None, Stage 5.5 is skipped.
        hatching_lines:  Hatching LineStrings from filter_hatching().removed.
                         A polygon containing a hatch midpoint is a 100%-
                         confidence wall; if the resolved thickness threshold
                         would mislabel any of them as rooms, the threshold
                         is raised so they all land on the wall side.
                         Calibration only ever raises the threshold, never
                         lowers it. None disables calibration.
        classification_primitives:
                         Non-boundary, non-opening primitives (stairs, toilets,
                         sinks, furniture, elevators, etc.). A polygon whose
                         interior contains the centroid of any such primitive
                         is locked as a 100%-room — bypasses noise/column/
                         sliver/threshold filters and goes straight into
                         result.rooms with `contained_semantics` populated.
                         Hatching wins ties: a polygon that is both hatched
                         and object-bearing stays a wall. None disables this.

    Returns:
        SeparatorResult with rooms, walls, merged walls, doors, windows,
        outline, and stats.
    """
    if config is None:
        config = PolygonizationConfig()

    result = SeparatorResult()
    if not polygons:
        return result

    candidates = list(polygons)

    # Stage 1: filter out door/window AABB polygons (the polygonize output
    # contains them when door/window edges were added for room closure).
    # Inputs are partial-SVG slices, so we don't pre-strip an "exterior
    # region" — every polygon is a real candidate.
    doors, windows, candidates = _split_off_openings(
        candidates, door_polygons or [], window_polygons or [],
    )
    result.doors = doors
    result.windows = windows
    result.stats.door_count = len(doors)
    result.stats.window_count = len(windows)


    if not candidates:
        return result

    # Stage 1.5: match hatching strokes to polygons. Hatching lives inside
    # wall thickness, so every hit is a 100%-confidence wall — we use this
    # at Stage 4 to calibrate (raise) the thickness threshold if it would
    # otherwise misclassify a hatched polygon as a room.
    hatched_ids: set[int] = set()
    if hatching_lines:
        hatched_ids = _match_hatching_to_polygons(candidates, hatching_lines)
        result.stats.hatched_wall_count = len(hatched_ids)

    # Stage 1.6: lock object-bearing polygons as 100%-rooms (Inc 12). A
    # polygon containing the centroid of a classification primitive (stairs,
    # toilet, sink, furniture, elevator, ...) is a real room regardless of
    # thickness. Hatching wins ties: an already-hatched polygon stays a wall
    # even if it happens to contain an object centroid.
    object_room_ids: set[int] = set()
    room_semantics: dict[int, dict[int, int]] = {}
    if classification_primitives:
        object_room_ids, room_semantics, overruled = _lock_object_rooms(
            candidates, classification_primitives, hatched_ids,
        )
        result.stats.locked_room_count = len(object_room_ids)
        result.stats.locked_overruled_by_hatching_count = overruled

    # Pull locked rooms out of the candidate stream — they bypass
    # noise/column/sliver/threshold and go straight to result.rooms.
    locked_polys: list[Polygon] = []
    candidates_remaining: list[Polygon] = []
    for p in candidates:
        if id(p) in object_room_ids:
            locked_polys.append(p)
        else:
            candidates_remaining.append(p)

    # Stage 2: min-area filter (small -> walls)
    small_walls: list[Polygon] = []
    sized: list[Polygon] = []
    for p in candidates_remaining:
        if p.area < config.separator_min_area:
            small_walls.append(p)
        else:
            sized.append(p)
    result.stats.small_wall_count = len(small_walls)

    # Stage 2.5: classify interior structural columns. They could otherwise
    # pass the thickness threshold and be misclassified as small rooms; we
    # detect them here and route them straight into the wall bucket so they
    # get merged with adjacent walls in Stage 6 (or stand alone if isolated).
    # column_count is kept in stats as a diagnostic.
    columns, sized = _classify_columns(sized, config)
    result.stats.column_count = len(columns)

    # Stage 3: thickness + min-thickness filter.
    # Polygons below min_thickness are noise slivers; classify as walls and
    # drop them from the dynamic threshold input so the gap detector latches
    # onto the walls-vs-rooms valley, not the noise-vs-walls valley.
    sliver_walls: list[Polygon] = []
    sized_after_thickness: list[Polygon] = []
    sized_thicknesses: list[float] = []
    for p in sized:
        t = _compute_thickness(p)
        if t < config.separator_min_thickness:
            sliver_walls.append(p)
        else:
            sized_after_thickness.append(p)
            sized_thicknesses.append(t)

    # Stage 4: dynamic threshold (on polygons that passed both filters)
    threshold, method, gap_size = _resolve_threshold(sized_thicknesses, config)
    result.stats.largest_gap_size = gap_size

    # Stage 4 calibration: raise threshold if any hatched polygon would
    # otherwise be classified as a room. Hatching is ground truth — those
    # polygons are walls by construction. Calibration is one-directional.
    if hatched_ids:
        hatched_thicknesses = [
            t for p, t in zip(sized_after_thickness, sized_thicknesses)
            if id(p) in hatched_ids
        ]
        if hatched_thicknesses:
            needed = max(hatched_thicknesses) + 1e-6
            if needed > threshold:
                threshold = needed
                method = f"{method}+hatching_calibrated"
                result.stats.hatching_calibrated = True

    result.stats.threshold_method = method
    result.stats.thickness_threshold = threshold

    # Stage 5: classify
    threshold_rooms: list[Polygon] = []
    big_walls: list[Polygon] = []
    for p, t in zip(sized_after_thickness, sized_thicknesses):
        if t < threshold:
            big_walls.append(p)
        else:
            threshold_rooms.append(p)

    # Wrap rooms (threshold-classified + Inc 12 locked) into RoomPolygon.
    # Threshold rooms have no objects inside by construction (any polygon
    # containing an object centroid was pulled out at Stage 1.6), so their
    # contained_semantics is {}.
    rooms_out: list[RoomPolygon] = [
        RoomPolygon(geometry=p, contained_semantics={}) for p in threshold_rooms
    ]
    for p in locked_polys:
        rooms_out.append(RoomPolygon(
            geometry=p,
            contained_semantics=room_semantics.get(id(p), {}),
        ))
    result.rooms = rooms_out
    result.walls = small_walls + sliver_walls + big_walls + columns
    result.stats.room_count = len(result.rooms)
    result.stats.wall_count = len(result.walls)

    # Stage 5.5: synthesize thin walls from single-line wall LineStrings.
    # Some walls in the source SVG are drawn as a single stroke that becomes
    # the shared boundary between two room polygons (no 2D wall sliver
    # forms). We detect those geometrically and buffer them into thin
    # rectangles so the wall map is complete.
    if wall_lines:
        room_geoms = [r.geometry for r in result.rooms if isinstance(r.geometry, Polygon)]
        synth = _synthesize_single_line_walls(
            wall_lines, result.walls, room_geoms, config,
        )
        result.walls.extend(synth)
        result.stats.synth_wall_count = len(synth)
        result.stats.wall_count = len(result.walls)

    # Thickness percentiles cover everything (including slivers) for diagnostics.
    all_thicknesses = sized_thicknesses + [
        _compute_thickness(p) for p in sliver_walls
    ]
    if all_thicknesses:
        arr = np.array(all_thicknesses)
        result.stats.thickness_p10 = float(np.percentile(arr, 10))
        result.stats.thickness_p50 = float(np.percentile(arr, 50))
        result.stats.thickness_p90 = float(np.percentile(arr, 90))

    # Stage 6: merge walls by connected components
    merged, components = _merge_walls_by_components(result.walls)
    result.merged_walls = merged
    result.wall_components = components
    result.stats.merged_wall_count = len(merged)

    # Stage 7: outline (columns are already inside result.walls).
    room_geoms = [r.geometry for r in result.rooms if isinstance(r.geometry, Polygon)]
    result.outline = _compute_outline(room_geoms, result.walls)

    logger.info(
        "Separator: small_walls=%d walls=%d rooms=%d columns_in_walls=%d "
        "merged=%d method=%s threshold=%.3f gap=%.3f",
        result.stats.small_wall_count,
        result.stats.wall_count - result.stats.small_wall_count,
        result.stats.room_count,
        result.stats.column_count,
        result.stats.merged_wall_count,
        method,
        threshold,
        gap_size,
    )
    return result


def _split_off_openings(
    candidates: list[Polygon],
    door_sources: list[RoomPolygon],
    window_sources: list[RoomPolygon],
) -> tuple[list[RoomPolygon], list[RoomPolygon], list[Polygon]]:
    """Provenance-based skip of door/window AABB polygons.

    The door/window AABBs are rectangles WE created in the closing stage.
    polygonize_full produces a polygon for each closed AABB (modulo small
    cleaning/noding shifts), so every output polygon either:
      - sits entirely inside a known AABB (it IS that AABB) -> skip from walls
      - has zero overlap with any AABB                      -> wall/room candidate

    A polygon "came from" an AABB if more than half of its area lies inside
    that AABB. This works because cleaning can shave small slivers off the
    AABB (dropping intersection area from 100% to ~75%) but cannot move the
    polygon outside the AABB. Random walls/rooms have ~0% overlap with any
    AABB, so the 50% boundary lies in a wide empty zone — robust against
    numerical noise.

    Each match wraps the cleaned polygon in a new RoomPolygon, copying the
    source's stroke and tagging facil_type="Door" / "Window" so the
    visualization can render it in the original SVG color.

    Returns:
        (doors, windows, remaining) where doors/windows are RoomPolygons
        and `remaining` is the subset of `candidates` that did not come
        from any opening AABB.
    """
    doors: list[RoomPolygon] = []
    windows: list[RoomPolygon] = []
    remaining: list[Polygon] = []

    if not door_sources and not window_sources:
        return doors, windows, list(candidates)

    door_aabbs = [d.geometry for d in door_sources]
    window_aabbs = [w.geometry for w in window_sources]
    door_tree = STRtree(door_aabbs) if door_aabbs else None
    window_tree = STRtree(window_aabbs) if window_aabbs else None

    def _match(poly: Polygon, aabbs: list, tree: STRtree | None) -> int | None:
        if tree is None or poly.area <= 0:
            return None
        for j in tree.query(poly):
            aabb = aabbs[int(j)]
            if not poly.intersects(aabb):
                continue
            if poly.intersection(aabb).area / poly.area > 0.5:
                return int(j)
        return None

    for p in candidates:
        di = _match(p, door_aabbs, door_tree)
        if di is not None:
            doors.append(RoomPolygon(
                geometry=p, facil_type="Door", stroke=door_sources[di].stroke,
            ))
            continue
        wi = _match(p, window_aabbs, window_tree)
        if wi is not None:
            windows.append(RoomPolygon(
                geometry=p, facil_type="Window", stroke=window_sources[wi].stroke,
            ))
            continue
        remaining.append(p)
    return doors, windows, remaining


def _classify_columns(
    candidates: list[Polygon],
    config: PolygonizationConfig,
) -> tuple[list[Polygon], list[Polygon]]:
    """Split candidates into structural columns vs everything else.

    A polygon is a column when both gates pass:
      - area <= config.separator_column_max_area
      - oriented-bounding-box aspect (short/long) >= config.separator_column_min_aspect

    Aspect is computed from minimum_rotated_rectangle so a 45-deg-rotated
    pillar isn't penalized like an axis-aligned aspect test would be.

    Returns:
        (columns, remaining)
    """
    max_area = config.separator_column_max_area
    min_aspect = config.separator_column_min_aspect
    columns: list[Polygon] = []
    remaining: list[Polygon] = []

    for p in candidates:
        if p.is_empty or p.area <= 0 or p.area > max_area:
            remaining.append(p)
            continue
        mrr = p.minimum_rotated_rectangle
        if not isinstance(mrr, Polygon):
            remaining.append(p)
            continue
        coords = list(mrr.exterior.coords)
        if len(coords) < 4:
            remaining.append(p)
            continue
        e1 = hypot(coords[1][0] - coords[0][0], coords[1][1] - coords[0][1])
        e2 = hypot(coords[2][0] - coords[1][0], coords[2][1] - coords[1][1])
        if e1 <= 0 or e2 <= 0:
            remaining.append(p)
            continue
        short_, long_ = (e1, e2) if e1 <= e2 else (e2, e1)
        if short_ / long_ < min_aspect:
            remaining.append(p)
            continue
        columns.append(p)

    return columns, remaining


def _instance_centroids(
    primitives: list[FloorplanPrimitive],
) -> list[tuple[int, "BaseGeometry"]]:
    """Group primitives by instance_id and return one (semantic_id, centroid)
    pair per instance.

    A single physical object (one toilet, one sofa, one elevator) is drawn
    as many primitives sharing the same instance_id. Per-primitive
    centroids scatter across the object's drawing strokes; the average is
    far more likely to land near the object's true center, which in turn
    is far more likely to be inside the room polygon — so an instance-
    level test is robust where a primitive-level test misses on edge
    pieces.

    Primitives without an instance_id (None or < 0) are treated as
    one-primitive instances so we don't drop them entirely.
    """
    groups: dict[tuple[int, int], list[FloorplanPrimitive]] = {}
    pseudo_id = -2
    for prim in primitives:
        if prim.geometry is None or prim.geometry.is_empty:
            continue
        sem = prim.semantic_id if prim.semantic_id is not None else -1
        if prim.instance_id is None or prim.instance_id < 0:
            key = (pseudo_id, sem)
            pseudo_id -= 1
        else:
            key = (prim.instance_id, sem)
        groups.setdefault(key, []).append(prim)

    out: list[tuple[int, "BaseGeometry"]] = []
    for (_, sem), prims in groups.items():
        try:
            union = unary_union([p.geometry for p in prims])
        except Exception:  # noqa: BLE001
            continue
        if union.is_empty:
            continue
        c = union.centroid
        if c.is_empty:
            continue
        out.append((sem, c))
    return out


def _lock_object_rooms(
    polygons: list[Polygon],
    primitives: list[FloorplanPrimitive],
    hatched_ids: set[int],
) -> tuple[set[int], dict[int, dict[int, int]], int]:
    """Inc 12: lock polygons containing object-instance centroids as
    100%-rooms.

    Tests one centroid per instance (see `_instance_centroids`) against
    polygon interiors with `contains`. Hatching wins ties: a polygon
    already in `hatched_ids` is skipped (stray-object case stays a wall).

    Returns:
        (locked_ids, semantics, overruled_count)
        locked_ids: set of id(polygon) classified as 100%-room.
        semantics: id(polygon) -> {semantic_id: count_of_instances} for
                   Inc 13 to consume. Counts are now per-instance, not
                   per-primitive.
        overruled_count: how many object-bearing polygons were rejected
                         because they were also hatched.
    """
    if not polygons or not primitives:
        return set(), {}, 0

    tree = STRtree(polygons)
    locked: set[int] = set()
    semantics: dict[int, dict[int, int]] = {}
    overruled: set[int] = set()

    for sem, pt in _instance_centroids(primitives):
        for idx in tree.query(pt):
            poly = polygons[int(idx)]
            if not poly.contains(pt):
                continue
            poly_key = id(poly)
            if poly_key in hatched_ids:
                overruled.add(poly_key)
                break
            locked.add(poly_key)
            counts = semantics.setdefault(poly_key, {})
            counts[sem] = counts.get(sem, 0) + 1
            break

    return locked, semantics, len(overruled)


def _match_hatching_to_polygons(
    polygons: list[Polygon],
    hatching: list[LineString],
) -> set[int]:
    """Return id() of polygons containing at least one hatch midpoint.

    Hatching strokes are short (~0.4 units) and drawn strictly inside wall
    thickness, so a single midpoint test is enough to classify whether a
    polygon "is a hatched wall". The STRtree is built on polygon AABBs to
    keep the per-hatch lookup O(log N).
    """
    if not polygons or not hatching:
        return set()

    tree = STRtree(polygons)
    hits: set[int] = set()

    for line in hatching:
        if not isinstance(line, LineString) or line.is_empty:
            continue
        mid = line.interpolate(0.5, normalized=True)
        for idx in tree.query(mid):
            poly = polygons[int(idx)]
            if poly.contains(mid):
                hits.add(id(poly))
                break

    return hits


def _synthesize_single_line_walls(
    wall_lines: list[LineString],
    walls: list[Polygon],
    rooms: list[Polygon],
    config: PolygonizationConfig,
) -> list[Polygon]:
    """Buffer wall LineStrings that bound polygons but aren't already in walls.

    A single-line wall is one stroke acting as the shared boundary between
    two rooms (or the boundary between a room and the visible-slice edge).
    It has no 2D area in polygonize_full's output. We buffer it to create
    a thin wall polygon ~ median_wall_thickness wide so the wall map
    reflects the drawing.

    Cases handled per line:
      (a) inside an existing wall polygon  -> skip (already represented)
      (b) on the exterior of any polygon   -> buffer (single-line wall)
      (c) far from any polygon boundary    -> skip (dangle / unused)
    """
    if not wall_lines:
        return []

    eps = 1e-3
    walls_region = unary_union(walls).buffer(eps) if walls else None
    boundary_pieces: list = []
    for poly in walls:
        boundary_pieces.append(poly.exterior)
    for poly in rooms:
        boundary_pieces.append(poly.exterior)
    if not boundary_pieces:
        return []
    boundary_union = unary_union(boundary_pieces)

    # Buffer radius: 0.2 * median wall thickness (synthesized walls extend
    # radius units into each adjacent room, so total width = 0.4 * median).
    # This keeps single-line walls visually present without eating too far
    # into the rooms they border. Falls back to config default when the
    # sample has no real wall polygons to derive a thickness from.
    radius = 0.0
    if walls:
        ts = [_compute_thickness(w) for w in walls]
        ts = [t for t in ts if t > 0]
        if ts:
            radius = float(np.median(ts)) * 0.2
    if radius <= 0:
        radius = float(config.synth_wall_default_radius)
    if radius <= 0:
        return []

    out: list[Polygon] = []
    for line in wall_lines:
        if not isinstance(line, LineString) or line.is_empty:
            continue
        # Case (a): already covered by a wall polygon -> skip
        if walls_region is not None and walls_region.contains(line):
            continue
        # Case (b): on the boundary of a polygon -> buffer
        # Case (c): far from any polygon boundary (dangle) -> skip
        if line.distance(boundary_union) >= eps:
            continue
        buf = line.buffer(radius, cap_style=2)
        if isinstance(buf, Polygon) and not buf.is_empty:
            out.append(buf)
    return out


def _compute_thickness(poly: Polygon) -> float:
    """Return 2 * inradius via shapely.maximum_inscribed_circle.

    The center of the inscribed disc lies on the medial axis, so 2*radius
    equals the local thickness at the widest point along the centerline.
    For uniform-thickness walls, this is the actual wall thickness.
    """
    if poly.is_empty or poly.area <= 0:
        return 0.0
    # Tolerance scales with polygon size for stable convergence.
    tol = max(0.01, (poly.area ** 0.5) * 0.005)
    try:
        mic = maximum_inscribed_circle(poly, tolerance=tol)
    except Exception:  # noqa: BLE001
        return 0.0
    if mic is None or mic.is_empty:
        return 0.0
    return float(2 * mic.length)


def _resolve_threshold(
    thicknesses: list[float],
    config: PolygonizationConfig,
) -> tuple[float, str, float]:
    """Pick a thickness cutoff between walls and rooms.

    Returns:
        (threshold, method, gap_size).
        method is one of: "manual" | "largest_gap" | "otsu" | "median" | "none".
        gap_size is the largest log10-gap found (0.0 when not applicable).
    """
    if config.separator_manual_thickness is not None:
        return float(config.separator_manual_thickness), "manual", 0.0

    positive = [t for t in thicknesses if t > 0]
    if len(positive) < 2:
        return 0.0, "none", 0.0

    # Primary: largest-gap on log10
    gap_threshold, gap_size = _largest_gap_threshold(positive)
    if gap_threshold is not None and gap_size >= config.separator_min_log_gap:
        return gap_threshold, "largest_gap", gap_size

    # Fallback
    fallback = config.separator_fallback_method
    if fallback == "otsu":
        return _otsu_threshold(positive), "otsu", gap_size
    if fallback == "median":
        return float(np.median(positive)), "median", gap_size
    # Manual fallback that wasn't set up earlier -> use otsu as safety net
    return _otsu_threshold(positive), "otsu", gap_size


def _largest_gap_threshold(values: list[float]) -> tuple[float | None, float]:
    """Find a bimodal-splitting gap in log10(values).

    Each candidate gap is scored by gap_size * min(count_left, count_right),
    so an outlier-driven gap at the extremes (where one side has count=1)
    loses to a smaller gap that actually splits the distribution into two
    balanced groups.

    Returns:
        (threshold, raw_gap_size). threshold is None if no positive gap.
        raw_gap_size is the log10-distance of the chosen gap (not the score),
        so it remains comparable against config.separator_min_log_gap.
    """
    log_vals = sorted(log10(v) for v in values if v > 0)
    n = len(log_vals)
    if n < 2:
        return None, 0.0
    best_score = 0.0
    best_gap = 0.0
    best_mid = log_vals[0]
    for i in range(n - 1):
        gap = log_vals[i + 1] - log_vals[i]
        if gap <= 0:
            continue
        # Mass on each side of the candidate split.
        left = i + 1
        right = n - left
        score = gap * min(left, right)
        if score > best_score:
            best_score = score
            best_gap = gap
            best_mid = (log_vals[i] + log_vals[i + 1]) / 2
    if best_gap <= 0:
        return None, 0.0
    return 10 ** best_mid, best_gap


def _otsu_threshold(values: list[float], nbins: int = 64) -> float:
    """1D Otsu's method on log10(values). Returns linear-space threshold."""
    log_vals = np.log10(np.array([v for v in values if v > 0]))
    if len(log_vals) < 2:
        return 0.0
    hist, edges = np.histogram(log_vals, bins=nbins)
    total = hist.sum()
    if total == 0:
        return 0.0
    bin_mids = (edges[:-1] + edges[1:]) / 2
    sum_total = (hist * bin_mids).sum()
    sum_b = 0.0
    w_b = 0
    max_var = 0.0
    thresh_log = edges[0]
    for i in range(nbins):
        w_b += hist[i]
        if w_b == 0:
            continue
        w_f = total - w_b
        if w_f == 0:
            break
        sum_b += hist[i] * bin_mids[i]
        m_b = sum_b / w_b
        m_f = (sum_total - sum_b) / w_f
        between = w_b * w_f * (m_b - m_f) ** 2
        if between > max_var:
            max_var = between
            thresh_log = edges[i + 1]
    return float(10 ** thresh_log)


def _merge_walls_by_components(
    walls: list[Polygon],
) -> tuple[list[Polygon], list[list[int]]]:
    """Group walls into connected components and union each.

    Two walls are connected if their geometries intersect (touch or overlap).
    Uses an STRtree for an O(N log N) adjacency query. Within each component,
    unary_union merges the touching slivers into one polygon.
    """
    n = len(walls)
    if n == 0:
        return [], []

    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    tree = STRtree(walls)
    for i, w in enumerate(walls):
        # query returns indices of candidate neighbors whose envelopes overlap.
        for j in tree.query(w):
            j = int(j)
            if j <= i:
                continue
            if walls[i].intersects(walls[j]):
                union(i, j)

    components: dict[int, list[int]] = {}
    for i in range(n):
        components.setdefault(find(i), []).append(i)

    merged: list[Polygon] = []
    component_indices: list[list[int]] = []
    for indices in components.values():
        if len(indices) == 1:
            merged.append(walls[indices[0]])
            component_indices.append(list(indices))
            continue
        geom = unary_union([walls[i] for i in indices])
        # MultiPolygon happens when the union-find grouped polygons that touch
        # only at an envelope edge (false neighbor); treat each piece as its
        # own merged wall, with the same source-index list (best-effort).
        if isinstance(geom, MultiPolygon):
            for piece in geom.geoms:
                if isinstance(piece, Polygon):
                    merged.append(piece)
                    component_indices.append(list(indices))
        elif isinstance(geom, Polygon):
            merged.append(geom)
            component_indices.append(list(indices))

    return merged, component_indices


def _compute_outline(
    rooms: list[Polygon],
    walls: list[Polygon],
) -> Polygon | None:
    """Compute the visible-slice silhouette as a single polygon.

    Outline = exterior ring of unary_union(rooms + walls). If the union
    is a MultiPolygon (disconnected pieces), pick the largest. Columns
    are already in `walls` (folded in by Stage 2.5), so they're covered.
    """
    if not rooms and not walls:
        return None
    union: BaseGeometry = unary_union(rooms + walls)
    if union.is_empty:
        return None
    if isinstance(union, MultiPolygon):
        biggest = max(union.geoms, key=lambda g: g.area)
    elif isinstance(union, Polygon):
        biggest = union
    else:
        return None
    # Drop holes; outline is the silhouette (exterior ring only).
    return Polygon(biggest.exterior.coords)
