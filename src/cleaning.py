"""Geometry cleaning pipeline: round, snap, node.

Prepares boundary LineStrings for polygonization by:
  1. Rounding coordinates to configurable precision.
  2. Removing degenerate (zero-length / too-short) segments.
  3. Snapping nearby endpoints via KD-tree + union-find.
  4. Noding at all intersections via unary_union.

Typical usage:

    from src.cleaning import clean_geometry
    from src.config import CleaningConfig

    result = clean_geometry(boundary_primitives, CleaningConfig())
    noded_lines = result.lines  # ready for polygonize()
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
from scipy.spatial import cKDTree
from shapely.geometry import LineString, Polygon
from shapely.ops import unary_union

from src.config import CleaningConfig
from src.models import FloorplanPrimitive

logger = logging.getLogger(__name__)


@dataclass
class CleaningStats:
    """Counts at each cleaning stage for diagnostics."""

    input_count: int = 0
    non_linestring_skipped: int = 0
    after_rounding: int = 0
    degenerates_removed: int = 0
    snap_groups: int = 0
    endpoints_snapped: int = 0
    after_snap: int = 0
    after_noding: int = 0


@dataclass
class CleaningResult:
    """Result of the geometry cleaning pipeline.

    Attributes:
        lines: Fully noded LineStrings ready for polygonize().
        stats: Counts at each pipeline stage.
    """

    lines: list[LineString] = field(default_factory=list)
    stats: CleaningStats = field(default_factory=CleaningStats)


def clean_geometry(
    primitives: list[FloorplanPrimitive],
    config: CleaningConfig | None = None,
) -> CleaningResult:
    """Run the full cleaning pipeline: round -> remove degenerates -> snap -> node.

    Args:
        primitives: Input primitives (typically boundary after hatching filter).
        config:     Cleaning thresholds. Uses CleaningConfig defaults if None.

    Returns:
        CleaningResult with noded lines and per-stage statistics.
    """
    if config is None:
        config = CleaningConfig()

    stats = CleaningStats()

    # Extract LineStrings + convert closed Polygons (e.g. circle/ellipse
    # boundary primitives) to ring LineStrings so they participate in
    # polygonization. A round structural column drawn as <circle> reaches
    # us as a buffered Point (Polygon); without this conversion, walls
    # ending at the column would be dangles and the adjacent room would
    # fail to close. The polygon's exterior ring becomes a closed
    # LineString that polygonize_full can use as a normal wall edge.
    lines: list[LineString] = []
    for p in primitives:
        g = p.geometry
        if isinstance(g, LineString):
            lines.append(g)
        elif isinstance(g, Polygon) and not g.is_empty:
            ring = LineString(g.exterior.coords)
            if not ring.is_empty:
                lines.append(ring)
        else:
            stats.non_linestring_skipped += 1
    stats.input_count = len(lines)

    if not lines:
        return CleaningResult(lines=[], stats=stats)

    # Stage 1: Round coordinates + remove degenerates
    lines, deg_count = _round_and_remove_degenerates(
        lines, config.round_precision, config.min_line_length
    )
    stats.degenerates_removed = deg_count
    stats.after_rounding = len(lines)
    logger.info(
        "Cleaning stage 1: %d -> %d lines (%d degenerates removed)",
        stats.input_count,
        stats.after_rounding,
        deg_count,
    )

    # Stage 2: Snap endpoints
    if config.snap_tolerance > 0 and lines:
        lines, snap_groups, snapped_count = _snap_endpoints(
            lines, config.snap_tolerance, config.round_precision
        )
        stats.snap_groups = snap_groups
        stats.endpoints_snapped = snapped_count

        # Second degenerate pass after snapping
        lines, deg2 = _round_and_remove_degenerates(
            lines, config.round_precision, config.min_line_length
        )
        stats.degenerates_removed += deg2
        stats.after_snap = len(lines)
        logger.info(
            "Cleaning stage 2: %d groups, %d endpoints snapped, %d lines remain",
            snap_groups,
            snapped_count,
            stats.after_snap,
        )
    else:
        stats.after_snap = len(lines)

    # Stage 3: Node at intersections
    if lines:
        lines = _node_lines(lines)
    stats.after_noding = len(lines)
    logger.info(
        "Cleaning stage 3: noding produced %d lines from %d",
        stats.after_noding,
        stats.after_snap,
    )

    return CleaningResult(lines=lines, stats=stats)


# ---------------------------------------------------------------------------
# Stage 1: Rounding + degenerate removal
# ---------------------------------------------------------------------------


def _round_and_remove_degenerates(
    lines: list[LineString],
    precision: int,
    min_length: float,
) -> tuple[list[LineString], int]:
    """Round coordinates and remove degenerate lines.

    Returns:
        (cleaned_lines, degenerates_removed_count)
    """
    cleaned = []
    removed = 0

    for line in lines:
        if line.is_empty:
            removed += 1
            continue

        # Round all coordinates
        rounded_coords = [
            (round(x, precision), round(y, precision))
            for x, y in line.coords
        ]

        # Deduplicate consecutive identical coordinates
        deduped = [rounded_coords[0]]
        for c in rounded_coords[1:]:
            if c != deduped[-1]:
                deduped.append(c)

        # Need at least 2 distinct points for a valid LineString
        if len(deduped) < 2:
            removed += 1
            continue

        new_line = LineString(deduped)
        if new_line.length < min_length:
            removed += 1
            continue

        cleaned.append(new_line)

    return cleaned, removed


# ---------------------------------------------------------------------------
# Stage 2: Endpoint snapping
# ---------------------------------------------------------------------------


class _UnionFind:
    """Disjoint-set with path compression and union by rank."""

    def __init__(self, n: int):
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, x: int, y: int) -> None:
        rx, ry = self.find(x), self.find(y)
        if rx == ry:
            return
        if self.rank[rx] < self.rank[ry]:
            rx, ry = ry, rx
        self.parent[ry] = rx
        if self.rank[rx] == self.rank[ry]:
            self.rank[rx] += 1


def _snap_endpoints(
    lines: list[LineString],
    tolerance: float,
    precision: int,
) -> tuple[list[LineString], int, int]:
    """Snap nearby endpoints using KD-tree + union-find.

    Only first and last coordinates of each LineString participate;
    internal coordinates of multi-point lines are untouched.

    Returns:
        (snapped_lines, snap_group_count, endpoints_snapped_count)
    """
    # Collect all endpoints (with duplicates)
    endpoints = []
    for line in lines:
        coords = list(line.coords)
        endpoints.append(coords[0])
        endpoints.append(coords[-1])

    pts = np.array(endpoints)
    tree = cKDTree(pts)
    pairs = tree.query_pairs(r=tolerance)

    if not pairs:
        return lines, 0, 0

    # Union-find to group transitively connected endpoints
    uf = _UnionFind(len(endpoints))
    for i, j in pairs:
        uf.union(i, j)

    # Group by representative and compute centroids
    from collections import defaultdict

    groups: dict[int, list[int]] = defaultdict(list)
    for i in range(len(endpoints)):
        groups[uf.find(i)].append(i)

    # Build snap map: original coord -> snapped coord
    snap_map: dict[tuple[float, float], tuple[float, float]] = {}
    multi_groups = 0
    total_snapped = 0

    for members in groups.values():
        if len(members) < 2:
            continue
        multi_groups += 1
        # Compute centroid of the group
        cx = sum(endpoints[i][0] for i in members) / len(members)
        cy = sum(endpoints[i][1] for i in members) / len(members)
        centroid = (round(cx, precision), round(cy, precision))
        for i in members:
            orig = (endpoints[i][0], endpoints[i][1])
            if orig != centroid:
                snap_map[orig] = centroid
                total_snapped += 1

    if not snap_map:
        return lines, 0, 0

    # Rebuild LineStrings with snapped endpoints
    snapped_lines = []
    for line in lines:
        coords = list(line.coords)
        start = coords[0]
        end = coords[-1]
        new_start = snap_map.get(start, start)
        new_end = snap_map.get(end, end)
        if new_start != start or new_end != end:
            new_coords = [new_start] + coords[1:-1] + [new_end]
            # Handle 2-point lines where start and end are the only coords
            if len(coords) == 2:
                new_coords = [new_start, new_end]
            snapped_lines.append(LineString(new_coords))
        else:
            snapped_lines.append(line)

    return snapped_lines, multi_groups, total_snapped


# ---------------------------------------------------------------------------
# Stage 3: Noding
# ---------------------------------------------------------------------------


def _node_lines(lines: list[LineString]) -> list[LineString]:
    """Split lines at all intersection points using unary_union.

    Returns fully noded individual LineString segments.
    """
    merged = unary_union(lines)

    # Extract individual LineStrings from the result
    result = []
    if merged.is_empty:
        return result
    if merged.geom_type == "LineString":
        result.append(merged)
    elif merged.geom_type == "MultiLineString":
        result.extend(merged.geoms)
    elif merged.geom_type == "GeometryCollection":
        for geom in merged.geoms:
            if geom.geom_type == "LineString":
                result.append(geom)

    return result
