"""First-pass polygonization of noded boundary lines.

Takes fully noded LineStrings from the cleaning pipeline and runs
shapely.ops.polygonize_full() to extract closed polygon regions.
This is a diagnostic first pass (Inc 9) — wall/room separation
and iterative gap-closing are handled in later increments.

Typical usage:

    from src.polygonizer import polygonize_lines
    from src.config import PolygonizationConfig

    result = polygonize_lines(noded_lines, PolygonizationConfig())
    rooms = result.polygons        # filtered by min_room_area
    gaps = result.dangles          # free-endpoint edges (gap indicators)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from shapely.geometry import LineString, Polygon
from shapely.geometry.base import BaseGeometry
from shapely.ops import polygonize_full

from src.config import PolygonizationConfig

logger = logging.getLogger(__name__)


@dataclass
class PolygonizeStats:
    """Counts and area metrics from polygonization."""

    raw_polygon_count: int = 0
    filtered_polygon_count: int = 0
    removed_small_count: int = 0
    dangle_count: int = 0
    cut_edge_count: int = 0
    invalid_ring_count: int = 0
    total_area: float = 0.0
    min_area: float = 0.0
    max_area: float = 0.0
    total_dangle_length: float = 0.0


@dataclass
class PolygonizeResult:
    """Result of polygonize_full with extracted geometries and stats.

    Attributes:
        polygons:      Polygons with area >= min_room_area.
        dangles:       Edges with at least one free endpoint (gap indicators).
        cut_edges:     Edges connected at both ends but not part of a polygon.
        invalid_rings: Rings that form invalid geometry (bowties, etc.).
        stats:         Counts and area metrics.
    """

    polygons: list[Polygon] = field(default_factory=list)
    dangles: list[LineString] = field(default_factory=list)
    cut_edges: list[LineString] = field(default_factory=list)
    invalid_rings: list[LineString] = field(default_factory=list)
    stats: PolygonizeStats = field(default_factory=PolygonizeStats)


def polygonize_lines(
    lines: list[LineString],
    config: PolygonizationConfig | None = None,
) -> PolygonizeResult:
    """Run polygonize_full on noded lines and filter by minimum area.

    Args:
        lines:  Fully noded LineStrings from clean_geometry().
        config: Polygonization thresholds. Uses defaults if None.

    Returns:
        PolygonizeResult with filtered polygons, byproducts, and stats.
    """
    if config is None:
        config = PolygonizationConfig()

    stats = PolygonizeStats()

    if not lines:
        return PolygonizeResult(stats=stats)

    # polygonize_full returns: (polygons, cut_edges, dangles, invalid_rings)
    polys_gc, cuts_gc, dangles_gc, invalids_gc = polygonize_full(lines)

    raw_polygons = _extract_polygons(polys_gc)
    dangles = _extract_linestrings(dangles_gc)
    cut_edges = _extract_linestrings(cuts_gc)
    invalid_rings = _extract_linestrings(invalids_gc)

    stats.raw_polygon_count = len(raw_polygons)
    stats.dangle_count = len(dangles)
    stats.cut_edge_count = len(cut_edges)
    stats.invalid_ring_count = len(invalid_rings)
    stats.total_dangle_length = sum(d.length for d in dangles)

    # Filter by minimum area
    filtered = [p for p in raw_polygons if p.area >= config.min_room_area]
    stats.filtered_polygon_count = len(filtered)
    stats.removed_small_count = stats.raw_polygon_count - len(filtered)

    if filtered:
        areas = [p.area for p in filtered]
        stats.total_area = sum(areas)
        stats.min_area = min(areas)
        stats.max_area = max(areas)

    logger.info(
        "Polygonize: %d raw -> %d filtered (%d small removed), "
        "%d dangles, %d cuts, %d invalid",
        stats.raw_polygon_count,
        stats.filtered_polygon_count,
        stats.removed_small_count,
        stats.dangle_count,
        stats.cut_edge_count,
        stats.invalid_ring_count,
    )

    return PolygonizeResult(
        polygons=filtered,
        dangles=dangles,
        cut_edges=cut_edges,
        invalid_rings=invalid_rings,
        stats=stats,
    )


def _extract_polygons(gc: BaseGeometry) -> list[Polygon]:
    """Extract Polygon geometries from a GeometryCollection."""
    if gc.is_empty:
        return []
    if gc.geom_type == "Polygon":
        return [gc]
    return [g for g in gc.geoms if g.geom_type == "Polygon"]


def _extract_linestrings(gc: BaseGeometry) -> list[LineString]:
    """Extract LineString geometries from a GeometryCollection."""
    if gc.is_empty:
        return []
    if gc.geom_type == "LineString":
        return [gc]
    return [g for g in gc.geoms if g.geom_type == "LineString"]
