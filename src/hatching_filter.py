"""Remove decorative wall hatching from boundary primitives.

FloorplanCAD wall elements (semanticId={33,34}) include both real wall
edges and decorative cross-hatching fill — short diagonal lines drawn
inside the wall thickness. Without filtering, these create thousands of
micro-triangles during polygonization (audit finding C1).

Hatching is identified by a simple geometric test: short length AND
diagonal orientation. Empirical analysis across 7 samples shows zero
overlap between the hatching cluster (length ~0.4, angle ~45°) and real
wall edges (length ~12, angle ~0°/90°), so a threshold filter achieves
perfect separation.

Typical usage:

    from src.hatching_filter import filter_hatching

    result = parse_svg(svg_path)
    filtered = filter_hatching(result.boundary)
    clean_boundary = filtered.kept  # use for polygonization
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field

from shapely.geometry import LineString

from src.config import HatchingFilterConfig
from src.models import FloorplanPrimitive

logger = logging.getLogger(__name__)


@dataclass
class FilterResult:
    """Result of the hatching filter, separating real edges from noise.

    Attributes:
        kept:          Primitives classified as real wall edges.
        removed:       Primitives classified as decorative hatching.
        total_count:   Total number of input primitives.
        removed_count: Number of primitives removed as hatching.
    """

    kept: list[FloorplanPrimitive] = field(default_factory=list)
    removed: list[FloorplanPrimitive] = field(default_factory=list)
    total_count: int = 0
    removed_count: int = 0


def filter_hatching(
    primitives: list[FloorplanPrimitive],
    config: HatchingFilterConfig | None = None,
) -> FilterResult:
    """Remove decorative hatching segments from a list of boundary primitives.

    Classifies each primitive as either a real wall edge (kept) or
    decorative hatching (removed) based on length and orientation angle.
    Non-LineString geometries (circles, ellipses) are always kept.

    Args:
        primitives: Boundary primitives from ParseResult.boundary.
        config:     Filter thresholds. Uses HatchingFilterConfig defaults
                    if None (min_length=1.5, max_diagonal_angle_deg=20.0).

    Returns:
        FilterResult with kept/removed lists and counts.
    """
    if config is None:
        config = HatchingFilterConfig()

    result = FilterResult(total_count=len(primitives))

    if not config.enabled:
        result.kept = list(primitives)
        logger.info("Hatching filter disabled, keeping all %d primitives", len(primitives))
        return result

    for prim in primitives:
        if _is_hatching(prim, config):
            result.removed.append(prim)
        else:
            result.kept.append(prim)

    result.removed_count = len(result.removed)

    pct = (result.removed_count / result.total_count * 100) if result.total_count > 0 else 0.0
    logger.info(
        "Hatching filter: removed %d/%d boundary primitives (%.1f%%)",
        result.removed_count,
        result.total_count,
        pct,
    )

    return result


def _is_hatching(prim: FloorplanPrimitive, config: HatchingFilterConfig) -> bool:
    """Classify a single primitive as hatching or real wall edge.

    A primitive is hatching if ALL of:
      1. Its geometry is a LineString (circles/ellipses are never hatching).
      2. Its length is below config.min_length.
      3. Its angle to the horizontal axis falls within the diagonal band
         [45 - max_diagonal_angle_deg, 45 + max_diagonal_angle_deg].

    Args:
        prim:   A boundary FloorplanPrimitive.
        config: Thresholds for length and angle.

    Returns:
        True if the primitive is decorative hatching.
    """
    if not isinstance(prim.geometry, LineString):
        return False

    if prim.geometry.length >= config.min_length:
        return False

    angle = _segment_angle(prim.geometry)
    low = 45.0 - config.max_diagonal_angle_deg
    high = 45.0 + config.max_diagonal_angle_deg

    return low <= angle <= high


def _segment_angle(geom: LineString) -> float:
    """Compute the angle of a LineString relative to the horizontal axis.

    Uses the first and last coordinates to determine the overall
    direction, then normalizes to the [0, 90] range. This makes the
    angle axis-symmetric: a segment at 10° and one at 170° both
    return 10°; a segment at 80° and one at 100° both return 80°.

    Args:
        geom: A Shapely LineString.

    Returns:
        Angle in degrees, normalized to [0, 90].
    """
    coords = list(geom.coords)
    x1, y1 = coords[0]
    x2, y2 = coords[-1]
    dx = x2 - x1
    dy = y2 - y1

    if dx == 0.0 and dy == 0.0:
        return 0.0

    angle_rad = math.atan2(abs(dy), abs(dx))
    return math.degrees(angle_rad)
