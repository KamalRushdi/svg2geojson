"""Parse FloorplanCAD-annotated SVG files into FloorplanPrimitive lists.

Reads SVG files containing <path>, <circle>, and <ellipse> elements with
custom FloorplanCAD attributes (semanticId, instanceId, primitiveId,
originalType, layer, stroke). Converts each element to a Shapely geometry
wrapped in a FloorplanPrimitive dataclass.

Supports four SVG primitive types:
  - Segments:  <path d="M x,y L x,y">              -> LineString (2 points)
  - Arcs:      <path d="M x,y A rx,ry rot l,s x,y"> -> LineString (discretized)
  - Circles:   <circle cx cy r>                     -> Polygon (buffered point)
  - Ellipses:  <ellipse cx cy rx ry>                -> Polygon (buffered point)

Public API:
    parse_svg(svg_path, config) -> ParseResult
"""

from __future__ import annotations

import logging
import math
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path

from shapely.geometry import LineString, Point

from src.config import ParsingConfig
from src.models import FloorplanPrimitive

logger = logging.getLogger(__name__)

# SVG namespace — ElementTree prefixes all tags with this.
_SVG_NS = "{http://www.w3.org/2000/svg}"

# Regex for segment paths: "M x1,y1 L x2,y2"
# Handles scientific notation (e.g. 5.82e-11), comma or space separators.
_SEGMENT_RE = re.compile(
    r"M\s*([-\d.eE+]+)[,\s]+([-\d.eE+]+)\s*"
    r"L\s*([-\d.eE+]+)[,\s]+([-\d.eE+]+)"
)

# Regex for arc paths: "M x1,y1 A rx,ry rotation large_arc_flag,sweep_flag x2,y2"
_ARC_RE = re.compile(
    r"M\s*([-\d.eE+]+)[,\s]+([-\d.eE+]+)\s*"
    r"A\s*([-\d.eE+]+)[,\s]+([-\d.eE+]+)\s+"
    r"([-\d.eE+]+)\s+"
    r"([01])[,\s]+([01])\s+"
    r"([-\d.eE+]+)[,\s]+([-\d.eE+]+)"
)


@dataclass
class ParseResult:
    """Container for parsed and grouped FloorplanPrimitives.

    Returned by parse_svg(). Primitives are grouped by their semantic role
    based on semanticId values and the ParsingConfig ID sets.

    Attributes:
        all_primitives:  Every successfully parsed primitive, including those
                         without a semanticId.
        boundary:        Primitives whose semantic_id is in the boundary set
                         (walls + curtain walls, default {33, 34}).
        doors:           Primitives whose semantic_id is in the door set.
        windows:         Primitives whose semantic_id is in the window set.
        classification:  All other primitives that have a semantic_id but
                         don't belong to boundary/door/window groups.
        skipped_count:   Number of elements without a semanticId attribute.
        viewbox_height:  Height from the SVG viewBox attribute, needed for
                         Y-flip and downstream coordinate reference.
    """

    all_primitives: list[FloorplanPrimitive] = field(default_factory=list)
    boundary: list[FloorplanPrimitive] = field(default_factory=list)
    doors: list[FloorplanPrimitive] = field(default_factory=list)
    windows: list[FloorplanPrimitive] = field(default_factory=list)
    classification: list[FloorplanPrimitive] = field(default_factory=list)
    skipped_count: int = 0
    viewbox_height: float = 0.0


def parse_svg(
    svg_path: Path,
    config: ParsingConfig | None = None,
) -> ParseResult:
    """Parse an SVG file into classified groups of FloorplanPrimitives.

    Reads all <path>, <circle>, and <ellipse> elements from the SVG,
    extracts their geometries and FloorplanCAD metadata attributes, and
    groups them by semantic role (boundary, door, window, classification).

    Args:
        svg_path: Path to the SVG file.
        config:   Parsing configuration (semantic ID sets, Y-flip flag,
                  arc resolution). Uses ParsingConfig defaults if None.

    Returns:
        ParseResult with primitives grouped by semantic role.

    Raises:
        FileNotFoundError: If svg_path does not exist.
        ValueError: If the SVG root element has no viewBox attribute.
    """
    if config is None:
        config = ParsingConfig()

    tree = ET.parse(svg_path)
    root = tree.getroot()

    vb = _parse_viewbox(root)
    viewbox_height = vb[3]

    result = ParseResult(viewbox_height=viewbox_height)

    # --- Parse <path> elements (segments, arcs) ---
    for elem in root.iter(f"{_SVG_NS}path"):
        prim = _parse_path_element(elem, arc_resolution=config.arc_resolution)
        if prim is not None:
            result.all_primitives.append(prim)

    # --- Parse <circle> elements ---
    for elem in root.iter(f"{_SVG_NS}circle"):
        prim = _parse_circle(elem)
        if prim is not None:
            result.all_primitives.append(prim)

    # --- Parse <ellipse> elements ---
    for elem in root.iter(f"{_SVG_NS}ellipse"):
        prim = _parse_ellipse(elem)
        if prim is not None:
            result.all_primitives.append(prim)

    # --- Translate to origin if viewBox has a non-zero offset ---
    # Working-plan SVGs come with georeferenced viewBoxes, e.g.
    #   viewBox="850192 3924494 4200 4200"  (HKÜ — UTM-ish)
    # which puts coordinates in the millions and breaks downstream
    # tolerances/heuristics (snap_tolerance, min_line_length, etc.) that
    # assume drawing-unit-scale numbers near origin. Shift all primitives
    # by -(vb_x, vb_y) so data lives in [0, W] × [0, H], then continue
    # with the standard y-flip on the height-W viewBox.
    if vb[0] != 0.0 or vb[1] != 0.0:
        result.all_primitives = [
            _translate(p, -vb[0], -vb[1]) for p in result.all_primitives
        ]
        vb = (0.0, 0.0, vb[2], vb[3])

    # --- Y-flip: SVG Y-down to geometry Y-up ---
    if config.y_flip:
        result.all_primitives = [
            _flip_y(p, viewbox_height) for p in result.all_primitives
        ]

    # --- Snap wall endpoints close to the SVG boundary onto it ---
    # Wall segments often end slightly short of the SVG edge (e.g., 0.3
    # units from y=140). Without this snap, those near-misses prevent
    # polygons from closing against the boundary rectangle.
    result.all_primitives = _snap_to_svg_boundary(
        result.all_primitives, vb, viewbox_height, config.y_flip,
        tolerance=config.boundary_snap_tolerance,
    )

    # --- Add boundary rectangle (4 LineStrings) ---
    boundary_prims = _create_boundary_rectangle(vb, viewbox_height, config.y_flip)
    result.all_primitives.extend(boundary_prims)

    # --- Group by semantic role ---
    for prim in result.all_primitives:
        if prim.semantic_id is None:
            result.skipped_count += 1
            continue
        if prim.semantic_id in config.boundary_semantic_ids:
            result.boundary.append(prim)
        elif prim.semantic_id in config.door_semantic_ids:
            result.doors.append(prim)
        elif prim.semantic_id in config.window_semantic_ids:
            result.windows.append(prim)
        else:
            result.classification.append(prim)

    logger.info(
        "Parsed %d primitives from %s "
        "(boundary=%d, doors=%d, windows=%d, classification=%d, skipped=%d)",
        len(result.all_primitives),
        svg_path.name,
        len(result.boundary),
        len(result.doors),
        len(result.windows),
        len(result.classification),
        result.skipped_count,
    )

    return result


def _parse_viewbox(root: ET.Element) -> tuple[float, float, float, float]:
    """Extract the viewBox dimensions from the SVG root element.

    The viewBox defines the SVG coordinate space. Its height is needed for
    the Y-flip transformation (SVG Y-down to geometry Y-up).

    Args:
        root: The root <svg> XML Element.

    Returns:
        Tuple of (min_x, min_y, width, height) as floats.

    Raises:
        ValueError: If the viewBox attribute is missing or doesn't have
                    exactly 4 numeric values.
    """
    vb_attr = root.get("viewBox")
    if vb_attr is None:
        raise ValueError("SVG root element has no viewBox attribute")

    parts = vb_attr.split()
    if len(parts) != 4:
        raise ValueError(f"viewBox has {len(parts)} values, expected 4: '{vb_attr}'")

    return tuple(float(p) for p in parts)  # type: ignore[return-value]


def _extract_attributes(elem: ET.Element) -> dict:
    """Extract FloorplanCAD custom attributes from an SVG element.

    Reads the annotation attributes that the FloorplanCAD export tool adds
    to every SVG element. Handles missing attributes gracefully.

    Args:
        elem: An XML Element (<path>, <circle>, or <ellipse>).

    Returns:
        Dict with keys:
            semantic_id   (int | None) — FloorplanCAD class, None if missing
            instance_id   (int | None) — object grouping, None if attr missing,
                                         -1 kept as int (means "uncountable")
            primitive_id  (int)        — unique element ID within the file
            original_type (str)        — "segment", "arc", "circle", or "ellipse"
            layer         (str)        — CAD layer name (for debug only)
            stroke        (str)        — CSS rgb color string

    Raises:
        KeyError: If primitiveId is missing (element is skipped by caller).
    """
    pid_raw = elem.get("primitiveId")
    if pid_raw is None:
        raise KeyError("Element has no primitiveId attribute")

    sem_raw = elem.get("semanticId")
    inst_raw = elem.get("instanceId")

    return {
        "semantic_id": int(sem_raw) if sem_raw is not None else None,
        "instance_id": int(inst_raw) if inst_raw is not None else None,
        "primitive_id": int(pid_raw),
        "original_type": elem.get("originalType", ""),
        "layer": elem.get("layer", ""),
        "stroke": elem.get("stroke", ""),
    }


def _parse_segment(d_attr: str) -> LineString | None:
    """Parse an SVG path d attribute containing M/L commands into a LineString.

    Handles the simple two-point pattern produced by FloorplanCAD:
        "M x1,y1 L x2,y2"
    where coordinates may use commas or spaces as separators and may
    include scientific notation (e.g. 5.82e-11).

    Args:
        d_attr: The 'd' attribute string from a <path> element.

    Returns:
        A Shapely LineString with two points [(x1,y1), (x2,y2)],
        or None if the d attribute doesn't match the expected pattern.
    """
    m = _SEGMENT_RE.match(d_attr.strip())
    if m is None:
        return None

    x1, y1, x2, y2 = (float(g) for g in m.groups())
    return LineString([(x1, y1), (x2, y2)])


def _arc_endpoint_to_center(
    x1: float,
    y1: float,
    rx: float,
    ry: float,
    phi_deg: float,
    large_arc: bool,
    sweep: bool,
    x2: float,
    y2: float,
) -> tuple[float, float, float, float, float, float] | None:
    """Convert SVG arc endpoint parameterization to center parameterization.

    Implements the W3C SVG specification, Section F.6.5:
    https://www.w3.org/TR/SVG/implnote.html#ArcConversionEndpointToCenter

    SVG arcs are defined by start/end points and radii (endpoint form).
    To discretize an arc into points we need the center, corrected radii,
    start angle, and sweep angle (center form).

    Handles degenerate cases per the spec:
      - start == end -> None (zero-length arc)
      - rx or ry == 0 -> None (degenerate, caller falls back to line)
      - radii too small to reach endpoints -> scaled up (F.6.6.3)

    Args:
        x1, y1:     Arc start point.
        rx, ry:     Ellipse radii (before correction).
        phi_deg:    X-axis rotation of the ellipse in degrees.
        large_arc:  True if the arc spans > 180 degrees.
        sweep:      True if the arc is drawn in the positive-angle direction.
        x2, y2:     Arc end point.

    Returns:
        Tuple of (cx, cy, rx_corrected, ry_corrected, theta1, d_theta)
        where theta1 is the start angle and d_theta is the angular sweep,
        both in radians. Returns None for degenerate arcs.
    """
    # F.6.2: If endpoints are identical, skip this arc.
    if x1 == x2 and y1 == y2:
        return None

    # F.6.6.1: Ensure radii are positive.
    rx = abs(rx)
    ry = abs(ry)
    if rx == 0.0 or ry == 0.0:
        return None

    phi = math.radians(phi_deg)
    cos_phi = math.cos(phi)
    sin_phi = math.sin(phi)

    # F.6.5.1: Compute (x1', y1') — rotated midpoint.
    dx = (x1 - x2) / 2.0
    dy = (y1 - y2) / 2.0
    x1p = cos_phi * dx + sin_phi * dy
    y1p = -sin_phi * dx + cos_phi * dy

    # F.6.6.2: Ensure radii are large enough.
    x1p_sq = x1p * x1p
    y1p_sq = y1p * y1p
    rx_sq = rx * rx
    ry_sq = ry * ry

    lam = x1p_sq / rx_sq + y1p_sq / ry_sq
    if lam > 1.0:
        # F.6.6.3: Scale radii up so the ellipse just reaches the endpoints.
        lam_sqrt = math.sqrt(lam)
        rx *= lam_sqrt
        ry *= lam_sqrt
        rx_sq = rx * rx
        ry_sq = ry * ry

    # F.6.5.2: Compute (cx', cy') — center in the rotated frame.
    num = rx_sq * ry_sq - rx_sq * y1p_sq - ry_sq * x1p_sq
    den = rx_sq * y1p_sq + ry_sq * x1p_sq

    if den == 0.0:
        return None

    sq = max(num / den, 0.0)
    sq = math.sqrt(sq)

    # Sign depends on large_arc and sweep being different.
    if large_arc == sweep:
        sq = -sq

    cxp = sq * rx * y1p / ry
    cyp = -sq * ry * x1p / rx

    # F.6.5.3: Compute (cx, cy) — center in the original frame.
    cx = cos_phi * cxp - sin_phi * cyp + (x1 + x2) / 2.0
    cy = sin_phi * cxp + cos_phi * cyp + (y1 + y2) / 2.0

    # F.6.5.5-6: Compute theta1 and d_theta.
    def _angle(ux: float, uy: float, vx: float, vy: float) -> float:
        """Signed angle between vectors (ux,uy) and (vx,vy) in radians."""
        dot = ux * vx + uy * vy
        length = math.sqrt((ux * ux + uy * uy) * (vx * vx + vy * vy))
        if length == 0.0:
            return 0.0
        cos_a = max(-1.0, min(1.0, dot / length))
        angle = math.acos(cos_a)
        if ux * vy - uy * vx < 0.0:
            angle = -angle
        return angle

    theta1 = _angle(1.0, 0.0, (x1p - cxp) / rx, (y1p - cyp) / ry)
    d_theta = _angle(
        (x1p - cxp) / rx,
        (y1p - cyp) / ry,
        (-x1p - cxp) / rx,
        (-y1p - cyp) / ry,
    )

    # Adjust d_theta to match sweep direction.
    two_pi = 2.0 * math.pi
    if not sweep and d_theta > 0.0:
        d_theta -= two_pi
    elif sweep and d_theta < 0.0:
        d_theta += two_pi

    return (cx, cy, rx, ry, theta1, d_theta)


def _parse_arc(d_attr: str, arc_resolution: int) -> LineString | None:
    """Parse an SVG arc path and discretize it into a polyline.

    Handles the pattern produced by FloorplanCAD:
        "M x1,y1 A rx,ry rotation large_arc_flag,sweep_flag x2,y2"

    Converts from SVG endpoint parameterization to center parameterization
    (W3C SVG spec F.6.5), then samples equally-spaced points along the arc
    to produce a polyline approximation.

    Args:
        d_attr:         The 'd' attribute string from a <path> element.
        arc_resolution: Number of intermediate points to sample along the arc.
                        Total points in the output = arc_resolution + 2
                        (including the start and end points).

    Returns:
        A Shapely LineString approximating the arc with arc_resolution + 2
        points, or None if parsing fails or the arc is degenerate.
    """
    m = _ARC_RE.match(d_attr.strip())
    if m is None:
        return None

    (x1, y1, rx, ry, phi_deg, large_arc_f, sweep_f, x2, y2) = (
        float(m.group(i)) for i in range(1, 10)
    )
    large_arc = large_arc_f != 0.0
    sweep = sweep_f != 0.0

    # Degenerate: zero radii -> treat as straight line.
    if rx == 0.0 or ry == 0.0:
        return LineString([(x1, y1), (x2, y2)])

    params = _arc_endpoint_to_center(
        x1, y1, rx, ry, phi_deg, large_arc, sweep, x2, y2
    )
    if params is None:
        return None

    cx, cy, rx_c, ry_c, theta1, d_theta = params
    phi = math.radians(phi_deg)
    cos_phi = math.cos(phi)
    sin_phi = math.sin(phi)

    # Sample points along the arc.
    n_points = arc_resolution + 2  # includes start and end
    points = []
    for i in range(n_points):
        t = theta1 + d_theta * i / (n_points - 1)
        cos_t = math.cos(t)
        sin_t = math.sin(t)
        x = cx + rx_c * cos_t * cos_phi - ry_c * sin_t * sin_phi
        y = cy + rx_c * cos_t * sin_phi + ry_c * sin_t * cos_phi
        points.append((x, y))

    return LineString(points)


def _translate(
    prim: FloorplanPrimitive, dx: float, dy: float,
) -> FloorplanPrimitive:
    """Shift a primitive's geometry by (dx, dy) without altering metadata.

    Used to move SVG-coordinate data to the origin when the viewBox has a
    non-zero (min_x, min_y) offset (common in georeferenced floor plans).
    """
    from shapely.affinity import affine_transform

    moved = affine_transform(prim.geometry, [1, 0, 0, 1, dx, dy])

    return FloorplanPrimitive(
        geometry=moved,
        semantic_id=prim.semantic_id,
        instance_id=prim.instance_id,
        primitive_id=prim.primitive_id,
        original_type=prim.original_type,
        layer=prim.layer,
        stroke=prim.stroke,
    )


def _flip_y(prim: FloorplanPrimitive, viewbox_height: float) -> FloorplanPrimitive:
    """Flip a primitive's Y coordinates from SVG space to geometry space.

    SVG Y-axis points downward; standard geometry/GIS Y-axis points upward.
    Applies the affine transform [1, 0, 0, -1, 0, h] which maps
    y -> viewbox_height - y, mirroring across y = h/2.

    Returns a new FloorplanPrimitive with the transformed geometry;
    all metadata fields are preserved unchanged.

    Args:
        prim:           The primitive whose geometry will be flipped.
        viewbox_height: The height value from the SVG viewBox attribute.

    Returns:
        A new FloorplanPrimitive with Y-flipped geometry.
    """
    from shapely.affinity import affine_transform

    flipped_geom = affine_transform(prim.geometry, [1, 0, 0, -1, 0, viewbox_height])

    return FloorplanPrimitive(
        geometry=flipped_geom,
        semantic_id=prim.semantic_id,
        instance_id=prim.instance_id,
        primitive_id=prim.primitive_id,
        original_type=prim.original_type,
        layer=prim.layer,
        stroke=prim.stroke,
    )


def _parse_circle(elem: ET.Element) -> FloorplanPrimitive | None:
    """Parse a <circle> SVG element into a FloorplanPrimitive with Polygon geometry.

    Reads cx, cy, r attributes and creates a buffered Point (Polygon)
    representing the circle. Circles can be large structural elements
    (columns) that participate in boundary polygonization, so they are
    stored as Polygon rather than Point.

    Args:
        elem: A <circle> XML Element with FloorplanCAD custom attributes.

    Returns:
        A FloorplanPrimitive with a Polygon geometry (buffered circle),
        or None if required attributes are missing.
    """
    try:
        attrs = _extract_attributes(elem)
    except KeyError:
        logger.warning("Skipping <circle> element without primitiveId")
        return None

    cx_raw = elem.get("cx")
    cy_raw = elem.get("cy")
    r_raw = elem.get("r")

    if cx_raw is None or cy_raw is None or r_raw is None:
        logger.warning(
            "Skipping circle primitiveId=%d: missing cx/cy/r", attrs["primitive_id"]
        )
        return None

    cx, cy, r = float(cx_raw), float(cy_raw), float(r_raw)

    if r <= 0.0:
        logger.warning(
            "Skipping circle primitiveId=%d: non-positive radius %f",
            attrs["primitive_id"],
            r,
        )
        return None

    geom = Point(cx, cy).buffer(r)

    return FloorplanPrimitive(
        geometry=geom,
        semantic_id=attrs["semantic_id"],
        instance_id=attrs["instance_id"],
        primitive_id=attrs["primitive_id"],
        original_type=attrs["original_type"],
        layer=attrs["layer"],
        stroke=attrs["stroke"],
    )


def _parse_ellipse(elem: ET.Element) -> FloorplanPrimitive | None:
    """Parse an <ellipse> SVG element into a FloorplanPrimitive with Polygon geometry.

    Reads cx, cy, rx, ry attributes and creates a scaled buffered Point.
    No ellipses exist in the current sample set, but they appear in other
    FloorplanCAD datasets. Uses Shapely's affine scaling to stretch a
    circle buffer into an ellipse.

    Args:
        elem: An <ellipse> XML Element with FloorplanCAD custom attributes.

    Returns:
        A FloorplanPrimitive with a Polygon geometry (ellipse shape),
        or None if required attributes are missing.
    """
    try:
        attrs = _extract_attributes(elem)
    except KeyError:
        logger.warning("Skipping <ellipse> element without primitiveId")
        return None

    cx_raw = elem.get("cx")
    cy_raw = elem.get("cy")
    rx_raw = elem.get("rx")
    ry_raw = elem.get("ry")

    if cx_raw is None or cy_raw is None or rx_raw is None or ry_raw is None:
        logger.warning(
            "Skipping ellipse primitiveId=%d: missing cx/cy/rx/ry",
            attrs["primitive_id"],
        )
        return None

    cx, cy = float(cx_raw), float(cy_raw)
    rx, ry = float(rx_raw), float(ry_raw)

    if rx <= 0.0 or ry <= 0.0:
        logger.warning(
            "Skipping ellipse primitiveId=%d: non-positive radii rx=%f ry=%f",
            attrs["primitive_id"],
            rx,
            ry,
        )
        return None

    # Create a unit circle at the center, then scale to ellipse dimensions.
    from shapely.affinity import scale

    geom = scale(Point(cx, cy).buffer(1.0), xfact=rx, yfact=ry, origin=(cx, cy))

    return FloorplanPrimitive(
        geometry=geom,
        semantic_id=attrs["semantic_id"],
        instance_id=attrs["instance_id"],
        primitive_id=attrs["primitive_id"],
        original_type=attrs["original_type"],
        layer=attrs["layer"],
        stroke=attrs["stroke"],
    )


def _parse_path_element(
    elem: ET.Element, arc_resolution: int = 20
) -> FloorplanPrimitive | None:
    """Parse a single <path> element into a FloorplanPrimitive.

    Dispatches to the appropriate geometry parser based on the
    originalType attribute:
      - "segment" -> _parse_segment()
      - "arc"     -> _parse_arc()

    Note: "circle" and "ellipse" types use dedicated SVG elements
    (<circle>, <ellipse>), not <path>. They are handled separately
    in parse_svg() (Inc 3).

    Args:
        elem:           A <path> XML Element with FloorplanCAD attributes.
        arc_resolution: Number of intermediate points for arc discretization.

    Returns:
        A FloorplanPrimitive with the parsed geometry and metadata,
        or None if the element cannot be parsed.
    """
    try:
        attrs = _extract_attributes(elem)
    except KeyError:
        logger.warning("Skipping <path> element without primitiveId")
        return None

    d_attr = elem.get("d", "")
    original_type = attrs["original_type"]

    geom = None
    if original_type == "segment":
        geom = _parse_segment(d_attr)
    elif original_type == "arc":
        geom = _parse_arc(d_attr, arc_resolution)
    else:
        logger.debug(
            "Skipping path primitiveId=%d with unsupported originalType='%s'",
            attrs["primitive_id"],
            original_type,
        )
        return None

    if geom is None:
        logger.warning(
            "Failed to parse d attribute for primitiveId=%d: '%s'",
            attrs["primitive_id"],
            d_attr[:80],
        )
        return None

    return FloorplanPrimitive(
        geometry=geom,
        semantic_id=attrs["semantic_id"],
        instance_id=attrs["instance_id"],
        primitive_id=attrs["primitive_id"],
        original_type=attrs["original_type"],
        layer=attrs["layer"],
        stroke=attrs["stroke"],
    )


def _snap_to_svg_boundary(
    prims: list[FloorplanPrimitive],
    viewbox: tuple[float, float, float, float],
    viewbox_height: float,
    y_flip: bool,
    tolerance: float,
) -> list[FloorplanPrimitive]:
    """Snap LineString endpoints close to SVG boundary edges onto those edges.

    SVG drawings often have wall segments that end ~0.3 units short of the
    SVG viewBox edge due to drawing imprecision. After we add the boundary
    rectangle for polygonization, those near-misses still leave tiny gaps —
    polygons can't close against the boundary because the wall endpoints
    aren't exactly on it.

    For each LineString primitive in `prims`, this function checks each
    coordinate against the 4 viewBox edges (left, right, top, bottom).
    If a coordinate is within `tolerance` of an edge, it's snapped to it.

    Args:
        prims:          List of FloorplanPrimitive to process.
        viewbox:        SVG viewBox tuple (min_x, min_y, width, height).
        viewbox_height: viewbox[3], passed for Y-flip math.
        y_flip:         Whether Y was flipped during parsing.
        tolerance:      Snap radius in SVG units.

    Returns:
        New list of FloorplanPrimitive (Line geometries snapped where
        applicable; non-LineString geometries pass through unchanged).
    """
    if tolerance <= 0:
        return prims

    min_x, min_y, width, height = viewbox
    max_x = min_x + width
    max_y = min_y + height

    if y_flip:
        x_lo, x_hi = min_x, max_x
        y_lo, y_hi = viewbox_height - max_y, viewbox_height - min_y
    else:
        x_lo, x_hi = min_x, max_x
        y_lo, y_hi = min_y, max_y

    def _snap(c: tuple[float, float]) -> tuple[float, float]:
        x, y = c[0], c[1]
        if abs(x - x_lo) < tolerance:
            x = x_lo
        elif abs(x - x_hi) < tolerance:
            x = x_hi
        if abs(y - y_lo) < tolerance:
            y = y_lo
        elif abs(y - y_hi) < tolerance:
            y = y_hi
        return (x, y)

    out: list[FloorplanPrimitive] = []
    for p in prims:
        if isinstance(p.geometry, LineString):
            new_coords = [_snap(c) for c in p.geometry.coords]
            # Drop degenerate results (all points snapped to same location)
            if len(set(new_coords)) >= 2:
                new_geom = LineString(new_coords)
                out.append(
                    FloorplanPrimitive(
                        geometry=new_geom,
                        semantic_id=p.semantic_id,
                        instance_id=p.instance_id,
                        primitive_id=p.primitive_id,
                        original_type=p.original_type,
                        layer=p.layer,
                        stroke=p.stroke,
                    )
                )
            else:
                out.append(p)  # keep original if snap made it degenerate
        else:
            out.append(p)
    return out


def _create_boundary_rectangle(
    viewbox: tuple[float, float, float, float],
    viewbox_height: float,
    y_flip: bool,
) -> list[FloorplanPrimitive]:
    """Create 4 LineStrings forming a bounding rectangle from the viewBox.

    Creates boundary lines for the edges of the SVG coordinate space.
    After Y-flip, these represent the actual boundary of the geometry space.

    Args:
        viewbox:       Tuple of (min_x, min_y, width, height) from SVG viewBox.
        viewbox_height: The height value from viewBox (same as viewbox[3]).
        y_flip:        Whether Y-flip has been applied.

    Returns:
        List of 4 FloorplanPrimitive objects with LineString geometries,
        one for each edge of the bounding rectangle. semantic_id is None
        so they don't get grouped into semantic categories.
    """
    min_x, min_y, width, height = viewbox
    max_x = min_x + width
    max_y = min_y + height

    if y_flip:
        # After Y-flip, the coordinates are transformed as (x, viewbox_height - y)
        # So the rectangle corners become:
        # (min_x, min_y) -> (min_x, viewbox_height - min_y)
        # (max_x, min_y) -> (max_x, viewbox_height - min_y)
        # (max_x, max_y) -> (max_x, viewbox_height - max_y)
        # (min_x, max_y) -> (min_x, viewbox_height - max_y)
        p1 = (min_x, viewbox_height - min_y)
        p2 = (max_x, viewbox_height - min_y)
        p3 = (max_x, viewbox_height - max_y)
        p4 = (min_x, viewbox_height - max_y)
    else:
        p1 = (min_x, min_y)
        p2 = (max_x, min_y)
        p3 = (max_x, max_y)
        p4 = (min_x, max_y)

    # 4 edges: bottom, right, top, left
    edges = [
        LineString([p1, p2]),  # bottom
        LineString([p2, p3]),  # right
        LineString([p3, p4]),  # top
        LineString([p4, p1]),  # left
    ]

    # Assign primitive IDs starting from a large number to avoid conflicts
    primitives = []
    for i, edge in enumerate(edges):
        primitives.append(
            FloorplanPrimitive(
                geometry=edge,
                semantic_id=None,
                instance_id=None,
                primitive_id=1000000 + i,
                original_type="boundary",
                layer="boundary",
                stroke="rgb(0, 0, 0)",
            )
        )

    return primitives
