"""Data models for the SVG-to-GeoJSON pipeline.

This module defines the core data structures that flow through every stage
of the pipeline. Using dataclasses instead of attaching attributes to Shapely
objects directly, because Shapely operations (buffer, affine_transform, etc.)
return NEW geometry objects and silently drop any attached attributes.

Pipeline data flow:
    SVG file
      -> svg_parser    -> list[FloorplanPrimitive]   (raw parsed elements)
      -> hatching_filter -> list[FloorplanPrimitive]  (noise removed)
      -> polygonizer   -> list[RoomPolygon]           (closed room shapes)
      -> classifier    -> list[RoomPolygon]           (with facil_type set)
      -> georeferencer -> GeoJSON file                (transformed to WGS84)
"""

from __future__ import annotations

from dataclasses import dataclass, field

from shapely.geometry.base import BaseGeometry


@dataclass
class FloorplanPrimitive:
    """A single parsed SVG primitive with its geometry and metadata.

    Created by svg_parser from each <path>, <circle>, or <ellipse> element.
    Carries all the custom XML attributes from the FloorplanCAD annotation
    scheme alongside the converted Shapely geometry.

    Attributes:
        geometry:      Shapely geometry (LineString for segments/arcs,
                       Point or Polygon for circles/ellipses).
        semantic_id:   FloorplanCAD class (1-35). None for background elements.
                       This is the PRIMARY classifier — not layer names.
        instance_id:   Groups primitives of the same physical object (e.g. all
                       segments and arcs of one door share an instance_id).
                       -1 or None means ungrouped.
        primitive_id:  Unique numeric ID within the SVG file.
        original_type: Source element kind: 'segment', 'arc', 'circle', 'ellipse'.
        layer:         CAD layer name. Stored for debugging but NOT used for
                       filtering — layer names are inconsistent across files.
        stroke:        CSS color string, e.g. 'rgb(167, 92, 32)'.
    """

    geometry: BaseGeometry
    semantic_id: int | None
    instance_id: int | None
    primitive_id: int
    original_type: str
    layer: str
    stroke: str


@dataclass
class RoomPolygon:
    """A room polygon produced by polygonization, then enriched by classification.

    Created by polygonizer from closed wall loops. Initially only geometry is set.
    The classifier then fills in facil_type based on which furniture/fixture
    primitives fall inside the polygon.

    Attributes:
        geometry:            Shapely Polygon representing the room boundary.
        facil_type:          Room type label: "Office", "WC", "Kitchen", "Stairs",
                             "Elevator", "Bedroom", "Walkways", "Deadzone", "Wall",
                             "Door", or None if not yet classified.
        facil_name:          Optional human-readable name (e.g. "Room 201").
        stroke:              Original SVG stroke color (CSS rgb string) of the
                             primitives that produced this polygon. Set for
                             Door/Window polygons so visualizations can match
                             the SVG's semantic coloring.
        contained_semantics: Map of {semantic_id: count} for primitives inside
                             this room. Used by the classifier to decide facil_type.
                             Example: {28: 3, 35: 1} means 3 stair primitives and
                             1 railing primitive are inside this room.
    """

    geometry: BaseGeometry
    facil_type: str | None = None
    facil_name: str | None = None
    stroke: str | None = None
    contained_semantics: dict[int, int] = field(default_factory=dict)
    has_arc: bool = False
    hinge: BaseGeometry | None = None  # Point at the arc circle center


# ---------------------------------------------------------------------------
# Semantic ID groupings from FloorplanCAD (poilabs subset)
#
# The full FloorplanCAD scheme has 36 classes (1-indexed). Our dataset
# removes these classes (0-indexed list from dataset config):
#   poilabs: [35, 3, 5, 7, 8, 9, 11, 14, 15, 17, 19, 20, 21, 22, 23]
#
# In 1-indexed semanticIds that means these will NEVER appear in our SVGs:
#   4 (folding door), 6 (rolling door), 8 (bay window), 9 (blind window),
#   10 (opening symbol), 12 (bed), 15 (TV cabinet), 16 (wardrobe),
#   18 (gas stove), 20 (refrigerator), 21 (airconditioner),
#   22 (bath), 23 (bath tub), 36 (background — never has semanticId attr)
#
# Only IDs that actually appear in our data are included below.
# ---------------------------------------------------------------------------

BOUNDARY_IDS = {33, 34}            # wall + curtain wall -> polygonization input
DOOR_IDS = {1, 2, 3, 5}           # single, double, sliding, revolving door
WINDOW_IDS = {7}                   # window (only type in our dataset)
OPENING_IDS = DOOR_IDS | WINDOW_IDS  # combined -> segments close wall gaps
STAIR_IDS = {28}
ELEVATOR_IDS = {29}
ESCALATOR_IDS = {30}
TOILET_IDS = {25, 26, 27}         # squat toilet, urinal, toilet
KITCHEN_IDS = {19}                 # sink (only kitchen item in our dataset)
FURNITURE_IDS = {11, 13, 14, 17}   # sofa, chair, table, cabinet
RAILING_IDS = {35}
PARKING_IDS = {32}
ROW_CHAIR_IDS = {31}
BACKGROUND_ID = 36                # no semanticId attr in SVG -> skip
