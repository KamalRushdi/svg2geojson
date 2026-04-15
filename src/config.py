"""Configuration system for the SVG-to-GeoJSON pipeline.

All tunable parameters are centralized here so they can be adjusted per-building
via YAML config files without editing code. Each pipeline phase has its own
config dataclass. Defaults work for most FloorplanCAD files.

Usage:
    # Use defaults:
    config = load_config()

    # Override from YAML:
    config = load_config(Path("config/building_A.yaml"))

    # Access parameters:
    config.cleaning.snap_tolerance  # 0.1
    config.parsing.boundary_semantic_ids  # {33, 34}

YAML file format — only include keys you want to override:
    parsing:
      boundary_semantic_ids: [33, 34]
      arc_resolution: 20
    cleaning:
      snap_tolerance: 0.5
    hatching:
      min_length: 3.0
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass
class ParsingConfig:
    """Controls which semantic IDs belong to which pipeline group.

    Attributes:
        boundary_semantic_ids: IDs treated as wall boundaries for polygonization.
                               Default {33, 34} = wall + curtain wall.
        door_semantic_ids:     IDs whose straight segments close door gaps.
        window_semantic_ids:   IDs whose straight segments close window gaps.
        y_flip:                Flip Y-axis (SVG Y goes down, geometry Y goes up).
        arc_resolution:        Number of sample points when discretizing SVG arcs.
    """

    boundary_semantic_ids: set[int] = field(default_factory=lambda: {33, 34})
    door_semantic_ids: set[int] = field(default_factory=lambda: {1, 2, 3, 5})
    window_semantic_ids: set[int] = field(default_factory=lambda: {7})
    y_flip: bool = True
    arc_resolution: int = 20


@dataclass
class HatchingFilterConfig:
    """Controls the geometry-based hatching noise filter.

    Wall hatching (cross-hatch fill inside wall thickness) has semanticId=33
    just like real wall edges, but consists of short diagonal lines. Without
    filtering, these create thousands of micro-triangles in polygonization.

    Attributes:
        min_length:             Segments shorter than this (in SVG units) AND
                                diagonal are classified as hatching. Wall edges
                                are typically 5-30+ units, hatching is <2 units.
        max_diagonal_angle_deg: How close to 45 degrees a segment's angle must
                                be to count as "diagonal". 15 means 30-60 degrees.
        enabled:                Set False to skip hatching filter entirely.
    """

    min_length: float = 10.0
    max_diagonal_angle_deg: float = 2.0
    enabled: bool = True


@dataclass
class CleaningConfig:
    """Controls coordinate rounding, endpoint snapping, and degenerate removal.

    Attributes:
        round_precision:  Decimal places for coordinate rounding. 3 means
                          85.30875300 becomes 85.309. Merges near-miss coords.
        snap_tolerance:   Maximum distance (SVG units) to snap endpoints together
                          using KD-tree. Increase if too few rooms form.
        min_line_length:  Lines shorter than this are removed as degenerate.
    """

    round_precision: int = 3
    snap_tolerance: float = 0.1
    min_line_length: float = 0.01


@dataclass
class PolygonizationConfig:
    """Controls polygon extraction and wall/room separation.

    Attributes:
        min_room_area:            Polygons with area below this are discarded.
        wall_aspect_threshold:    Aspect ratio (short/long edge) of minimum
                                  rotated rectangle. Below this -> classified as
                                  wall strip, not a room.
        max_snap_retry:           How many times to retry polygonization with
                                  increasing snap tolerance on low room count.
        snap_tolerance_increment: How much to increase snap_tolerance per retry.
    """

    min_room_area: float = 1.0
    wall_aspect_threshold: float = 0.08
    max_snap_retry: int = 3
    snap_tolerance_increment: float = 0.2


@dataclass
class ClassificationConfig:
    """Controls room type assignment based on contained furniture/fixtures.

    Each set maps semantic IDs to the room type they indicate. Priority order
    in the classifier: stairs > elevator > WC > kitchen > geometric.

    Only IDs present in the poilabs dataset subset are included.
    Removed from FloorplanCAD: 12 (bed), 16 (wardrobe), 18 (gas stove),
    20 (refrigerator), 22 (bath), 23 (bath tub).

    Attributes:
        walkway_aspect_ratio: Rooms more elongated than this -> "Walkways".
        deadzone_max_area:    Rooms smaller than this with no furniture -> "Deadzone".
    """

    stair_ids: set[int] = field(default_factory=lambda: {28})
    elevator_ids: set[int] = field(default_factory=lambda: {29})
    toilet_ids: set[int] = field(default_factory=lambda: {25, 26, 27})
    kitchen_ids: set[int] = field(default_factory=lambda: {19})
    walkway_aspect_ratio: float = 4.0
    deadzone_max_area: float = 2.0


@dataclass
class ExportConfig:
    """Controls GeoJSON output properties.

    Attributes:
        floor_level:       Value for level_id and floorLevel GeoJSON properties.
        default_facil_type: Fallback room type when no classification rule matches.
    """

    floor_level: str = "0"
    default_facil_type: str = "Office"


@dataclass
class PipelineConfig:
    """Top-level config aggregating all pipeline phase configs.

    Create with defaults via PipelineConfig(), or load from YAML via load_config().
    """

    parsing: ParsingConfig = field(default_factory=ParsingConfig)
    hatching: HatchingFilterConfig = field(default_factory=HatchingFilterConfig)
    cleaning: CleaningConfig = field(default_factory=CleaningConfig)
    polygonization: PolygonizationConfig = field(default_factory=PolygonizationConfig)
    classification: ClassificationConfig = field(default_factory=ClassificationConfig)
    export: ExportConfig = field(default_factory=ExportConfig)


def load_config(path: Path | None = None) -> PipelineConfig:
    """Load pipeline config from a YAML file, merging with defaults.

    Any key present in the YAML overrides the default. Keys not in the YAML
    keep their default values. If path is None or doesn't exist, returns
    all defaults.

    Args:
        path: Path to a YAML config file, or None for defaults.

    Returns:
        PipelineConfig with merged values.

    Example YAML::

        cleaning:
          snap_tolerance: 0.5
        hatching:
          enabled: false
    """
    if path is None or not path.exists():
        return PipelineConfig()

    with open(path) as f:
        raw = yaml.safe_load(f) or {}

    config = PipelineConfig()

    if "parsing" in raw:
        p = raw["parsing"]
        if "boundary_semantic_ids" in p:
            config.parsing.boundary_semantic_ids = set(p["boundary_semantic_ids"])
        if "door_semantic_ids" in p:
            config.parsing.door_semantic_ids = set(p["door_semantic_ids"])
        if "window_semantic_ids" in p:
            config.parsing.window_semantic_ids = set(p["window_semantic_ids"])
        if "y_flip" in p:
            config.parsing.y_flip = p["y_flip"]
        if "arc_resolution" in p:
            config.parsing.arc_resolution = p["arc_resolution"]

    if "hatching" in raw:
        h = raw["hatching"]
        if "min_length" in h:
            config.hatching.min_length = h["min_length"]
        if "max_diagonal_angle_deg" in h:
            config.hatching.max_diagonal_angle_deg = h["max_diagonal_angle_deg"]
        if "enabled" in h:
            config.hatching.enabled = h["enabled"]

    if "cleaning" in raw:
        c = raw["cleaning"]
        if "round_precision" in c:
            config.cleaning.round_precision = c["round_precision"]
        if "snap_tolerance" in c:
            config.cleaning.snap_tolerance = c["snap_tolerance"]
        if "min_line_length" in c:
            config.cleaning.min_line_length = c["min_line_length"]

    if "polygonization" in raw:
        pg = raw["polygonization"]
        if "min_room_area" in pg:
            config.polygonization.min_room_area = pg["min_room_area"]
        if "wall_aspect_threshold" in pg:
            config.polygonization.wall_aspect_threshold = pg["wall_aspect_threshold"]
        if "max_snap_retry" in pg:
            config.polygonization.max_snap_retry = pg["max_snap_retry"]
        if "snap_tolerance_increment" in pg:
            config.polygonization.snap_tolerance_increment = pg["snap_tolerance_increment"]

    if "classification" in raw:
        cl = raw["classification"]
        for attr in ("stair_ids", "elevator_ids", "toilet_ids", "kitchen_ids"):
            if attr in cl:
                setattr(config.classification, attr, set(cl[attr]))
        if "walkway_aspect_ratio" in cl:
            config.classification.walkway_aspect_ratio = cl["walkway_aspect_ratio"]
        if "deadzone_max_area" in cl:
            config.classification.deadzone_max_area = cl["deadzone_max_area"]

    if "export" in raw:
        e = raw["export"]
        if "floor_level" in e:
            config.export.floor_level = str(e["floor_level"])
        if "default_facil_type" in e:
            config.export.default_facil_type = e["default_facil_type"]

    return config
