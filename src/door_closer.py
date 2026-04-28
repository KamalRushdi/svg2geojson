"""Close door and window gaps in the boundary set with simple bounding rectangles.

FloorplanCAD walls stop at door/window openings, leaving gaps that prevent
polygonize() from forming closed room loops. This module constructs an
axis-aligned bounding rectangle for each door/window instance and adds
the 4 rectangle edges to the boundary set so polygonize_full can close
the wall loops.

Wall-thickness matching is deferred to a later step (door_resizer, post-Inc 10).
After polygonization, each door/window's bounding rectangle will be resized
to align with the adjacent wall polygon's thickness.

Algorithm per door/window instance:
  1. Collect endpoints of segment primitives (skip arcs — those are swing arms).
  2. Build axis-aligned bounding rectangle from those endpoints.
  3. Filter mega-instances by area ratio (rect_area / total_area).
  4. Reject rectangles thinner than min_rect_thickness.
  5. Emit 4 boundary LineStrings + one Door/Window RoomPolygon.

Typical usage:

    from src.door_closer import close_openings

    closing = close_openings(result.doors, result.windows, boundary)
    sealed_boundary = boundary + closing.door_edges + closing.window_edges
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np
from shapely.geometry import LineString, MultiPoint, Polygon, box

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
    """Close door/window gaps with simple axis-aligned bounding rectangles.

    For each door/window instance (grouped by instance_id), builds an AABB
    from the segment-primitive endpoints, validates it, and emits the 4
    edges as boundary primitives plus the rectangle as a Door/Window polygon.

    Args:
        doors:    Door primitives from ParseResult.doors.
        windows:  Window primitives from ParseResult.windows.
        boundary: Boundary primitives — used only for total-area normalization
                  in the mega-instance filter.
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

    wall_lines = [
        p.geometry for p in boundary if isinstance(p.geometry, LineString)
    ]
    total_area = _compute_total_area(wall_lines)

    # Process doors
    door_instances = _group_by_instance(doors)
    for iid, prims in door_instances.items():
        rect = _build_bbox_rect(prims, total_area, config)
        if rect is None:
            result.skipped_instances.append((iid, "could not build rectangle"))
            continue

        edges = _extract_rect_edges(rect, iid, prims[0].semantic_id)
        result.door_edges.extend(edges)
        result.door_polygons.append(
            RoomPolygon(
                geometry=rect, facil_type="Door", stroke=prims[0].stroke
            )
        )
        result.door_count += 1

    # Process windows — apply spatial multi-leg splitting if enabled
    window_instances = _group_by_instance(windows)
    for iid, prims in window_instances.items():
        if config.split_l_shaped_windows:
            subgroups = _split_spatially(prims, config)
        else:
            subgroups = [prims]

        for sub_prims in subgroups:
            rect = _build_bbox_rect(sub_prims, total_area, config)
            if rect is None:
                result.skipped_instances.append(
                    (iid, "could not build rectangle")
                )
                continue

            edges = _extract_rect_edges(rect, iid, sub_prims[0].semantic_id)
            result.window_edges.extend(edges)
            result.window_polygons.append(
                RoomPolygon(
                    geometry=rect,
                    facil_type="Window",
                    stroke=sub_prims[0].stroke,
                )
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


def _split_spatially(
    prims: list[FloorplanPrimitive],
    config: DoorClosingConfig,
) -> list[list[FloorplanPrimitive]]:
    """Detect L/U/T-shaped instances via concavity and split via k-means.

    Algorithm:
      1. Concavity check — `convex_hull(endpoints).area / aabb.area`. If
         the shape is convex enough, return [prims] (no split).
      2. K-means clustering on segment midpoints for k=2..max_clusters,
         picking the k with the highest silhouette score.
      3. Reject the split if best silhouette < min_silhouette
         (clusters too overlapping → not a real multi-leg shape).
      4. Partition primitives by cluster; non-segment primitives assigned
         to the cluster of the spatially-nearest segment.

    Returns:
        Single-element list [prims] when no split, else N-element list.
    """
    seg_prims: list[FloorplanPrimitive] = []
    midpoints: list[tuple[float, float]] = []
    for p in prims:
        if not (
            isinstance(p.geometry, LineString)
            and p.original_type == "segment"
        ):
            continue
        coords = list(p.geometry.coords)
        if len(coords) < 2:
            continue
        mx = (coords[0][0] + coords[-1][0]) / 2
        my = (coords[0][1] + coords[-1][1]) / 2
        seg_prims.append(p)
        midpoints.append((mx, my))

    if len(seg_prims) < 4:
        return [prims]

    # Step 1: instance-area check — convex hull of all segment endpoints
    # divided by AABB area. For a rectangle (even with sparse internal
    # segments), endpoints reach the AABB corners → hull ≈ AABB → ratio
    # near 1.0. An L/U/T-shape leaves part of the AABB unreachable by
    # any endpoint → hull < AABB → ratio drops.
    all_endpoints: list[tuple[float, float]] = []
    for p in seg_prims:
        all_endpoints.extend(p.geometry.coords)
    xs = [c[0] for c in all_endpoints]
    ys = [c[1] for c in all_endpoints]
    aabb_area = (max(xs) - min(xs)) * (max(ys) - min(ys))
    if aabb_area <= 0:
        return [prims]
    try:
        hull_area = MultiPoint(all_endpoints).convex_hull.area
    except Exception:
        return [prims]
    if hull_area / aabb_area >= config.area_ratio_threshold:
        return [prims]

    # Step 2: try k=2..max_clusters, pick the LOWEST k whose silhouette
    # exceeds min_silhouette. Prefer parsimony — for an L, k=2 already
    # captures the structure; k=3 might score slightly higher by over-
    # segmenting one leg, but that's not what we want.
    pts = np.array(midpoints, dtype=float)
    best_k = None
    best_score = -1.0
    best_labels: np.ndarray | None = None
    for k in range(2, config.max_clusters + 1):
        if k > len(pts):
            break
        labels = _kmeans(pts, k)
        counts = np.bincount(labels, minlength=k)
        if counts.min() < 2:
            continue
        score = _silhouette(pts, labels)
        if score >= config.min_silhouette:
            best_k = k
            best_labels = labels
            best_score = score
            break  # take the lowest k that qualifies

    if best_labels is None:
        return [prims]

    # Step 3: partition primitives by cluster assignment
    final_clusters: list[list[FloorplanPrimitive]] = [
        [] for _ in range(best_k)
    ]
    seg_id_to_cluster: dict[int, int] = {}
    for sp, lbl in zip(seg_prims, best_labels):
        ci = int(lbl)
        final_clusters[ci].append(sp)
        seg_id_to_cluster[id(sp)] = ci

    # Distribute non-segment primitives by nearest segment-midpoint
    cluster_centroids = np.array(
        [pts[best_labels == ci].mean(axis=0) for ci in range(best_k)]
    )
    for p in prims:
        if id(p) in seg_id_to_cluster:
            continue
        try:
            ctr = p.geometry.centroid
            cx, cy = ctr.x, ctr.y
        except Exception:
            final_clusters[0].append(p)
            continue
        d2 = ((cluster_centroids - np.array([cx, cy])) ** 2).sum(axis=1)
        final_clusters[int(d2.argmin())].append(p)

    # Drop empty clusters (shouldn't happen given the min-2 check above)
    final_clusters = [c for c in final_clusters if c]
    if len(final_clusters) <= 1:
        return [prims]
    return final_clusters


def _kmeans(points: np.ndarray, k: int, max_iter: int = 30) -> np.ndarray:
    """Lightweight k-means for small point sets. Returns int label per point."""
    n = len(points)
    if k >= n:
        return np.arange(n) % k

    # Initialize centroids: farthest-first traversal
    centroids = [int(np.argmax(np.linalg.norm(points - points.mean(axis=0), axis=1)))]
    for _ in range(1, k):
        d2 = np.min(
            np.array([
                ((points - points[c]) ** 2).sum(axis=1) for c in centroids
            ]),
            axis=0,
        )
        centroids.append(int(np.argmax(d2)))
    cs = points[centroids].copy()

    labels = np.zeros(n, dtype=int)
    for _ in range(max_iter):
        # Assign each point to nearest centroid
        d2 = ((points[:, None, :] - cs[None, :, :]) ** 2).sum(axis=2)
        new_labels = d2.argmin(axis=1)
        if np.array_equal(new_labels, labels):
            break
        labels = new_labels
        # Update centroids
        for ci in range(k):
            mask = labels == ci
            if mask.any():
                cs[ci] = points[mask].mean(axis=0)
    return labels


def _silhouette(points: np.ndarray, labels: np.ndarray) -> float:
    """Silhouette score in [-1, 1]. Higher = better-separated clusters."""
    n = len(points)
    unique = np.unique(labels)
    if len(unique) < 2:
        return 0.0
    # Pairwise distance matrix
    diff = points[:, None, :] - points[None, :, :]
    dist = np.sqrt((diff ** 2).sum(axis=2))
    scores = []
    for i in range(n):
        same = (labels == labels[i]) & (np.arange(n) != i)
        if not same.any():
            scores.append(0.0)
            continue
        a = dist[i, same].mean()
        b = float("inf")
        for c in unique:
            if c == labels[i]:
                continue
            mask = labels == c
            if mask.any():
                b = min(b, dist[i, mask].mean())
        denom = max(a, b)
        scores.append((b - a) / denom if denom > 0 else 0.0)
    return float(np.mean(scores))


# ---------------------------------------------------------------------------
# Per-instance rectangle construction
# ---------------------------------------------------------------------------


def _build_bbox_rect(
    prims: list[FloorplanPrimitive],
    total_area: float,
    config: DoorClosingConfig,
) -> Polygon | None:
    """Build a simple axis-aligned bounding rectangle for one instance.

    Pipeline:
      1. Collect endpoints from segment primitives (skip arcs).
      2. Compute AABB.
      3. Reject if too small.
      4. Mega-instance filter (area ratio).
      5. Reject if thinner than min_rect_thickness.
    """
    # Collect endpoints from segment primitives only — arcs are swing arms
    # that extend far into the room and would distort the AABB.
    pts = []
    for p in prims:
        if isinstance(p.geometry, LineString) and p.original_type == "segment":
            pts.extend(list(p.geometry.coords))
    # Fallback to all linestring primitives if no segments
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

    rect_area = width * height
    if total_area > 0 and rect_area / total_area > config.max_area_ratio:
        logger.debug(
            "Skipping mega-instance: area ratio %.3f > %.3f",
            rect_area / total_area,
            config.max_area_ratio,
        )
        return None

    rect = box(min_x, min_y, max_x, max_y)

    if not rect.is_valid or rect.area < 0.01:
        return None

    short, _long = _rect_dimensions(rect)
    if short < config.min_rect_thickness:
        return None

    return rect


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
    coords = list(rect.exterior.coords)
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
