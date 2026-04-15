# Implementation Increments

Each increment is a small, testable step. Mark `[x]` when done.

---

## Phase 0: Foundation

- [x] **Inc 0.1: Project setup** — pyproject.toml, dependencies, src/ package structure, `models.py` with `FloorplanFeature` dataclass
- [x] **Inc 0.2: Config system** — `config.py` with YAML loader, default config with all tunable parameters (snap_tolerance, round_precision, wall_aspect_threshold, min_line_length, etc.)
- [x] **Inc 0.3: Debug visualizer** — `visualizer.py` with reusable matplotlib helpers: plot_lines(lines, color_by), plot_polygons(polys, color_by), side_by_side comparison

---

## Phase 1: SVG Parsing

- [X] **Inc 1: Parse path segments** — Parse `<path d="M x,y L x,y">` elements, extract semanticId/instanceId/primitiveId/originalType/layer attributes, return `list[FloorplanFeature]` with LineString geometries
  - Checkpoint: parse sample SVG, count walls (semanticId=33) > 100, plot all segments colored by semanticId
- [x] **Inc 2: Parse arcs** — Handle `M x,y A rx ry rot large sweep x,y` path commands, discretize to ~20-point polylines
  - Checkpoint: arcs render as smooth curves (door swings), compare visual with/without arcs
- [x] **Inc 3: Parse circles and ellipses** — `<circle>` → Point or buffered Polygon, `<ellipse>` → scaled buffered Polygon
  - Checkpoint: total primitives ~3500 for AB2 sample
- [x] **Inc 4: Y-flip and semanticId grouping** — Flip Y (`140 - y`), group into boundary (33,34), doors (1-6), windows (7-10), classification (rest), skip no-semanticId with warning
  - Checkpoint: plot boundary lines only — should look like floor plan WITH gaps at doors/windows

---

## Phase 1.5: Hatching Filter (CRITICAL — audit finding C1)

- [x] **Inc 4.5: Hatching filter** — `hatching_filter.py`: remove short diagonal wall segments that are decorative fill, not boundary edges. Use length threshold + orientation angle. Keep filter configurable.
  - Checkpoint: boundary count drops ~60-80% on files with hatching (AB2, ARI_3, DBLOK). Files without hatching (HKÜ, ESKİŞEHİR) unaffected. Plot before/after.

---

## Phase 1 continued: Door/Window Closing

- [ ] **Inc 5: Door and window closing** — Add straight frame segments (originalType='segment') from door/window primitives to boundary set. Group by instanceId for context. Skip arc primitives (swing visuals).
  - Checkpoint: boundary count increases, gaps at doors/windows visually sealed. Side-by-side plot: before (gaps) vs after (sealed).
- [ ] **Inc 5.5: Synthetic closing (Strategy B)** — After polygonization attempt, find dangling wall endpoints within door-width distance, add synthetic closing lines for remaining gaps
  - Checkpoint: dangle count decreases

---

## Phase 2: Geometry Cleaning

- [x] **Inc 6: Coordinate rounding + degenerate removal** — Round to configurable precision, remove zero-length and invalid geometries
  - Checkpoint: small number of degenerates removed, print before/after counts
- [x] **Inc 7: Endpoint snapping (KD-tree)** — Find endpoint pairs within tolerance, snap to centroid using union-find
  - Checkpoint: print snap count, plot zoomed corner showing gap closure
- [x] **Inc 8: Merge + node pipeline** — Chain: round → clean → snap → unary_union. Produce fully noded MultiLineString.
  - Checkpoint: every intersection is a vertex, print segment counts

---

## Phase 3: Polygonization

- [ ] **Inc 9: First polygonize attempt** — Run `polygonize_full` on noded lines. Print polygon/dangle/cut/invalid counts. Plot all polygons with random colors.
  - Checkpoint: visually verify rooms are forming. This may require multiple iterations.
- [ ] **Inc 10: Separate outline, rooms, walls** — Identify building outline (union of all → exterior ring), filter wall-thickness polygons by configurable aspect ratio + area, keep rest as rooms
  - Checkpoint: plot outline (blue), rooms (green), walls (gray). Manual verification.
- [ ] **Inc 11: Iterative gap-closing** — If room count too low, retry with higher snap tolerance. Report: tolerance → polygon_count → room_count for different values.
  - Checkpoint: find sweet spot where room count stabilizes

---

## Phase 4: Room Classification

- [ ] **Inc 12: Point-in-polygon assignment** — For each room, find classification primitives inside it using `intersects()`. Group by instanceId for furniture.
  - Checkpoint: rooms with stairs/toilet/kitchen primitives correctly identified
- [ ] **Inc 13: Rule-based classification** — Apply priority rules: stairs > elevator > WC > kitchen > default (no bed/bath in poilabs dataset)
  - Checkpoint: Counter of facil_types shows expected distribution. Plot rooms colored by type.
- [ ] **Inc 14: Geometric heuristics for unlabeled rooms** — Elongated → Walkways, tiny → Deadzone, default → Office
  - Checkpoint: zero rooms with facil_type "Unknown"

---

## Phase 5: Georeferencing & Export

- [ ] **Inc 15: Affine transform computation** — `compute_affine(svg_pts, geo_pts)` → 6-param matrix. Note: geo_points are (longitude, latitude) order.
  - Checkpoint: test with known points, center of building lands at expected lat/lon
- [ ] **Inc 16: Apply transform to all polygons** — Transform every FloorplanFeature geometry from SVG to WGS84
  - Checkpoint: all coordinates near expected geographic location
- [ ] **Inc 17: GeoJSON export** — Build FeatureCollection with properties: id, facil_type, facil_name, level_id, floorLevel, isClickable, clickId
  - Checkpoint: valid GeoJSON, loads in geojson.io, rooms overlay on satellite imagery

---

## Phase 6: Integration

- [ ] **Inc 18: Full pipeline CLI** — `python main.py input.svg output.geojson --control-points cp.yaml --config config.yaml --floor-level 2`
  - Checkpoint: single command produces valid GeoJSON
- [ ] **Inc 19: Test on second SVG file** — Run pipeline on a different building. Document which parameters need per-file tuning.
  - Checkpoint: output GeoJSON loads correctly, rooms labeled reasonably
- [ ] **Inc 20: Batch processing** — Process multiple SVGs (multiple floors/buildings) in one run from batch config YAML
  - Checkpoint: multi-floor output with correct level_id per floor
- [ ] **Inc 21: Programmatic validation** — Automated checks: valid GeoJSON schema, no self-intersecting polygons, no overlapping rooms, area sanity checks
  - Checkpoint: validation passes on all processed files

---

## Dependency Graph

```
Inc 0.1-0.3 (foundation)
  └─→ Inc 1 (segments) → Inc 2 (arcs) → Inc 3 (circles/ellipses)
       └─→ Inc 4 (Y-flip + grouping)
            └─→ Inc 4.5 (hatching filter) ★ CRITICAL
                 └─→ Inc 5 (door/window closing)
                      │
                      ├─→ Inc 6 → Inc 7 → Inc 8 (cleaning pipeline)
                      │    └─→ Inc 9 (first polygonize) ★ HARDEST
                      │         └─→ Inc 10 (separation) → Inc 11 (gap-closing)
                      │              └─→ Inc 5.5 (synthetic closing, if needed)
                      │
                      └─→ Inc 12 (point-in-polygon) → Inc 13 (rules) → Inc 14 (heuristics)

Inc 11 + Inc 14 → Inc 15 (affine) → Inc 16 (transform) → Inc 17 (export)
                   └─→ Inc 18 (CLI) → Inc 19 (2nd file) → Inc 20 (batch) → Inc 21 (validation)

Note: Inc 15 (affine math) can be developed in parallel with anything.
      Inc 6-11 (cleaning/polygonize) and Inc 12-14 (classification) are parallel tracks.
```
