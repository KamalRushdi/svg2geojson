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

- [x] **Inc 5: Door and window closing (revised)** — Build a simple AABB rectangle per door/window instance from segment endpoints. Emit 4 LineStrings (rectangle edges) into the boundary set so polygonize_full can close adjacent wall loops. Wall-thickness matching is deferred to a post-Inc 10 resize step. See [plan/inc5_rework.md](inc5_rework.md).
  - Checkpoint: [output/inc5_v2/](../output/inc5_v2/) — single bbox per instance.
- [x] **Inc 5 v3: Multi-leg window splitting** — For window instances whose convex_hull/aabb area ratio is < 0.75 (concave envelope, indicating L/U/T-shape), run k-means on segment midpoints (k=2..3) and pick the lowest k whose silhouette score exceeds 0.3. Each cluster becomes its own bbox + 4 edges + RoomPolygon.
  - Checkpoint: [output/inc5_v3/](../output/inc5_v3/) — GaziUni inst 10 splits into 2 thin rectangles, no false positives on regular rectangles.
- [x] **Inc 5 SVG-boundary anchor** — Add a 4-LineString rectangle for the SVG viewBox (in [src/svg_parser.py](../src/svg_parser.py) `_create_boundary_rectangle`). Snap wall endpoints within `boundary_snap_tolerance` (default 0.5 SVG units) of any viewBox edge onto that edge so polygons can close cleanly against the SVG boundary.
  - Checkpoint: total area of polygonization equals viewBox area on samples without internal voids; dangles drop dramatically.
- [ ] **Inc 5 resize (deferred)** — After Inc 10 wall identification, find each door/window polygon's adjacent wall polygons and resize the door/window short axis to match the wall thickness. Re-polygonize so room polygons recover the area previously eaten by oversized door bboxes. See "Future Ideas" in [plan/inc5_rework.md](inc5_rework.md).
  - Checkpoint: door/window polygons sit exactly within wall thickness; rooms expand to true size.

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

- [x] **Inc 9: First polygonize attempt** — Run `polygonize_full` on noded lines. Print polygon/dangle/cut/invalid counts. Plot all polygons with random colors.
  - Checkpoint: [output/inc9/](../output/inc9/) — rooms forming on all 5 samples.
- [x] **Inc 9 + 5v3: Full pipeline checkpoint** — End-to-end pipeline (parse → hatching filter → close openings with multi-leg splitting → snap to SVG boundary → clean → polygonize). 3-panel plot: full building (semantic colors) | boundary only | polygonization with door/window overlays.
  - Checkpoint: [output/inc9_inc5v3/](../output/inc9_inc5v3/).
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
       └─→ Inc 4 (Y-flip + grouping) + SVG boundary rect + boundary-snap
            └─→ Inc 4.5 (hatching filter) ★ CRITICAL
                 └─→ Inc 5 (bbox closing, no wall-snap)
                      └─→ Inc 5 v3 (multi-leg window splitting via concavity + k-means)
                           │
                           ├─→ Inc 6 → Inc 7 → Inc 8 (cleaning pipeline)
                           │    └─→ Inc 9 (first polygonize) ★ HARDEST
                           │         └─→ Inc 10 (separation) → Inc 5 resize → Inc 11 (gap-closing)
                           │
                           └─→ Inc 12 (point-in-polygon) → Inc 13 (rules) → Inc 14 (heuristics)

Inc 11 + Inc 14 → Inc 15 (affine) → Inc 16 (transform) → Inc 17 (export)
                   └─→ Inc 18 (CLI) → Inc 19 (2nd file) → Inc 20 (batch) → Inc 21 (validation)

Note: Inc 15 (affine math) can be developed in parallel with anything.
      Inc 6-11 (cleaning/polygonize) and Inc 12-14 (classification) are parallel tracks.
```
