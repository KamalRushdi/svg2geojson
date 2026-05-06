# Phase 4 Room Classification — Decisions and Pipeline

This document captures the design decisions for Inc 12, Inc 13, and Inc 14 (room classification) made in the session that built them on top of Inc 10. It is a working reference, not a spec — when behavior changes, update both the code and this file.

---

## The full decision tree

For every polygon coming out of `polygonize_full`, the pipeline assigns one of:

`Door | Window | Wall | Column | Stairs | Escalator | Elevator | WC | Kitchen | Auditorium | Office | Parking | Walkways | Deadzone`

The order below is the **order of decisions**, not the priority of types. Earlier stages "claim" a polygon and remove it from later consideration.

### Stage 1 — Inc 10 separation ([src/separator.py](../src/separator.py))

| Step | What happens |
|------|--------------|
| 1.0 split openings | Polygons that match a door/window AABB by IoU > 0.5 → `result.doors`/`result.windows`. |
| 1.5 hatching match | Polygons containing a hatching-stroke midpoint → ground-truth walls (`hatched_ids`). |
| **1.6 object-room lock** *(Inc 12, see below)* | Polygons containing an object **instance centroid** → 100%-rooms, bypass all later wall filters. |
| 2 noise filter | Remaining polygons with `area < 25` → walls. |
| 2.5 columns | Remaining polygons that are small AND ~square → walls. |
| 3 sliver filter | Remaining polygons with thickness < min → walls. |
| 4 threshold | Remaining: largest-gap on log10(thickness), Otsu fallback, hatching-calibrated upward. |
| 5 classify | Polygon thickness above threshold → room; below → wall. |
| 5.5 synth thin walls | Single-stroke wall LineStrings buffered into thin polygons. |
| 6 merge | Connected wall components unioned. |

After Stage 5, `result.rooms` holds `RoomPolygon` objects with `facil_type=None`. Locked rooms also carry their `contained_semantics` dict.

### Stage 2 — Inc 13 priority classification ([src/classifier.py](../src/classifier.py))

For every room with non-empty `contained_semantics`, walk this priority list and assign `facil_type` from the first match:

```
1. Stairs        (semantic_id 28)
2. Escalator     (30)
3. Elevator      (29)
4. WC            (25, 26, 27)
5. Kitchen       (19)
6. Auditorium    (31 — row chairs)
7. Office        (11, 13, 14, 17 — sofa, chair, table, cabinet)
8. Parking       (32)
```

Anything that only contained a railing (35) or an unrecognized id falls through to `Deadzone`.

**Rationale**: "functional rooms beat generic rooms". A stair landing that happens to contain a chair is still a stair landing. A WC with a handwashing sink stays a WC because WC is higher than Kitchen. Presence-is-enough — one toilet primitive is sufficient to call a room WC; we don't require multiple.

### Stage 3 — Inc 13 refinement: kitchen-near-WC

A "Kitchen" assigned in Stage 2 means "this room contains a sink primitive (id 19)". Sinks adjacent to WCs in real buildings are handwashing sinks. So any room that:
- Is currently `facil_type == "Kitchen"`, AND
- Lies within `buffer_distance` (default 5.0 SVG units, ≈ one wall thickness) of a WC polygon

…gets reclassified to WC.

Implementation: buffer all WC polygons, take their union, test each Kitchen polygon for `intersects` against that buffered union. Single linear sweep.

### Stage 4 — Inc 14 geometric heuristics

Rooms still unclassified after Stage 3 have no detectable objects inside. They get Walkway or Office based on two signals — **either** one triggers Walkway:

| Signal | Default threshold | What it catches |
|--------|-------------------|-----------------|
| **Thickness** = `2 × maximum_inscribed_circle.radius` | `< 14.0` | Narrow corridors regardless of length. ARI 8 corridors (12.86 wide) end up as Walkways. |
| **Solidity** = `polygon.area / convex_hull.area` | `< 0.8` | L/U/T-shaped polygons — a corridor that turns a corner has solidity ~0.7, a U-shape ~0.6, a rectangle ~1.0. |

Both thresholds are kwargs of `classify_by_geometry`; tune per data scale.

**Why both signals**: thickness alone misses wide L-shaped corridors; aspect ratio alone is fragile to polygonize noise. Solidity is the L/U/T detector and it's cheap.

---

## Why rooms get locked early (Inc 12)

The original Inc 10 pipeline was wall-conservative: if a polygon's thickness fell below threshold, it became a wall. That misclassified small-but-real rooms (tiny WCs, narrow stair landings) as walls, then Stage 6 merged them into the wall mass, making them unrecoverable.

The fix is a **symmetric counterpart to the existing hatching ground-truth**:
- Hatching primitive midpoint inside a polygon → ground-truth wall (existing).
- Object-instance centroid inside a polygon → ground-truth room (Inc 12).

Wall priority on ties: a polygon both hatched AND object-bearing stays a wall. Reasoning: hatching is structural drawing inside walls — if hatching is present, it's a wall regardless of any spurious object centroid that happens to land there.

### Why **instance** centroids, not per-primitive centroids

A single physical sofa is drawn as ~50 line segments. Each segment has its own centroid, scattered across the sofa's drawing strokes. Many fall on the room boundary or just outside it. With per-primitive testing, we either:
- Lock too aggressively (any single segment outside the polygon = miss for that segment), or
- Lock too noisily (a thin polygon between two real rooms catches a stray sofa segment).

`_instance_centroids()` groups primitives by `instance_id`, computes `unary_union(geometries).centroid` for each instance, and returns one test point per real object. The centroid of all sofa parts lands at the sofa's true center, which is reliably inside the room polygon.

---

## Why Inc 14 needs both thickness AND solidity

Tested four heuristics on the 5 reference samples:

1. **Aspect ratio** (short/long of OBB): catches narrow corridors well, but fragile when polygonize produces near-rectangular polygons that aren't quite axis-aligned.
2. **Thickness alone**: works for straight corridors (low thickness), misses L-shaped corridors (their inscribed circle still fits the corridor width).
3. **Solidity alone**: catches L/U/T-shapes, but misses straight corridors (solidity ~1.0 for a long thin rectangle).
4. **Thickness OR solidity**: catches both, defaults work across all 5 samples.

Default thresholds chosen by inspection of the 5 samples:
- `walkway_max_thickness = 14.0` — separates ARI 8's 12.86-wide corridors from its 15.43-wide smallest offices.
- `walkway_max_solidity = 0.8` — clean rectangles (~1.0) stay Office, L/U/T-shapes (~0.7-0.6) become Walkway.

---

## Color scheme — derived from SVG, not invented

`TYPE_COLORS` at [src/classifier.py:38-58](../src/classifier.py#L38-L58) maps each `facil_type` to a hex color. Each color is the **actual stroke color of the dominant semantic primitive** sampled across the 5 reference floorplans:

| Type | Color | Source |
|------|-------|--------|
| Stairs | `#f7ce4b` yellow | id 28 (stair primitive stroke) |
| Escalator | `#f7ce4b` yellow | not present in data; reuses Stairs |
| Elevator | `#ed702d` orange | id 29 |
| WC | `#ee7ca2` pink | id 27 toilet (canonical for 25/26/27) |
| Kitchen | `#719752` green | id 19 sink |
| Auditorium | `#426b51` dark green | not present; reuses chair color (id 13) |
| Office | `#7ab591` light green | id 11 sofa (canonical for 11/13/14/17) |
| Parking | `#66433e` brown | id 32 |
| Walkways | `#e5e7eb` smoked white | no semantic source — neutral fallback |
| Deadzone | `#cbd5e1` light slate | no semantic source — neutral fallback |

For room types that aggregate multiple semantic IDs (WC, Office), we picked the most recognizable/most-common single ID as the canonical color.

---

## Visualization gotchas (in `tests/inc13_checkpoint.py`)

### `_fill_polygon` hole-rendering
The original helper drew a polygon with holes by:
1. Filling the exterior at `alpha=0.85`,
2. Then painting each hole with **opaque white** to "subtract" it.

That second step also painted opaque white over any room layer drawn underneath, making rooms invisible behind walls. Fixed by switching to `matplotlib.path.Path` + `PathPatch` with the even-odd fill rule. Holes are now truly transparent.

### Render order
Rooms are sorted **by descending area** before painting. The largest polygons (typically the SVG-exterior leftover) paint first; smaller real rooms paint on top so they remain visible. This is the only mitigation for the SVG-exterior issue right now — the dedicated filter is documented separately in [svg_exterior_filter.md](svg_exterior_filter.md) and stays disabled until we work on full-building SVGs.

### Centroid overlay (Panel 3)
Panel 3 of the Inc 13 plot overlays one colored dot per object instance — same point the locker tested. If a dot lands on a wall or in a Walkway/Deadzone polygon, it means Inc 12 didn't lock the room we expected. This is the visual diagnostic we rely on when classifications look off.

---

## Knobs to tune per dataset

If a future dataset has different scale or unusual layouts, here are the levers:

| Lever | Default | Effect |
|-------|---------|--------|
| `reclassify_kitchen_near_wc(buffer_distance=)` | 5.0 | Smaller → only adjacent rooms (sharing wall) reclassify. Larger → 2-rooms-away kitchens also flip to WC. |
| `classify_by_geometry(walkway_max_thickness=)` | 14.0 | Higher → wider rooms become Walkways. |
| `classify_by_geometry(walkway_max_solidity=)` | 0.8 | Higher → more rooms with notches/columns flip to Walkway. |
| Priority list at [src/classifier.py:31-40](../src/classifier.py#L31-L40) | as listed | Reorder entries to change tie-breaking. |

---

## Final breakdown across the 5 reference samples

| Sample | Stairs | Elevator | WC | Office | Walkways | Total rooms |
|--------|-------:|---------:|---:|-------:|---------:|------------:|
| ABLOK_12KAT_parça1 | 1 | 2 | 2 | 5 | 7 | 17 |
| HKÜ_HUKUK_3KAT_parça1 | 0 | 0 | 0 | 4 | 2 | 6 |
| ARI 8_1.kat_parça1_layered | 1 | 0 | 0 | 7 | 12 | 20 |
| ESKİŞEHİR_ADLİYESİ_1.KAT_parça15 | 0 | 0 | 0 | 3 | 1 | 4 |
| GaziUni_2Kat_parça2 | 1 | 0 | 6 | 3 | 2 | 12 |

Verify with `python -m tests.inc13_checkpoint` — outputs to [output/inc13/](../output/inc13/).
