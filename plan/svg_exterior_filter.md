# SVG-Exterior Polygon Filter (deferred)

When the pipeline runs on a **full-building SVG** (not a partial slice), you'll need this filter to hide the canvas-spanning leftover polygon that `polygonize_full` produces. This note captures what it did, why it existed, and the exact code so you can drop it back in when the time comes.

## What the filter does

`polygonize_full` runs on the noded line set, which includes the SVG viewBox rectangle Inc 5 added as a boundary anchor. That gives you N+1 polygons:

- **N rooms** inside the building.
- **1 leftover polygon** filling the area between the building outline and the canvas edge — bounds span the full viewBox, no objects inside, just empty space "outside the building".

For HKÜ this leftover looks like:

```
ROOM area=7609.9   bounds=(0.0-140.0, 0.0-140.0)   contained_semantics={}
```

## Why we need to filter it (in visualization)

Without filtering:

1. Inc 12 doesn't lock it (no object centroids inside).
2. Inc 13 leaves `facil_type=None` (empty `contained_semantics`).
3. **Inc 14 measures its OBB aspect ratio ≈ 1.0** (canvas is roughly square), assigns `facil_type = "Office"`.
4. Plot paints it as a giant green rectangle covering the entire canvas. Real rooms still paint over it (so they remain visible thanks to the size-descending sort), but the margin around the building shows up as misleading "Office green".

## Filter code

Insertion point: in `plot_three_panel` at [tests/inc13_checkpoint.py](../tests/inc13_checkpoint.py), inside the Panel 3 room-rendering loop. Replace the simple iteration with this:

```python
# ---- Panel 3: classified rooms + walls + door/window overlay ----
# Compute a "this polygon is the SVG-exterior leftover" detector: a
# polygon whose bbox spans most of the canvas AND has no semantics.
# Painting it would cover the real rooms with a misleading fill.
bbox_span = max(viewbox_height, 1e-6)

def _is_svg_exterior(geom: Polygon) -> bool:
    b = geom.bounds
    width = b[2] - b[0]
    height = b[3] - b[1]
    return (width >= 0.85 * bbox_span and height >= 0.85 * bbox_span)

# Sort: classified rooms last, so they paint over unclassified.
rooms_in_z_order = sorted(
    sep.rooms,
    key=lambda r: (1 if (r.facil_type and r.facil_type in TYPE_COLORS) else 0)
)
for r in rooms_in_z_order:
    if not isinstance(r.geometry, Polygon):
        continue
    if _is_svg_exterior(r.geometry) and not r.contained_semantics:
        continue   # don't paint the canvas-spanning exterior leftover
    if r.facil_type and r.facil_type in TYPE_COLORS:
        color = TYPE_COLORS[r.facil_type]
        _fill_polygon(ax3, r.geometry, facecolor=color, alpha=0.65,
                      edgecolor=color, linewidth=0.4)
    else:
        _fill_polygon(ax3, r.geometry, facecolor=UNCLASSIFIED_FILL,
                      alpha=0.5, edgecolor=UNCLASSIFIED_EDGE,
                      linewidth=0.4)
```

## How the detector decides

Two conditions, both required:

1. **Bbox covers ≥ 85% of viewbox in both dimensions.** The 85% (instead of 100%) gives slack for buildings that extend right up to one canvas edge.
2. **Empty `contained_semantics`.** Guard against false positives — if a polygon legitimately spans most of the canvas AND has objects inside (e.g., one big open hall), it's a real room and stays.

## Important: render-only, not data

The filter only skips the polygon during plotting. The polygon **is still in `sep.rooms` and still counted in the breakdown**. That's why you'd see e.g. `Office=5` in the stats while only 4 Office rooms appear in the panel — the 5th is the hidden canvas-spanning exterior.

## Cleaner long-term fix (when shipping)

The filter belongs deeper in the pipeline. Better to drop the SVG-exterior polygon **inside the separator** so it never reaches `result.rooms`. Candidates:

- [src/separator.py](../src/separator.py) Stage 1 — recognize the SVG-boundary rectangle anchor and drop the polygon whose envelope matches it.
- [src/separator.py](../src/separator.py) Stage 2 — add a "drop polygons whose bbox covers >85% of viewbox AND has no contained_semantics" rule, after Inc 12 has already populated the lock map.

That way the breakdown counts also reflect reality (no phantom Office) and any downstream consumer (Inc 17 GeoJSON export, etc.) doesn't have to reapply the filter.
