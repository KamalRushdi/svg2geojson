# svg2geojson

Convert CAD floorplan SVGs into GeoJSON room polygons.

> Experimental. The pipeline (parsing → cleaning → polygonization → room separation → classification) works on the included sample floorplans, but the API, CLI flags, and output schema are still in flux.

## Requirements

- Python 3.13
- [uv](https://docs.astral.sh/uv/) for environment and dependency management

## Setup

```bash
# Clone, then from the project root:
uv sync
```

`uv sync` creates `.venv/` and installs everything in `pyproject.toml` (shapely, geopandas, scipy, numpy, pyproj, matplotlib, pyyaml).

## Usage

The CLI takes one or more SVG inputs and writes a single GeoJSON `FeatureCollection`.

```bash
uv run python -m main \
    --input "input/sample/AB2_2KAT_parça18.svg=2" \
    --input "input/sample/ABLOK_12KAT_parça1.svg=12" \
    --output output/demo.geojson \
    --name "demo_export"
```

### Arguments

| Flag | Description |
| --- | --- |
| `--input path[=level_id]` | SVG to process. Repeat for multi-floor exports. If `level_id` is omitted, it's inferred from the filename (e.g. `_12KAT_` → `"12"`, `_1BODRUM_` → `"-1"`, `_ZeminKat_` → `"0"`). |
| `--output` | Path to the output `.geojson` file. |
| `--name` | Value for the top-level GeoJSON `name` field. Default: `svg2geojson_export`. |
| `--window-as` | Emit window features as `Wall` (default) or `Window`. |

### Example: single floor with inferred level

```bash
uv run python -m main \
    --input input/sample/ARI_3_10Kat_4.Parça.svg \
    --output output/ari3_10kat.geojson
```

The level id is read from `_10Kat_` in the filename, so no `=...` is needed.

## Running the test / checkpoint scripts

The `tests/` folder holds incremental "checkpoint" scripts — each one runs a stage of the pipeline on the sample SVGs and writes diagnostic plots / SVGs / GeoJSON into `output/<inc-name>/`. They are run as modules from the project root:

```bash
# Latest checkpoint (Inc 15 — arc-door absorption + classification)
uv run python -m tests.inc15_checkpoint

# Earlier checkpoints
uv run python -m tests.inc13_checkpoint
uv run python -m tests.inc14_checkpoint
uv run python -m tests.inc10_checkpoint
```

You can also list and run any of them:

```bash
ls tests/*_checkpoint.py
uv run python -m tests.<checkpoint_name>   # e.g. tests.inc9_inc5v3_checkpoint
```

Outputs are written to `output/<inc-name>/` — open the generated PNGs / SVGs to inspect each stage visually.

## Notes

- Coordinates are currently emitted using a placeholder affine derived from the SVG viewBox. Real-world georeferencing requires control points and `compute_affine` from `src/georeferencer.py`.
- Sample SVGs in `input/sample/` are partial slices of larger Turkish CAD floorplans (AB2, ABLOK, ARI series).
- Inputs and outputs are gitignored; only the pipeline code is tracked.

## Project layout

```
main.py            # CLI entry point
src/               # pipeline modules (parser, cleaning, polygonizer, classifier, ...)
tests/             # checkpoint scripts for incremental development
input/sample/      # sample SVG floorplans
output/            # generated GeoJSON and debug PNGs
```
