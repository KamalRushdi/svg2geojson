import json
import math
import argparse
from svgelements import SVG, Path, Line, Polyline, Polygon, Rect, Circle, Ellipse


def dist(p1, p2):
    return math.hypot(p2[0] - p1[0], p2[1] - p1[1])


def transform_point(x, y, height=None, flip_y=False):
    """
    Convert SVG point to output point.
    If flip_y is True, transform from SVG screen coordinates to Cartesian-like coordinates.
    """
    if flip_y and height is not None:
        return [float(x), float(height - y)]
    return [float(x), float(y)]


def ring_is_closed(coords, eps=1e-9):
    if len(coords) < 4:
        return False
    return dist(coords[0], coords[-1]) <= eps


def close_ring(coords):
    if not coords:
        return coords
    if coords[0] != coords[-1]:
        coords.append(coords[0])
    return coords


def sample_path(path, svg_height=None, flip_y=False, samples_per_segment=20):
    """
    Approximate an SVG path as a list of points.
    Curved segments are sampled.
    """
    coords = []

    for seg in path.segments():
        n = samples_per_segment

        # For straight lines, fewer samples are enough
        try:
            length = abs(seg.length())
            if length < 1e-9:
                continue
            if hasattr(seg, "start") and hasattr(seg, "end"):
                if seg.__class__.__name__.lower() == "line":
                    n = 2
                else:
                    # Slight adaptive behavior
                    n = max(8, min(samples_per_segment, int(length / 10) + 2))
        except Exception:
            n = samples_per_segment

        for i in range(n):
            t = i / (n - 1) if n > 1 else 0
            pt = seg.point(t)
            x = pt.real if isinstance(pt, complex) else pt.x
            y = pt.imag if isinstance(pt, complex) else pt.y
            coords.append(transform_point(x, y, svg_height, flip_y))

    # Remove consecutive duplicates
    cleaned = []
    for c in coords:
        if not cleaned or cleaned[-1] != c:
            cleaned.append(c)

    return cleaned


def polyline_to_coords(elem, svg_height=None, flip_y=False):
    coords = []
    for p in elem:
        coords.append(transform_point(p.x, p.y, svg_height, flip_y))
    return coords


def rect_to_coords(elem, svg_height=None, flip_y=False):
    x, y, w, h = elem.x, elem.y, elem.width, elem.height
    coords = [
        transform_point(x, y, svg_height, flip_y),
        transform_point(x + w, y, svg_height, flip_y),
        transform_point(x + w, y + h, svg_height, flip_y),
        transform_point(x, y + h, svg_height, flip_y),
    ]
    return close_ring(coords)


def circle_to_coords(elem, svg_height=None, flip_y=False, samples=64):
    coords = []
    for i in range(samples):
        t = 2.0 * math.pi * i / samples
        x = elem.cx + elem.rx * math.cos(t)
        y = elem.cy + elem.ry * math.sin(t)
        coords.append(transform_point(x, y, svg_height, flip_y))
    return close_ring(coords)


def ellipse_to_coords(elem, svg_height=None, flip_y=False, samples=64):
    coords = []
    for i in range(samples):
        t = 2.0 * math.pi * i / samples
        x = elem.cx + elem.rx * math.cos(t)
        y = elem.cy + elem.ry * math.sin(t)
        coords.append(transform_point(x, y, svg_height, flip_y))
    return close_ring(coords)


def element_properties(elem):
    props = {
        "svg_type": elem.__class__.__name__,
        "id": getattr(elem, "id", None),
        "stroke": str(getattr(elem, "stroke", "")) if getattr(elem, "stroke", None) is not None else None,
        "fill": str(getattr(elem, "fill", "")) if getattr(elem, "fill", None) is not None else None,
    }

    # Optional common attributes from custom SVG pipelines
    for attr in [
        "values",
        "class",
        "layer",
        "semanticId",
        "instanceId",
        "primitiveId",
        "originalType",
    ]:
        if hasattr(elem, attr):
            try:
                val = getattr(elem, attr)
                if val is not None:
                    props[attr] = str(val)
            except Exception:
                pass

    return {k: v for k, v in props.items() if v is not None}


def svg_to_geojson(svg_path, output_path, flip_y=False, samples_per_segment=20):
    svg = SVG.parse(svg_path)

    # Try to get SVG height for y-flip
    svg_height = None
    try:
        if svg.height is not None:
            svg_height = float(svg.height)
    except Exception:
        pass

    # Fall back to viewBox height if needed
    if svg_height is None:
        try:
            if svg.viewbox is not None:
                svg_height = float(svg.viewbox[3])
        except Exception:
            pass

    features = []

    for elem in svg.elements():
        if elem is None:
            continue

        geometry = None
        props = element_properties(elem)

        try:
            if isinstance(elem, Line):
                coords = [
                    transform_point(elem.x1, elem.y1, svg_height, flip_y),
                    transform_point(elem.x2, elem.y2, svg_height, flip_y),
                ]
                geometry = {
                    "type": "LineString",
                    "coordinates": coords,
                }

            elif isinstance(elem, Polyline):
                coords = polyline_to_coords(elem, svg_height, flip_y)
                if len(coords) >= 2:
                    geometry = {
                        "type": "LineString",
                        "coordinates": coords,
                    }

            elif isinstance(elem, Polygon):
                coords = polyline_to_coords(elem, svg_height, flip_y)
                coords = close_ring(coords)
                if len(coords) >= 4:
                    geometry = {
                        "type": "Polygon",
                        "coordinates": [coords],
                    }

            elif isinstance(elem, Rect):
                coords = rect_to_coords(elem, svg_height, flip_y)
                geometry = {
                    "type": "Polygon",
                    "coordinates": [coords],
                }

            elif isinstance(elem, Circle):
                coords = circle_to_coords(elem, svg_height, flip_y)
                geometry = {
                    "type": "Polygon",
                    "coordinates": [coords],
                }

            elif isinstance(elem, Ellipse):
                coords = ellipse_to_coords(elem, svg_height, flip_y)
                geometry = {
                    "type": "Polygon",
                    "coordinates": [coords],
                }

            elif isinstance(elem, Path):
                coords = sample_path(
                    elem,
                    svg_height=svg_height,
                    flip_y=flip_y,
                    samples_per_segment=samples_per_segment,
                )
                if len(coords) >= 2:
                    if ring_is_closed(coords):
                        coords = close_ring(coords)
                        geometry = {
                            "type": "Polygon",
                            "coordinates": [coords],
                        }
                    else:
                        geometry = {
                            "type": "LineString",
                            "coordinates": coords,
                        }

            if geometry is not None:
                features.append({
                    "type": "Feature",
                    "properties": props,
                    "geometry": geometry,
                })

        except Exception as e:
            features.append({
                "type": "Feature",
                "properties": {
                    **props,
                    "conversion_error": str(e),
                },
                "geometry": None,
            })

    geojson = {
        "type": "FeatureCollection",
        "features": features,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(geojson, f, indent=2)

    print(f"Wrote {len(features)} features to {output_path}")


def main():
    # parser = argparse.ArgumentParser(description="Convert SVG to local-coordinate GeoJSON.")
    # parser.add_argument("svg_path", help="Input SVG file")
    # parser.add_argument("output_path", help="Output GeoJSON file")
    # parser.add_argument(
    #     "--flip-y",
    #     action="store_true",
    #     help="Flip SVG y-axis so output behaves more like a Cartesian/GIS coordinate system",
    # )
    # parser.add_argument(
    #     "--samples-per-segment",
    #     type=int,
    #     default=20,
    #     help="Sampling density for curved path segments",
    # )
    # args = parser.parse_args()
    input_svg = "Arı6_1Kat_parça1_layered.svg"
    output_geojson = "Arı6_1Kat_parça1_layered_plus_y_flip.geojson"
    flip_y = True
    samples_per_segment = 20
    svg_to_geojson(
        input_svg,
        output_geojson,
        flip_y=flip_y,
        samples_per_segment=samples_per_segment,
    )


if __name__ == "__main__":
    main()