"""Debug visualization helpers for the SVG-to-GeoJSON pipeline.

Provides matplotlib-based plotting for every pipeline stage. Each function
accepts either raw Shapely geometries or the pipeline's own dataclasses
(FloorplanPrimitive, RoomPolygon), making it easy to visualize intermediate
results during development and parameter tuning.

Typical usage at each pipeline stage:

    # After parsing (Inc 1-4): see all wall segments colored by semantic ID
    plot_lines(boundary_primitives, color_by="semantic_id", title="Walls")

    # After hatching filter (Inc 4.5): compare before/after
    plot_side_by_side(before, after, "Before filter", "After filter")

    # After polygonization (Inc 9-10): see rooms colored by type
    plot_polygons(room_polygons, color_by="facil_type", title="Rooms")

All functions return the matplotlib Axes object for further customization
and optionally save to a file via save_path.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon as MplPolygon
from shapely.geometry import LineString, Polygon

from src.models import FloorplanPrimitive, RoomPolygon

# Semantic ID -> hex color (derived from FloorplanCAD RGB values).
# Only IDs present in the poilabs dataset subset are included.
# Removed classes (0-indexed): [35,3,5,7,8,9,11,14,15,17,19,20,21,22,23]
SEMANTIC_COLORS = {
    # Boundaries
    33: "#a75c20",  # wall
    34: "#7968b2",  # curtain wall
    # Doors (4=folding, 6=rolling removed)
    1: "#e03e9b",   # single door
    2: "#9d2265",   # double door
    3: "#e8745b",   # sliding door
    5: "#ac6b85",   # revolving door
    # Windows (8=bay, 9=blind, 10=opening removed)
    7: "#604ef5",   # window
    # Furniture (12=bed, 15=TV, 16=wardrobe removed)
    11: "#7ab591",  # sofa
    13: "#426b51",  # chair
    14: "#7bb572",  # table
    17: "#91b670",  # cabinet
    # Fixtures (18=stove, 20=fridge, 21=ac, 22=bath, 23=bathtub removed)
    19: "#719752",  # sink
    24: "#504893",  # washing machine
    25: "#646c3b",  # squat toilet
    26: "#b6aa70",  # urinal
    27: "#ee7ca2",  # toilet
    # Vertical transport
    28: "#f7ce4b",  # stairs
    29: "#ed702d",  # elevator
    30: "#e93b2e",  # escalator
    # Other
    31: "#ac6b97",  # row chairs
    32: "#66433e",  # parking spot
    35: "#403469",  # railing
}

# Room type -> color for classified polygon plots.
# No "Bedroom" — bed/wardrobe removed from poilabs dataset.
FACIL_COLORS = {
    "Wall": "#888888",
    "Door": "#e03e9b",
    "Stairs": "#f7ce4b",
    "Elevator": "#ed702d",
    "WC": "#6baed6",
    "Kitchen": "#fd8d3c",
    "Office": "#74c476",
    "Walkways": "#d9d9d9",
    "Deadzone": "#bdbdbd",
}

DEFAULT_COLOR = "#333333"


def plot_lines(
    lines: Sequence[LineString | FloorplanPrimitive],
    color_by: str = "semantic_id",
    title: str = "Lines",
    ax: plt.Axes | None = None,
    save_path: Path | None = None,
) -> plt.Axes:
    """Plot LineString geometries on a matplotlib axes.

    Args:
        lines:     List of Shapely LineStrings or FloorplanPrimitive dataclasses.
        color_by:  "semantic_id" to color by FloorplanCAD class (uses
                   SEMANTIC_COLORS map), or "uniform" for all-same color.
        title:     Plot title string.
        ax:        Existing axes to draw on, or None to create a new figure.
        save_path: If set, saves the figure as a PNG at this path.

    Returns:
        The matplotlib Axes with the plotted lines.
    """
    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=(14, 14))

    for item in lines:
        if isinstance(item, FloorplanPrimitive):
            geom = item.geometry
            if color_by == "semantic_id" and item.semantic_id is not None:
                color = SEMANTIC_COLORS.get(item.semantic_id, DEFAULT_COLOR)
            else:
                color = DEFAULT_COLOR
        else:
            geom = item
            color = DEFAULT_COLOR

        if isinstance(geom, LineString):
            coords = np.array(geom.coords)
            ax.plot(coords[:, 0], coords[:, 1], color=color, linewidth=0.5)

    ax.set_aspect("equal")
    ax.set_title(title)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    return ax


def plot_polygons(
    polygons: Sequence[Polygon | RoomPolygon],
    color_by: str = "facil_type",
    title: str = "Polygons",
    ax: plt.Axes | None = None,
    save_path: Path | None = None,
    show_legend: bool = True,
) -> plt.Axes:
    """Plot Shapely Polygons or RoomPolygons on a matplotlib axes.

    Args:
        polygons:     List of Shapely Polygons or RoomPolygon dataclasses.
        color_by:     "facil_type" to color by room classification (uses
                      FACIL_COLORS map), or "random" for random colors.
        title:        Plot title string.
        ax:           Existing axes to draw on, or None to create a new figure.
        save_path:    If set, saves the figure as a PNG at this path.
        show_legend:  Whether to show the facil_type color legend.

    Returns:
        The matplotlib Axes with the plotted polygons.
    """
    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=(14, 14))

    seen_types = set()

    for item in polygons:
        if isinstance(item, RoomPolygon) and color_by == "facil_type":
            geom = item.geometry
            ft = item.facil_type or "Unknown"
            color = FACIL_COLORS.get(ft, DEFAULT_COLOR)
        else:
            geom = item.geometry if isinstance(item, RoomPolygon) else item
            ft = "Unknown"
            color = np.random.rand(3)

        if isinstance(geom, Polygon):
            coords = np.array(geom.exterior.coords)
            patch = MplPolygon(coords, alpha=0.4, facecolor=color, edgecolor="black", linewidth=0.3)
            ax.add_patch(patch)
            if ft not in seen_types and show_legend:
                ax.plot([], [], color=color, label=ft, linewidth=5)
                seen_types.add(ft)

    ax.set_aspect("equal")
    ax.autoscale_view()
    ax.set_title(title)
    if show_legend and seen_types:
        ax.legend(loc="upper right", fontsize=8)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    return ax


def plot_side_by_side(
    left: Sequence,
    right: Sequence,
    left_title: str = "Before",
    right_title: str = "After",
    plot_fn: str = "lines",
    save_path: Path | None = None,
) -> tuple[plt.Axes, plt.Axes]:
    """Plot two datasets side-by-side for before/after comparison.

    Useful for visualizing the effect of filtering or cleaning steps.
    Both panels share the same plot type (lines or polygons).

    Args:
        left:        Data for the left panel (list of LineStrings/Primitives
                     or Polygons/RoomPolygons).
        right:       Data for the right panel (same types as left).
        left_title:  Title for the left panel.
        right_title: Title for the right panel.
        plot_fn:     "lines" to use plot_lines, "polygons" to use plot_polygons.
        save_path:   If set, saves the combined figure as a PNG.

    Returns:
        Tuple of (left_axes, right_axes).
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(28, 14))

    if plot_fn == "lines":
        plot_lines(left, title=left_title, ax=ax1)
        plot_lines(right, title=right_title, ax=ax2)
    else:
        plot_polygons(left, title=left_title, ax=ax1)
        plot_polygons(right, title=right_title, ax=ax2)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    return ax1, ax2
