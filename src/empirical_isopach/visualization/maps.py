"""Offline cartography, complete survey frames and explicit contour extents."""
from functools import lru_cache
import gzip
import json
from pathlib import Path

import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter
from src.Laea import make_laea_transform
from src.probabilistic_isopach.global_volume import EarthDomain

INK = "#183843"
TEAL = "#087E8B"
AMBER = "#B8672C"
SOURCE = "#C34E32"
ASSET = Path(__file__).parents[1]/"assets/ne_10m_coastline.geojson.gz"
THICKNESS_LEVELS = np.array([v*10.**k for k in range(-1, 6) for v in (1, 3)])


@lru_cache(maxsize=1)
def coast_rings():
    """Losslessly compressed, pinned Natural Earth 1:10-million coastline."""
    with gzip.open(ASSET, "rt") as stream:
        features = json.load(stream)["features"]
    rings = []
    for feature in features:
        g = feature["geometry"]
        coordinates = [g["coordinates"]] if g["type"] == "LineString" else g["coordinates"]
        rings.extend(np.asarray(c, float) for c in coordinates if len(c) >= 2)
    return tuple(rings)


@lru_cache(maxsize=12)
def projected_coasts(vent):
    forward, _, _ = make_laea_transform(*vent)
    return tuple(np.column_stack(forward.transform(*ring.T)) for ring in coast_rings())


def coastlines(ax, vent, bounds=None):
    """Fine shorelines without changing map limits or adding geological evidence."""
    if bounds is None:
        bounds = np.array([*ax.get_xlim(), *ax.get_ylim()])[[0, 2, 1, 3]]
    x0, y0, x1, y1 = bounds
    lines = []
    for xy in projected_coasts(tuple(map(float, vent))):
        a, b = xy[:-1], xy[1:]
        valid = (np.isfinite(a).all(1) & np.isfinite(b).all(1)
                 & (np.linalg.norm(b-a, axis=1) < 500)
                 & (np.maximum(a[:, 0], b[:, 0]) >= x0)
                 & (np.minimum(a[:, 0], b[:, 0]) <= x1)
                 & (np.maximum(a[:, 1], b[:, 1]) >= y0)
                 & (np.minimum(a[:, 1], b[:, 1]) <= y1))
        # Retain contiguous sequences; do not draw hundreds of thousands of
        # independent SVG paths or connect islands across a rejected segment.
        cuts = np.flatnonzero(np.diff(np.r_[False, valid, False]))
        lines.extend(xy[start:end+1] for start, end in cuts.reshape(-1, 2))
    artist = LineCollection(lines, colors="#596F73", linewidths=.48,
                            alpha=.82, zorder=3, rasterized=True)
    artist.set_gid("natural-earth-10m-coastlines")
    ax.add_collection(artist, autolim=False)
    return artist


def frame_bounds(points, margin=.09, include_source=True):
    points = np.asarray(points, float)
    if include_source:
        points = np.vstack((points, [0., 0.]))
    low, high = points.min(0), points.max(0)
    span = np.maximum(high-low, max(float(np.ptp(points, axis=0).max())*.05, 1.))
    return np.r_[low-margin*span, high+margin*span]


def grid_axes(bounds, longest=321, minimum=101):
    span = np.asarray(bounds)[2:]-np.asarray(bounds)[:2]
    counts = np.maximum(minimum, np.ceil(longest*span/span.max()).astype(int))
    axes = []
    for lo, hi, n in zip(bounds[:2], bounds[2:], counts):
        v = np.linspace(lo, hi, n)
        if lo < 0 < hi:
            v = np.unique(np.r_[v, 0.])
        axes.append(v)
    return axes


def panel_layout(bounds, panels=2):
    """Wide footprints get stacked panels instead of unreadably small maps."""
    width, height = np.asarray(bounds)[2:]-np.asarray(bounds)[:2]
    ratio = width/height
    if panels == 2 and ratio > 1.6:
        return (2, 1), (12.0, 2*max(3.0, 10.6/ratio)+1.4)
    return (1, panels), (6.1*panels, min(10., max(5.5, 5.2/ratio+1.4)))


def setup_map(ax, bounds, vent):
    ax.set(xlim=(bounds[0], bounds[2]), ylim=(bounds[1], bounds[3]), aspect="equal",
           xlabel="Easting from source (km)", ylabel="Northing from source (km)")
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:,.0f}"))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:,.0f}"))
    ax.tick_params(labelsize=8)
    ax.grid(alpha=.14, linewidth=.5)
    coastlines(ax, vent, bounds)


def observations(ax, data, size=18, *, meta=None):
    interpreted = (meta or {}).get("interpreted_upper_limits", [])
    marked = np.isin(data.row_ids, [s["row_id"] for s in interpreted])
    data = data.reported_bounds()
    for mask, marker in ((~data.censored, "o"), (data.lower_censored & ~data.interval, "^"), (data.interval, "s"),
                         (data.censored & ~data.lower_censored & ~marked, "v")):
        if mask.any():
            ax.scatter(*data.xy[mask].T, s=size, marker=marker, facecolors="white",
                       edgecolors=INK, linewidths=.65, zorder=7)
    if marked.any():
        ax.scatter(*data.xy[marked].T, s=max(32, size*1.5), marker="D", facecolors="#f5be45",
                   edgecolors="#332807", linewidths=.8, zorder=8)
    ax.scatter(0, 0, marker="*", s=105, c=SOURCE, edgecolors="white", lw=.7, zorder=8)


def observation_legend(fig, data, extra=(), *, meta=None, columns=None):
    handles = [Line2D([], [], marker="o", color=INK, markerfacecolor="white", ls="", label="Exact thickness"),
               Line2D([], [], marker="*", color=SOURCE, ls="", markersize=9, label="Source")]
    data = data.reported_bounds()
    for present, marker, label in ((np.any(data.lower_censored & ~data.interval), "^", "Lower bound"),
                                   (data.interval.any(), "s", "Thickness interval"),
                                   (np.any(data.censored & ~data.lower_censored), "v", "Upper bound")):
        if present:
            handles.append(Line2D([], [], marker=marker, color=INK, markerfacecolor="white", ls="", label=label))
    if (meta or {}).get("interpreted_upper_limits"):
        handles.append(Line2D([], [], marker="D", color="#332807", markerfacecolor="#f5be45",
                              ls="", label="Interpreted archive limit"))
    entries = [*handles, *extra]
    fig.legend(handles=entries, loc="outside lower center", ncols=columns or min(5,len(entries)),
               fontsize=8, frameon=False)


def level_labels(value):
    number = 10.**value
    return f"{number:g}" if number < 1000 else f"{number:,.0f}"


def contour_levels(values):
    finite = np.asarray(values)[np.isfinite(values)]
    if not len(finite):
        return np.array([])
    levels = np.log10(THICKNESS_LEVELS)
    return levels[(levels > finite.min()) & (levels < finite.max())]


def boundary_points(bounds, n=257):
    x0, y0, x1, y1 = bounds
    x, y = np.linspace(x0, x1, n), np.linspace(y0, y1, n)
    return np.vstack((np.column_stack((x, np.full(n, y0))), np.column_stack((x, np.full(n, y1))),
                      np.column_stack((np.full(n, x0), y)), np.column_stack((np.full(n, x1), y))))


def global_frame(data_xy, latitude, evaluate, minimum_cm=.1):
    """Search the entire physical chart, then pad and check the display edge.

    `evaluate` returns central log10 thickness. This only chooses a viewport:
    neither this threshold nor any raster enters a volume calculation.
    Finite grids cannot certify unseen features; an explicit edge audit and
    whole-Earth companion are always published.
    """
    domain = EarthDomain.wgs84(latitude)
    radius, axis = domain.rho_max, domain.axis_scale
    earth = np.array([-radius*axis, -radius/axis, radius*axis, radius/axis])
    theta = np.linspace(0, 2*np.pi, 360, endpoint=False)
    # Both uniform-area coverage and dense near-source radii are used so an
    # isolated, displaced fan cannot be missed by a local perimeter search.
    radial = np.unique(np.r_[np.linspace(0, radius*.999999, 81),
                             np.geomspace(max(1., np.linalg.norm(data_xy, axis=1).min()/8), radius*.999999, 64)])
    rr, tt = np.meshgrid(radial, theta)
    xy = np.column_stack((rr.ravel()*np.sin(tt.ravel())*axis,
                          rr.ravel()*np.cos(tt.ravel())/axis))
    z = evaluate(xy)
    chosen = xy[z >= np.log10(minimum_cm)]
    points = np.vstack((data_xy, [[0., 0.]], chosen))
    bounds = frame_bounds(points, margin=.1)
    bounds[:2] = np.maximum(bounds[:2], earth[:2])
    bounds[2:] = np.minimum(bounds[2:], earth[2:])
    checks = []
    for iteration in range(9):
        edge = boundary_points(bounds)
        physical = domain.rho(edge) < radius*.999999
        values = evaluate(edge[physical]) if physical.any() else np.array([])
        maximum = None if not len(values) else float(np.max(values))
        checks.append({"bounds_km": bounds.tolist(), "maximum_edge_log10_cm": maximum})
        if maximum is None or maximum < np.log10(minimum_cm) or iteration == 8:
            break
        span = bounds[2:]-bounds[:2]
        expanded = np.r_[np.maximum(bounds[:2]-.15*span, earth[:2]),
                          np.minimum(bounds[2:]+.15*span, earth[2:])]
        if np.array_equal(expanded, bounds):
            break
        bounds = expanded
    rim = xy[rr.ravel() >= radius*.99999]
    rim_max = float(np.max(evaluate(rim)))
    max_edge = checks[-1]["maximum_edge_log10_cm"]
    return bounds, {"minimum_display_contour_cm": minimum_cm, "search_points": len(xy),
        "edge_refinements": checks, "physical_rim_maximum_log10_cm": rim_max,
        "lowest_contour_within_chart": bool((max_edge is None or max_edge < np.log10(minimum_cm))
                                             and rim_max < np.log10(minimum_cm)),
        "scope": "Display-only finite-grid extent audit, not contour-support validation or an integration boundary."}
