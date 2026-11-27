"""Place labels, true geographic grid and regional locator for observation plates."""
import csv
from functools import lru_cache
import gzip
from hashlib import sha256
from pathlib import Path

from matplotlib.patches import Rectangle
from matplotlib.ticker import MaxNLocator
from matplotlib.transforms import Bbox
import numpy as np

from benchmarks.observation_relief import terrain_rgb
from src.Laea import make_laea_transform
from src.empirical_isopach.visualization.deposit_maps import regular_axes
from src.empirical_isopach.visualization.shorelines import ASSETS as SHORELINE_ASSETS
from src.empirical_isopach.visualization.shorelines import TIERS, index, local_mask

ROOT = Path(__file__).resolve().parents[1]
PLACES = ROOT / "src/empirical_isopach/assets/ne_10m_places.csv.gz"
MUTED = "#436473"
ACCENT = "#912c55"
MAP_RESOLUTION = 960
LOCATOR_RESOLUTION = 440
SHORELINE_LIMITS = {"f": .3, "h": 1.5}


def shoreline_tier(bounds, resolution):
    spacing = max(bounds[2:] - bounds[:2]) / resolution
    return next((tier for tier, limit in SHORELINE_LIMITS.items() if spacing <= limit), "i")


def shoreline_provenance():
    pack_index = SHORELINE_ASSETS / "index.json"
    return {
        "source": index()["version"],
        "archive_sha256": index()["source_sha256"],
        "pack_index": pack_index.relative_to(ROOT).as_posix(),
        "pack_index_sha256": sha256(pack_index.read_bytes()).hexdigest(),
        "map_samples_long_side": MAP_RESOLUTION,
        "locator_samples_long_side": LOCATOR_RESOLUTION,
        "km_per_sample_max_by_tier": {TIERS[tier]: limit for tier, limit in SHORELINE_LIMITS.items()},
    }


def base(ax, bounds, vent, resolution=MAP_RESOLUTION):
    ax.set(xlim=(bounds[0], bounds[2]), ylim=(bounds[1], bounds[3]), aspect="equal")
    bounds = np.asarray(bounds)
    x, y = regular_axes(bounds, resolution)
    _, inverse, _ = make_laea_transform(*vent)
    lon, lat = inverse.transform(*np.meshgrid(x, y))
    tier = shoreline_tier(bounds, resolution)
    mask = local_mask(x, y, vent, tier)
    rgb = terrain_rgb(lon, lat, mask, x, y)
    dx, dy = (x[1] - x[0]) / 2, (y[1] - y[0]) / 2
    ax.imshow(rgb, extent=(x[0]-dx, x[-1]+dx, y[0]-dy, y[-1]+dy),
              origin="lower", interpolation="nearest", zorder=0)
    for spine in ax.spines.values():
        spine.set_color("#a3b5bb")
    if (mask == 0).any() and (mask == 1).any():
        ax.contour(x, y, np.ma.masked_less(mask, 0), levels=[.5], colors="#596f73",
                   linewidths=.40, alpha=.82, zorder=1.5)


def degree(value, positive, negative, precision):
    if negative == "W":
        value = (value + 180) % 360 - 180
    number = f"{abs(value):.{precision}f}"
    return f"{number}°{positive if value >= 0 else negative}"


def graticule(ax, bounds, vent):
    """Draw real WGS84 meridians and parallels through the vent-centred projection."""
    _, inverse, _ = make_laea_transform(*vent)
    xx, yy = np.meshgrid(*regular_axes(np.asarray(bounds), 95))
    lon, lat = inverse.transform(xx, yy)
    lon = vent[0] + (lon - vent[0] + 180) % 360 - 180
    for coordinates, hemispheres in ((lon, ("E", "W")), (lat, ("N", "S"))):
        finite = coordinates[np.isfinite(coordinates)]
        lo, hi = min(finite), max(finite)
        ticks = MaxNLocator(4, steps=[1, 2, 2.5, 5, 10]).tick_values(lo, hi)
        levels = ticks[(ticks > lo) & (ticks < hi)]
        if not len(levels):
            continue
        step = float(ticks[1] - ticks[0])
        precision = max(0, -int(np.floor(np.log10(step))))
        scaled_step = step * 10. ** precision
        if not np.isclose(scaled_step, round(scaled_step)):
            precision += 1
        lines = ax.contour(xx, yy, coordinates, levels=levels, colors="#73959e", alpha=.72,
                           linestyles="dotted", linewidths=.75, zorder=2)
        labels = ax.clabel(lines, fmt=lambda value: degree(value, *hemispheres, precision),
                           fontsize=7, colors=MUTED, inline=True, inline_spacing=3, zorder=3)
        to_frame = ax.transData + ax.transAxes.inverted()
        for label in labels:
            position = to_frame.transform(label.get_position())
            if not ((position > .025) & (position < .975)).all():
                label.remove()


@lru_cache(maxsize=12)
def projected_places(vent):
    with gzip.open(PLACES, "rt", encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    names = [r["name"] for r in rows]
    rank = np.array([int(r["scale_rank"]) for r in rows])
    population = np.array([int(r["population"]) for r in rows])
    coordinates = np.array([(float(r["longitude"]), float(r["latitude"])) for r in rows])
    forward, _, _ = make_laea_transform(*vent)
    xy = np.column_stack(forward.transform(*coordinates.T))
    return xy, names, rank, population


def places(ax, bounds, vent, *, obstacles=(), count=3):
    xy, names, rank, population = projected_places(vent)
    limits = np.reshape(bounds, (2, 2))
    visible = np.flatnonzero(((xy >= limits[0]) & (xy <= limits[1])).all(axis=1))
    if not len(visible):
        return
    ranked = sorted(visible, key=lambda i: (rank[i], -population[i], names[i]))
    transform = ax.transData
    ax.figure.canvas.draw()
    renderer = ax.figure.canvas.get_renderer()
    used = [artist.get_window_extent(renderer).expanded(1.14, 1.22) for artist in ax.texts]
    reserved = transform.transform(np.asarray(obstacles).reshape(-1, 2))
    for i in ranked:
        point = xy[i]
        x, y = transform.transform(point)
        if any(box.contains(x, y) for box in used):
            continue
        if len(reserved) and np.linalg.norm(reserved - [x, y], axis=1).min() < 10:
            continue
        for dx, dy, ha in ((5, 3, "left"), (-5, 3, "right"), (5, -8, "left"), (-5, -8, "right")):
            label = ax.annotate(names[i], point, xytext=(dx, dy), textcoords="offset points",
                                ha=ha, va="center", fontsize=7.5, style="italic", color=MUTED, zorder=4)
            size = label.get_window_extent(renderer).expanded(1.05, 1.13)
            if (ax.bbox.contains(size.x0, size.y0) and ax.bbox.contains(size.x1, size.y1)
                    and not any(size.overlaps(box) for box in used)
                    and not any(size.overlaps(Bbox.from_bounds(px-5, py-5, 10, 10)) for px, py in reserved)):
                ax.plot(*point, marker="+", markersize=4.5, color=MUTED, mew=.95, ls="", zorder=4)
                used.append(size)
                count -= 1
                break
            label.remove()
        if count == 0:
            return


def locator_extent(bounds, vent):
    points, _, _, _ = projected_places(vent)
    low, high = np.reshape(bounds, (2, 2))
    center = (low + high) / 2
    nearest = np.linalg.norm(points - center, axis=1)
    nearest = nearest[np.isfinite(nearest)]
    # Include nearby geography even on a completely offshore detail.
    half = max(67., 1.07 * (high - low).max(), 1.46 * nearest.min())
    return np.r_[center - [half * 1.40, half], center + [half * 1.40, half]]


def locator(fig, bounds, vent):
    low, high = np.reshape(bounds, (2, 2))
    center = (low + high) / 2
    extent = locator_extent(bounds, vent)
    ax = fig.add_axes([.070, .088, .218, .190])
    base(ax, extent, vent, resolution=LOCATOR_RESOLUTION)
    ax.set(xticks=[], yticks=[])
    ax.set_title("Regional locator", loc="left", fontsize=9, color=MUTED, pad=5)
    ax.add_patch(Rectangle(low, *(high - low), fill=False, edgecolor=ACCENT,
                           linestyle=(0, (3, 1.5)), linewidth=1.25, zorder=5))
    ax.plot(*center, marker="+", markersize=8, mew=1.3, color=ACCENT, ls="", zorder=5)
    ax.plot(0, 0, marker="*", markersize=8, color=ACCENT, markerfacecolor="none", ls="", zorder=5)
    places(ax, extent, vent, obstacles=[center, (0, 0)], count=3)
    _, inverse, _ = make_laea_transform(*vent)
    lon, lat = inverse.transform(*center)
    fig.text(.304, .208, "Detail centre", fontsize=9, weight="bold", color=MUTED)
    fig.text(.304, .184, f"{degree(lat, 'N', 'S', 3)}, {degree(lon, 'E', 'W', 3)}",
             fontsize=9, color=MUTED)
    fig.text(.304, .147, "Magenta + marks the centre\nof the boxed detail.", fontsize=8.5, color=MUTED)
    fig.text(.304, .107, "Blue-grey + marks a named place.", fontsize=8.5, color=MUTED)
