"""Measure topology, turning, orientation, spacing and lobe changes on the actual fitted contours."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import shapely

from .directional_scale_maps import ARCHIVE, read_grid
from .footprint_morphology import outlines
from src.empirical_isopach.core.catalog import datasets
from src.empirical_isopach.contour_support import level_polygons
from src.empirical_isopach.visualization.deposit_maps import regular_axes
from src.empirical_isopach.visualization.deposit_field import doubling_levels


def rings(polygon):
    for part in shapely.get_parts(polygon):
        for ring in (part.exterior, *part.interiors):
            yield ring


def moments(shape):
    """Axis of the entire filled area, subtracting holes and retaining disconnected patches."""
    origin = np.array(shape.centroid.coords[0])
    sums = np.zeros(3)
    for part in shapely.get_parts(shape):
        for j, ring in enumerate((part.exterior, *part.interiors)):
            p = np.asarray(ring.coords) - origin
            x, y = p[:-1].T
            u, v = p[1:].T
            cross = x * v - u * y
            sign = np.sign(cross.sum()) * (1 if j == 0 else -1)
            sums += sign * np.array(
                [
                    cross @ (x * x + x * u + u * u) / 12,
                    cross @ (y * y + y * v + v * v) / 12,
                    cross @ (2 * x * y + x * v + u * y + 2 * u * v) / 24,
                ]
            )
    xx, yy, xy = sums / shape.area
    values, vectors = np.linalg.eigh([[xx, xy], [xy, yy]])
    return float(np.sqrt(max(0.0, 1 - values[0] / values[1]))), float(
        np.rad2deg(np.arctan2(*vectors[:, -1])) % 180
    )


def turning(ring):
    """Resample at equal arc length. A convex boundary has one absolute turn."""
    pts = shapely.get_coordinates(shapely.line_interpolate_point(ring, np.arange(144) / 144, normalized=True))
    delta = np.roll(pts, -1, axis=0) - pts
    phi = np.arctan2(delta[:, 1], delta[:, 0])
    changes = (np.roll(phi, -1) - phi + np.pi) % (2 * np.pi) - np.pi
    return float(abs(changes).sum() / (2 * np.pi))


def level_metrics(field, level, size):
    radius = field.outer_crossings(level)
    mask = np.isfinite(radius)
    if not mask.any() or np.nanmax(radius) >= field.radii[-1]:
        return dict(status="no finite closed boundary")
    x = radius[mask] * np.sin(field.angles[mask]) * field.axis
    y = radius[mask] * np.cos(field.angles[mask]) / field.axis
    pad = np.nanmax(radius) * max(0.05, 3 * 2 * np.pi / len(radius))
    bounds = np.array(
        [min(0.0, min(x)) - pad, min(0.0, min(y)) - pad, max(0.0, max(x)) + pad, max(0.0, max(y)) + pad]
    )
    x, y = regular_axes(bounds, size)
    rec, poly = level_polygons((x, y, field.on_grid(x, y) * np.log(10.0)), [level])
    shape = poly.get(level)
    if shape is None or shape.is_empty:
        return dict(status=rec[0]["status"])
    parts = list(shapely.get_parts(shape))
    lengths = np.array([r.length for r in rings(shape)])
    turns = [turning(r) for r in rings(shape)]
    eccentricity, heading = moments(shape)
    det = outlines(field, level)
    return dict(
        status="closed",
        components=len(parts),
        holes=sum(len(p.interiors) for p in parts),
        area_km2=shape.area,
        perimeter_km=shape.length,
        convex_hull_fraction=shape.area / shape.convex_hull.area,
        length_weighted_absolute_turns=float(np.average(turns, weights=lengths)),
        largest_connected_area_fraction=max(p.area for p in parts) / shape.area,
        vent_inside=bool(shape.covers(shapely.Point(0, 0))),
        centroid_x_km=shape.centroid.x,
        centroid_y_km=shape.centroid.y,
        eccentricity=eccentricity,
        axis_clockwise_from_north_deg=heading,
        **{
            k: v
            for k, v in (det or {}).items()
            if k
            in (
                "prominent_lobes_20pct",
                "high_harmonic_log_r_rms",
                "worst_indentation",
                "vent_connected_fraction",
            )
        },
    )


def fields(slug, release):
    home = Path("eruptions") / slug / "results/directional"
    previous = ARCHIVE / slug
    if not (previous / "median-grid.npz").exists():
        previous = home
    return {
        "previous": read_grid(previous / "median-grid.npz"),
        "multiscale": read_grid(release / slug / "median-grid.npz"),
    }


def study(release, output):
    rows = []
    for rec in datasets():
        slug = rec["slug"]
        for method, field in fields(slug, release).items():
            levels = doubling_levels(10 ** field.log10_cm.max())
            for size in (193, 385):
                previous = None
                for cm in levels:
                    metrics = level_metrics(field, cm, size)
                    r = field.outer_crossings(cm)
                    if (
                        previous is not None
                        and metrics["status"] == "closed"
                        and previous[1]["status"] == "closed"
                    ):
                        oldr, prev, old_spacing = previous
                        valid = np.isfinite(oldr) & np.isfinite(r) & (r > 0)
                        spacing = np.log(oldr[valid] / r[valid])
                        metrics["common_bearings"] = int(valid.sum())
                        if valid.any():
                            q = np.quantile(spacing, [0.1, 0.5, 0.9])
                            metrics.update(
                                log_spacing_p10=q[0], log_spacing_median=q[1], log_spacing_p90=q[2]
                            )
                            if old_spacing is not None:
                                metrics["successive_log_spacing_change"] = float(abs(q[1] - old_spacing))
                        if min(prev["eccentricity"], metrics["eccentricity"]) >= 0.4:
                            metrics["successive_axial_rotation_deg"] = abs(
                                (
                                    metrics["axis_clockwise_from_north_deg"]
                                    - prev["axis_clockwise_from_north_deg"]
                                    + 90
                                )
                                % 180
                                - 90
                            )
                    previous = (r, metrics, metrics.get("log_spacing_median"))
                    rows.append(
                        dict(slug=slug, method=method, level_cm=cm, cartesian_long_side=size, **metrics)
                    )
    pd.DataFrame(rows).to_csv(output, index=False, compression={"method": "gzip", "mtime": 0})


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--release", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    study(a.release, a.output)
