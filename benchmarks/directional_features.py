"""Locate secondary patches and holes, match actual evidence, and re-evaluate suspect features."""

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path

import numpy as np
import pandas as pd
import shapely

from .directional_scale_maps import read_grid
from src.empirical_isopach.contour_support import level_polygons
from src.empirical_isopach.core.audit import audited_data
from src.empirical_isopach.core.catalog import datasets
from src.empirical_isopach.models.directional.multiscale.posterior import DirectionalField
from src.empirical_isopach.visualization.deposit_field import digest, doubling_levels
from src.empirical_isopach.visualization.deposit_maps import regular_axes, frames


def features(shape):
    parts = sorted(shapely.get_parts(shape), key=lambda p: -p.area)
    for i, part in enumerate(parts):
        if not part.covers(shapely.Point(0, 0)):
            yield "patch", part
        for ring in part.interiors:
            yield "hole", shapely.Polygon(ring)


def supporting_rows(data, shape, kind, cm):
    inside = shapely.contains_xy(shape, *data.xy.T)
    exact = ~data.censored
    if kind == "patch":
        supported = (exact | data.interval) & (data.log_value >= np.log(cm))
    else:
        supported = (exact | (data.censored & ~data.lower_censored)) & (data.log_value < np.log(cm))
        supported |= data.interval & (data.interval_log_upper < np.log(cm))
    return data.row_ids[inside & supported].tolist(), data.row_ids[inside].tolist()


def inspect_case(job):
    rec, release, baseline = job
    data, meta = audited_data(rec["path"])
    rows = []
    for method, root in (("previous", baseline), ("multiscale", release)):
        home = root / rec["slug"]
        field = read_grid(home / "median-grid.npz")
        bounds = frames(field, data, 0.5)[0]
        levels = doubling_levels(10 ** field.log10_cm.max())
        pending = []
        for resolution in (193, 385):
            x, y = regular_axes(bounds, resolution)
            _, polygons = level_polygons((x, y, field.on_grid(x, y) * np.log(10)), levels)
            for cm, shape in polygons.items():
                for kind, feature in features(shape):
                    support, inside = supporting_rows(data, feature, kind, cm)
                    center = feature.representative_point()
                    row = dict(
                        slug=rec["slug"],
                        method=method,
                        input_sha256=meta["input_sha256"],
                        posterior_sha256=digest(home / "multiscale/posterior.npz"),
                        level_cm=cm,
                        kind=kind,
                        cartesian_long_side=resolution,
                        area_km2=feature.area,
                        x_km=center.x,
                        y_km=center.y,
                        supporting_row_ids=json.dumps(support),
                        enclosed_row_ids=json.dumps(inside),
                        local_direct_check="not required: contains supporting evidence"
                        if support
                        else "coarse mesh flag",
                    )
                    rows.append(row)
                    if resolution == 385 and not support:
                        pending.append((row, feature))
        if not pending:
            continue
        model = DirectionalField.load(home / "multiscale/posterior.npz")
        for row, feature in pending:
            box = np.array(feature.bounds)
            pad = max(2.0, 0.5 * max(box[2:] - box[:2]))
            box += pad * np.array([-1.0, -1.0, 1.0, 1.0])
            refined = []
            for resolution in (97, 193):
                x, y = regular_axes(box, resolution)
                xy = np.column_stack([a.ravel() for a in np.meshgrid(x, y)])
                z = model.central(xy).reshape(len(y), len(x)) * np.log(10)
                # For a hole, invert the field so the thin area is a filled component.
                sign = 1 if row["kind"] == "patch" else -1
                level = row["level_cm"] ** sign
                _, parts = level_polygons((x, y, sign * z), [level])
                shape = parts.get(level)
                matched = (
                    [] if shape is None else [p for p in shapely.get_parts(shape) if p.intersects(feature)]
                )
                union = shapely.union_all(matched)
                support, inside = supporting_rows(data, union, row["kind"], row["level_cm"])
                refined.append(
                    dict(
                        resolution=resolution,
                        area_km2=union.area,
                        supporting_row_ids=support,
                        enclosed_row_ids=inside,
                        min_cm=float(np.exp(z.min())),
                        max_cm=float(np.exp(z.max())),
                    )
                )
            row["local_direct_check"] = json.dumps(refined)
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--release", type=Path, required=True)
    p.add_argument("--baseline", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--jobs", type=int, default=2)
    a = p.parse_args()
    with ProcessPoolExecutor(a.jobs) as pool:
        rows = [
            row
            for result in pool.map(inspect_case, [(r, a.release, a.baseline) for r in datasets()])
            for row in result
        ]
    pd.DataFrame(rows).to_csv(a.output, index=False, compression={"method": "gzip", "mtime": 0})


if __name__ == "__main__":
    main()
