"""Extract actual closed level sets; never join open ends at a map boundary."""
import numpy as np
from contourpy import contour_generator

from .report import domain_bounds


DEFAULT_LEVELS = (.1, 1., 2., 5., 10., 20., 50., 100., 200., 500.)


def predict_chunks(model, xy, chunk_size=4096):
    means, variances = [], []
    for start in range(0, len(xy), chunk_size):
        m, v = model.predict_log(xy[start:start+chunk_size])
        means.append(m)
        variances.append(v)
    return np.concatenate(means), np.concatenate(variances)


def boundary_maxima(model, bounds, samples=257):
    x0, y0, x1, y1 = bounds
    x, y = np.linspace(x0, x1, samples), np.linspace(y0, y1, samples)
    edges = [np.column_stack((np.full(samples, x0), y)),
             np.column_stack((x, np.full(samples, y0))),
             np.column_stack((np.full(samples, x1), y)),
             np.column_stack((x, np.full(samples, y1)))]
    return np.array([model.predict_log(p)[0].max() for p in edges])


def closed_domain(model, levels=DEFAULT_LEVELS, max_extent_km=6000.):
    """Expand only the necessary edges, then try a thicker outer contour.

    The extent is a reporting limit in projected coordinates, not an assertion
    of zero deposition. A level still crossing it is explicitly omitted.
    """
    levels = np.asarray(levels, float)
    if (levels.ndim != 1 or not len(levels) or not np.isfinite(levels).all()
            or np.any(levels <= 0) or np.any(np.diff(levels) <= 0)):
        raise ValueError("Contour levels must be finite, positive and strictly increasing.")
    if not np.isfinite(max_extent_km) or not 100 <= max_extent_km <= 8000:
        raise ValueError("Maximum projected map extent must be between 100 and 8000 km.")
    initial = domain_bounds(model.data.xy)
    if np.max(np.abs(model.data.xy)) >= max_extent_km:
        raise ValueError("Map extent must contain all observations; increase --max-extent-km.")
    for level in levels:
        bounds = np.clip(initial, -max_extent_km, max_extent_km)
        for _ in range(30):
            edges = boundary_maxima(model, bounds)
            if np.all(edges < np.log(level)-.12):
                return bounds, float(level)
            old = bounds.copy()
            spans = bounds[2:]-bounds[:2]
            for j in range(4):
                if edges[j] >= np.log(level)-.12:
                    bounds[j] += (-1 if j < 2 else 1)*.35*spans[j % 2]
            bounds = np.clip(bounds, -max_extent_km, max_extent_km)
            if np.array_equal(bounds, old):
                break
    raise ValueError("No requested contour closes within the map limit; request thicker levels.")


def ring_area(points):
    return .5*np.sum(points[:-1, 0]*points[1:, 1]-points[1:, 0]*points[:-1, 1])


def extract_levels(x, y, log_thickness, levels):
    generator = contour_generator(x=x, y=y, z=log_thickness,
                                  line_type="Separate", fill_type="OuterOffset")
    boundary = np.r_[log_thickness[0], log_thickness[-1],
                     log_thickness[:, 0], log_thickness[:, -1]].max()
    records, rings = [], {}
    for level in levels:
        threshold = np.log(level)
        row = {"level_cm": float(level), "status": "closed", "area_km2": None,
               "components": 0}
        if threshold <= boundary:
            row["status"] = "extends_beyond_map"
        elif threshold >= log_thickness.max():
            row["status"] = "above_resolved_peak"
        else:
            paths = generator.lines(threshold)
            closed = bool(paths) and all(len(p) >= 4 and np.allclose(p[0], p[-1], rtol=0, atol=1e-7)
                                         for p in paths)
            if not closed:
                row["status"] = "open_or_unresolved"
            else:
                # OuterOffset identifies islands and holes explicitly. Summing
                # absolute line areas alone would incorrectly fill all holes.
                points, offsets = generator.filled(threshold, log_thickness.max()+1.)
                area = 0.
                for p, offset in zip(points, offsets):
                    areas = [abs(ring_area(p[a:b])) for a, b in zip(offset[:-1], offset[1:])]
                    area += areas[0]-sum(areas[1:])
                row.update(area_km2=float(area), components=len(points))
                rings[float(level)] = paths
        records.append(row)
    return records, rings


def build_contours(model, levels=DEFAULT_LEVELS, resolution=481, max_extent_km=6000.):
    if resolution < 65 or resolution > 1201 or resolution % 2 != 1:
        raise ValueError("Contour resolution must be an odd number from 65 to 1201.")
    bounds, outer = closed_domain(model, levels, max_extent_km)
    x = np.linspace(bounds[0], bounds[2], resolution)
    y = np.linspace(bounds[1], bounds[3], resolution)
    X, Y = np.meshgrid(x, y)
    mean, variance = predict_chunks(model, np.column_stack((X.ravel(), Y.ravel())))
    mean, variance = mean.reshape(X.shape), variance.reshape(X.shape)
    records, rings = extract_levels(x, y, mean, levels)
    coarse, _ = extract_levels(x[::2], y[::2], mean[::2, ::2], levels)
    for row, previous in zip(records, coarse):
        row["area_grid_relative_change"] = (
            abs(row["area_km2"]-previous["area_km2"])/row["area_km2"]
            if row["area_km2"] and previous["area_km2"] else None)
        row["area_grid_scope"] = "whole map"
        row["area_grid_resolution"] = resolution
        # Resolve small proximal rings locally instead of forcing the full
        # continent-scale map onto a prohibitively dense uniform grid.
        if row["area_grid_relative_change"] is not None and row["area_grid_relative_change"] > .005:
            points = np.vstack(rings[row["level_cm"]])
            padding = np.maximum(np.ptp(points, axis=0)*.2, [3*(x[1]-x[0]), 3*(y[1]-y[0])])
            low = np.maximum(points.min(axis=0)-padding, bounds[:2])
            high = np.minimum(points.max(axis=0)+padding, bounds[2:])
            local_x, local_y = np.linspace(low[0], high[0], 481), np.linspace(low[1], high[1], 481)
            lx, ly = np.meshgrid(local_x, local_y)
            local_mean, _ = predict_chunks(model, np.column_stack((lx.ravel(), ly.ravel())))
            local_mean = local_mean.reshape(lx.shape)
            finer, fine_rings = extract_levels(local_x, local_y, local_mean, [row["level_cm"]])
            previous_local, _ = extract_levels(local_x[::2], local_y[::2], local_mean[::2, ::2], [row["level_cm"]])
            if finer[0]["status"] == "closed" and previous_local[0]["status"] == "closed":
                row.update(finer[0])
                row["area_grid_relative_change"] = abs(row["area_km2"]-previous_local[0]["area_km2"])/row["area_km2"]
                row["area_grid_scope"] = "local rectangle around all globally resolved components"
                row["area_grid_resolution"] = 481
                rings[row["level_cm"]] = fine_rings[row["level_cm"]]
    return {"bounds_km": bounds.tolist(), "outer_level_cm": outer,
            "resolution": resolution, "max_extent_km": max_extent_km,
            "levels": records, "rings": rings, "x": x, "y": y,
            "mean": mean, "variance": variance}


def split_antimeridian(lonlat):
    """Split geographic contour lines at ±180°, retaining boundary vertices."""
    segments, current = [], [lonlat[0].tolist()]
    for a, b in zip(lonlat[:-1], lonlat[1:]):
        if abs(b[0]-a[0]) > 180:
            adjusted = b[0]-np.sign(b[0]-a[0])*360
            edge = 180. if a[0] > 0 else -180.
            fraction = (edge-a[0])/(adjusted-a[0])
            latitude = float(a[1]+fraction*(b[1]-a[1]))
            current.append([edge, latitude])
            segments.append(current)
            current = [[-edge, latitude]]
        current.append(b.tolist())
    if len(current) > 1:
        segments.append(current)
    return segments


def geographic_contours(contours, inverse):
    features = []
    for row in contours["levels"]:
        if row["status"] != "closed":
            continue
        lines = []
        for ring in contours["rings"][row["level_cm"]]:
            lon, lat = inverse.transform(*ring.T)
            lines.extend(split_antimeridian(np.column_stack((lon, lat))))
        features.append({"type": "Feature", "properties": row,
                         "geometry": {"type": "MultiLineString", "coordinates": lines}})
    return {"type": "FeatureCollection", "features": features}
