"""Screen complete level sets before using them as area/thickness evidence.

Closure is necessary, not sufficient. No open line is joined at a map edge,
and no candidate level becomes an independent observation. Selection is a
conservative reproducible development policy, not a calibrated probability.
"""
import numpy as np
from contourpy import contour_generator
from shapely import Polygon, contains_xy, get_parts, union_all

from .core.distributions import predictive_distribution
from .support import POLICIES, SupportMesh, TriangleRegion
from .region import integrate_region


def candidate_levels(data, count=25):
    low, high = np.quantile(data.log_value[~data.censored], [.05, .95])
    if count < 3 or high-low <= 1e-10:
        return np.array([])
    return np.exp(np.linspace(low, high, count))


def median_grid(model, bounds, resolution=257):
    if resolution < 33 or resolution % 2 != 1:
        raise ValueError("Contour resolution must be an odd integer >=33.")
    x, y = [np.linspace(bounds[k], bounds[k+2], resolution) for k in (0, 1)]
    xx, yy = np.meshgrid(x, y)
    xy = np.column_stack((xx.ravel(), yy.ravel()))
    values = [predictive_distribution(model, xy[i:i+2048]).quantile(.5)
              for i in range(0, len(xy), 2048)]
    return x, y, np.concatenate(values).reshape(xx.shape)


def level_polygons(grid, levels):
    x, y, z = grid
    generator = contour_generator(x=x, y=y, z=z, line_type="Separate", fill_type="OuterOffset")
    boundary = np.r_[z[0], z[-1], z[:, 0], z[:, -1]].max()
    records, polygons = [], {}
    for level in levels:
        t = np.log(level)
        record = {"level_cm": float(level), "status": "closed", "area_km2": None, "components": 0}
        if t <= boundary:
            record["status"] = "map_boundary_above_level"
        elif t >= z.max():
            record["status"] = "above_resolved_peak"
        else:
            paths = generator.lines(t)
            if not paths or not all(len(p) >= 4 and np.array_equal(p[0], p[-1]) for p in paths):
                record["status"] = "open_or_unresolved"
            else:
                points, offsets = generator.filled(t, z.max()+1.)
                components = []
                for p, offset in zip(points, offsets):
                    rings = [p[a:b] for a, b in zip(offset[:-1], offset[1:])]
                    components.append(Polygon(rings[0], rings[1:]))
                polygon = union_all(components)
                if not polygon.is_valid or polygon.is_empty or polygon.area <= 0:
                    record["status"] = "invalid_polygon"
                else:
                    polygons[float(level)] = polygon
                    record.update(area_km2=float(polygon.area), components=len(components))
        records.append(record)
    return records, polygons


def shape_change(a, b):
    if a is None or b is None:
        return None
    return float(a.symmetric_difference(b).area/a.union(b).area)


def bracket_counts(data, polygon, log_level):
    """Count distinct exact sites on both sides; bounds cannot become exacts."""
    exact = ~data.censored
    xy, y = data.xy[exact], data.log_value[exact]
    def count(mask):
        return len(np.unique(xy[mask], axis=0))
    inside = contains_xy(polygon, xy[:, 0], xy[:, 1])
    components = []
    holes = []
    for part in get_parts(polygon):
        in_part = contains_xy(part, xy[:, 0], xy[:, 1])
        components.append(count(in_part & (y >= log_level)))
        for ring in part.interiors:
            in_hole = contains_xy(Polygon(ring), xy[:, 0], xy[:, 1])
            holes.append(count(in_hole & (y < log_level)))
    return {"above_inside_each_component": components,
            "below_outside": count(~inside & (y < log_level)), "below_inside_each_hole": holes}


def screen_contours(data, grid, deletion_grids, meshes, levels):
    records, polygons = level_polygons(grid, levels)
    coarse = (grid[0][::2], grid[1][::2], grid[2][::2, ::2])
    _, coarse_polygons = level_polygons(coarse, levels)
    deletion_polygons = [level_polygons(g, levels)[1] for g in deletion_grids]
    for row in records:
        t = row["level_cm"]
        polygon = polygons.get(t)
        row["policies"] = {}
        if polygon is not None:
            row["grid_shape_change"] = shape_change(polygon, coarse_polygons.get(t))
            row["deletion_shape_changes"] = [shape_change(polygon, p.get(t)) for p in deletion_polygons]
            row["bracketing"] = bracket_counts(data, polygon, np.log(t))
        for name, mesh in meshes.items():
            reasons, details = [], {}
            policy = mesh.policy
            if polygon is None:
                reasons.append(row["status"])
            else:
                missing = polygon.boundary.difference(mesh.geometry).length/polygon.length
                details["unsupported_perimeter_fraction"] = float(missing)
                details["unsupported_enclosed_area_fraction"] = float(polygon.difference(mesh.geometry).area/polygon.area)
                # Geometric tolerances only; this is effectively an entire-line requirement.
                if missing > 1e-8:
                    reasons.append("perimeter_leaves_local_support")
                grid_change = row["grid_shape_change"]
                if grid_change is None or grid_change > policy.grid_shape_tolerance:
                    reasons.append("contour_not_grid_resolved")
                brackets = row["bracketing"]
                if (min(brackets["above_inside_each_component"], default=0) < 3
                        or brackets["below_outside"] < 3
                        or min(brackets["below_inside_each_hole"], default=3) < 3):
                    reasons.append("fewer_than_three_bracketing_sites")
                changes = row["deletion_shape_changes"]
                if len(changes) != 4 or any(c is None for c in changes):
                    reasons.append("not_closed_after_every_geographic_deletion")
                elif max(changes) > policy.deletion_shape_tolerance:
                    reasons.append("unstable_under_geographic_deletion")
            row["policies"][name] = {**details, "eligible": not reasons, "reasons": reasons}
    return records, polygons


def contiguous_bands(records, polygons, mesh):
    """All contiguous eligible runs, split wherever the intervening area has a gap.

A supported perimeter can enclose an unsupported proximal core. A band may
exclude that core, but must not bridge unsupported holes between its endpoints.
"""
    runs, run = [], []
    for row in records:
        if not row["policies"][mesh.policy.name]["eligible"]:
            if run:
                runs.append(run)
            run = []
            continue
        if run:
            outer, inner = polygons[run[-1]], polygons[row["level_cm"]]
            annulus = outer.difference(inner)
            if (annulus.area <= 0 or annulus.difference(mesh.geometry).area > 1e-8*annulus.area
                    or inner.difference(outer).area > 1e-8*outer.area):
                runs.append(run)
                run = []
        run.append(row["level_cm"])
    if run:
        runs.append(run)
    return [{"levels_cm": r, "span_factor": r[-1]/r[0],
             "eligible_for_extension": len(r) >= 3 and r[-1]/r[0] >= 2,
             "reason": "eligible" if len(r) >= 3 and r[-1]/r[0] >= 2 else "need_three_contours_spanning_a_factor_of_two"}
            for r in runs]


def layer_cake_bounds(levels, areas):
    """Deterministic Riemann brackets for a geographic thickness band, cm*km².

Integral over {l <= T < h}: l*A(l) - h*A(h) + integral_l^h A(t) dt.
The integral of A alone is a vertical slab, not the geographic band's volume.
These are numerical bounds conditional on exact areas, NOT statistical CIs.
"""
    t, a = np.asarray(levels, float), np.asarray(areas, float)
    if (t.ndim != 1 or len(t) < 2 or a.shape != t.shape or not np.isfinite(t).all()
            or not np.isfinite(a).all() or np.any(t <= 0) or np.any(a <= 0)
            or np.any(np.diff(t) <= 0) or np.any(np.diff(a) > 1e-8*a[:-1])):
        raise ValueError("Need increasing positive thickness and decreasing positive superlevel area.")
    boundary = t[0]*a[0]-t[-1]*a[-1]
    return {"lower_km3": float((boundary+np.diff(t)@a[1:])*1e-5),
            "upper_km3": float((boundary+np.diff(t)@a[:-1])*1e-5),
            "meaning": "Numerical layer-cake bracket conditional on fitted contour areas; not a confidence interval."}


def continuation_scenarios(levels, areas, band_volume):
    """Endpoint-matched proximal/distal assumptions, never a preferred total.

Secants across one third of the *log-thickness range* avoid treating denser
contour sampling as more independent evidence. No outer distance cutoff.
"""
    t, a = np.asarray(levels, float), np.asarray(areas, float)
    layer_cake_bounds(t, a)  # shared input checks
    if not np.isfinite(band_volume) or band_volume < 0:
        raise ValueError("Band volume must be finite and nonnegative.")
    x, logt = np.sqrt(a), np.log(t)
    low_third, high_third = logt[0]+(logt[-1]-logt[0])*np.array([1/3, 2/3])
    xd, xp = np.interp([low_third, high_third], logt, x)
    if x[0] <= xd or xp <= x[-1]:
        return {"status": "unresolved_endpoint_slope", "scenarios": [], "preferred_global_km3": None}
    kd = (low_third-logt[0])/(x[0]-xd)
    kp = (logt[-1]-high_third)/(xp-x[-1])
    if not np.isfinite([kd, kp]).all() or min(kd, kp) <= 0:
        return {"status": "unresolved_endpoint_slope", "scenarios": [], "preferred_global_km3": None}
    p = kd*x[0]
    def representable(value):
        return float(value) if np.isfinite(value) else None
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        distal = {"exponential": representable(2*t[0]*(x[0]/kd+1/kd**2)*1e-5),
                  "slope_matched_power": None if p <= 2 else representable(2*t[0]*a[0]/(p-2)*1e-5)}
    z = kp*x[-1]
    # Preserve the constant-cap limit without catastrophic cancellation.
    with np.errstate(over="ignore", invalid="ignore"):
        factor = 1+z/3+z*z/12+z**3/60+z**4/360 if z < 1e-3 else (
            np.inf if z > 700 else 2*(np.expm1(z)-z)/z**2)
        prox_exp = representable(t[-1]*a[-1]*factor*1e-5)
    proximal = {"constant_cap": float(t[-1]*a[-1]*1e-5), "exponential_cap": prox_exp}
    scenarios = []
    for prox, vp in proximal.items():
        for tail, vt in distal.items():
            finite = vp is not None and vt is not None
            scenarios.append({"proximal": prox, "distal": tail, "proximal_km3": vp,
                "band_median_surface_km3": float(band_volume), "distal_km3": vt,
                "total_km3": representable(vp+band_volume+vt) if finite else None,
                "status": "conditional_scenario" if finite else (
                    "nonintegrable_power_tail" if tail == "slope_matched_power" and p <= 2 else "numerical_overflow")})
    return {"status": "conditional_scenarios_only", "preferred_global_km3": None,
            "distal_slope_per_sqrt_km2": float(kd), "proximal_slope_per_sqrt_km2": float(kp),
            "distal_power_exponent": float(p), "scenarios": scenarios,
            "scope": "Bulk volume of an infinite-area empirical continuation of summed superlevel areas, not a geographically reconstructed or validated complete-Earth volume. Conditional on band choice and endpoint slopes; no tail cutoff or eruption-specific wind parameters.",
            "identifiability": "General positivity, smoothness and finite mass do not provide a finite data-identified upper bound. These four scenarios are not exhaustive and their range is not a confidence interval."}


def audit_contours(model, data, bounds, deletion_models, meshes=None, resolution=257, level_count=25):
    meshes = meshes or {p.name: SupportMesh(data, p) for p in POLICIES}
    dense_levels = candidate_levels(data, 2*level_count-1)
    grid = median_grid(model, bounds, resolution)
    deletion_grids = [median_grid(m, bounds, resolution) for m in deletion_models]
    dense_records, polygons = screen_contours(data, grid, deletion_grids, meshes, dense_levels)
    records = dense_records[::2]
    policies = {}
    for name, mesh in meshes.items():
        runs = contiguous_bands(records, polygons, mesh)
        for run in runs:
            l, h = run["levels_cm"][0], run["levels_cm"][-1]
            run["level_refinement_verified"] = all(row["policies"][name]["eligible"]
                for row in dense_records if l <= row["level_cm"] <= h)
            if not run["level_refinement_verified"]:
                run["eligible_for_extension"] = False
                run["reason"] = "intermediate_level_fails_refinement_check"
        candidates = [r for r in runs if r["eligible_for_extension"]]
        best = max(candidates, key=lambda r: (r["span_factor"], len(r["levels_cm"])), default=None)
        result = {"contiguous_runs": runs, "selected_band": None,
                  "extension": {"status": "no_supported_stable_band", "preferred_global_km3": None}}
        if best is not None:
            l, h = best["levels_cm"][0], best["levels_cm"][-1]
            polygon = polygons[l].difference(polygons[h])
            region = TriangleRegion.from_polygon(polygon)
            volume = integrate_region(model, region)
            # Many derived levels refine an integral, never its statistical sample size.
            fine_levels = np.geomspace(l, h, 257)
            fine_rows, _ = level_polygons(grid, fine_levels)
            if all(r["status"] == "closed" for r in fine_rows):
                areas = [r["area_km2"] for r in fine_rows]
                bracket = layer_cake_bounds(fine_levels, areas)
                result["selected_band"] = {**best, "area_km2": region.area,
                    "volume": volume, "layer_cake_check": bracket,
                    "proximal_area_km2": polygons[h].area,
                    "unsupported_proximal_area_fraction": polygons[h].difference(mesh.geometry).area/polygons[h].area,
                    "interpretation": "Geographic band selected from the fitted median surface; selection-adjusted uncertainty is not available."}
                if volume["converged"] and bracket["lower_km3"]*.99 <= volume["median_surface_km3"] <= bracket["upper_km3"]*1.01:
                    result["extension"] = continuation_scenarios(fine_levels, areas, volume["median_surface_km3"])
                else:
                    result["extension"]["status"] = "band_integral_not_resolved"
        policies[name] = result
    report = {"protocol": "supported-isopachs-v1", "resolution": resolution,
        "level_rule": f"{level_count} equally spaced log-thickness levels between exact-observation 5th and 95th percentiles.",
        "levels": records, "policies": policies,
        "refined_levels": dense_records,
        "level_density_sensitivity": {"level_count": len(dense_levels), "bands": {
            name: contiguous_bands(dense_records, polygons, mesh) for name, mesh in meshes.items()}},
        "preferred_global_km3": None,
        "scope": "Entire-contour geometric support, three-site bracketing, grid refinement and four fully retuned geographic deletions. Eligibility is a development screen, not a probability or guarantee that unsampled fallout is absent.",
        "uncertainty": "No selection-adjusted band or global confidence interval is claimed. Interpolation-footprint uncertainty is reported separately from extrapolation scenarios."}
    return report, grid, polygons, deletion_grids
