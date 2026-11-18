"""Positive polar integration to the antipode with no calculation window.

A ray intersects a triangle over one radial interval. Gaps use the analytic
radial thinning integral. Local/transition intervals have positive Gauss
weights. Split angular panels at mesh vertices and compass-sector boundaries.
"""
import numpy as np
from scipy.special import expit, exprel
from numpy.polynomial.legendre import leggauss

from .geometry import angular_weights, polar, TAU
from .inference import flat, log_profile
from ..mass_mixture.geometry import RADIUS

PARTS = ("footprint", "transition", "proximal_background", "gap_background", "distal_background")


def radial_integral(lo, hi, scale, shape):
    """Closed radial mass for any inverse power, including exponential and q=2."""
    shape = np.asarray(shape)
    # For q>2.25, use the log survival of the normalized radial density.
    # This stays accurate in the exponential limit without subtracting terms
    # proportional to q squared. The heavy-tail branch lives on finite Earth.
    x = np.minimum(shape, 1/2.25)
    def survival(r):
        z = r/scale
        return np.log1p((1-x)*z)+log_profile(r, scale, x)+np.log1p(x*z)
    log_lo, log_hi = survival(lo), survival(hi)
    light = (scale**2/((1-x)*(1-2*x))*np.exp(log_lo)*(-np.expm1(np.minimum(log_hi-log_lo, 0.))))
    if np.all(shape <= 1/2.25):
        return light
    q = 1/np.maximum(shape, 1/2.25)
    s = q*scale
    v = np.log1p((hi-lo)/(s+lo))
    heavy = ((1+lo/s)**(-q)*(s+lo)*v*
             ((s+lo)*exprel((2-q)*v)-s*exprel((1-q)*v)))
    return np.maximum(np.where(shape <= 1/2.25, light, heavy), 0.)


def make_rule(g, angular_order, radial_order):
    uv0 = np.einsum("tij,tj->ti", g.inverse, -g.triangles[:, 0])
    intercept = np.column_stack((1-uv0.sum(1), uv0))
    vertices = g.triangles.reshape(-1, 2)
    panels = np.r_[np.linspace(0, TAU, 33), polar(vertices, g.axis)[1],
                   (g.centers-g.width) % TAU, g.centers, (g.centers+g.width) % TAU]
    panels = np.unique(panels)
    lower, length = panels[:-1], np.diff(panels)
    keep = length > 1e-13
    u, aw = leggauss(angular_order)
    angles = (lower[keep, None]+length[keep, None]*(1+u)/2).ravel()
    aw = (length[keep, None]*aw/2).ravel()
    mid, rw = leggauss(radial_order)
    sector = np.floor((angles+np.pi/8)/(np.pi/4)).astype(int) % 8
    exact_r = polar(g.mesh.xy, g.axis)[0]
    inner, outer = exact_r.min(), exact_r.max()
    gr, ga, gb, gp = [], [], [], []
    local = {k: [] for k in ("ray", "idx", "weight", "r", "bary", "part")}
    for j, angle in enumerate(angles):
        d = np.array([np.sin(angle)*g.axis, np.cos(angle)/g.axis])
        uv_slope = g.inverse @ d
        slope = np.column_stack((-uv_slope.sum(1), uv_slope))
        with np.errstate(divide="ignore", invalid="ignore"):
            cross = -intercept/slope
        lo = np.maximum(0., np.where(slope > 0, cross, -np.inf).max(1))
        hi = np.minimum(2*RADIUS, np.where(slope < 0, cross, np.inf).min(1))
        hit = np.flatnonzero((hi > lo) & ~np.any((slope == 0) & (intercept < 0), axis=1))
        hit = hit[np.argsort(lo[hit])]
        stops = np.r_[hi[hit], 2*RADIUS]
        starts = np.r_[0., stops[:-1]]
        ends = np.r_[lo[hit], 2*RADIUS]
        if np.any(ends < starts-1e-5):
            raise ArithmeticError("Overlapping ray intervals in the continuation mesh")
        for a, b, kind in ((0., inner, 2), (inner, outer, 3), (outer, 2*RADIUS, 4)):
            x, y = np.maximum(starts, a), np.minimum(ends, b)
            mask = y > x
            if mask.any():
                gr.extend([j]*mask.sum()); ga.extend(x[mask]); gb.extend(y[mask]); gp.extend([kind]*mask.sum())
        if len(hit):
            widths = (hi-lo)[hit, None]/2
            radius = lo[hit, None]+widths*(1+mid)
            bary = np.clip(intercept[hit, None, :]+slope[hit, None, :]*radius[..., None], 0, 1)
            bary /= bary.sum(axis=-1, keepdims=True)
            ids = g.node_ids[hit]
            local["ray"].append(np.full(radius.size, j))
            local["idx"].append(np.repeat(ids+1, radial_order, axis=0))
            local["r"].append(radius.ravel())
            local["weight"].append((aw[j]*widths*rw*radius).ravel())
            local["bary"].append((bary*(ids[:, None, :] >= 0)).reshape(-1, 3))
            local["part"].append(np.repeat(np.where(g.inside[hit], 0, 1), radial_order))
    shapes = {"idx": (0, 3), "bary": (0, 3)}
    local = {k: np.concatenate(v) if v else np.empty(shapes.get(k, (0,)), dtype=int if k in ("idx","ray","part") else float)
             for k, v in local.items()}
    gr = np.asarray(gr, int)
    lgroup = 8*local["part"]+sector[local["ray"]]
    ggroup = 8*np.asarray(gp, int)+sector[gr]
    return {"angles": angles, "angles_w": angular_weights(angles, g.centers, g.width),
            "gap_ray": gr, "gap_lo": np.asarray(ga), "gap_hi": np.asarray(gb), "gap_aw": aw[gr],
            "local": local, "lgroup": lgroup, "ggroup": ggroup,
            "area": np.bincount(lgroup, weights=local["weight"], minlength=16).reshape(2, 8)}


def _sum_by(values, group, dest):
    for idx in np.unique(group):
        dest[:, idx] += np.sum(values[:, group == idx], axis=1)


def integrate(draws, g, rule):
    params = flat(draws)
    local = rule["local"]
    result = np.zeros((len(params["source"]), 40))
    # Limit temporary arrays for dense or multi-island footprints.
    batch = min(64, max(1, 350_000//max(1, len(local["r"]))))
    for start in range(0, len(result), batch):
        sl = slice(start, start+batch)
        out = result[sl]
        core = np.exp(params["log_range"][sl, None])
        h = core+np.exp(params["fan_log_range"][sl]) @ rule["angles_w"].T
        mix = expit(params["fan_logit"][sl, None])
        amplitude = params["source"][sl, None]
        shape = params["tail_shape"][sl, None]
        gap_h = h[:, rule["gap_ray"]]
        gap = ((1-mix)*radial_integral(rule["gap_lo"], rule["gap_hi"], core, shape)
               + mix*radial_integral(rule["gap_lo"], rule["gap_hi"], gap_h, shape))*np.exp(amplitude)*rule["gap_aw"]
        _sum_by(gap, rule["ggroup"], out)
        if len(local["r"]):
            nodes = np.pad(params["nodes"][sl], ((0, 0), (1, 0)))
            w, ids = local["bary"], local["idx"]
            back = np.logaddexp(np.log1p(-mix)+log_profile(local["r"], core, shape),
                                np.log(mix)+log_profile(local["r"], h[:, local["ray"]], shape))
            logt = (1-w.sum(1)).clip(0, 1)*(amplitude+back)
            for vertex in range(3):
                logt += nodes[:, ids[:, vertex]]*w[:, vertex]
            _sum_by(np.exp(logt)*local["weight"], rule["lgroup"], out)
    return (result*1e-5).reshape((*draws["source"].shape, 5, 8))


def integrate_checked(draws, g, n_per_chain=256):
    # A stratified deterministic sample of complete fields preserves chains.
    n = draws["source"].shape[1]
    idx = np.unique(np.linspace(0, n-1, min(n, n_per_chain)).round().astype(int))
    selected = {k: v[:, idx] for k, v in draws.items()}
    previous = None
    history = []
    for angular, radial in ((1, 3), (2, 5), (4, 8), (8, 12), (12, 18)):
        rule = make_rule(g, angular, radial)
        mass = integrate(selected, g, rule)
        total = mass.sum(axis=(-1, -2))
        area = np.array([g.mesh.area, g.area[~g.inside].sum()])
        area_err = float((np.abs(rule["area"].sum(1)-area)/np.maximum(area, 1.)).max())
        entry = {"angular_order": angular, "radial_order": radial, "area_relative_error": area_err,
                 "ray_count": len(rule["angles"])}
        if previous is not None:
            # L1 over the entire 5 x 8 partition prevents compensating errors.
            delta = np.abs(mass-previous).sum(axis=(-1, -2))/total
            entry["max_partition_L1_change_over_total"] = float(delta.max())
        history.append(entry)
        if previous is not None and entry["max_partition_L1_change_over_total"] < .005 and area_err < .001:
            break
        previous = mass
    passed = entry["max_partition_L1_change_over_total"] < .005 and area_err < .001
    if not np.isfinite(mass).all() or np.any(mass < 0):
        raise ArithmeticError("Nonfinite or negative mass")
    return mass, idx, {"passed": bool(passed), "history": history, "area_by_compass_km2": rule["area"].tolist(),
                       "volume_draws_per_chain": len(idx)}

