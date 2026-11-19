"""Integrate the full posterior deposits in the matched B-Tm thinning experiment."""
import argparse
from dataclasses import replace
import json
from pathlib import Path

import numpy as np
from scipy.special import exprel, expit

from benchmarks.btm_thinning import audit_input
from benchmarks.btm_thinning_fit import VARIANTS, warp
from src.empirical_isopach.models.directional.geometry import BASE, Geometry
from src.empirical_isopach.models.directional.inference import flat, log_profile
from src.empirical_isopach.models.directional.integration import (
    _sum_by, integrate, integrate_checked, make_rule, radial_integral,
)
from src.empirical_isopach.models.directional.study import digest, write_json
from src.empirical_isopach.models.mass_mixture.study import audited_data


def segment_integral(lo, hi, scale, shape, breaks, log_ratios):
    """Integrate r K(w(r)) dr in true equal-area radius, splitting at each slope change."""
    result = 0.
    start = 0.
    effective_start = 0.
    slope = 1.
    for k in range(breaks.shape[-1] + 1):
        left = np.maximum(lo, start)
        stop = breaks[..., k, None] if k < breaks.shape[-1] else hi
        width = np.maximum(0., np.minimum(hi, stop) - left)
        effective_left = effective_start + slope * (left - start)
        # K(w(left)+s*x, h) = K(w(left), h) K(x, (h+shape*w(left))/s).
        local_scale = (scale + shape * effective_left) / slope
        log_end = log_profile(width, local_scale, shape)
        zeroth_moment = -local_scale * log_end * exprel((1. - shape) * log_end)
        first_moment = radial_integral(0., width, local_scale, shape)
        result = result + np.exp(log_profile(effective_left, scale, shape)) * (
            left * zeroth_moment + first_moment)
        if k < breaks.shape[-1]:
            effective_start = effective_start + slope * (stop - start)
            slope = slope * np.exp(log_ratios[..., k, None])
            start = stop
    return result


def integrate_broken(draws, rule):
    """Use the production ray mesh and local quadrature with each draw's own radial breaks."""
    params = flat(draws)
    local = rule["local"]
    result = np.zeros((len(params["source"]), 40))
    batch = max(1, min(64, 350_000 // max(1, len(local["r"]), len(rule["gap_lo"]))))
    for start in range(0, len(result), batch):
        sl = slice(start, start + batch)
        p = {k: v[sl] for k, v in params.items()}
        knots, ratios = p["break_km"], p["log_slope_change"]
        core = np.exp(p["log_range"][:, None])
        scales = core + np.exp(p["fan_log_range"]) @ rule["angles_w"].T
        mix = expit(p["fan_logit"][:, None])
        source = p["source"][:, None]
        shape = p["tail_shape"][:, None]
        lo, hi = rule["gap_lo"], rule["gap_hi"]
        gap = (1. - mix) * segment_integral(lo, hi, core, shape, knots, ratios)
        gap += mix * segment_integral(lo, hi, scales[:, rule["gap_ray"]], shape, knots, ratios)
        _sum_by(gap * np.exp(source) * rule["gap_aw"], rule["ggroup"], result[sl])
        if len(local["r"]):
            nodes = np.pad(p["nodes"], ((0, 0), (1, 0)))
            w, ids = local["bary"], local["idx"]
            r = warp(local["r"], knots, ratios)
            back = np.logaddexp(np.log1p(-mix) + log_profile(r, core, shape),
                                np.log(mix) + log_profile(r, scales[:, local["ray"]], shape))
            logt = np.clip(1. - w.sum(1), 0., 1.) * (source + back)
            for vertex in range(3):
                logt += nodes[:, ids[:, vertex]] * w[:, vertex]
            _sum_by(np.exp(logt) * local["weight"], rule["lgroup"], result[sl])
    return (result * 1e-5).reshape((*draws["source"].shape, 5, 8))


def integrate_full(draws, geometry):
    if "break_km" not in draws:
        mass, _, details = integrate_checked(draws, geometry, n_per_chain=draws["source"].shape[1])
        return mass, details
    previous = None
    history = []
    exact_area = np.array([geometry.mesh.area, geometry.area[~geometry.inside].sum()])
    for angular, radial in ((1, 3), (2, 5), (4, 8), (8, 12), (12, 18)):
        rule = make_rule(geometry, angular, radial)
        mass = integrate_broken(draws, rule)
        totals = mass.sum(axis=(-1, -2))
        if not np.isfinite(mass).all() or np.any(mass < 0) or np.any(totals <= 0):
            raise ArithmeticError("Invalid integrated thickness")
        area_error = np.max(abs(rule["area"].sum(1) - exact_area) / np.maximum(exact_area, 1.))
        item = {"angular_order": angular, "radial_order": radial,
                "area_relative_error": float(area_error), "ray_count": len(rule["angles"])}
        history.append(item)
        if previous is not None:
            change = np.max(np.abs(mass - previous).sum(axis=(-1, -2)) / totals)
            item["max_partition_L1_change_over_total"] = float(change)
            if change < .005 and area_error < .001:
                break
        previous = mass
    return mass, {"history": history, "passed": bool(change < .005 and area_error < .001),
                  "volume_draws_per_chain": draws["source"].shape[1]}


def verify_radial_identity():
    """Compare analytic integrals to adaptive quadrature, including both exponential limits."""
    from scipy.integrate import quad
    breaks = np.array([6., 24.])
    for tail in (0., .001, .32, .5, 1.):
        for ratios in (np.log([1., 1.]), np.log([.125, 3.5]), np.log([4., .36])):
            def integrand(r):
                t = float(warp([r], breaks, ratios)[0])
                return r * float(np.exp(log_profile(t, 11., tail)))
            for lo, hi in ((0., 5.), (5., 19.), (21., 147.), (0., 12742.), (230., 490.)):
                ref = quad(integrand, lo, hi, points=breaks[(lo < breaks) & (breaks < hi)],
                           epsabs=1e-10, epsrel=2e-11, limit=200)[0]
                actual = segment_integral(lo, hi, 11., np.asarray(tail), breaks, ratios)
                np.testing.assert_allclose(actual, ref, rtol=2e-9, atol=1e-10)


def load_full(work, variant, smooth_cache=None):
    record = json.loads((work/variant/"result.json").read_text())
    if record["principal_input_sha256"] != digest(audit_input()):
        raise ValueError("The fitted and current source-audit inputs differ")
    latest = record["full_fit"]["attempts"][-1]
    folder = smooth_cache if variant == "smooth" and smooth_cache is not None else work/variant/"full"
    path = folder / latest.get("file", f"attempt-{latest['attempt']}.npz")
    if digest(path) != latest["posterior_sha256"]:
        raise ValueError(f"Posterior checksum mismatch in {folder}")
    with np.load(path) as archive:
        draws = dict(archive)
    data, meta = audited_data(audit_input())
    policy = replace(BASE, name="exponential", fixed_tail_shape=0.) if VARIANTS[variant][1] else BASE
    geometry = Geometry(data.canonical(), meta["vent_lonlat"][1], policy)
    return draws, geometry, latest


def run(work, variant, output, smooth_cache):
    import arviz as az
    verify_radial_identity()
    draws, geometry, latest = load_full(work, variant, smooth_cache)
    selected = {k:v[:, :2] for k,v in draws.items()}
    neutral = dict(selected)
    shape = (*selected["source"].shape, 2)
    neutral["break_km"] = np.broadcast_to(np.array([30., 180.]), shape)
    neutral["log_slope_change"] = np.zeros(shape)
    rule = make_rule(geometry, 2, 5)
    np.testing.assert_allclose(integrate_broken(neutral, rule), integrate(selected, geometry, rule),
                               rtol=2e-8, atol=1e-10)
    mass, checks = integrate_full(draws, geometry)
    total = mass.sum(axis=(-1,-2))
    posterior = az.from_dict(posterior={"log_volume": np.log(total)})
    rhat = float(az.rhat(posterior).log_volume)
    bulk = float(az.ess(posterior, method="bulk").log_volume)
    tail = float(az.ess(posterior, method="tail").log_volume)
    output.mkdir(parents=True, exist_ok=True)
    path = output/(variant+".npz")
    np.savez_compressed(path, total_km3=total)
    probabilities = [.025, .5, .975]
    summary = {"variant": variant, "input_sha256": digest(audit_input()),
               "posterior_sha256": latest["posterior_sha256"], "totals_sha256": digest(path),
               "volume_draw_shape": list(total.shape), "sampler_passed": latest["diagnostics"]["passed"],
               "quadrature": checks, "volume_chain": {"rhat": rhat, "bulk_ess": bulk, "tail_ess": tail,
               "log_quantile_mcse": [float(az.mcse(posterior, method="quantile", prob=p).log_volume) for p in probabilities],
               "passed": rhat < 1.01 and min(bulk,tail) >= 400},
               "volume_km3": dict(zip(("lower95","median","upper95"), np.quantile(total,probabilities).tolist(), strict=True))}
    write_json(output/(variant+".json"), summary)
    print(variant, summary["volume_km3"], checks["passed"], summary["volume_chain"]["passed"], flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--variant", choices=VARIANTS, required=True)
    parser.add_argument("--smooth-cache", type=Path, default=None,
                        help="Existing signed smooth-reference full-fit directory if reused in the thinning experiment")
    args = parser.parse_args()
    run(args.work, args.variant, args.output, args.smooth_cache)
