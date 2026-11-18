"""Integrate the median map and independently integrate whole posterior fields."""

import numpy as np
from scipy.stats import qmc

from ..estimate import estimate
from ..geometry import RADIUS, polar
from ..integration import integrate_checked
from .fields import draw_weights, inverse_cdf, latent_arrays
from .members import directory


SEEDS = (41791, 80627)
ORDERS = (11, 12, 13, 14, 15, 16)
SUMMARY = ("central", "uniform", "without_extra_fan", "with_extra_fan", "even_draws", "odd_draws", "mean_field")
PROBS = (.025, .5, .975)


def interval(values, weights):
    q = inverse_cdf(np.asarray(values), weights, PROBS)
    return dict(zip(("lower95","median","upper95"), q.tolist(), strict=True))


def coherent(model, data, latitude, work, full):
    masses, map_means, checks, members = [], [], [], {}
    for m in model.members:
        policy = m.geometry.policy
        folder = directory(work, policy)
        if full:
            detail = estimate(data, latitude, policy, folder)
            with np.load(folder/"mass.npz") as a:
                if not np.array_equal(a["draw_index"], np.arange(a["mass"].shape[1])):
                    raise ValueError("Expected complete chain volume histories")
                mass = a["mass"]
            selected = mass[:, m.index]
            passed = detail["passed"]
        else:
            mass, _, quadrature = integrate_checked(m.draws, m.geometry, n_per_chain=len(m.index))
            selected = mass
            passed = m.records[-1]["diagnostics"]["passed"] and quadrature["passed"]
            detail = {"attempts":m.records,"quadrature":quadrature,
                      "posterior_diagnostics_passed":m.records[-1]["diagnostics"]["passed"]}
        members[policy.name] = {k:detail[k] for k in ("attempts","quadrature","posterior_diagnostics_passed",
                                                      "volume_diagnostics","signature") if k in detail}
        members[policy.name]["passed"] = bool(passed)
        map_means.append(selected.sum(axis=(-1,-2)).mean())
        masses.append(mass.reshape(-1,5,8))
        checks.append(passed)
    weights = draw_weights(model.model_weights, [len(a) for a in masses])
    mass = np.concatenate(masses)
    totals = mass.sum(axis=(1,2))
    for m,v in zip(model.members,masses,strict=True):
        vals = v.sum(axis=(1,2))
        members[m.geometry.policy.name]["volume_km3"] = interval(vals,np.ones(len(vals)))
    return {"volume_km3":interval(totals,weights), "parts":[interval(x,weights) for x in mass.sum(2).T],
            "sectors":[interval(x,weights) for x in mass.sum(1).T],
            "uniform_volume_km3":interval(totals,draw_weights(np.ones(len(masses)),[len(a) for a in masses])),
            "field_mean_volume_km3":float(np.dot(model.model_weights,map_means)),
            "all_members_passed":bool(all(checks)), "members":members}, masses


def polar_nodes(uv, axis):
    """Log-radius parametrization of the entire equal-area disk. The scale only places nodes."""
    angle = uv[:,0]*2*np.pi
    L = np.log1p(2*RADIUS/50.)
    r = 50.*np.expm1(uv[:,1]*L)
    xy = np.column_stack((r*np.sin(angle)*axis, r*np.cos(angle)/axis))
    return xy, 2*np.pi*L*r*(r+50.), r, angle


def weight_vectors(model):
    w = model.model_weights
    extra = np.array([m.geometry.policy.hidden_fan for m in model.members])
    even = np.concatenate([np.tile(np.arange(len(m.index))%2==0, m.draws["source"].shape[0]) for m in model.members])
    return (model.weights, draw_weights(np.ones(len(w)), model.sizes),
            draw_weights(w*(~extra), model.sizes), draw_weights(w*extra, model.sizes),
            model.weights*even, model.weights*(~even))


def integrate_block(model, uv, weight_sets, levels=()):
    g = model.members[0].geometry
    totals = np.zeros((len(SUMMARY),40))
    areas = np.zeros((len(levels),5))
    inner, outer = np.quantile(polar(g.mesh.xy, g.axis)[0], [0.,1.])
    for start in range(0, len(uv), 192):
        xy, area, r, angle = polar_nodes(uv[start:start+192],g.axis)
        t, _, _ = g.locate(xy)
        part = np.where(r < inner, 2, np.where(r <= outer, 3, 4))
        hit = t >= 0
        part[hit] = np.where(g.inside[t[hit]], 0, 1)
        groups = 8*part + np.floor((angle+np.pi/8)/(np.pi/4)).astype(int)%8
        fields, _ = latent_arrays(model.members, xy)
        v = np.ascontiguousarray(np.concatenate(fields).T)
        order = np.argsort(v, axis=1, kind="stable")
        stats = [np.exp(inverse_cdf(v, w, order=order)[0]) for w in weight_sets]
        stats.append(np.exp(v) @ model.weights)
        for out, x in zip(totals, stats, strict=True):
            out += np.bincount(groups,weights=x*area*1e-5,minlength=40)
        for j, level in enumerate(levels):
            central, first, second = (stats[k] >= level for k in (0,4,5))
            union = first | second | central
            disjoint = first != second
            areas[j] += [area@central, area@disjoint, area@union, (area*area)@union, union.sum()]
    return totals.reshape(-1,5,8), areas


def median_integral(model, expected_mean, levels=()):
    weights = weight_vectors(model)
    generators = [qmc.Sobol(d=2,scramble=True,seed=s) for s in SEEDS]
    accum = np.zeros((len(SEEDS),len(SUMMARY),5,8))
    area_accum = np.zeros((len(SEEDS),len(levels),5))
    previous = None
    history = []
    for i, power in enumerate(ORDERS):
        for rep, gen in enumerate(generators):
            points = gen.random_base2(power if i == 0 else power-1)
            vol, area = integrate_block(model, points, weights, levels)
            accum[rep] += vol
            area_accum[rep] += area
        reps = accum/(2**power)
        avg = reps.mean(0)
        total = avg[0].sum()
        spread = np.abs(reps[0,0]-reps[1,0]).sum()/total
        change = None if previous is None else np.abs(avg[0]-previous).sum()/total
        mean_error = abs(avg[-1].sum()/expected_mean-1)
        split = np.max(np.abs(avg[4:6].sum(axis=(1,2))/total-1))
        item = {"points_per_scramble":2**power,"central_km3":float(total),
                "scramble_partition_L1_relative":float(spread),"nested_partition_L1_relative":change,
                "independent_same_draw_mean_relative_error":float(mean_error),
                "split_draw_central_relative_change":float(split)}
        level_checks = []
        for level, a in zip(levels, area_accum.sum(0), strict=True):
            ess = a[2]**2/a[3] if a[3] > 0 else 0.
            level_checks.append({"level_cm":float(level), "median_exceedance_area_km2":float(a[0]/(2*2**power)),
                "split_symmetric_difference_over_union":float(a[1]/a[2]) if a[2] else None,
                "weighted_union_ess":float(ess), "union_points":int(a[4]),"resolved_for_split_check":bool(ess>=64)})
        # Report sparse levels individually. A resolved level can trigger a larger draw packet.
        level_pass = all(a["split_symmetric_difference_over_union"] < .10
                         for a in level_checks if a["resolved_for_split_check"])
        history.append(item)
        spatial = change is not None and change < .015 and spread < .025 and mean_error < .01
        if spatial:
            break
        previous = avg[0].copy()
    numerical = {"history":history, "spatial_passed":bool(spatial), "draw_split_passed":bool(split < .025),
                 "level_split_checks":level_checks, "resolved_level_splits_passed":bool(level_pass),
                 "thresholds":{"resolved_level_split_symmetric_difference":.10,"minimum_weighted_union_ess":64,"nested_partition_L1":.015, "scramble_partition_L1":.025,
                               "mean_integral_relative":.01,"split_draw_relative":.025},
                 "scramble_seeds":list(SEEDS), "maximum_power":max(ORDERS)}
    return {"volume_km3":float(total), "parts_sectors_km3":avg[0].tolist(),
            "weight_sensitivity_km3":{k:float(a.sum()) for k,a in zip(SUMMARY[:4],avg[:4],strict=True)},
            "numerical":numerical,"passed":bool(spatial and split < .025 and level_pass)}
