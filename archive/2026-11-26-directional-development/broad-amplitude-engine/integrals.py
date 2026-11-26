"""Whole-field volume, angular mass and independent conditional uncertainty checks."""

import numpy as np
from scipy.stats import qmc

from ....support import SupportMesh
from ..ensemble.integrals import polar_nodes
from .geometry import LEGACY_SCHEMA


def trials(model, data, parameter_seed, point_seed, draws, order):
    params = model.sample_parameters(draws, parameter_seed)
    u = qmc.Sobol(2, scramble=True, seed=point_seed).random_base2(order)
    xy, w, r, bearing = polar_nodes(u, model.domain.axis_scale)
    sector = ((bearing + np.pi / 8) // (np.pi / 4)).astype(int) % 8
    mesh = SupportMesh(data)
    face = mesh.triangulation.find_simplex(xy)
    inside = (face >= 0) & mesh.accepted[face.clip(min=0)]
    site_r = model.domain.rho(data.xy[~data.censored])
    classes = np.stack([inside, r < site_r.min(), r > site_r.max()])
    mass = np.zeros((draws, 8))
    regions = np.zeros((draws, 3))
    for start in range(0, len(r), 256):
        ix = slice(start, start + 256)
        flux = np.exp(model.draw_fields(xy[ix], params)) * (1e-5 * w[ix] / len(w))
        for j in range(8):
            mass[:, j] += flux[:, sector[ix] == j].sum(1)
        regions += flux @ classes[:, ix].T
    ids = np.empty(draws, dtype=int)
    for i, ix, _ in params:
        ids[ix] = i
    return mass, regions, ids


def integrate(model, data, path):
    history = []
    previous = None
    legacy = getattr(model, "protocol", LEGACY_SCHEMA) == LEGACY_SCHEMA
    order = 11
    for n in (512, 1024, 2048, 4096, 8192, 16384, 32768):
        first, regions, ids = trials(model, data, 34141, 51071, n, order)
        same, same_regions, _ = trials(model, data, 34141, 68389, n, order)
        other, region2, ids2 = trials(model, data, 79031, 68389, n, order)
        qparts = np.array([np.quantile(x.sum(1), [0.025, 0.5, 0.975]) for x in (first, same, other)])
        totals = np.concatenate(((first + same) / 2, other)).sum(1)
        q = np.quantile(totals, [0.025, 0.5, 0.975])
        geo = float(np.max(abs(qparts[0] - qparts[1]) / q))
        sample = float(np.max(abs(qparts[1] - qparts[2]) / q))
        change = float(np.max(abs(previous / q - 1))) if previous is not None else None
        passed = change is not None and geo < 0.04 and sample < 0.10 and change < 0.08
        history.append(
            dict(
                draws_per_population=n,
                points_per_spatial_scramble=2**order,
                conditional_quantiles_km3=q.tolist(),
                spatial_relative_difference=geo,
                population_relative_difference=sample,
                refinement_relative_difference=change,
                passed=passed,
            )
        )
        if passed:
            break
        previous = q
        if legacy:
            if n == 8192:
                break
            order += 1
        else:
            # Spatial and parameter uncertainty need different effort. Once
            # the two spatial rules agree to 1%, refine the parameter tail
            # without also doubling a converged spatial grid. Recheck both
            # spatial scrambles on every enlarged parameter population.
            order = min(16, max(12, order + int(geo >= 0.01)))
    masses = np.concatenate(((first + same) / 2, other))
    regions = np.concatenate(((regions + same_regions) / 2, region2))
    ids = np.r_[ids, ids2]
    np.savez_compressed(path, volume=totals, sector=masses, allocation=regions, component=ids)
    sensitivity = {}
    keys = ("scale", "evolving", "length", "spread", "tail", "noise")
    if all(hasattr(c, "amplitude") for c in model.components):
        keys += ("amplitude",)
    for key in keys:
        values = np.array([getattr(c, key) for c in model.components])
        sensitivity[key] = [
            {
                "value": float(v),
                "posterior_mass": float(model.weights[values == v].sum()),
                "draws": int((values[ids] == v).sum()),
                "conditional_interval95_km3": (
                    np.quantile(totals[values[ids] == v], [0.025, 0.975]).tolist()
                    if (values[ids] == v).sum() >= 32
                    else None
                ),
            }
            for v in np.unique(values)
        ]
    return dict(
        independent_draw_integration=history,
        conditional_interval95_km3=q[[0, 2]].tolist(),
        integration_passed=bool(passed),
        structure_sensitivity=sensitivity,
        sector_order=["N", "NE", "E", "SE", "S", "SW", "W", "NW"],
        sector_volume_quantiles_km3=np.quantile(masses, [0.025, 0.5, 0.975], axis=0).T.tolist(),
        allocation_order=[
            "retained exact-site triangles",
            "inside nearest exact radius",
            "beyond farthest exact radius",
        ],
        allocation_fraction_quantiles=np.quantile(
            regions / totals[:, None], [0.025, 0.5, 0.975], axis=0
        ).T.tolist(),
    )
