"""Whole-Earth volumes and coverage diagnostics for the geographic nested model."""
import numpy as np
from scipy.stats import qmc

from src.probabilistic_isopach.global_volume import EarthDomain
from ...support import SupportMesh
from ..directional.ensemble.integrals import polar_nodes
from .geometry import positions
from .reach import PROBS

LEVELS = np.array([.01, .1, 1., 10., 100.])
SECTORS = ("N", "NE", "E", "SE", "S", "SW", "W", "NW")
SEEDS = (28439, 95717)


def nodes(order, seed, latitude):
    u = qmc.Sobol(2, scramble=True, seed=seed).random_base2(order)
    return polar_nodes(u, EarthDomain.wgs84(latitude).axis_scale)


def integrate(model, data, order, seed, levels=LEVELS):
    xy, weights, r, theta = nodes(order, seed, model.latitude)
    weights /= len(r)
    exact = ~data.censored
    limits = positions(data.xy[exact], model.latitude)[0]
    mesh = SupportMesh(data)
    tri = mesh.triangulation.find_simplex(xy)
    inside = (tri >= 0) & mesh.accepted[np.maximum(tri, 0)]
    outside_r = r > np.max(limits)
    sector = np.floor((theta + np.pi / 8) / (np.pi / 4)).astype(int) % 8
    n = len(model.law)
    mass = np.zeros((n, 8))
    allocation = np.zeros((n, 3))
    parent_sum = np.zeros(n)
    areas = np.zeros((len(levels), 8))
    missed_support = np.zeros(len(levels))
    map_mass = 0.
    for ids in np.array_split(np.arange(len(r)), max(1, len(r) // 192)):
        base = model.parent.field(xy[ids])
        field = base + model.corridors.penalties(r[ids], theta[ids])[model.law]
        weighted = np.exp(field) * (1e-5 * weights[ids])
        parent_sum += np.exp(base) @ (1e-5 * weights[ids])
        median = np.median(field, axis=0)
        map_mass += np.exp(median) @ (1e-5 * weights[ids])
        exceeds = (median >= np.log(levels)[:, None]) * weights[ids]
        missed_support += exceeds[:, ~inside[ids]].sum(axis=1)
        for k in range(8):
            select = sector[ids] == k
            mass[:, k] += weighted[:, select].sum(axis=1)
            areas[:, k] += exceeds[:, select].sum(axis=1)
        allocation[:, 0] += weighted[:, inside[ids]].sum(axis=1)
        allocation[:, 1] += weighted[:, outside_r[ids]].sum(axis=1)
        allocation[:, 2] += weighted[:, r[ids] < np.min(limits)].sum(axis=1)
    return mass, allocation, areas, missed_support, parent_sum, map_mass


def checked(model, data):
    """Two scrambles, an exact parent integral and independent particle populations."""
    history = []
    previous = None
    for n, order in ((512, 12), (1024, 13), (2048, 14), (4096, 15)):
        part = model.reduced(n)
        trials = [integrate(part, data, order, seed) for seed in SEEDS]
        mass, allocation, areas, unsupported, reference, map_mass = [np.mean(x, axis=0) for x in zip(*trials)]
        volume = mass.sum(axis=1)
        q = np.quantile(volume, PROBS)
        pair = np.stack([np.quantile(x[0].sum(axis=1), PROBS) for x in trials])
        pare = np.stack([x[4] / part.parent.volume() for x in trials])
        scramble = float(np.max(np.ptp(pair, axis=0) / q))
        parent_error = float(np.quantile(abs(pare - 1.), .95))
        change = None if previous is None else float(np.max(abs(q / previous - 1.)))
        passed = change is not None and change < .10 and scramble < .04 and parent_error < .05
        history.append({"draws":len(volume), "nodes_per_scramble":2**order, "scrambles":list(SEEDS),
                        "volume_km3":q.tolist(), "scramble_difference":scramble,
                        "change_from_previous":change, "parent_q95_relative_error":parent_error, "passed":passed})
        if passed:
            break
        previous = q
    group_q = np.array([np.quantile(x, PROBS) for x in np.array_split(volume, 4)])
    spread = np.ptp(np.log(group_q), axis=0)
    totals = areas.sum(axis=1)
    report = {"volume_km3":q.tolist(), "median_map_volume_km3":float(map_mass),
              "volume_population_quantiles":group_q.tolist(),
              "volume_populations_passed":bool(spread[1] < np.log(1.35) and max(spread[[0, 2]]) < np.log(2.)),
              "quadrature":history, "level_cm":LEVELS.tolist(), "median_map_area_km2":totals.tolist(),
              "median_map_area_by_sector_km2":areas.T.tolist(),
              "area_outside_retained_exact_triangles_km2":unsupported.tolist(),
              "sectors":list(SECTORS), "sector_volume_km3":np.quantile(mass, PROBS, axis=0).T.tolist(),
              "sector_volume_fraction":np.quantile(mass / volume[:, None], PROBS, axis=0).T.tolist(),
              "allocation_order":["inside_retained_exact_triangles","beyond_farthest_exact","inside_nearest_exact"],
              "allocation_fraction":np.quantile(allocation / volume[:, None], PROBS, axis=0).T.tolist()}
    return report, {"volume_km3":volume,"sector_km3":mass,"allocation_km3":allocation,"law":part.law}
