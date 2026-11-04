"""Matched convex-hull volumes; never extrapolate to the whole Earth."""
import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.linalg import cholesky
from scipy.spatial import ConvexHull
from scipy.special import logsumexp

from .core.distributions import predictive_distribution


class SurveyRegion:
    def __init__(self, xy):
        xy = np.unique(np.asarray(xy, float), axis=0)
        if xy.ndim != 2 or xy.shape[1] != 2 or not np.isfinite(xy).all() or len(xy) < 3:
            raise ValueError("A survey region needs at least three finite locations.")
        if np.linalg.matrix_rank(xy-xy.mean(axis=0)) < 2:
            raise ValueError("A survey region needs non-collinear locations.")
        self.hull = ConvexHull(xy)
        self.vertices = xy[self.hull.vertices]
        self.area = float(self.hull.volume)

    def contains(self, xy):
        equations = self.hull.equations
        return np.all(np.asarray(xy)@equations[:, :2].T+equations[:, 2] <= 1e-8, axis=1)

    def quadrature(self, order):
        if not isinstance(order, int) or order < 2:
            raise ValueError("Triangle quadrature needs an integer order >=2.")
        nodes, weights = leggauss(order)
        nodes, weights = (nodes+1)/2, weights/2
        u, v = np.meshgrid(nodes, nodes)
        u, v = u.ravel(), v.ravel()
        w = np.outer(weights, weights).ravel()*(1-u)
        center = self.vertices.mean(axis=0)
        points, areas = [], []
        for b, c in zip(self.vertices, np.roll(self.vertices, -1, axis=0)):
            ab, ac = b-center, c-center
            jacobian = abs(ab[0]*ac[1]-ab[1]*ac[0])
            points.append(center+u[:, None]*ab+((1-u)*v)[:, None]*ac)
            areas.append(w*jacobian)
        return np.concatenate(points), np.concatenate(areas)

    def describe(self):
        return {"definition": "Convex hull of all accepted measurement and bound locations; no vent or pseudo-points added.",
                "vertices_xy_km": self.vertices.tolist(), "area_km2": self.area,
                "interpretation": "Bulk fallout inside the surveyed footprint, including unsampled gaps; not total eruption volume."}


def integrate_region(model, region, orders=(8, 12, 20, 32, 48, 64), tolerance=.001):
    history, previous, consecutive = [], None, 0
    for order in orders:
        xy, weights = region.quadrature(order)
        # Bound prediction memory for triangulated supports with many components.
        expected, medians = [], []
        for start in range(0, len(xy), 2048):
            dist = predictive_distribution(model, xy[start:start+2048])
            expected.append(dist.log_expected_thickness())
            medians.append(dist.quantile(.5))
        logs = np.array([logsumexp(np.log(weights)+np.concatenate(expected)),
                         logsumexp(np.log(weights)+np.concatenate(medians))])+np.log(1e-5)
        if not np.isfinite(logs).all() or np.any(logs > 700):
            raise RuntimeError("Regional volume is not numerically representable; inspect unconstrained surface variance.")
        values = np.exp(logs)
        change = None if previous is None else float(np.max(abs(values-previous)/np.maximum(values, 1e-300)))
        history.append({"order": order, "n_points": len(xy), "expected_km3": float(values[0]),
                        "median_surface_km3": float(values[1]), "relative_change": change})
        consecutive = consecutive+1 if change is not None and change < tolerance else 0
        if consecutive >= 2:
            break
        previous = values
    return {"expected_km3": float(values[0]), "median_surface_km3": float(values[1]),
            "converged": consecutive >= 2, "tolerance": tolerance, "refinement": history,
            "unit_conversion": "cm * km^2 * 1e-5 = km^3",
            "scope": "Integral on the explicitly supplied regional geometry; no global integral.",
            "median_surface_note": "Integral of pointwise median thickness; not the median of the volume distribution."}


def regional_draws(model, region, reference, draws=2000, seed=1729):
    """Joint latent draws, with quadrature checked against expected volume.

Intervals are conditional, not selection-adjusted. Sampling-grid agreement of
the mean does not prove convergence of tail quantiles; refinement is reported.
"""
    if not isinstance(draws, int) or draws < 100 or seed < 0:
        raise ValueError("Use at least 100 joint draws and a nonnegative seed.")
    if not np.isfinite(reference) or reference <= 0:
        raise ValueError("Reference expected volume must be positive and finite.")
    grid_checks = []
    for order in (4, 6, 8, 10, 12, 16):
        xy, weights = region.quadrature(order)
        dist = predictive_distribution(model, xy)
        mean = float(np.exp(logsumexp(np.log(weights)+dist.log_expected_thickness()))*1e-5)
        relative = abs(mean/reference-1)
        grid_checks.append({"order": order, "n_points": len(xy), "expected_km3": mean,
                            "relative_error_to_refined_mean": relative})
        if relative < .005:
            break
    rng = np.random.default_rng(seed)
    log_weights = np.log(weights)[:, None]+np.log(1e-5)
    if hasattr(model, "draw_coefficients"):
        coefficients = model.draw_coefficients(draws, rng)
        log_surfaces = model.basis.design(xy)@coefficients
        log_volumes = logsumexp(log_weights+log_surfaces, axis=0)
        jitter = 0.
    else:
        models = getattr(model, "models", [model])
        component_weights = getattr(model, "weights", np.ones(1))
        components = rng.choice(len(models), size=draws, p=component_weights)
        log_volumes = np.empty(draws)
        jitter = 0.
        for index, fitted in enumerate(models):
            mask = components == index
            if not mask.any():
                continue
            mean, covariance = fitted.predict_log(xy, covariance=True)
            extra = max(float(np.max(np.diag(covariance))), 1.)*1e-10
            L = cholesky((covariance+covariance.T)/2+np.eye(len(xy))*extra, lower=True)
            jitter = max(jitter, extra)
            values = mean[:, None]+L@rng.standard_normal((len(xy), mask.sum()))
            log_volumes[mask] = logsumexp(log_weights+values, axis=0)
    if not np.isfinite(log_volumes).all() or np.max(log_volumes) > 700:
        raise RuntimeError("Joint regional-volume draws overflow; no interval can be reported.")
    volumes = np.exp(log_volumes)
    # Order-statistic sampling precision, conditional on this quadrature/model.
    from scipy.stats import binom
    sorted_values = np.sort(volumes)
    mc = {}
    for p in (.025, .5, .975):
        indices = np.clip(binom.ppf([.05, .95], draws, p).astype(int), 0, draws-1)
        mc[str(p)] = sorted_values[indices].tolist()
    return {"conditional_quantiles_km3": dict(zip(("0.025", "0.5", "0.975"),
                                                 np.quantile(volumes, [.025, .5, .975]).tolist())),
            "draws": draws, "seed": seed, "integration_order": order,
            "sampling_mean_km3": float(volumes.mean()), "grid_checks": grid_checks,
            "expected_grid_converged": relative < .005, "covariance_jitter_log": jitter,
            "quantile_monte_carlo_90_ranges_km3": mc,
            "interpretation": "Conditional joint-surface intervals. Omit fitted-parameter and selection uncertainty. Mean-grid and Monte Carlo checks do not establish tail-quantile quadrature convergence."}
