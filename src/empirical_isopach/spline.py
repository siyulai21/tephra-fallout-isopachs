"""Cubic tensor splines with normalized trend/derivative penalties.

No artificial observation enters the likelihood. Gaussian coefficient uncertainty
is exact conditional on the fitted trend, noise and penalties for exact data;
censoring uses a conditional Laplace approximation. It is not full Bayes.
"""
from dataclasses import asdict, dataclass
from functools import lru_cache

import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.interpolate import BSpline
from scipy.linalg import cho_factor, cho_solve, solve_triangular
from scipy.optimize import minimize

from src.probabilistic_isopach.model import censored_terms


def survey_bounds(xy):
    """Known sampling geometry, with a fixed 5% margin; no thickness input."""
    xy = np.asarray(xy, float)
    if xy.ndim != 2 or xy.shape[1] != 2 or not np.isfinite(xy).all():
        raise ValueError("Coordinates must be a finite (n, 2) array.")
    lo, hi = xy.min(axis=0), xy.max(axis=0)
    span = hi - lo
    if np.any(span <= 0) or np.linalg.matrix_rank(xy - xy.mean(axis=0)) < 2:
        raise ValueError("Regional surfaces require non-collinear sampling locations.")
    return np.r_[lo - .05*span, hi + .05*span]


def checked_bounds(bounds):
    bounds = np.asarray(bounds, float)
    if bounds.shape != (4,) or not np.isfinite(bounds).all() or np.any(bounds[2:] <= bounds[:2]):
        raise ValueError("Bounds must be finite xmin, ymin, xmax, ymax with positive area.")
    return bounds


@dataclass(frozen=True)
class SplineCandidate:
    tail: str = "exponential"
    knot_intervals: int = 4  # intervals on the longer side of the rectangle
    tension: float = .5
    smoothing: float = .01
    trend_strength: float = 1.
    noise_ratio: float = .75

    def __post_init__(self):
        if self.tail not in ("exponential", "power"):
            raise ValueError("Unknown thinning trend.")
        if (not isinstance(self.knot_intervals, (int, np.integer))
                or not 1 <= self.knot_intervals <= 16):
            raise ValueError("Knot intervals must be an integer in [1, 16].")
        if not np.isfinite(self.tension) or not 0 <= self.tension <= 1:
            raise ValueError("Tension must be in [0, 1].")
        for name in ("smoothing", "noise_ratio"):
            if not np.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive and finite.")
        if not np.isfinite(self.trend_strength) or self.trend_strength < 0:
            raise ValueError("Trend strength must be nonnegative and finite.")


class DirectionalTrend:
    """Training-only constrained mean; no GP residual or kernel is evaluated.

Exact measurements estimate the target; bounds act through the final spline
likelihood. Both regional trend families only require outward decay, not a
global-integrability condition. Residual scale is an empirical plug-in estimate.
"""
    def __init__(self, data, tail):
        self.tail = tail
        det = ~data.censored
        radii = np.linalg.norm(data.xy[det], axis=1)
        self.radius_scale = max(float(np.median(radii[radii > 0])), 1e-6)
        H, y = self.features(data.xy[det]), data.log_value[det]
        # Estimate the same target even when noisy reported limits are added.
        constraint = {"type": "ineq",
                      "fun": lambda b: b[1]-np.linalg.norm(b[2:])-1e-6,
                      "jac": lambda b: np.r_[0., 1., -b[2:]/max(np.linalg.norm(b[2:]), 1e-12)]}
        result = minimize(lambda b: (.5*np.sum((H@b-y)**2), H.T@(H@b-y)),
                          np.array([np.mean(y), 1., 0., 0.]), jac=True,
                          constraints=constraint, method="SLSQP",
                          options={"ftol": 1e-10, "maxiter": 300})
        if not result.success or constraint["fun"](result.x) < -1e-6:
            raise RuntimeError(f"Spline trend did not converge: {result.message}")
        self.beta = result.x
        self.amplitude = max(float(np.sqrt(np.mean((y-H@self.beta)**2))), .1)
        self.on_boundary = bool(constraint["fun"](self.beta) < 1e-4)

    def features(self, xy):
        xy = np.asarray(xy, float)
        radius = np.linalg.norm(xy, axis=1)
        direction = xy/np.maximum(radius[:, None], np.finfo(float).eps)
        h = radius/self.radius_scale
        if self.tail == "power":
            h = np.log1p(h)
        return np.column_stack((np.ones(len(xy)), -h, direction*h[:, None]))

    def predict(self, xy):
        return self.features(xy)@self.beta


def axis_basis(width, intervals, order):
    edges = np.linspace(0., width, intervals+1)
    knots = np.r_[np.repeat(edges[0], 4), edges[1:-1], np.repeat(edges[-1], 4)]
    spline = BSpline(knots, np.eye(len(knots)-4), 3, extrapolate=False)
    nodes, weights = leggauss(order)
    x = ((edges[:-1, None]+edges[1:, None])/2
         + np.diff(edges)[:, None]*nodes/2).ravel()
    w = (np.diff(edges)[:, None]*weights/2).ravel()
    values = [spline(x, nu=k) for k in range(3)]
    grams = [b.T@(w[:, None]*b) for b in values]
    return spline, x, w, grams


class TensorBasis:
    def __init__(self, bounds, intervals, quadrature_order=6):
        self.bounds = checked_bounds(bounds)
        self.scale = float(max(self.bounds[2:]-self.bounds[:2]))
        widths = (self.bounds[2:]-self.bounds[:2])/self.scale
        self.intervals = tuple(max(1, int(np.floor(intervals*w+.5))) for w in widths)
        bx, x, wx, gx = axis_basis(widths[0], self.intervals[0], quadrature_order)
        by, y, wy, gy = axis_basis(widths[1], self.intervals[1], quadrature_order)
        self.axes = bx, by
        self.n_coefficients = len(bx.c)*len(by.c)
        area = np.prod(widths)
        # Gauss order >=4 exactly integrates all products of cubic derivatives.
        self.mass = np.kron(gy[0], gx[0])/area
        self.gradient = (np.kron(gy[0], gx[1])+np.kron(gy[1], gx[0]))/area
        self.curvature = (np.kron(gy[0], gx[2])+2*np.kron(gy[1], gx[1])
                          + np.kron(gy[2], gx[0]))/area
        xx, yy = np.meshgrid(x, y)
        self.quadrature_xy = self.bounds[:2] + self.scale*np.column_stack((xx.ravel(), yy.ravel()))
        self.quadrature_weights = np.outer(wy, wx).ravel()/area
        self.quadrature_basis = self.design(self.quadrature_xy)

    def design(self, xy):
        xy = np.asarray(xy, float)
        if xy.ndim != 2 or xy.shape[1] != 2 or not np.isfinite(xy).all():
            raise ValueError("Prediction coordinates must be a finite (n, 2) array.")
        if np.any(xy < self.bounds[:2]) or np.any(xy > self.bounds[2:]):
            raise ValueError("Spline predictions outside its fitted rectangle are undefined.")
        uv = (xy-self.bounds[:2])/self.scale
        # Clip only round-off at an already checked boundary.
        upper = (self.bounds[2:]-self.bounds[:2])/self.scale
        uv = np.clip(uv, 0, upper)
        return np.einsum("ni,nj->nij", self.axes[1](uv[:, 1]),
                         self.axes[0](uv[:, 0])).reshape(len(xy), self.n_coefficients)


@lru_cache(maxsize=12)
def tensor_basis(bounds, intervals, quadrature_order=6):
    return TensorBasis(bounds, intervals, quadrature_order)


class SplineWorkspace:
    """Cache response-dependent quantities only within one training subset."""
    def __init__(self, data, bounds=None):
        self.data = data.canonical()
        det = ~self.data.censored
        if det.sum() < 6 or len(np.unique(self.data.xy[det], axis=0)) < 4:
            raise ValueError("Need six exact measurements at four distinct locations.")
        if np.linalg.matrix_rank(self.data.xy[det]-self.data.xy[det].mean(axis=0)) < 2:
            raise ValueError("Exact measurements must not all be collinear.")
        self.bounds = checked_bounds(survey_bounds(data.xy) if bounds is None else bounds)
        if np.any(data.xy < self.bounds[:2]) or np.any(data.xy > self.bounds[2:]):
            raise ValueError("Fit rectangle must contain every training location.")
        self.trends, self.designs, self.targets = {}, {}, {}

    def prepare(self, candidate, order):
        if candidate.tail not in self.trends:
            self.trends[candidate.tail] = DirectionalTrend(self.data, candidate.tail)
        trend = self.trends[candidate.tail]
        basis = tensor_basis(tuple(self.bounds), candidate.knot_intervals, order)
        key = (candidate.knot_intervals, order)
        if key not in self.designs:
            self.designs[key] = basis.design(self.data.xy)
        tkey = (*key, candidate.tail)
        if tkey not in self.targets:
            self.targets[tkey] = basis.quadrature_basis.T@(
                basis.quadrature_weights*trend.predict(basis.quadrature_xy))
        return trend, basis, self.designs[key], self.targets[tkey]


class TrendRegularizedSpline:
    def __init__(self, candidate=SplineCandidate(), quadrature_order=6):
        if not isinstance(quadrature_order, int) or not 4 <= quadrature_order <= 20:
            raise ValueError("Quadrature order must be an integer from 4 to 20.")
        self.candidate, self.quadrature_order = candidate, quadrature_order

    def fit(self, data, bounds=None, workspace=None):
        workspace = SplineWorkspace(data, bounds) if workspace is None else workspace
        self.data = data = workspace.data
        self.bounds = workspace.bounds.copy()
        c = self.candidate
        self.trend, self.basis, B, target = workspace.prepare(c, self.quadrature_order)
        self.noise = max(c.noise_ratio*self.trend.amplitude, .05)
        self.sd = np.sqrt(self.noise**2+data.log_sd**2)
        det = ~data.censored
        s2 = self.trend.amplitude**2
        self.penalty = (c.smoothing*(c.tension*self.basis.gradient
                        +(1-c.tension)*self.basis.curvature)
                        +c.trend_strength*self.basis.mass)/s2
        self.target_rhs = c.trend_strength*target/s2
        self.penalty = (self.penalty+self.penalty.T)/2
        W = 1/self.sd[det]**2
        base_hessian = self.penalty+B[det].T@(W[:, None]*B[det])
        rhs = self.target_rhs+B[det].T@(W*data.log_value[det])
        factor = cho_factor(base_hessian, lower=True)
        coefficients = cho_solve(factor, rhs)

        def objective(beta, hessian=False):
            residual = (B[det]@beta-data.log_value[det])/self.sd[det]
            loss = .5*(residual@residual+beta@self.penalty@beta)-self.target_rhs@beta
            gradient = base_hessian@beta-rhs
            H = base_hessian.copy() if hessian else None
            if (~det).any():
                terms, grad, curvature = censored_terms(B[~det]@beta,
                    data.log_value[~det], self.sd[~det], data.lower_censored[~det])
                loss += terms.sum()
                gradient += B[~det].T@grad
                if hessian:
                    H += B[~det].T@(curvature[:, None]*B[~det])
            return loss, gradient, H

        iterations = 0
        if (~det).any():
            for iterations in range(80):
                loss, gradient, H = objective(coefficients, True)
                factor = cho_factor(H, lower=True)
                step = cho_solve(factor, gradient)
                decrement = float(gradient@step)
                if decrement < 1e-12:
                    break
                length = 1.
                while length > 2**-25:
                    proposed = coefficients-length*step
                    if objective(proposed)[0] <= loss-1e-4*length*decrement:
                        coefficients = proposed
                        break
                    length /= 2
                else:
                    if decrement < 1e-8:
                        break
                    raise RuntimeError("Censored spline Newton line search failed.")
            else:
                raise RuntimeError("Censored spline did not converge in 80 Newton steps.")
        _, gradient, H = objective(coefficients, True)
        self.factor = cho_factor(H, lower=True)
        self.newton_decrement = float(gradient@cho_solve(self.factor, gradient))
        if not np.isfinite(coefficients).all() or self.newton_decrement > 1e-7:
            raise RuntimeError("Spline solution failed its stationarity check.")
        self.coefficients = coefficients
        self.iterations = iterations
        self.effective_df_exact = float(np.trace(cho_solve(self.factor, B[det].T@(W[:, None]*B[det]))))
        return self

    def predict_log(self, xy, covariance=False, observation=False, log_sd=None):
        B = self.basis.design(xy)
        mean = B@self.coefficients
        solved = solve_triangular(self.factor[0], B.T, lower=True)
        var = solved.T@solved if covariance else np.sum(solved**2, axis=0)
        if observation:
            known = np.zeros(len(B)) if log_sd is None else np.asarray(log_sd, float)
            if known.shape != (len(B),) or not np.isfinite(known).all() or np.any(known < 0):
                raise ValueError("Known log errors must be finite, nonnegative and match predictions.")
            if covariance:
                var[np.diag_indices_from(var)] += self.noise**2+known**2
            else:
                var += self.noise**2+known**2
        return mean, var

    def draw_coefficients(self, draws, rng):
        """Draw one complete correlated spline surface per coefficient vector."""
        z = rng.standard_normal((len(self.coefficients), draws))
        return self.coefficients[:, None]+solve_triangular(self.factor[0].T, z, lower=False)

    def describe(self):
        return {"method": "spline", "candidate": asdict(self.candidate),
                "bounds_km": self.bounds.tolist(), "knot_intervals_xy": list(self.basis.intervals),
                "n_coefficients": len(self.coefficients), "noise_log": self.noise,
                "amplitude_log": self.trend.amplitude, "trend_beta": self.trend.beta.tolist(),
                "trend_radius_scale_km": self.trend.radius_scale,
                "trend_on_constraint_boundary": self.trend.on_boundary,
                "effective_df_exact": self.effective_df_exact,
                "newton_decrement": self.newton_decrement, "newton_iterations": self.iterations,
                "quadrature_order": self.quadrature_order,
                "uncertainty": "Conditional Laplace" if self.data.censored.any() else "Conditional Gaussian",
                "uncertainty_omits": ["fitted trend", "noise and smoothing estimation", "knot/trend selection",
                                      "sampling and preservation bias", "vent location error"],
                "pseudo_observations": 0}
