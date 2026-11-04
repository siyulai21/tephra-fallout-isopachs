"""Conditional empirical-Bayes spatial model with a censored likelihood.

Trend and covariance selection are estimated from observations. Reported Gaussian
uncertainty is conditional on these estimates, NOT a full Bayesian posterior.
"""
from dataclasses import dataclass, asdict

import numpy as np
from scipy.linalg import cho_factor, cho_solve, cholesky, solve_triangular
from scipy.optimize import minimize
from scipy.spatial.distance import cdist, pdist
from scipy.special import log_ndtr

from ...core.observations import Observations
from ...core.likelihood import censored_terms


@dataclass(frozen=True)
class Candidate:
    tail: str = "exponential"
    length_factor: float = 0.5
    noise_ratio: float = 0.5

    def __post_init__(self):
        if self.tail not in {"exponential", "power"}:
            raise ValueError("tail must be exponential or power")
        if not np.isfinite(self.length_factor) or self.length_factor <= 0:
            raise ValueError("length_factor must be positive and finite")
        if not np.isfinite(self.noise_ratio) or self.noise_ratio <= 0:
            raise ValueError("noise_ratio must be positive and finite")


def matern32(a, b, length, amplitude):
    d = np.sqrt(3.) * cdist(a, b) / length
    return amplitude**2 * (1 + d) * np.exp(-d)


class SpatialModel:
    def __init__(self, candidate=Candidate()):
        self.candidate = candidate

    def _features(self, xy):
        xy = np.asarray(xy, dtype=float)
        r = np.linalg.norm(xy, axis=1)
        direction = xy / np.maximum(r[:, None], np.finfo(float).eps)
        h = r / self.radius_scale
        if self.candidate.tail == "power":
            h = np.log1p(h)
        return np.column_stack((np.ones(len(xy)), -h, direction*h[:, None]))

    def fit(self, data: Observations):
        data = data.canonical()
        det = ~data.censored
        if det.sum() < 6 or len(np.unique(data.xy[det], axis=0)) < 4:
            raise ValueError("Need at least six positive measurements at four distinct locations.")
        distances = pdist(data.xy[det])
        positive_distances = distances[distances > 0]
        if not len(positive_distances):
            raise ValueError("Measurements need spatial extent.")
        self.data = data
        radii = np.linalg.norm(data.xy[det], axis=1)
        self.radius_scale = max(float(np.median(radii[radii > 0])), 1e-6)
        self.length = self.candidate.length_factor * float(np.median(positive_distances))
        H = self._features(data.xy)
        # Convex constrained trend: slope stays decreasing on every ray. The
        # power tail has exponent >2 so its planar mean integral is finite.
        floor = 2.000001 if self.candidate.tail == "power" else 1e-6
        constraint = {"type": "ineq", "fun": lambda b: b[1]-np.linalg.norm(b[2:])-floor,
                      "jac": lambda b: np.r_[0., 1., -b[2:]/max(np.linalg.norm(b[2:]), 1e-12)]}
        hd, yd = H[det], data.log_value[det]
        initial = np.array([np.mean(yd), floor+1., 0., 0.])
        result = minimize(lambda b: (0.5*np.sum((hd@b-yd)**2), hd.T@(hd@b-yd)), initial,
                          jac=True, constraints=constraint, method="SLSQP",
                          options={"ftol": 1e-10, "maxiter": 300})
        if not result.success or constraint["fun"](result.x) < -1e-6:
            raise RuntimeError(f"Trend fit did not converge: {result.message}")
        self.beta = result.x
        residual = yd - hd@self.beta
        # Floors are recorded numerical/model defaults; they are not observations.
        self.amplitude = max(float(np.sqrt(np.mean(residual**2))), 0.1)
        self.noise = max(self.candidate.noise_ratio*self.amplitude, 0.05)
        self.sd = np.sqrt(self.noise**2 + data.log_sd**2)
        self.mean_train = H@self.beta
        K = matern32(data.xy, data.xy, self.length, self.amplitude)
        self.jitter = 1e-9*self.amplitude**2
        K.flat[::len(K)+1] += self.jitter
        if not np.any(data.censored):
            C = K + np.diag(self.sd**2)
            self.factor = cho_factor(C, lower=True)
            self.alpha = cho_solve(self.factor, data.log_value-self.mean_train)
            self.censored = False
        else:
            # Whiten the prior: f=m+Lv, v~N(0,I). The mixed likelihood is
            # log-concave, so the unique mode can be found without MCMC.
            L = cholesky(K, lower=True)
            def objective(v):
                f = self.mean_train + L@v
                grad = np.zeros(len(f))
                delta = (f[det]-data.log_value[det])/self.sd[det]
                loss = .5*(v@v + delta@delta)
                grad[det] = delta/self.sd[det]
                terms, g, _ = censored_terms(f[~det], data.log_value[~det], self.sd[~det], data.lower_censored[~det])
                grad[~det] = g
                return loss+terms.sum(), v+L.T@grad
            opt = minimize(objective, np.zeros(len(K)), jac=True, method="BFGS",
                           options={"gtol": 1e-7, "maxiter": 500})
            if np.max(np.abs(objective(opt.x)[1])) > 2e-5:
                raise RuntimeError(f"Censored posterior fit did not converge: {opt.message}")
            mode = self.mean_train + L@opt.x
            w = 1/self.sd**2
            w[~det] = censored_terms(mode[~det], data.log_value[~det], self.sd[~det], data.lower_censored[~det])[2]
            sw = np.sqrt(w)
            B = np.eye(len(K)) + sw[:, None]*K*sw[None, :]
            self.factor = cho_factor(B, lower=True)
            self.sw = sw
            self.alpha = solve_triangular(L.T, opt.x, lower=False)
            self.censored = True
        self.trend_boundary = bool(self.beta[1]-np.linalg.norm(self.beta[2:])-floor < 1e-4)
        return self

    def predict_log(self, xy, covariance=False, observation=False, log_sd=None):
        xy = np.asarray(xy, dtype=float)
        if xy.ndim != 2 or xy.shape[1] != 2 or not np.isfinite(xy).all():
            raise ValueError("Prediction coordinates must be a finite (n,2) array.")
        k = matern32(self.data.xy, xy, self.length, self.amplitude)
        mean = self._features(xy)@self.beta + k.T@self.alpha
        a = self.sw[:, None]*k if self.censored else k
        solved = cho_solve(self.factor, a)
        if covariance:
            var = matern32(xy, xy, self.length, self.amplitude) - a.T@solved
            var = (var+var.T)/2
            diagonal = np.diag_indices_from(var)
            var[diagonal] = np.maximum(var[diagonal], 0.)
        else:
            var = np.maximum(self.amplitude**2 - np.sum(a*solved, axis=0), 0.)
        if observation:
            known = 0. if log_sd is None else np.asarray(log_sd, dtype=float)**2
            if covariance:
                var[np.diag_indices_from(var)] += self.noise**2 + known
            else:
                var += self.noise**2 + known
        return mean, var

    def describe(self):
        return {"candidate": asdict(self.candidate), "beta": self.beta.tolist(),
                "radius_scale_km": self.radius_scale, "length_km": self.length,
                "amplitude_log": self.amplitude, "noise_log": self.noise,
                "jitter": self.jitter, "trend_on_constraint_boundary": self.trend_boundary,
                "uncertainty": "conditional Laplace" if self.censored else "conditional Gaussian"}


def condition_independent_evidence(mean, covariance, indices, values, error_covariance):
    """Condition a Gaussian surface on genuinely independent external evidence.

    values are log thicknesses, with a full error covariance. Never use this to
    reinsert predictions derived from the same observations. Correlated errors
    prevent replicated expert statements from becoming independent data.
    """
    mean, covariance = np.asarray(mean, float), np.asarray(covariance, float)
    indices, values, R = np.asarray(indices, int), np.asarray(values, float), np.asarray(error_covariance, float)
    n, m = len(mean), len(indices)
    if covariance.shape != (n, n) or values.shape != (m,) or R.shape != (m, m):
        raise ValueError("Evidence shapes do not match.")
    if m == 0 or np.any(indices < 0) or np.any(indices >= n):
        raise ValueError("Evidence indices are empty or out of bounds.")
    for mat in (covariance, R):
        if (not np.isfinite(mat).all() or not np.allclose(mat, mat.T)
                or np.linalg.eigvalsh(mat).min() < -1e-8):
            raise ValueError("Covariances must be finite, symmetric and positive semidefinite.")
    if not np.isfinite(mean).all() or not np.isfinite(values).all():
        raise ValueError("Evidence and means must be finite.")
    cross = covariance[:, indices]
    # Pseudoinverse also handles exact duplicates with perfectly correlated error.
    S = covariance[np.ix_(indices, indices)] + R
    residual = values-mean[indices]
    inverse = np.linalg.pinv(S, hermitian=True)
    if not np.allclose(S@inverse@residual, residual, atol=1e-7):
        raise ValueError("Contradictory evidence with singular error covariance.")
    gain = cross@inverse
    posterior = covariance-gain@cross.T
    return mean+gain@residual, (posterior+posterior.T)/2
