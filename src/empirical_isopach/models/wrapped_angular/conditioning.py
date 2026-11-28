"""Condition on exact beds first, then integrate correlated bounded observations."""

import numpy as np
from scipy.linalg import solve_triangular
from scipy.special import log_ndtr, ndtri, ndtri_exp, logsumexp
from scipy.stats import qmc
from ...core.likelihood import normal_log_event, censored_terms
from ..directional.multiscale.conditioning import condition


def single_event(design, offset, data, noise):
    loading = design.item()
    error = np.hypot(data.log_sd, noise)
    target = data.log_value - offset
    hi = data.interval_log_upper - offset
    loss, gradient, curvature = censored_terms(
        np.zeros(1), target, np.hypot(loading, error), data.lower_censored, hi
    )
    mean = -loading * gradient
    variance = np.maximum(1.0e-12, 1.0 - loading**2 * curvature)
    return dict(
        X=design,
        mean=mean,
        cov=variance.reshape(1, 1),
        sd=error,
        target=target,
        hi=hi,
        lower=data.lower_censored,
        tau=(1.0 / variance - 1.0) / loading**2,
        eta=mean / (loading * variance),
        cs=-loss - 0.5 * np.log(variance) - 0.5 * mean**2 / variance,
        logz=-loss.item(),
        ep_residual=0.0,
    )


class BedConditioner:
    def __init__(self, data):
        self.data = data
        self.events = data.subset(data.censored)
        self.cache = {}

    def evaluate(self, gid, gram, offset, noise):
        data = self.data
        exact, bound = ~data.censored, data.censored
        if gid not in self.cache:
            kernel = gram + np.eye(len(gram)) * 1.0e-9
            cov = kernel[np.ix_(exact, exact)] + np.diag(noise**2 + data.log_sd[exact] ** 2)
            chol = np.linalg.cholesky(cov)
            projection = solve_triangular(chol, kernel[np.ix_(exact, bound)], lower=True)
            remaining = kernel[np.ix_(bound, bound)] - projection.T @ projection
            small = np.linalg.cholesky(0.5 * (remaining + remaining.T) + np.eye(bound.sum()) * 1.0e-11)
            self.cache[gid] = chol, projection, small
        chol, projection, small = self.cache[gid]
        residual = solve_triangular(chol, data.log_value[exact] - offset[exact], lower=True)
        logz = (
            -0.5 * residual @ residual - np.log(chol.diagonal()).sum() - 0.5 * exact.sum() * np.log(2 * np.pi)
        )
        if not bound.any():
            return dict(logz=float(logz), ep_residual=0.0)
        conditional_mean = offset[bound] + residual @ projection
        solve = single_event if len(small) == 1 else condition
        state = solve(small, conditional_mean, self.events, noise)
        return dict(
            logz=float(logz + state["logz"]),
            ep_residual=state["ep_residual"],
            bound_state=state,
            conditional_mean=conditional_mean,
        )


def bounded_normal(a, b, u):
    """Inverse truncated normal using log survival probabilities in the positive tail."""
    right = a > 0
    lo, hi = np.where(right, -b, a), np.where(right, -a, b)
    q = np.logaddexp(np.log1p(-u) + log_ndtr(lo), np.log(u) + log_ndtr(hi))
    return np.where(right, -1.0, 1.0) * ndtri_exp(q)


def correction(model, power, seed):
    """Importance-correct the proposal and sample the unobserved thickness within each bound."""
    bound = model.data.censored
    count = int(bound.sum())
    if not count:
        parts = [(g, j, model.data.log_value) for g, j, _ in model.parts]
        return (
            parts,
            model.weights,
            dict(
                seed=seed,
                power=0,
                relative_ess=1.0,
                log_evidence_correction=0.0,
                largest_weight=float(model.weights.max()),
            ),
        )
    u = np.clip(
        qmc.Sobol(1 + 2 * count, scramble=True, seed=seed).random_base2(power), 1.0e-14, 1.0 - 1.0e-14
    )
    ids = np.minimum(np.searchsorted(model.weights.cumsum(), u[:, 0]), len(model.weights) - 1)
    normal = ndtri(u[:, 1 : count + 1])
    parts = [None] * len(u)
    log_ratio = np.empty(len(u))
    for component in np.unique(ids):
        indices = np.flatnonzero(ids == component)
        g, j, state = model.parts[component]
        proposal = state["bound_state"]
        X, sd, y, lower, hi = (proposal[k] for k in ("X", "sd", "target", "lower", "hi"))
        beta = proposal["mean"] + normal[indices] @ np.linalg.cholesky(proposal["cov"]).T
        f = beta @ X.T
        ll = normal_log_event(y, f, sd, np.ones(count, bool), lower, hi)
        log_ratio[indices] = (ll + 0.5 * f * f * proposal["tau"] - f * proposal["eta"] - proposal["cs"]).sum(
            axis=1
        )
        two = np.isfinite(hi)
        lo, up = np.where(lower | two, y, -np.inf), np.where(two, hi, np.where(lower, np.inf, y))
        completed = f + sd * bounded_normal((lo - f) / sd, (up - f) / sd, u[indices, count + 1 :])
        for k, row in zip(indices, completed, strict=True):
            y_full = model.data.log_value.copy()
            y_full[bound] = state["conditional_mean"] + row
            parts[k] = g, j, y_full
    norm = logsumexp(log_ratio)
    weights = np.exp(log_ratio - norm)
    return (
        parts,
        weights,
        dict(
            seed=seed,
            power=power,
            relative_ess=float(1 / (len(u) * (weights @ weights))),
            log_evidence_correction=float(norm - np.log(len(u))),
            largest_weight=float(weights.max()),
        ),
    )
