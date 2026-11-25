"""Gaussian evidence and expectation propagation proposals for documented bounded events."""

import numpy as np
from scipy.linalg import cho_solve
from scipy.special import log_ndtr, ndtri, ndtri_exp, logsumexp
from scipy.stats import qmc

from ....core.likelihood import censored_terms, normal_log_event


def gaussian(precision, linear):
    chol = np.linalg.cholesky(precision)
    cov = cho_solve((chol, True), np.eye(len(chol)), check_finite=False)
    return cov @ linear, cov, float(np.log(np.diag(chol)).sum())


def condition(B, offset, data, noise):
    measurable = ~data.censored
    error = np.hypot(data.log_sd, noise)
    y = data.log_value - offset
    A = B[measurable] / error[measurable, None]
    z = y[measurable] / error[measurable]
    P = np.eye(B.shape[1]) + A.T @ A
    h = A.T @ z
    constant = -0.5 * (len(z) * np.log(2 * np.pi) + z @ z) - np.log(error[measurable]).sum()
    X = B[~measurable]
    sd = error[~measurable]
    target = y[~measurable]
    lower = data.lower_censored[~measurable]
    hi = (data.interval_log_upper - offset)[~measurable]
    tau, eta, cs = np.zeros((3, len(X)))
    residual = 0.0
    for iteration in range(64):
        precision = P + (X.T * tau) @ X
        linear = h + X.T @ eta
        mean, cov, logdet = gaussian(precision, linear)
        if not len(X):
            break
        mu = X @ mean
        variance = np.maximum(1e-10, np.sum((X @ cov) * X, axis=1))
        cavity = 1.0 / np.maximum(1e-10, 1.0 / variance - tau)
        center = cavity * (mu / variance - eta)
        loss, g, curv = censored_terms(center, target, np.sqrt(cavity + sd**2), lower, hi)
        denom = np.maximum(1e-10, 1 - cavity * curv)
        proposed = curv / denom
        nat = (center * curv - g) / denom
        c = (
            -loss
            - 0.5 * np.log(denom)
            - 0.5 * ((center - cavity * g) ** 2 / (cavity * denom) - center**2 / cavity)
        )
        residual = max(
            np.max(abs(proposed - tau) / (1 + abs(proposed))),
            np.max(abs(nat - eta) / (1 + abs(nat))),
            np.max(abs(c - cs) / (1 + abs(c))),
        )
        if residual < 3e-6 or iteration == 63:
            break
        tau += 0.55 * (proposed - tau)
        eta += 0.55 * (nat - eta)
        cs += 0.55 * (c - cs)
    logz = constant + 0.5 * (linear @ mean) - logdet + cs.sum()
    full_precision = P + (X.T / (sd**2)) @ X
    complete_cov = cho_solve((np.linalg.cholesky(full_precision), True), np.eye(len(P)), check_finite=False)
    return dict(
        mean=mean,
        cov=cov,
        h=h,
        X=X,
        sd=sd,
        target=target,
        lower=lower,
        hi=hi,
        tau=tau,
        eta=eta,
        cs=cs,
        complete_cov=complete_cov,
        logz=float(logz),
        ep_residual=float(residual),
        iterations=iteration + 1,
    )


def bounded_normal(a, b, u):
    """Inverse truncated-normal CDF, using survival probabilities in the right tail."""
    right = a > 0
    lo = np.where(right, -b, a)
    hi = np.where(right, -a, b)
    q = np.logaddexp(np.log1p(-u) + log_ndtr(lo), np.log(u) + log_ndtr(hi))
    return np.where(right, -1.0, 1.0) * ndtri_exp(q)


def corrected(state, power, seed):
    """Importance correction of both evidence and the conditional law; no event is a point value."""
    mean = state["mean"]
    X = state["X"]
    if not len(X):
        return mean[None, :], np.ones(1), 0.0, 1.0
    sd, y, lower, hi = (state[k] for k in ("sd", "target", "lower", "hi"))
    rng = qmc.Sobol(len(mean) + len(X), scramble=True, seed=seed)
    u = np.clip(rng.random_base2(power), 1e-14, 1.0 - 1e-14)
    beta = mean + ndtri(u[:, : len(mean)]) @ np.linalg.cholesky(state["cov"]).T
    f = beta @ X.T
    events = normal_log_event(y, f, sd, np.ones(len(y), bool), lower, hi)
    ratio = (events + 0.5 * f * f * state["tau"] - f * state["eta"] - state["cs"]).sum(axis=1)
    norm = logsumexp(ratio)
    w = np.exp(ratio - norm)
    two = np.isfinite(hi)
    lower_edge = np.where(lower | two, y, -np.inf)
    upper_edge = np.where(two, hi, np.where(lower, np.inf, y))
    completed = f + sd * bounded_normal((lower_edge - f) / sd, (upper_edge - f) / sd, u[:, len(mean) :])
    centers = (state["h"] + (completed / (sd**2)) @ X) @ state["complete_cov"]
    return centers, w, float(norm - np.log(len(u))), float(1.0 / (len(w) * (w @ w)))
