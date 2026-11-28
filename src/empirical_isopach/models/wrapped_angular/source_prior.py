"""Marginalize the source-only prior when the mapped source contains no observations."""

import numpy as np
from scipy.special import ndtr, ndtri
from scipy.stats import qmc


def sample_prior(weights, power, seed):
    u = qmc.Sobol(5, scramble=True, seed=seed).random_base2(power)
    cumulative = np.cumsum(weights)
    cumulative[-1] = 1.0
    return (
        np.searchsorted(cumulative, u[:, 0]),
        ndtri(u[:, 1]),
        np.sqrt(u[:, 2]),
        np.cos(2.0 * np.pi * u[:, 3]),
        np.exp(0.7 * ndtri(u[:, 4])),
    )


def prior_quantile(regional, center, cross_covariance, radius, window, draws, probability):
    """Integrate lognormal regional thickness plus the spike-and-slab local excess.

    Within each mixture component, X is regional log thickness at the location
    and V is regional log thickness at the centroid plus the random log amplitude.
    Conditional on D = X - V, the log sum is normal with a known mean shift.
    The radial prior permits measuring the random centre angle relative to the
    evaluation bearing. This integrates the same circular prior at every point.
    """
    ix, normal, offset, cos_angle, width = draws
    mu_x, var_x = regional.means[ix], regional.variances[ix]
    mu_v = center.means[ix]
    var_v = center.variances[ix] + np.log(10.0) ** 2
    cov_xd = var_x - cross_covariance[ix]
    sd_d = np.sqrt(var_x + var_v - 2.0 * cross_covariance[ix])
    d = mu_x - mu_v + normal[:, None] * sd_d
    conditional_mean = mu_x + (cov_xd / sd_d) * normal[:, None]
    conditional_sd = np.sqrt(np.maximum(var_x - (cov_xd / sd_d) ** 2, 1.0e-15))
    distance2 = radius**2 + offset[:, None] ** 2 - 2.0 * (offset * cos_angle)[:, None] * radius
    with np.errstate(divide="ignore"):
        shape = np.log(window) - distance2 / (2.0 * width[:, None] ** 2)
    increment = np.logaddexp(0.0, shape - d)

    def cdf(y):
        # The zero branch has probability 1/2. The exact marginal CDF is
        # also a control variate, restoring the exact field at the boundary.
        z = (y - conditional_mean) / conditional_sd
        change = ndtr(z) - ndtr(z - increment / conditional_sd)
        return regional.cdf(y) - 0.5 * change.mean(axis=0)

    lo = regional.quantile(probability)
    hi = lo + 1.0
    for _ in range(16):
        short = cdf(hi) < probability
        if not short.any():
            break
        hi[short] += hi[short] - lo[short]
    else:
        raise ArithmeticError("Source quantile is unbracketed")
    for _ in range(25):
        mid = (lo + hi) / 2.0
        lower = cdf(mid) < probability
        lo, hi = np.where(lower, mid, lo), np.where(lower, hi, mid)
    answer = (lo + hi) / 2.0
    if np.max(np.abs(cdf(answer) - probability)) > 2.0e-6:
        raise ArithmeticError("Source quantile inversion did not converge")
    return answer
