"""A source-area excess conditioned on a regional field fitted to exterior stations."""

import numpy as np
from scipy.special import i0e, logsumexp, ndtri, roots_legendre
from scipy.stats import qmc


def student_logcdf(z):
    r = np.hypot(2.0, z)
    h = np.abs(z) / r
    small = 2.0 * (np.log(4.0) - 2.0 * np.log(r) - np.log1p(h)) + np.log(2.0 + h) - np.log(4.0)
    return np.where(z < 0.0, small, np.log1p(-np.exp(small)))


def student_events(value, upper, kind, center, sd):
    """Point density, upper limit or two-sided interval under four t degrees."""
    z = (value - center) / sd
    # Supply a real interval in all branches to keep autodiff finite even on
    # rows whose selected likelihood is a density or a single tail.
    endpoint = np.where(kind == 2, upper, value + 1.0)
    width = np.where(kind == 2, upper - value, 1.0) / sd
    hi = (endpoint - center) / sd
    positive = z > 0.0
    a = np.where(positive, -hi, z)
    b = np.where(positive, -z, hi)
    lhi = student_logcdf(b)
    llo = student_logcdf(a)
    interval = lhi + np.log(-np.expm1(np.minimum(llo - lhi, -np.finfo(float).eps)))
    log_density = np.log(0.375) - 5.0 * np.log(np.hypot(1.0, 0.5 * z))
    # The endpoints of a real interval can round to the same standardized
    # value far in a tail. The midpoint density times the original width is
    # accurate to rounding precision when relative width is this small.
    narrow = width < 1.0e-8 * np.maximum(1.0, np.abs(z))
    middle = z + 0.5 * width
    local = np.log(0.375) - 5.0 * np.log(np.hypot(1.0, 0.5 * middle)) + np.log(width)
    interval = np.where(narrow, local, interval)
    point = log_density - np.log(sd)
    return np.where(kind == 0, point, np.where(kind == 1, student_logcdf(z), interval))


def coordinates(xy, source):
    delta = np.asarray(xy) - source["centroid_km"]
    return 0.5 * np.linalg.solve(np.linalg.cholesky(source["covariance_km2"]), delta.T).T


def proposals(power, seed, with_branch=False):
    """Relative amplitude, peak centre, width and between-section scatter."""
    u = qmc.Sobol(6, scramble=True, seed=seed).random_base2(power)
    z = ndtri(np.clip(u, np.finfo(float).eps, 1.0 - np.finfo(float).eps))
    # A spike at zero allows the data to favor regional deposition alone.
    log_amp = np.where(u[:, 0] < 0.5, -np.inf, np.log(10.0) * z[:, 1])
    radial = np.sqrt(u[:, 2])
    angle = 2.0 * np.pi * u[:, 3]
    params = np.column_stack(
        (
            log_amp,
            radial * np.cos(angle),
            radial * np.sin(angle),
            np.exp(0.7 * z[:, 4]),
            np.exp(np.log(0.65) + 0.65 * z[:, 5]),
        )
    )
    # Half the positive branch may use a data-informed proposal. Its density
    # ratio restores the same normal prior on log amplitude.
    return (params, u[:, 0] >= 0.75) if with_branch else params


def amplitude_proposal(params, branch, regional, reference, unit_shape, observations):
    """Defensive importance proposal for the positive source amplitude."""
    useful = (~observations.censored) & (observations.log_value > regional)
    ratio = np.zeros(len(params))
    if not np.any(useful):
        return params, ratio
    y = observations.log_value[useful]
    excess = y + np.log(-np.expm1(regional[useful] - y))
    center = np.median(excess - reference - unit_shape[:, useful], axis=1)
    prior_sd = np.log(10.0)
    scale = np.hypot(0.35, params[:, 4] / np.sqrt(useful.sum()))
    ans = params.copy()
    ans[branch, 0] = center[branch] + scale[branch] * params[branch, 0] / prior_sd
    positive = np.isfinite(ans[:, 0])
    a = ans[positive, 0]
    prior = -0.5 * (a / prior_sd) ** 2 - np.log(prior_sd)
    shifted = -0.5 * ((a - center[positive]) / scale[positive]) ** 2 - np.log(scale[positive])
    ratio[positive] = np.log(2.0) + prior - np.logaddexp(prior, shifted)
    return ans, ratio


def taper(radius, width):
    if not np.isfinite(width) or width <= 0:
        raise ValueError("Source-neighborhood width must be positive")
    u = np.clip(2.0 * radius / width - 1.0, 0.0, 1.0)
    return (1.0 - u) ** 3 * (1.0 + 3.0 * u + 6.0 * u * u)


def log_shape(params, points, width):
    """C2 taper on the local excess. The regional fallout continues everywhere."""
    window = taper(np.linalg.norm(points, axis=1), width)
    with np.errstate(divide="ignore"):
        window = np.log(window)
    dx = points[None, :, 0] - params[:, None, 1]
    dy = points[None, :, 1] - params[:, None, 2]
    return params[:, None, 0] - 0.5 * (dx * dx + dy * dy) / params[:, None, 3] ** 2 + window


def excess_volume(params, reference, source, width, order):
    """Integrate the source excess over the equal-area plane in cubic kilometres."""
    nodes, weights = roots_legendre(order)
    m = np.hypot(params[:, 1], params[:, 2])[:, None]
    variance = params[:, None, 3] ** 2
    area = np.zeros(len(params))
    for low, high in ((0.0, width / 2.0), (width / 2.0, width)):
        r = 0.5 * (low + high + (high - low) * nodes)
        # Analytical angular mean of the off-centre Gaussian. The scaled
        # Bessel function stays finite when the peak is narrow.
        circle = np.exp(-0.5 * (r - m) ** 2 / variance) * i0e(r * m / variance)
        area += circle @ (np.pi * (high - low) * weights * r * taper(r, width))
    jacobian = 4.0 * np.sqrt(np.linalg.det(source["covariance_km2"]))
    return 1.0e-5 * jacobian * area * np.exp(reference + params[:, 0])


def conditional_weights(regional, reference, shape, params, observations, log_prior_over_proposal=0.0):
    """Update only the local parameters, conditional on this regional realization."""
    if not len(observations.log_value):
        return np.full(len(params), -np.log(len(params)))
    mu = np.logaddexp(regional, reference + shape)
    sd = np.hypot(params[:, None, 4], observations.log_sd)
    kind = observations.censored.astype(int) + observations.interval.astype(int)
    ll = (
        student_events(observations.log_value, observations.interval_log_upper, kind, mu, sd).sum(axis=1)
        + log_prior_over_proposal
    )
    return ll - logsumexp(ll)


def predictive(log_weight, regional, reference, shape, params, target):
    center = np.logaddexp(regional, reference + shape)
    sd = np.hypot(params[:, None, 4], target.log_sd)
    kind = target.censored.astype(int) + target.interval.astype(int)
    ll = student_events(target.log_value, target.interval_log_upper, kind, center, sd)
    lcdf = student_logcdf((target.log_value - center) / sd)
    return logsumexp(log_weight[:, None] + ll, axis=0), logsumexp(log_weight[:, None] + lcdf, axis=0)
