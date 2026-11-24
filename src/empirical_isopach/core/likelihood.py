"""Gaussian density and probability of documented thickness events in log cm."""
import numpy as np
from scipy.special import log_ndtr


def interval_log_probability(lo, hi, *, xp=np, logcdf=log_ndtr):
    """Log normal mass between standardized endpoints, including in either tail."""
    width = hi - lo
    mid = lo + width / 2
    # For very narrow bins, integrating the density around its midpoint avoids
    # subtracting indistinguishable CDFs. Keep inactive JAX branches finite.
    narrow = width * (1 + xp.abs(mid)) < 1e-4
    left = xp.where(narrow, -1., xp.where(lo > 0, -hi, lo))
    right = xp.where(narrow, 0., xp.where(lo > 0, -lo, hi))
    log_right = logcdf(right)
    mass = log_right + xp.log(-xp.expm1(logcdf(left) - log_right))
    w = xp.where(narrow, width, 1.)
    m = xp.where(narrow, mid, 0.)
    small = -.5 * (m*m + np.log(2*np.pi)) + xp.log(w) + xp.log1p((m*m-1)*w*w/24)
    return xp.where(narrow, small, mass)


def normal_log_event(y, mu, sd, censored, lower, upper=None, *, xp=np, logcdf=log_ndtr):
    """Score one observed event per site, broadcasting over prediction draws."""
    z = (y - mu) / sd
    exact = -.5 * (z*z + np.log(2*np.pi)) - xp.log(sd)
    if xp is np and np.ndim(z):
        # For particle and mixture matrices evaluate CDFs only at the sites
        # which need them. In particular, never expand a rare interval into
        # several scratch arrays covering every draw and every measured site.
        n = z.shape[-1]
        bounds = np.broadcast_to(censored, (n,))
        lower = np.broadcast_to(lower, (n,))
        high = np.broadcast_to(np.nan if upper is None else upper, (n,))
        interval = np.isfinite(high)
        upper_only, lower_only = bounds & ~lower, lower & ~interval
        exact[..., upper_only] = logcdf(z[..., upper_only])
        exact[..., lower_only] = logcdf(-z[..., lower_only])
        if interval.any():
            mu_i = np.broadcast_to(mu, z.shape)[..., interval]
            sd_i = np.broadcast_to(sd, z.shape)[..., interval]
            hi = (high[interval]-mu_i)/sd_i
            exact[..., interval] = interval_log_probability(z[..., interval], hi)
        return exact
    scores = xp.where(censored, logcdf(xp.where(lower, -z, z)), exact)
    if upper is None:
        return scores
    interval = xp.isfinite(upper)
    if xp is np and not np.any(interval):
        return scores
    lo = xp.where(interval, z, 0.)
    hi = (xp.where(interval, upper, mu) - mu) / sd
    hi = xp.where(interval, hi, 1.)
    return xp.where(interval, interval_log_probability(lo, hi, xp=xp, logcdf=logcdf), scores)


def observation_log_event(data, mu, sd):
    return normal_log_event(data.log_value, mu, sd, data.censored, data.lower_censored,
                            data.interval_log_upper)


def censored_terms(f, log_limit, sd, lower=False, interval_log_upper=None):
    """Negative log event probability and its first two derivatives in latent f."""
    f, log_limit, sd = np.broadcast_arrays(f, log_limit, sd)
    sign = np.where(lower, -1., 1.)
    z = sign * (log_limit - f) / sd
    lp = log_ndtr(z)
    mills = np.exp(-0.5*z*z - 0.5*np.log(2*np.pi) - lp)
    value, grad = -lp, sign*mills/sd
    curvature = np.maximum(mills*(mills+z)/(sd*sd), 0.)
    if interval_log_upper is None:
        return value, grad, curvature
    interval_log_upper = np.broadcast_to(interval_log_upper, f.shape)
    mask = np.isfinite(interval_log_upper)
    if np.any(mask):
        value, grad, curvature = map(np.asarray, (value, grad, curvature))
        s = sd[mask]
        lo, hi = (log_limit[mask]-f[mask])/s, (interval_log_upper[mask]-f[mask])/s
        mass = interval_log_probability(lo, hi)
        width = hi-lo
        mid = lo+width/2
        narrow = width*(1+abs(mid)) < 1e-4
        # For a standard normal truncated to [lo, hi], derivatives are
        # -E[Z]/s and (1-Var[Z])/s². Use their local expansions in a narrow bin.
        a = np.exp(-.5*(lo*lo+np.log(2*np.pi))-mass)
        b = np.exp(-.5*(hi*hi+np.log(2*np.pi))-mass)
        mean = np.where(narrow, mid*(1.-width**2/12), a-b)
        weight = np.where(narrow, 1.-width**2/12, mean**2 + hi*b - lo*a)
        value[mask], grad[mask], curvature[mask] = -mass, -mean/s, np.clip(weight, 0., 1.)/s**2
    return value, grad, curvature
