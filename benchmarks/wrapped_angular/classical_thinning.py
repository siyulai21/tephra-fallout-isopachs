"""Power-law and continuous multi-segment exponential thinning on sqrt(area)."""

from dataclasses import dataclass

import numpy as np
from scipy.optimize import brentq, minimize_scalar, nnls

from .area_laws import (
    Law,
    exponential_integral,
    fit_law,
    line_at,
    normalized_power,
)


def ordered_ranks(x, thickness):
    x, t = np.asarray(x, float), np.asarray(thickness, float)
    if (
        x.ndim != 1
        or x.shape != t.shape
        or not len(x)
        or np.any(x <= 0)
        or np.any(t <= 0)
        or not np.isfinite(x + t).all()
    ):
        raise ValueError("Need finite positive area ranks and thicknesses")
    u, inverse, count = np.unique(x, return_inverse=True, return_counts=True)
    # Coincident contour areas have one numerically resolved rank.
    return u, np.bincount(inverse, weights=np.log(t)) / count


@dataclass(frozen=True)
class Power:
    reference: float
    log_thickness: float
    exponent: float

    def predict_log(self, x):
        return self.log_thickness - self.exponent * np.log(np.asarray(x) / self.reference)

    def integral(self, lo, hi=np.inf, anchor_cm=None):
        if lo < 0 or hi < lo or (anchor_cm is not None and (lo == 0 or anchor_cm <= 0)):
            raise ValueError("Invalid power-law limits or anchor")
        if hi == lo:
            return 0.0
        p = 2.0 - self.exponent
        if abs(p) <= 32 * np.finfo(float).eps:
            p = 0.0
        if lo == 0:
            if p <= 0 or np.isinf(hi):
                return np.inf
            return 2e-5 * np.exp(self.log_thickness) * self.reference**2 * (hi / self.reference) ** p / p
        thick = np.exp(self.predict_log(lo)) if anchor_cm is None else anchor_cm
        if np.isinf(hi):
            factor = -1.0 / p if p < 0 else np.inf
        else:
            log_ratio = np.log(hi / lo)
            factor = np.expm1(p * log_ratio) / p if p else log_ratio
        return 2e-5 * thick * lo**2 * factor


@dataclass(frozen=True)
class Segments:
    starts: np.ndarray
    log_thickness: np.ndarray
    rates: np.ndarray
    counts: np.ndarray

    def predict_log(self, x):
        x = np.asarray(x, float)
        i = np.searchsorted(self.starts[1:], x, side="right")
        return self.log_thickness[i] - self.rates[i] * (x - self.starts[i])

    def integral(self, lo, hi=np.inf, anchor_cm=None):
        if lo < 0 or hi < lo or (anchor_cm is not None and (lo == 0 or anchor_cm <= 0)):
            raise ValueError("Invalid segmented exponential limits or anchor")
        if hi == lo:
            return 0.0
        shift = 0.0 if anchor_cm is None else np.log(anchor_cm) - self.predict_log(lo)
        cuts = self.starts[1:]
        edges = np.r_[lo, cuts[(cuts > lo) & (cuts < hi)], hi]
        mass = 0.0
        for start, end in zip(edges[:-1], edges[1:], strict=True):
            j = np.searchsorted(cuts, start, side="right")
            mass += exponential_integral(self.predict_log(start) + shift, start, end, self.rates[j])
        return mass


def exponential_segments(x, logt):
    """Return each identifiable count of consecutive exponential regimes.

    The dynamic program minimizes independent log-linear residuals for each
    allocation. Adjacent lines meet in their intervening rank gap; if their
    unconstrained intersection leaves that gap, profile all slopes jointly
    at the clipped junctions to preserve continuity and decreasing thickness.
    """
    n = len(x)
    if n < 2:
        return {}
    span = x[-1] - x[0]
    u = (x - x[0]) / span
    cost = np.full((n + 1, n + 1), np.inf)
    slope = np.zeros_like(cost)
    intercept = np.zeros_like(cost)
    for i in range(n - 1):
        xx = u[i:] - u[i]
        yy = logt[i:] - logt[i]
        m = np.arange(2, n - i + 1)
        sx = np.cumsum(xx)[1:]
        sy = np.cumsum(yy)[1:]
        vx = np.cumsum(xx * xx)[1:] - sx * sx / m
        cov = np.cumsum(xx * yy)[1:] - sx * sy / m
        vy = np.cumsum(yy * yy)[1:] - sy * sy / m
        rate = np.minimum(0.0, np.divide(cov, vx, out=np.zeros_like(cov), where=vx > 0))
        ends = i + m
        slope[i, ends] = rate
        intercept[i, ends] = logt[i] + sy / m - rate * (u[i] + sx / m)
        cost[i, ends] = np.maximum(0.0, vy - 2 * rate * cov + rate * rate * vx)
        cost[i, i + 2] = 0.0 if yy[1] <= 0 else cost[i, i + 2]
    # dp[j] is the minimum residual ending before observation j.
    dp = np.full(n + 1, np.inf)
    dp[0] = 0.0
    predecessors = []
    profiles = {}
    for count in range(1, n // 2 + 1):
        candidates = dp[:, None] + cost
        prev = np.argmin(candidates, axis=0)
        dp = candidates[prev, np.arange(n + 1)]
        predecessors.append(prev)
        end = n
        cuts = [n]
        for predecessor in reversed(predecessors):
            end = predecessor[end]
            cuts.append(end)
        cuts = np.asarray(cuts[::-1])
        m = slope[cuts[:-1], cuts[1:]]
        c = intercept[cuts[:-1], cuts[1:]]
        low, high = u[cuts[1:-1] - 1], u[cuts[1:-1]]
        with np.errstate(divide="ignore", invalid="ignore"):
            cross = (c[1:] - c[:-1]) / (m[:-1] - m[1:])
        cross = np.where(np.isfinite(cross), cross, (low + high) / 2.0)
        knots = np.clip(cross, low, high)
        starts = np.r_[0.0, knots]
        widths = np.r_[np.diff(starts), np.inf]
        travel = np.clip(u[:, None] - starts, 0.0, widths)
        design = travel - travel.mean(axis=0)
        target = logt.mean() - logt
        rates = np.linalg.lstsq(design, target, rcond=None)[0]
        if np.any(rates < 0):
            rates = nnls(design, target)[0]
        initial = np.mean(logt + travel @ rates)
        values = np.r_[initial, initial - np.cumsum(rates[:-1] * np.diff(starts))]
        profiles[count] = Segments(x[0] + span * starts, values, rates / span, np.diff(cuts))
    return profiles


def joined_segments(x, logt):
    """Least-squares polyline with junctions on observed contours, through N−1 segments.

    The end ranks are always retained. Adjacent segments share their junction
    observation. At full resolution this is the published pairwise exponential.
    """
    n = len(x)
    cost = np.full((n, n), np.inf)
    for start in range(n - 1):
        dx = x[start + 1 :] - x[start]
        dy = logt[start + 1 :] - logt[start]
        slopes = dy / dx
        # Column j evaluates the chord from the start to end j at all
        # intervening observations. Both endpoints have exactly zero cost.
        residual = dy[:, None] - dx[:, None] * slopes
        cost[start, start + 1 :] = np.sum(np.triu(residual, k=1) ** 2, axis=0)
    dp = np.full(n, np.inf)
    dp[0] = 0.0
    predecessors = []
    profiles = {}
    for k in range(1, n):
        choices = dp[:, None] + cost
        previous = np.argmin(choices, axis=0)
        dp = choices[previous, np.arange(n)]
        predecessors.append(previous)
        end = n - 1
        selected = [end]
        for step in reversed(predecessors):
            end = step[end]
            selected.append(end)
        nodes = np.array(selected[::-1])
        rates = -np.diff(logt[nodes]) / np.diff(x[nodes])
        profiles[k] = Segments(x[nodes[:-1]], logt[nodes[:-1]], rates, np.diff(nodes) + 1)
    return profiles


def models(x, t):
    x, y = ordered_ranks(x, t)
    if len(x) < 3:
        return {}
    out = {f"exp-{k}": m for k, m in exponential_segments(x, y).items()}
    joined = joined_segments(x, y)
    out.update({f"joined-{k}": m for k, m in joined.items()})
    out["joined-all"] = joined[len(x) - 1]
    ref = np.mean(np.log(x))
    m, base = line_at(np.log(x), y, ref)
    out["power"] = Power(np.exp(ref), base, max(0.0, -m))
    out["weibull"] = fit_law("weibull", x, np.exp(y))
    out["weibull-outward"] = outward_weibull(x, np.exp(y), out["weibull"])
    return out


def outward_weibull(x, t, base=None):
    """Allow shape >2 when the unobserved mode precedes every fitted area rank."""
    x, y = ordered_ranks(x, t)
    if base is None:
        base = fit_law("weibull", x, np.exp(y))
    error = float(np.sum((base.predict_log(x) - y) ** 2))
    logx = np.log(x / x[0])
    span = logx[-1]
    allowance = y[0] - y[-1] + np.sqrt(2 * error)
    if allowance <= 0:
        return base

    # For n>2 the log derivative is n−2−k*(x/x_min)**n, hence k>=n−2.
    # The minimum possible endpoint drop grows monotonically with n.
    # Beyond the root, the two endpoint residuals alone exceed the incumbent SSE.
    def bound(n):
        with np.errstate(over="ignore"):
            return (n - 2.0) * (np.expm1(n * span) / n - span) - allowance

    high = 4.0
    while bound(high) < 0:
        high *= 2.0
    upper = brentq(bound, 2.0, high)

    def trial(n):
        d = normalized_power(logx, n)
        target = y - (n - 2.0) * logx
        dc = d - d.mean()
        rate = max(n - 2.0, -dc @ (target - target.mean()) / (dc @ dc))
        alpha = np.mean(target + rate * d)
        residual = alpha - rate * d - target
        return residual @ residual, Law("weibull", x[0], alpha, rate, n)

    grid = np.geomspace(2.0, upper, 33)
    values = [trial(n) for n in grid]
    fits = [(error, base), *values]
    for i, (value, _) in enumerate(values):
        lo, hi = max(i - 1, 0), min(i + 1, len(grid) - 1)
        if grid[hi] > grid[lo] and value <= min(values[lo][0], values[hi][0]):
            opt = minimize_scalar(
                lambda n: trial(n)[0],
                bounds=(grid[lo], grid[hi]),
                method="bounded",
                options={"xatol": 1e-11},
            )
            fits.append(trial(opt.x))
    return min(fits, key=lambda v: v[0])[1]


def choose_count(x, t, shared=False):
    """Rolling forward validation on only the retained training contour ranks."""
    x, y = ordered_ranks(x, t)
    n = len(x)
    if n < 3:
        return None, {}
    ends = np.unique(np.r_[np.clip(np.ceil(n * np.array([0.55, 0.70, 0.85])).astype(int), 2, n - 1), n])
    max_count = ends[0] - 1 if shared else ends[0] // 2
    losses = {k: [] for k in range(1, max_count + 1)}
    for start, end in zip(ends[:-1], ends[1:], strict=True):
        constructor = joined_segments if shared else exponential_segments
        fitted = constructor(x[:start], y[:start])
        for k, terms in losses.items():
            f = fitted[k]
            forecast = f.predict_log(x[start:end]) - f.predict_log(x[start - 1]) + y[start - 1]
            terms.append(np.mean(abs(forecast - y[start:end])))
    scores = {k: float(np.mean(v)) for k, v in losses.items()}
    best = min(scores.values())
    return min(k for k, v in scores.items() if v <= best + 1e-9), scores
