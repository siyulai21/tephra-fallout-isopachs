"""Conventional thinning fits on x = sqrt(superlevel area in km²), T in cm."""

from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares, minimize_scalar


FAMILIES = ("exponential", "segmented", "weibull")
SCREENS = {
    "hull99": (0, 0.99),
    "local99": (1, 0.99),
    "hull90": (0, 0.90),
    "nearby90": (3, 0.90),
    "radial90": (4, 0.90),
}


def fitting_points(data):
    """Point fits include lower-only values at their registered numeric minima."""
    return ~data.censored


def supported_band(loglevels, support, brackets, screen):
    k, limit = SCREENS[screen]
    ok = (support[2] > 0) & (support[k] >= limit * support[2]) & (brackets.min(axis=1) >= 3)
    runs = np.split(np.flatnonzero(ok), np.flatnonzero(np.diff(np.flatnonzero(ok)) > 1) + 1)
    valid = [v for v in runs if len(v) >= 3 and loglevels[v[-1]] - loglevels[v[0]] >= np.log(2.0)]
    return max(
        valid,
        key=lambda v: (loglevels[v[-1]] - loglevels[v[0]], -v[0]),
        default=np.array([], int),
    )


def volume_curve(z, weights, loglevels):
    # Sort by descending thickness. A disconnected fan counts towards the same
    # total area, and a hole below the level contributes neither area nor mass.
    order = np.argsort(-z)
    descending, w = z[order], weights[order]
    stops = np.searchsorted(-descending, -loglevels, side="right")
    area = np.r_[0.0, np.cumsum(w)][stops]
    volume = np.r_[0.0, np.cumsum(np.exp(descending) * w * 1.0e-5)]
    return area, volume[stops], volume[-1]


def normalized_power(log_u, n):
    """(u**n - 1)/n, including its logarithmic limit."""
    return np.expm1(n * log_u) / n if n else log_u


@dataclass(frozen=True)
class Law:
    family: str
    reference: float
    log_thickness: float
    rate: float
    shape: float = np.nan
    breakpoint: float = 0.0
    inner_rate: float = 0.0

    def predict_log(self, x):
        x = np.asarray(x, dtype=float)
        if self.family == "weibull":
            u = np.log(x / self.reference)
            return self.log_thickness + (self.shape - 2) * u - self.rate * normalized_power(u, self.shape)
        path = self.rate * (x - self.reference)
        if self.family == "segmented":
            path += (self.inner_rate - self.rate) * (
                np.minimum(x, self.breakpoint) - min(self.reference, self.breakpoint)
            )
        return self.log_thickness - path

    def integral(self, x, stop=np.inf, anchor_cm=None):
        """Integral of 2 x T(x) dx, optionally rescaled to hit (x, anchor_cm)."""
        if x < 0 or stop < x or (anchor_cm is not None and (anchor_cm <= 0 or x == 0)):
            raise ValueError("Invalid thinning integration limits or anchor")
        if x == stop:
            return 0.0
        if self.family == "weibull":
            n = self.shape
            if x == 0:
                if n == 0:
                    return np.inf
                # Mass inside the reference contour, added to or subtracted from
                # its continuation. Avoids extrapolating a singular T(0).
                ref = self.reference
                k = self.rate
                if k == 0:
                    return 2e-5 * np.exp(self.log_thickness) * ref**2 * (stop / ref) ** n / n
                with np.errstate(over="ignore"):
                    return (
                        2e-5
                        * np.exp(self.log_thickness + k / n)
                        * ref**2
                        / k
                        * -np.expm1(-(k / n) * (stop / ref) ** n)
                    )
            log_t = self.predict_log(x) if anchor_cm is None else np.log(anchor_cm)
            k = self.rate * (x / self.reference) ** n
            if np.isinf(stop):
                factor = 1 / k if k > 0 else np.inf
            else:
                spread = normalized_power(np.log(stop / x), n)
                factor = -np.expm1(-k * spread) / k if k > 0 else spread
            return 2e-5 * np.exp(log_t) * x * x * factor
        log_t = self.predict_log(x) if anchor_cm is None else np.log(anchor_cm)
        if x < self.breakpoint:
            end = min(stop, self.breakpoint)
            inner = exponential_integral(log_t, x, end, self.inner_rate)
            if end == stop:
                return inner
            log_t -= (end - x) * self.inner_rate
            x = end
        return exponential_integral(log_t, x, stop, self.rate)


def exponential_integral(log_thickness, start, end, rate):
    if rate == 0:
        return 1e-5 * np.exp(log_thickness) * (end**2 - start**2)
    if rate < 0 and np.isinf(end):
        return np.inf
    d = (end - start) * rate
    if np.isinf(d):
        a = b = 1.0
    else:
        a = -np.expm1(-d)
        b = d * d * (0.5 - d / 3 + d * d / 8 - d**3 / 30) if abs(d) < 1e-3 else a - d * np.exp(-d)
    return 2e-5 * np.exp(log_thickness) * (start * a / rate + b / rate**2)


def line_at(x, y, reference):
    centered = x - x.mean()
    slope = centered @ (y - y.mean()) / (centered @ centered)
    return slope, y.mean() + slope * (reference - x.mean())


def fit_law(family, x, t, relative=False):
    """Minimize log residuals, or squared relative residuals for the Weibull sensitivity."""
    x, t = np.asarray(x, float), np.asarray(t, float)
    if (
        x.shape != t.shape
        or x.ndim != 1
        or not len(x)
        or np.any(~np.isfinite(x + t))
        or min(x.min(), t.min()) <= 0
    ):
        raise ValueError("Thinning fits require finite positive areas and thicknesses")
    order = np.argsort(x)
    x, y = x[order], np.log(t[order])
    if len(np.unique(x)) < (6 if family == "segmented" else 3):
        raise ValueError("Too few distinct area ranks for the requested thinning law")
    ref = np.exp(np.mean(np.log(x)))
    if family == "exponential":
        slope, intercept = line_at(x, y, ref)
        if slope >= 0:
            raise ValueError("Exponential thickness does not decrease")
        return Law(family, ref, intercept, -slope)
    if family == "segmented":
        fits = []
        scale = np.ptp(x)
        for cut in range(3, len(x) - 2):
            if x[cut - 1] == x[cut] or len(np.unique(x[:cut])) < 3 or len(np.unique(x[cut:])) < 3:
                continue
            origin = x[cut - 1]
            left, right = (
                line_at(x[:cut], y[:cut], origin),
                line_at(x[cut:], y[cut:], origin),
            )
            cross = origin + (right[1] - left[1]) / (left[0] - right[0]) if left[0] != right[0] else origin
            b = np.clip(cross, x[cut - 1], x[cut])
            distance = (x - b) / scale
            A = np.column_stack((np.ones(len(x)), np.minimum(distance, 0), np.maximum(distance, 0)))
            beta = np.linalg.lstsq(A, y, rcond=None)[0]
            if max(beta[1:]) < 0:
                fits.append(
                    (
                        np.sum((A @ beta - y) ** 2),
                        Law(
                            family,
                            b,
                            beta[0],
                            -beta[2] / scale,
                            breakpoint=b,
                            inner_rate=-beta[1] / scale,
                        ),
                    )
                )
        if not fits:
            raise ValueError("No continuous decreasing two-segment fit")
        return min(fits, key=lambda a: a[0])[1]
    if family != "weibull":
        raise ValueError(f"Unknown thickness law {family}")
    u = np.log(x / ref)

    def profile(n):
        d = normalized_power(u, n)
        target = y - (n - 2) * u
        dc = d - d.mean()
        k = max(0.0, -dc @ (target - target.mean()) / (dc @ dc))
        a = target.mean() + k * d.mean()
        return np.sum((a - k * d - target) ** 2), a, k, n

    grid = np.linspace(0, 2, 17)
    trials = [profile(n) for n in grid]
    for i in range(len(grid)):
        lo, hi = max(0, i - 1), min(len(grid) - 1, i + 1)
        if trials[i][0] <= min(trials[lo][0], trials[hi][0]):
            opt = minimize_scalar(
                lambda n: profile(n)[0],
                bounds=tuple(grid[[lo, hi]]),
                method="bounded",
                options={"xatol": 1e-11},
            )
            trials.append(profile(opt.x))
    _, a, k, n = min(trials, key=lambda s: s[0])
    if relative:

        def residual(p):
            al, rate, shape = p
            return np.expm1(np.minimum(al + (shape - 2) * u - rate * normalized_power(u, shape) - y, 50))

        candidates = [(residual([a, k, n]) @ residual([a, k, n]), (a, k, n))]
        for _, initial_a, initial_k, initial_n in [profile(m) for m in (n, 0.3, 1, 1.7)]:
            result = least_squares(
                residual,
                (initial_a, initial_k, initial_n),
                bounds=([-np.inf, 0, 0], [np.inf, np.inf, 2]),
                max_nfev=400,
                ftol=1e-11,
                xtol=1e-11,
                gtol=1e-11,
            )
            if result.success:
                candidates.append((result.fun @ result.fun, result.x))
        _, (a, k, n) = min(candidates, key=lambda s: s[0])
        if k < 1e-10:
            k = 0.0
        if n < 1e-10:
            n = 0.0
    return Law(family, ref, a, k, n)
