"""Vent-centred transport mean and wrapped angular covariance on an equal-area Earth."""

from functools import lru_cache

import numpy as np
from scipy.optimize import brentq
from scipy.spatial.distance import cdist

SCALES = (3.0, 12.0, 48.0, 192.0)
NOISES = (0.3, 0.55, 0.9, 1.4)
BANDS = 3.0 * 4.0 ** np.arange(7)
REFERENCE_ORDERS = np.arange(1, 4)
SHAPES = [
    (reach, e, a)
    for reach in (30.0, 100.0, 300.0, 1000.0, 3000.0, 10000.0)
    for e in (0.0, 0.6, 0.9, 0.98)
    for a in (np.arange(16) * 2 * np.pi / 16 if e else [0.0])
]
PRIOR = np.array([1 / (16 if e else 1) for reach, e, a in SHAPES])
PRIOR /= PRIOR.sum()
LOGPRIOR = np.log(PRIOR)


def transported_mean(xy, axis, scale):
    p = np.asarray(xy) * [1 / axis, axis]
    r2 = np.sum(p * p, axis=1)
    q = np.sqrt(scale * scale + r2)
    rim = np.sqrt(np.maximum(0, 1 - r2 / (2 * 6371.007180918) ** 2))
    reach, e, a = np.array(SHAPES).T
    alpha = 1 / (reach * (1 - e))
    along = (np.sin(a)[:, None] * p[:, 0] + np.cos(a)[:, None] * p[:, 1]) * rim
    return (
        np.log(100.0)
        + np.log1p(alpha[:, None] * q)
        - np.log1p(alpha[:, None] * scale)
        - 3 * np.log(q / scale)
        - alpha[:, None] * ((q - scale) - e[:, None] * along)
    )


def local_kernel(a, b, scale):
    def sphere(xy):
        r2 = np.sum(xy * xy, axis=1)
        R = 6371.007180918
        rim = np.sqrt(np.maximum(0.0, 1 - r2 / (2 * R) ** 2))
        return np.column_stack((xy * rim[:, None], -r2 / (2 * R)))

    d = cdist(sphere(a), sphere(b)) * np.sqrt(3.0) / scale
    return (1 + d) * np.exp(-d)


@lru_cache(maxsize=4)
def heat_weights(power):
    """Wrapped Gaussian with the reference variance and mean-square angular derivative."""
    orders = np.arange(9)
    original = np.r_[1.0, REFERENCE_ORDERS.astype(float) ** (-2.0 * power)]
    variance = original.sum()
    curvature = original @ np.arange(4.0) ** 2 / variance

    def spectrum(scale):
        return np.r_[1.0, 2.0 * np.exp(-0.5 * (scale * orders[1:]) ** 2)]

    def error(scale):
        v = spectrum(scale)
        return v @ orders**2 / v.sum() - curvature

    scale = brentq(error, 0.2, 4.0)
    v = spectrum(scale)
    return np.sqrt(variance * v / v.sum())


def basis(xy, axis, scale, amplitude, power, covariance="wrapped"):
    """Seven cumulative radial transitions with a periodic angular covariance.

    The taper removes the bearing singularity at both poles of the equal-area disk.
    The reference spectrum is a fixed sensitivity with matching local variance.
    """
    points = np.asarray(xy) * [1.0 / axis, axis]
    radius = np.linalg.norm(points, axis=1)
    square = radius[:, None] ** 2
    step = square / (square + BANDS**2) * (amplitude / np.sqrt(3.0))
    rim = np.sqrt(np.maximum(0.0, 1.0 - (radius / (2.0 * 6371.007180918)) ** 2))
    h = radius / np.hypot(radius, scale) * rim
    theta = np.arctan2(points[:, 0], points[:, 1])
    if covariance == "wrapped":
        amplitudes = heat_weights(float(power))
        reference_variance = 1.0 + sum(h ** (2 * m) / m ** (2 * power) for m in REFERENCE_ORDERS)
        variance = sum(a * a * h ** (2 * m) for m, a in enumerate(amplitudes))
        match = np.sqrt(reference_variance / variance)
    elif covariance == "three-harmonic":
        amplitudes = np.r_[1.0, REFERENCE_ORDERS.astype(float) ** (-power)]
        match = np.ones_like(h)
    else:
        raise ValueError(f"Unknown angular covariance {covariance}")
    columns = [np.full((len(points), 1), 4.0), step * (match * amplitudes[0])[:, None]]
    for m, a in enumerate(amplitudes[1:], 1):
        angular = step * (match * a * h**m)[:, None]
        columns.extend((angular * np.cos(m * theta)[:, None], angular * np.sin(m * theta)[:, None]))
    return np.column_stack(columns)
