"""A broad, training-only prior for finite transport beyond positive evidence."""

import numpy as np
from scipy.spatial.distance import cdist

from ..geometry import polar
from ...mass_mixture.geometry import RADIUS
from .geometry import WARP_KM, chart


def footprint(data, axis):
    """Coefficients of the fourth-power cosine kernel; no site supplies a ray of ash."""
    points = np.unique(chart(data.xy[~data.censored | data.lower_censored], axis), axis=0)
    if not len(points):
        # A fold containing only upper bounds has no positive range information.
        return np.array([WARP_KM**4] + [0.0] * 8)
    r = np.linalg.norm(points, axis=1)
    angle = np.arctan2(points[:, 0], points[:, 1])
    width = np.hypot(r, WARP_KM)
    density = np.exp(
        -cdist(points, points, "sqeuclidean") / (0.18**2 * (width[:, None] ** 2 + width[None, :] ** 2))
    ).sum(1)
    weights = 1 / density
    weights /= weights.sum()
    moment = weights * r**4
    # cos(delta/2)^8 = (35 + 56 cos(delta) + 28 cos(2delta) + 8 cos(3delta) + cos(4delta))/128.
    # The floor regularises the zero-radius and exactly upwind cases.
    v = [WARP_KM**4 + (0.12**4 + 35.0 / 128) * moment.sum()]
    for order, c in enumerate((56, 28, 8, 1), 1):
        v.extend([c / 128 * (moment @ np.cos(order * angle)), c / 128 * (moment @ np.sin(order * angle))])
    return np.array(v)


def attenuation(xy, axis, coefficients):
    r, t = polar(xy, axis)
    rim = np.maximum(0.0, 1.0 - (r / (2.0 * RADIUS)) ** 2)
    reach = np.full(len(r), coefficients[0])
    for k in range(1, 5):
        a, b = coefficients[2 * k - 1 : 2 * k + 1]
        reach += rim ** (0.5 * k) * (a * np.cos(k * t) + b * np.sin(k * t))
    return np.log1p(r**4 / np.maximum(reach, WARP_KM**4))
