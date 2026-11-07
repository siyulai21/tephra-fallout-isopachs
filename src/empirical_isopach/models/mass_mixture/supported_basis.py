"""Response-independent radial bases at measured sites, rebuilt in every fold."""
import numpy as np
from .geometry import to_sphere, RADIUS


def design(data, latitude, count):
    unit = to_sphere(data.xy, latitude)
    # Canonical lexicographic ordering removes CSV-order/tie dependence. Repeated
    # sites do not create repeated knots. Bounds remain legitimate observation sites.
    unit = np.unique(unit, axis=0)
    n = min(count, len(unit))
    first = int(np.argmin(np.sum((unit-unit.mean(axis=0))**2, axis=1)))
    chosen = [first]
    nearest = np.sum((unit-unit[first])**2, axis=1)
    for _ in range(1, n):
        chosen.append(int(np.argmax(nearest)))
        nearest = np.minimum(nearest, np.sum((unit-unit[chosen[-1]])**2, axis=1))
    centers = unit[chosen]
    if n > 1:
        distances = RADIUS*np.linalg.norm(centers[:, None]-centers[None, :], axis=-1)
        np.fill_diagonal(distances, np.inf)
        scales = np.min(distances, axis=1)
    else:
        scales = np.array([np.median(RADIUS*np.linalg.norm(unit-[0., 0., 1.], axis=1))])
    if np.any(scales <= 0):
        raise ValueError("Supported bases need noncoincident observation geometry.")
    return centers, scales
