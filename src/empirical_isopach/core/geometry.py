"""Common survey geometry; no response values or model preference."""
import numpy as np


def survey_bounds(xy):
    """Known sampling geometry, with a fixed 5% margin; no thickness input."""
    xy = np.asarray(xy, float)
    if xy.ndim != 2 or xy.shape[1] != 2 or not np.isfinite(xy).all():
        raise ValueError("Coordinates must be a finite (n, 2) array.")
    lo, hi = xy.min(axis=0), xy.max(axis=0)
    span = hi - lo
    if np.any(span <= 0) or np.linalg.matrix_rank(xy - xy.mean(axis=0)) < 2:
        raise ValueError("Regional surfaces require non-collinear sampling locations.")
    return np.r_[lo - .05*span, hi + .05*span]
