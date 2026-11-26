"""Smooth source-relative thinning, uncertain fan headings, and local spatial departures."""

from dataclasses import dataclass
import numpy as np
from scipy.linalg import block_diag, solve_triangular
from scipy.spatial.distance import cdist

from ..geometry import polar
from ...mass_mixture.geometry import RADIUS

SCHEMA = "directional-multiscale-v6"
LEGACY_SCHEMA = "directional-multiscale-v5"
RANGES = (8.0, 32.0, 128.0, 512.0)
TAILS = (2.75, 4.5, 7.0)
NOISE = (0.26, 0.48, 0.8, 1.2)
LENGTHS = (0.20, 0.50, 1.1)
SPREAD = (0.0,)
WARP_KM = 35.0
LOCAL_SD = 1.15
LOCAL_KM = WARP_KM / 3.0


def chart(xy, axis):
    return np.asarray(xy, dtype=float) * [1.0 / axis, axis]


def matern(a, b, length):
    """Original v5 covariance, retained only for exact historical field replay."""
    ra = np.hypot(WARP_KM, np.linalg.norm(a, axis=1))[:, None]
    rb = np.hypot(WARP_KM, np.linalg.norm(b, axis=1))[None, :]
    square = (ra * ra + rb * rb) / 2
    d = cdist(a, b) * np.sqrt(5.0 / square) / length
    # Two physical correlation lengths let adjacent groups of stations share a
    # departure while allowing finer differences within each surveyed group.
    short = 3.0 * d
    return (ra * rb / square) * (
        0.80 * (1.0 + d + d * d / 3.0) * np.exp(-d)
        + 0.20 * (1.0 + short + short * short / 3.0) * np.exp(-short)
    )


def separated_covariance(centers, points, length):
    """Independent broad dispersal and fixed physical-scale local processes.

    The two blocks have separate coefficient priors. Broad fluctuations grow
    with dispersal distance; local exposure-scale variation does not spread
    across hundreds of kilometres merely because a station is distal.
    """
    n = len(centers) // 2
    ra = np.hypot(WARP_KM, np.linalg.norm(centers[:n], axis=1))[:, None]
    rb = np.hypot(WARP_KM, np.linalg.norm(points, axis=1))[None, :]
    square = (ra * ra + rb * rb) / 2
    broad = cdist(centers[:n], points) * np.sqrt(5.0 / square) / length
    fine = cdist(centers[n:], points) * np.sqrt(5.0) / (LOCAL_KM * length)
    return np.vstack(
        (
            0.8 * ra * rb / square * (1 + broad + broad * broad / 3) * np.exp(-broad),
            0.2 * (1 + fine + fine * fine / 3) * np.exp(-fine),
        )
    )


def independent_modes(covariance, coords, cap):
    """Deterministic, rotation-equivariant pivoting on remaining variance."""
    n = len(coords)
    remaining = np.diag(covariance).copy()
    features = np.zeros((n, min(n, cap)))
    chosen = []
    for j in range(features.shape[1]):
        idx = int(np.argmax(remaining))
        if j == 0 and np.ptp(remaining) < 1e-10:
            idx = int(np.argmin(np.sum((coords - coords.mean(axis=0)) ** 2, axis=1)))
        if remaining[idx] < 1e-5:
            break
        chosen.append(idx)
        residual = covariance[:, idx] - features[:, :j] @ features[idx, :j]
        features[:, j] = residual / np.sqrt(remaining[idx])
        remaining = np.maximum(0.0, remaining - features[:, j] ** 2)
        remaining[chosen] = 0.0
    return np.array(chosen, dtype=int)


@dataclass
class LocalBasis:
    centers: np.ndarray
    length: float
    averaging: np.ndarray
    kernel: str = "separated-matern-v6"

    @classmethod
    def fit(cls, xy, axis, length, shared_fraction=0.5):
        if not 0 <= shared_fraction <= 1:
            raise ValueError("Expected a coherent fraction between zero and one")
        points = np.unique(chart(xy, axis), axis=0)
        n = len(points)
        centers = np.vstack((points, points))
        covariance = separated_covariance(centers, points, length)
        broad, fine = covariance[:n], covariance[n:]
        # There is no station-count cap. The remaining variance controls rank,
        # so a densely surveyed eruption cannot exhaust all contrast capacity.
        ids = independent_modes(broad, points, n)
        factor = np.linalg.cholesky(broad[np.ix_(ids, ids)] + 1e-10 * np.eye(len(ids)))
        broad_projector = solve_triangular(factor, np.eye(n)[ids], lower=True, check_finite=False)
        groups = np.argsort(-fine, axis=1, kind="stable")[:, :3]
        eligible = (fine[np.arange(n), groups[:, -1]] >= 0.20 * 0.2) & (groups.shape[1] == 3)
        groups = np.unique(np.sort(groups[eligible], axis=1), axis=0)
        averaging = np.zeros((len(groups), n))
        averaging[np.arange(len(groups))[:, None], groups] = 1.0 / 3.0
        averaged = averaging @ fine @ averaging.T
        chosen = independent_modes(averaged, averaging @ points, n // 3)
        a = averaging[chosen]
        at = averaged[np.ix_(chosen, chosen)]
        factor = np.linalg.cholesky(at + 1e-10 * np.eye(len(a)))
        regional = (
            solve_triangular(factor, a, lower=True, check_finite=False) if len(a) else np.empty((0, n))
        )
        # The finer conditional process is orthogonal to the regional averages.
        # Split its variance between the coherent deposit and site heterogeneity.
        z = fine @ regional.T
        remainder = fine - z @ z.T
        ids = independent_modes(remainder, points, n) if shared_fraction else np.array([], int)
        if len(ids):
            precision = np.linalg.cholesky(remainder[np.ix_(ids, ids)] + 1e-10 * np.eye(len(ids)))
            contrast = np.eye(n)[ids] - z[ids] @ regional
            residual = solve_triangular(precision, contrast, lower=True, check_finite=False)
            projector = np.vstack((regional, np.sqrt(shared_fraction) * residual))
        else:
            projector = regional
        return cls(centers, length, block_diag(broad_projector, projector))

    def cross_covariance(self, xy, axis):
        if self.kernel == "matern-v5":
            covariance = matern
        elif self.kernel == "separated-matern-v6":
            covariance = separated_covariance
        else:
            raise ValueError("Unrecognized directional covariance")
        return covariance(self.centers, chart(xy, axis), self.length)

    def features(self, xy, axis):
        return (self.averaging @ self.cross_covariance(xy, axis)).T

    def unresolved(self, xy, axis):
        columns = self.features(xy, axis)
        return np.maximum(0.0, 1.0 - np.sum(columns * columns, axis=1))


def trend(xy, axis, scale, complexity, *, legacy=False):
    r, theta = polar(xy, axis)
    x = r / scale
    # The finite core has a differentiable summit. At large range thinning is a power law.
    distance = np.log1p(x * x / (np.hypot(x, 0.12) + 0.12))
    h = x / np.hypot(1.0, x)
    rim = np.sqrt(np.maximum(0.0, 1.0 - (r / (2.0 * RADIUS)) ** 2))
    turn = x * x / (16.0 + x * x)
    columns = [np.full(len(x), 2.8), 1.2 * h * h, 1.1 * h * h * (1.0 - turn)]
    amplitudes = (2.7, 1.25, 0.60) if complexity else (2.7, 0.65)
    for k, a in enumerate(amplitudes, 1):
        # r**(k+2) times a harmonic is smooth in Cartesian coordinates at the
        # source. The old r**(k+1) gate had direction-dependent curvature there.
        gate = (h * rim) ** k * (h if legacy else h * h)
        for func in (np.cos, np.sin):
            angular = a * gate * func(k * theta)
            columns.append(angular)
            if complexity:
                columns.append(0.85 * angular * turn)
    return np.column_stack(columns), distance, rim


def design(xy, axis, scale, complexity, local):
    columns, distance, rim = trend(xy, axis, scale, complexity)
    local_part = local.features(xy, axis) * rim[:, None] ** 2
    return np.column_stack((columns, LOCAL_SD * local_part)), distance
