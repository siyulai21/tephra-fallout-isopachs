"""A quadrature-normalized prior for users retaining the cubic spline model.

This is a data-adaptive penalized estimator, not an independent-data likelihood
or a calibrated Bayesian posterior. It supplies point estimates only. Its prior
strength is selected using real held-out measurements, with zero included.
"""
import numpy as np
from scipy.linalg import solveh_banded
from scipy.sparse import csr_matrix
from src.spline_isopach.lsf_bsplines2d import LsfBsplines2d, BSP3
from src.probabilistic_isopach.model import Candidate, SpatialModel
from src.probabilistic_isopach.report import domain_bounds, grid_cells
from src.probabilistic_isopach.validation import spatial_folds, training_mask


def design_matrix(spline, xy):
    mx, my = spline.mx0*2**spline.n2, spline.my0*2**spline.n2
    hx, hy = (spline.xmax-spline.xmin)/mx, (spline.ymax-spline.ymin)/my
    rows, cols, values = [], [], []
    for row, (x, y) in enumerate(xy):
        if not spline.xmin <= x <= spline.xmax or not spline.ymin <= y <= spline.ymax:
            raise ValueError("Prior quadrature must lie inside the spline domain.")
        i = min(int((x-spline.xmin)/hx), mx-1)
        j = min(int((y-spline.ymin)/hy), my-1)
        bx, by = np.zeros(4), np.zeros(4)
        BSP3((x-spline.xmin)/hx-i, hx, 0, bx)
        BSP3((y-spline.ymin)/hy-j, hy, 0, by)
        for dy in range(4):
            for dx in range(4):
                rows.append(row)
                cols.append((j+dy)*(mx+3)+i+dx)
                values.append(bx[3-dx]*by[3-dy])
    return csr_matrix((values, (rows, cols)), shape=(len(xy), (mx+3)*(my+3)))


class RegularizedSpline:
    def __init__(self, strength=1., quadrature_resolution=24, tau=.25, rou=3.2, tail="exponential"):
        if not np.isfinite(strength) or strength < 0:
            raise ValueError("Prior strength must be finite and nonnegative.")
        self.strength, self.resolution = strength, quadrature_resolution
        self.tau, self.rou, self.tail = tau, rou, tail

    def fit(self, data, quadrature=None, quadrature_weights=None, bounds=None):
        data = data.canonical()
        if data.censored.any():
            raise ValueError("The spline estimator supports positive measurements only; use SpatialModel for upper limits.")
        self.trend = SpatialModel(Candidate(self.tail)).fit(data)
        self.data = data
        self.spline = s = LsfBsplines2d(data.xy[:, 0], data.xy[:, 1], np.exp(data.log_value),
                                       np.sqrt(self.trend.noise**2+data.log_sd**2), self.tau, self.rou, {})
        # Domain, knot counts, scales, and smoothing are determined by real data.
        # Quadrature sites never influence them.
        self.bounds = domain_bounds(data.xy) if bounds is None else np.asarray(bounds, float)
        grid_cells(self.bounds, 2)  # validate rectangle
        if np.any(data.xy < self.bounds[:2]) or np.any(data.xy > self.bounds[2:]):
            raise ValueError("The fixed domain must contain all training measurements.")
        s.xmin, s.ymin, s.xmax, s.ymax = self.bounds
        mx, my = s.mx0*2**s.n2, s.my0*2**s.n2
        A, b, width, n = s._assemble_coarse(mx, my, self.rou)
        band = A.reshape((width, n), order="F").copy()
        q = grid_cells(self.bounds, self.resolution)[0] if quadrature is None else np.asarray(quadrature, float)
        if q.ndim != 2 or q.shape[1] != 2 or not len(q) or not np.isfinite(q).all():
            raise ValueError("Quadrature sites must be a nonempty finite (n,2) array.")
        weights = np.full(len(q), 1/len(q)) if quadrature_weights is None else np.asarray(quadrature_weights, float)
        if weights.shape != (len(q),) or not np.isfinite(weights).all() or np.any(weights < 0) or not np.isclose(weights.sum(), 1.):
            raise ValueError("Quadrature weights must be finite, nonnegative and sum to one.")
        B = design_matrix(s, q)
        target = self.trend._features(q)@self.trend.beta
        precision = self.strength*weights/self.trend.amplitude**2
        G = (B.T@B.multiply(precision[:, None])).tocoo()
        lower = G.row >= G.col
        offsets = G.row[lower]-G.col[lower]
        if np.any(offsets >= width):
            raise RuntimeError("Prior Gram matrix exceeds the spline bandwidth.")
        np.add.at(band, (offsets, G.col[lower]), G.data[lower])
        b += np.asarray(B.T@(precision*target)).ravel()
        s.cij = solveh_banded(band, b, lower=True)
        self.prior_total_precision = float(precision.sum())
        return self

    def predict_log(self, xy):
        xy = np.asarray(xy, float)
        if (np.any(xy < self.bounds[:2]) or np.any(xy > self.bounds[2:])):
            raise ValueError("Spline predictions outside its fitted domain are undefined; use SpatialModel for extrapolation.")
        return np.asarray(design_matrix(self.spline, xy)@self.spline.cij)


def select_spline_prior(data, strengths=(0., 1., 10., 100.), n_folds=4, quadrature_resolution=24):
    """Tune total prior strength on real data, with fold-specific refitting.

    A fixed survey rectangle is determined from ALL sampling coordinates, never
    from thicknesses. Trend, scale, knots, and errors use training folds only.
    This is a known-survey-domain validation design, distinct from GP extrapolation.
    """
    data = data.canonical()
    folds = spatial_folds(data.xy, n_folds)
    records = []
    bounds = domain_bounds(data.xy)
    # The fixed geographical domain is design information, not a measured response.
    for strength in strengths:
        errors, failures = [], []
        for fold in range(n_folds):
            train, test = training_mask(data, folds, fold, 0.)
            try:
                fitted = RegularizedSpline(strength, quadrature_resolution).fit(data.subset(train), bounds=bounds)
                errors.extend((fitted.predict_log(data.xy[test])-data.log_value[test])**2)
            except (ValueError, RuntimeError, np.linalg.LinAlgError) as exc:
                failures.append({"fold": fold, "error": str(exc)})
        records.append({"strength": float(strength), "mse_log": None if failures else float(np.mean(errors)),
                        "failures": failures})
    valid = [r for r in records if r["mse_log"] is not None]
    if not valid:
        raise ValueError(f"No valid spline CV candidate: {records[0]['failures'] if records else 'empty strength set'}")
    selected = min(valid, key=lambda r: r["mse_log"])
    return RegularizedSpline(selected["strength"], quadrature_resolution).fit(data), records
