"""Compatibility functions for the original spline API.

All solving now uses the tested complete banded system. Low-level fd values are
already transformed; fit() accepts positive thickness and takes its logarithm.
"""
import numpy as np
from .lsf_bsplines2d import LsfBsplines2d, BSP3, SHFTI, MCHLBA_exact


def domain_divisions(xd, yd, fd):
    model = LsfBsplines2d(xd, yd, np.ones_like(fd), np.zeros_like(fd), .5, 1., {})
    model.fd = np.asarray(fd, dtype=float)
    return model.domain_divisions()


def fit_bsplines2d(xd, yd, fd, sd, xmin=0., ymin=0., xmax=1., ymax=1.,
                   mx0=8, my0=8, n2=2, unl=.1, und=.1, rou=1., tau=.5,
                   nit=200, omg=1.2, use_scipy=True, seed=None):
    if xmin >= xmax or ymin >= ymax or mx0 < 1 or my0 < 1 or n2 < 0 or unl <= 0 or und <= 0:
        raise ValueError("Invalid spline domain, grid, or normalization scales.")
    model = LsfBsplines2d(xd, yd, np.ones_like(fd), sd, tau, rou, {})
    model.fd = np.asarray(fd, dtype=float)
    if not np.isfinite(model.fd).all():
        raise ValueError("Transformed values must be finite.")
    model.xmin, model.ymin, model.xmax, model.ymax = xmin, ymin, xmax, ymax
    model.mx0, model.my0, model.n2 = mx0, my0, n2
    model.unl, model.und = unl, und
    model.fit_bsplines2d(nit, omg, use_scipy, seed)
    return model.cij, mx0*2**n2, my0*2**n2


def eval_cubic_bspline_surface(x, y, xmin, ymin, xmax, ymax, mx, my, cij):
    # This compatibility surface returns the transformed value, including valid
    # negative logarithms for sub-centimetre deposits.
    model = object.__new__(LsfBsplines2d)
    model.xmin, model.ymin, model.xmax, model.ymax = xmin, ymin, xmax, ymax
    model.mx0, model.my0, model.n2 = mx, my, 0
    model.cij = cij
    return model.eval_cubic_bspline_surface(x, y)


def eval_surface_grid(mx, my, cij, nx=100, ny=100, xmin=0., ymin=0., xmax=1., ymax=1.):
    xs = np.linspace(xmin, xmax, nx, endpoint=False)
    ys = np.linspace(ymin, ymax, ny, endpoint=False)
    return np.array([[np.exp(eval_cubic_bspline_surface(x, y, xmin, ymin, xmax, ymax, mx, my, cij))
                      for x in xs] for y in ys])


def fit(xd, yd, fd, sd, rou, tau):
    model = LsfBsplines2d(xd, yd, fd, sd, tau, rou, {})
    model.fit()
    return model.cij, model.mx0*2**model.n2, model.my0*2**model.n2
