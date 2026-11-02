import numpy as np
import pytest
from src.spline_isopach.lsf_bsplines2d import LsfBsplines2d, BSP3


def spline():
    rng = np.random.default_rng(1)
    x, y = rng.uniform(-10, 10, (2, 25))
    return LsfBsplines2d(x, y, np.exp(2-.05*x+.02*y), np.ones(25), .25, 3.2, {})


def dense_from_band(flat, width, n):
    band = flat.reshape((width, n), order="F")
    dense = np.zeros((n, n))
    for j in range(n):
        for k in range(min(width, n-j)):
            dense[j+k, j] = dense[j, j+k] = band[k, j]
    return dense


def test_data_gram_includes_every_basis_pair():
    s = spline()
    s.mx0 = s.my0 = 4
    s.und = 1.
    A, b, width, n = s._assemble_coarse(rou=1e30)
    hx, hy = (s.xmax-s.xmin)/4, (s.ymax-s.ymin)/4
    design = np.zeros((len(s.xd), n))
    for row, (x, y) in enumerate(zip(s.xd, s.yd)):
        i = min(int((x-s.xmin)/hx), 3)
        j = min(int((y-s.ymin)/hy), 3)
        bx, by = np.zeros(4), np.zeros(4)
        BSP3((x-s.xmin)/hx-i, hx, 0, bx)
        BSP3((y-s.ymin)/hy-j, hy, 0, by)
        for dy in range(4):
            for dx in range(4):
                design[row, (j+dy)*7+i+dx] = bx[3-dx]*by[3-dy]
    np.testing.assert_allclose(dense_from_band(A, width, n), design.T@design, atol=1e-12)
    np.testing.assert_allclose(b, design.T@s.fd, atol=1e-12)


def test_direct_final_grid_satisfies_normal_equations():
    s = spline()
    s.mx0 = s.my0 = 4
    s.n2 = 1
    s.und = 1.
    s.fit()
    A, b, width, n = s._assemble_coarse(mx=8, my=8, rou=s.rou)
    dense = dense_from_band(A, width, n)
    np.testing.assert_allclose(dense@s.cij, b, atol=1e-10)
    reference = s.cij.copy()
    s.fit_bsplines2d(use_scipy=False)
    np.testing.assert_allclose(s.cij, reference, atol=1e-8)


def test_constant_surface_remains_constant():
    s = spline()
    s.fd[:] = np.log(3.)
    s.und = 1.
    s.fit()
    np.testing.assert_allclose(s.eval_surface_grid(8, 8), 3., atol=1e-7)


def test_functional_api_preserves_negative_log_thickness():
    from src.spline_isopach import Model
    s = spline()
    transformed = np.full(len(s.fd), np.log(.2))
    cij, mx, my = Model.fit_bsplines2d(s.xd, s.yd, transformed, s.sd,
                                     s.xmin, s.ymin, s.xmax, s.ymax, 4, 4, 0, 1., 1.)
    assert (mx, my) == (4, 4)
    value = Model.eval_cubic_bspline_surface(0., 0., s.xmin, s.ymin, s.xmax, s.ymax, mx, my, cij)
    np.testing.assert_allclose(value, np.log(.2), atol=1e-7)


def test_contour_area_subtracts_holes_and_preserves_fractional_levels():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    axis = np.linspace(-4000, 4000, 401)
    X, Y = np.meshgrid(axis, axis)
    r = np.hypot(X, Y)/1000
    fig, ax = plt.subplots()
    cs = ax.contour(X, Y, 1.5-(r-2)**2, levels=[.5])
    result = spline().area_compute(cs)
    plt.close(fig)
    assert result[0][1] == .5
    assert result[0][0]**2 == pytest.approx(8*np.pi, rel=.001)


def test_open_contour_does_not_create_artificial_area():
    from types import SimpleNamespace
    cs = SimpleNamespace(allsegs=[[np.array([[0., 0.], [1000., 500.], [1000., 1000.]])]], levels=[.5])
    with pytest.raises(ValueError, match='boundary'):
        spline().area_compute(cs)
