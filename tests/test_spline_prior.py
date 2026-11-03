import numpy as np
from src.probabilistic_isopach import Observations
from src.research.trend_regularized_spline import RegularizedSpline, select_spline_prior
from src.probabilistic_isopach.report import grid_cells


def data():
    rng = np.random.default_rng(19)
    xy = rng.uniform(-100, 100, (35, 2))
    y = 4-.025*np.linalg.norm(xy, axis=1)+rng.normal(0, .2, len(xy))
    return Observations(xy, y, np.zeros(len(xy), bool), np.zeros(len(xy)), np.arange(len(xy)))


def test_duplicate_quadrature_does_not_add_information():
    d = data()
    a = RegularizedSpline(strength=10, quadrature_resolution=8).fit(d)
    q, _ = grid_cells(a.bounds, 8)
    b = RegularizedSpline(strength=10).fit(d, quadrature=np.repeat(q, 2, axis=0))
    np.testing.assert_allclose(a.spline.cij, b.spline.cij, atol=1e-9)
    assert abs(a.prior_total_precision-b.prior_total_precision) < 1e-10


def test_prior_grid_converges_without_increasing_strength():
    d = data()
    a = RegularizedSpline(strength=10, quadrature_resolution=24).fit(d)
    b = RegularizedSpline(strength=10, quadrature_resolution=48).fit(d)
    np.testing.assert_allclose(a.predict_log(d.xy), b.predict_log(d.xy), atol=.03)
    assert abs(a.prior_total_precision-b.prior_total_precision) < 1e-10


def test_zero_prior_is_quadrature_invariant():
    d = data()
    a = RegularizedSpline(strength=0, quadrature_resolution=4).fit(d)
    b = RegularizedSpline(strength=0, quadrature_resolution=32).fit(d)
    np.testing.assert_allclose(a.spline.cij, b.spline.cij, atol=1e-12)


def test_strength_selection_refits_and_includes_zero():
    fitted, records = select_spline_prior(data(), strengths=(0., 10.), n_folds=3, quadrature_resolution=8)
    assert {r['strength'] for r in records} == {0., 10.}
    assert all(r['mse_log'] is not None for r in records)
    assert np.isfinite(fitted.predict_log(data().xy)).all()


def test_previous_spline_imports_remain_compatible():
    from src.probabilistic_isopach import regularized_spline as previous
    from src.research import trend_regularized_spline as current
    assert previous.RegularizedSpline is current.RegularizedSpline
    assert previous.select_spline_prior is current.select_spline_prior
    assert previous.design_matrix is current.design_matrix
