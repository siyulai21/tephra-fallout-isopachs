from dataclasses import replace

import numpy as np
import pytest
from numpy.testing import assert_allclose

from src.probabilistic_isopach.data import Observations
from src.empirical_isopach import SplineCandidate, TrendRegularizedSpline, select_spline
from src.empirical_isopach.spline import TensorBasis, survey_bounds
from src.empirical_isopach.region import SurveyRegion, integrate_region, regional_draws
from src.empirical_isopach.validation import automatic_buffer


def sample(seed=19, n=40):
    rng = np.random.default_rng(seed)
    xy = rng.uniform(-100, 100, (n, 2))
    y = 4-.025*np.linalg.norm(xy, axis=1)+rng.normal(0, .2, n)
    return Observations(xy, y, np.zeros(n, bool), np.zeros(n), np.arange(n))


def test_penalty_integrals_partition_and_polynomial_nullspaces():
    bounds = [-140., -70., 110., 90.]
    a, b = TensorBasis(bounds, 4, 4), TensorBasis(bounds, 4, 10)
    for key in ("mass", "gradient", "curvature"):
        assert_allclose(getattr(a, key), getattr(b, key), rtol=1e-10, atol=1e-10)
        assert np.linalg.eigvalsh(getattr(a, key)).min() > -1e-9
    assert_allclose(a.quadrature_weights.sum(), 1, atol=1e-13)
    assert_allclose(a.quadrature_basis.sum(axis=1), 1, atol=1e-13)
    ones = np.ones(a.n_coefficients)
    assert_allclose(ones@a.mass@ones, 1, atol=1e-13)
    assert_allclose(a.gradient@ones, 0, atol=1e-12)
    assert_allclose(a.curvature@ones, 0, atol=1e-10)
    # An affine field has zero curvature, including the mixed derivative term.
    target = 2+.3*a.quadrature_xy[:, 0]-.2*a.quadrature_xy[:, 1]
    coefficients = np.linalg.lstsq(a.quadrature_basis, target, rcond=None)[0]
    assert_allclose(a.curvature@coefficients, 0, atol=1e-8)


def test_gaussian_coefficients_and_joint_covariance_match_dense_reference():
    model = TrendRegularizedSpline().fit(sample())
    data, basis = model.data, model.basis
    B = basis.design(data.xy)
    H = B.T@np.diag(1/model.sd**2)@B+model.penalty
    rhs = B.T@(data.log_value/model.sd**2)+model.target_rhs
    expected = np.linalg.solve(H, rhs)
    query = np.array([[0., 0.], [50., -30.]])
    X = basis.design(query)
    mean, covariance = model.predict_log(query, covariance=True)
    assert_allclose(mean, X@expected, atol=1e-10)
    assert_allclose(covariance, X@np.linalg.inv(H)@X.T, atol=1e-10)
    assert_allclose(np.diag(covariance), model.predict_log(query)[1], atol=1e-12)
    observed = model.predict_log(query, covariance=True, observation=True, log_sd=[.1, .3])[1]
    assert_allclose(observed-covariance, np.diag(model.noise**2+np.array([.1, .3])**2), atol=1e-12)


@pytest.mark.parametrize("lower", [False, True])
def test_bound_likelihood_changes_fit_in_correct_direction_and_loose_limit_vanishes(lower):
    data = sample()
    bounds = survey_bounds(data.xy)
    original = TrendRegularizedSpline().fit(data, bounds=bounds)
    q = np.array([[0., 0.]])
    before = original.predict_log(q)[0][0]
    delta = 2 if lower else -2
    augmented = Observations(np.vstack([data.xy, q]), np.r_[data.log_value, before+delta],
                             np.r_[data.censored, True], np.r_[data.log_sd, 0],
                             np.r_[data.row_ids, 999], np.r_[data.lower_censored, lower])
    fitted = TrendRegularizedSpline().fit(augmented, bounds=bounds)
    after = fitted.predict_log(q)[0][0]
    assert (after-before)*delta > 0
    assert fitted.newton_decrement < 1e-7
    assert np.linalg.eigvalsh(fitted.predict_log(data.xy[:8], covariance=True)[1]).min() > -1e-10
    augmented.log_value[-1] = before-50 if lower else before+50
    loose = TrendRegularizedSpline().fit(augmented, bounds=bounds)
    assert_allclose(loose.predict_log(q)[0], before, atol=1e-9)


def test_no_extrapolation_and_invalid_inputs_fail_explicitly():
    model = TrendRegularizedSpline().fit(sample())
    with pytest.raises(ValueError, match="outside"):
        model.predict_log([[10000, 10000]])
    for xy in ([[np.nan, 0]], [1, 2]):
        with pytest.raises(ValueError, match="finite"):
            model.predict_log(xy)
    with pytest.raises(ValueError):
        SplineCandidate(trend_strength=-1)
    with pytest.raises(ValueError, match="non-collinear"):
        SurveyRegion([[0, 0], [1, 1], [2, 2]])


def test_coordinate_and_thickness_units_do_not_change_fitted_shape():
    data = sample()
    original = TrendRegularizedSpline().fit(data)
    scaled = TrendRegularizedSpline().fit(replace(data, xy=data.xy*1000))
    assert_allclose(original.predict_log(data.xy), scaled.predict_log(data.xy*1000), atol=1e-8)
    changed = TrendRegularizedSpline().fit(replace(data, log_value=data.log_value+np.log(100)))
    mean, variance = original.predict_log(data.xy)
    other_mean, other_var = changed.predict_log(data.xy)
    assert_allclose(other_mean, mean+np.log(100), atol=1e-6)
    assert_allclose(other_var, variance, atol=1e-7)


def test_surface_draws_preserve_joint_coefficient_uncertainty():
    model = TrendRegularizedSpline().fit(sample())
    q = np.array([[0., 0.], [10., 10.]])
    mean, covariance = model.predict_log(q, covariance=True)
    samples = model.basis.design(q)@model.draw_coefficients(30000, np.random.default_rng(13))
    assert_allclose(samples.mean(axis=1), mean, atol=.004)
    assert_allclose(np.cov(samples), covariance, rtol=.04, atol=.0003)


def test_prior_projection_quadrature_refinement():
    data = sample()
    candidate = SplineCandidate(trend_strength=100)
    coarse = TrendRegularizedSpline(candidate, quadrature_order=6).fit(data)
    fine = TrendRegularizedSpline(candidate, quadrature_order=12).fit(data)
    assert_allclose(coarse.predict_log(data.xy)[0], fine.predict_log(data.xy)[0], atol=.015)
    # Refining quadrature never increases prior mass/precision.
    assert_allclose(coarse.penalty, fine.penalty, rtol=1e-10, atol=1e-8)


def test_selection_is_permutation_invariant_and_no_validation_responses_fit(monkeypatch):
    data = sample()
    candidates = (SplineCandidate(trend_strength=0), SplineCandidate(trend_strength=100, tension=1))
    calls = []
    actual = TrendRegularizedSpline.fit
    def traced(self, obs, *args, **kwargs):
        workspace = kwargs.get("workspace")
        fitted_data = workspace.data if workspace is not None else obs
        calls.append(set(fitted_data.row_ids))
        return actual(self, obs, *args, **kwargs)
    monkeypatch.setattr(TrendRegularizedSpline, "fit", traced)
    a, report = select_spline(data, n_folds=3, candidates=candidates)
    ordered = data.canonical()
    labels = np.array(report["folds"])
    for index, ids in enumerate(calls[:-1]):
        assert ids.isdisjoint(ordered.row_ids[labels == index % 3])
    assert calls[-1] == set(data.row_ids)
    b, reversed_report = select_spline(data.subset(np.arange(len(data.xy))[::-1]), n_folds=3, candidates=candidates)
    assert report["selected_index"] == reversed_report["selected_index"]
    assert report["candidates"] == reversed_report["candidates"]
    assert_allclose(a.predict_log(data.xy), b.predict_log(data.xy), atol=1e-12)


def test_hull_quadrature_area_linear_function_and_units():
    region = SurveyRegion([[0, 0], [100, 0], [100, 100], [0, 100], [40, 60]])
    for order in (2, 4, 8):
        xy, weights = region.quadrature(order)
        assert_allclose(weights.sum(), 10000, rtol=1e-13)
        assert_allclose(weights@(xy[:, 0]+xy[:, 1]), 1000000, rtol=1e-13)
        assert region.contains(xy).all()
    class Constant:
        def predict_log(self, xy):
            return np.full(len(xy), np.log(10)), np.zeros(len(xy))
    volume = integrate_region(Constant(), region)
    assert volume["converged"]
    assert_allclose(volume["expected_km3"], 1, rtol=1e-12)
    assert_allclose(volume["median_surface_km3"], 1, rtol=1e-12)


def test_automatic_buffer_uses_geometry_and_kinds_only():
    data = sample(n=50)
    a, rule_a = automatic_buffer(data)
    b, rule_b = automatic_buffer(replace(data, log_value=data.log_value*20+100))
    assert a == b and rule_a == rule_b
    assert rule_a["attempts"][-1]["feasible"]


def test_joint_volume_is_reproducible_and_mean_matches_integral():
    data = sample()
    model = TrendRegularizedSpline().fit(data)
    region = SurveyRegion(data.xy)
    expected = integrate_region(model, region)
    a = regional_draws(model, region, expected["expected_km3"], draws=10000, seed=2)
    b = regional_draws(model, region, expected["expected_km3"], draws=10000, seed=2)
    assert a == b
    assert a["expected_grid_converged"]
    assert abs(a["sampling_mean_km3"]/expected["expected_km3"]-1) < .015


def test_ensemble_volume_selects_one_component_for_the_entire_field():
    from src.probabilistic_isopach.distributions import LogMixture
    class Constant:
        def __init__(self, thickness):
            self.thickness = thickness
        def predict_log(self, xy, covariance=False):
            return np.full(len(xy), np.log(self.thickness)), np.zeros((len(xy), len(xy)) if covariance else len(xy))
    class Mixture:
        models = [Constant(10), Constant(100)]
        weights = np.array([.5, .5])
        def distribution(self, xy, observation=False, log_sd=None, calibrated=False):
            return LogMixture(self.weights, np.array([m.predict_log(xy)[0] for m in self.models]), np.zeros((2, len(xy))))
    region = SurveyRegion([[0, 0], [100, 0], [100, 100], [0, 100]])
    report = regional_draws(Mixture(), region, reference=5.5, draws=2000)
    # Whole-field component choice gives two volumes, 1 and 10 km³. Independent
    # cell-by-cell choices would spuriously concentrate both endpoints near 5.5.
    quantiles = report["conditional_quantiles_km3"]
    assert_allclose([quantiles["0.025"], quantiles["0.975"]], [1., 10.], rtol=1e-4)


def test_nested_comparison_uses_identical_outer_rows_and_excludes_them_from_tuning(monkeypatch):
    from src.empirical_isopach import validation
    from src.probabilistic_isopach.model import Candidate
    from src.probabilistic_isopach.validation import select_model
    data = sample(n=50)
    calls = []
    def small_fit(method, train, bounds, folds, buffer):
        calls.append((method, set(train.row_ids)))
        if method == "spline":
            return select_spline(train, bounds=bounds, n_folds=folds, buffer_km=buffer,
                                 candidates=(SplineCandidate(), SplineCandidate(trend_strength=0)))
        return select_model(train, candidates=(Candidate(),), n_folds=folds, buffer_km=buffer, method="stacked")
    monkeypatch.setattr(validation, "fit_method", small_fit)
    result = validation.nested_comparison(data, SurveyRegion(data.xy), buffer_km=0)
    assert len(calls) == 8
    for index in range(4):
        assert calls[index][1] == calls[index+4][1]
        for method in validation.METHODS:
            fold = result["methods"][method]["selections"][index]
            assert set(fold["training_row_ids"]).isdisjoint(fold["withheld_row_ids"])
            assert set(fold["selection"]["row_ids"]) == set(fold["training_row_ids"])
    assert sorted(result["row_ids"]) == sorted(data.row_ids)


def test_unestimable_censored_simulations_are_recorded_not_dropped(tmp_path, monkeypatch):
    from benchmarks import regional
    original = regional.synthetic_case
    def insufficient(scenario, seed):
        data, region, truth, heldout, actual = original(scenario, seed)
        # There are too few exact observations for any allowed nested split.
        data.censored[:] = True
        data.censored[:5] = False
        data.lower_censored[:] = False
        return data, region, truth, heldout, actual
    monkeypatch.setattr(regional, "synthetic_case", insufficient)
    results = regional.synthetic_benchmark(tmp_path, repeats=1)
    assert len(results) == 4
    assert all(r["status"] == "insufficient_training_support" for r in results)
    assert all(r["methods"][m]["error"] for r in results for m in ("stacked", "spline"))
    assert (tmp_path/"synthetic.json").is_file()
