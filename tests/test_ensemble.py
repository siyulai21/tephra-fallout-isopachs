"""Check mixture mathematics, data isolation and the limits of calibration."""
import json
import numpy as np
import pytest
from scipy.special import ndtri, log_ndtr

from src.probabilistic_isopach.data import Observations
from src.probabilistic_isopach.distributions import LogMixture, predictive_distribution
from src.probabilistic_isopach.ensemble import StackedModel, stack_weights, observation_discrepancy
from src.probabilistic_isopach.model import Candidate
from src.probabilistic_isopach.validation import select_model, nested_validate, spatial_folds
from src.probabilistic_isopach.report import integrate, expected_volume


def observations(n=32):
    rng = np.random.default_rng(17)
    xy = rng.uniform(-100, 100, (n, 2))
    values = 4-.02*np.linalg.norm(xy, axis=1)+.006*xy[:, 0]+rng.normal(0, .3, n)
    return Observations(xy, values, np.zeros(n, bool), np.zeros(n), np.arange(n)).canonical()


def test_mixture_summaries_are_not_moment_matched_lognormal():
    d = LogMixture([.3, .7], [[0., 1.], [2., 3.]], [[.4, .4], [.2, .2]])
    for p in (.025, .5, .975):
        np.testing.assert_allclose(d.cdf(d.quantile(p)), p, atol=1e-10)
    mean, variance = d.moments()
    expected = .3*np.exp(np.array([0., 1.])+.2)+.7*np.exp(np.array([2., 3.])+.1)
    np.testing.assert_allclose(np.exp(d.log_expected_thickness()), expected)
    assert not np.allclose(np.exp(mean+variance/2), expected)
    one = LogMixture([1.], [[1., 2.]], [[.4, .8]])
    np.testing.assert_allclose(one.quantile(.975), [1., 2.]+ndtri(.975)*np.sqrt([.4, .8]))
    np.testing.assert_allclose(one.log_score([0., 0.], [True, True]), log_ndtr(-np.array([1., 2.])/np.sqrt([.4, .8])))


def test_stacking_recovers_complementary_experts_and_extreme_scores():
    scores = np.log([[.9, .1], [.1, .9]])-10000
    weights = stack_weights(scores, [.5, .5])
    np.testing.assert_allclose(weights, [.5, .5], atol=1e-7)
    assert stack_weights(scores, [.99, .01])[0] > .99


class ConstantComponent:
    """Perfectly correlated spatial fields, widely separated model means."""
    def __init__(self, mean, data):
        self.data, self.mean, self.amplitude = data, mean, .1
        self.candidate, self.trend_boundary = Candidate(), False

    def predict_log(self, xy, covariance=False, observation=False, log_sd=None):
        variance = np.full((len(xy), len(xy)), .01) if covariance else np.full(len(xy), .01)
        return np.full(len(xy), self.mean), variance


def test_whole_field_draws_and_volume_preserve_model_disagreement():
    data = observations()
    models = [ConstantComponent(0., data), ConstantComponent(np.log(10.), data)]
    stacked = StackedModel(models, [.4, .6], discrepancy_variance=4.)
    xy = np.array([[0., 0.], [1., 1.]])
    draws = stacked.sample_log_fields(xy, 10000, 123)
    # Choosing a component independently at each location would destroy this.
    assert np.max(abs(draws[0]-draws[1])) < .001
    np.testing.assert_array_equal(draws, stacked.sample_log_fields(xy, 10000, 123))
    m, cov = stacked.predict_log(xy, covariance=True)
    assert cov[0, 1] == pytest.approx(.01+.4*.6*np.log(10.)**2)
    np.testing.assert_allclose(stacked.predict_log(xy)[1], np.diag(cov))
    expected = (.4+.6*10)*np.exp(.005)*4e-5
    assert expected_volume(stacked, [-1, -1, 1, 1], 8) == pytest.approx(expected)
    result = integrate(stacked, [-1, -1, 1, 1], 8, 10000, 123)
    assert result['mean_km3'] == pytest.approx(expected)
    assert result['conditional_quantiles_km3']['0.025'] < 5e-5
    assert result['conditional_quantiles_km3']['0.975'] > 4e-4
    unadjusted = predictive_distribution(stacked, xy, observation=True)
    adjusted = predictive_distribution(stacked, xy, observation=True, calibrated=True)
    np.testing.assert_allclose(adjusted.variances-unadjusted.variances, 4.)
    stacked.discrepancy_variance = 100.
    np.testing.assert_array_equal(draws, stacked.sample_log_fields(xy, 10000, 123))


def test_discrepancy_recovers_single_normal_variance_mle():
    data = observations()
    means = np.zeros((len(data.xy), 1))
    variances = np.full_like(means, .1)
    variance, diagnostic = observation_discrepancy(data, np.ones(1), means, variances, np.ones(len(means))/len(means))
    assert variance == pytest.approx(np.mean(data.log_value**2)-.1, rel=1e-6)
    assert diagnostic['training_cv_score_after'] < diagnostic['training_cv_score_before']


SMALL_CANDIDATES = (Candidate('exponential', .75, .75), Candidate('power', .75, .75))


def test_stacking_is_reproducible_and_only_real_observations_are_used():
    data = observations()
    model, report = select_model(data, SMALL_CANDIDATES, n_folds=2, method='stacked')
    reordered = data.subset(np.arange(len(data.xy))[::-1])
    again, repeated = select_model(reordered, SMALL_CANDIDATES, n_folds=2, method='stacked')
    assert report == repeated
    np.testing.assert_allclose(model.predict_log(data.xy), again.predict_log(data.xy))
    assert all(np.array_equal(component.data.row_ids, data.row_ids) for component in model.models)


def test_outer_responses_cannot_change_their_own_weights_or_calibration():
    data = observations()
    folds = spatial_folds(data.xy, 2)
    altered = Observations(data.xy, data.log_value+3*(folds == 0), data.censored, data.log_sd, data.row_ids)
    before = nested_validate(data, SMALL_CANDIDATES, outer_folds=2, inner_folds=2, method='stacked')
    after = nested_validate(altered, SMALL_CANDIDATES, outer_folds=2, inner_folds=2, method='stacked')
    assert before['selections'][0] == after['selections'][0]
    for key in ('median_log', 'lower95_log', 'upper95_log', 'discrepancy_adjusted_upper95_log'):
        np.testing.assert_array_equal(np.array(before[key])[folds == 0], np.array(after[key])[folds == 0])
    assert before['mean_log_score'] != after['mean_log_score']


def test_cli_exports_a_joint_mixture_and_separate_observation_intervals(tmp_path):
    from src.probabilistic_isopach.cli import main
    main(['Tambora', '--output', str(tmp_path/'run'), '--method', 'stacked', '--no-publication',
          '--grid', '8', '--volume-grid', '8', '--draws', '20'])
    output = tmp_path/'run'
    report = json.loads((output/'report.json').read_text())
    assert report['settings']['method'] == 'stacked'
    joint = np.load(output/'virtual_predictions.npz')
    weights = joint['component_weights']
    means = joint['component_mean_log']
    mean = weights@means
    covariance = sum(w*(cov+np.outer(m-mean, m-mean)) for w, m, cov in
                     zip(weights, means, joint['component_covariance_log']))
    np.testing.assert_allclose(joint['mean_log'], mean)
    np.testing.assert_allclose(joint['covariance_log'], covariance)
    assert 'only moments' in str(joint['distribution'])
    assert 'diagnostic_adjusted_observation_upper95_cm' in (output/'predictions.csv').read_text().splitlines()[0]
