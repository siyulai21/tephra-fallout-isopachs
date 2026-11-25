import numpy as np
import pytest
from scipy.integrate import quad
from scipy.stats import norm

from src.empirical_isopach.core.observations import Observations
from src.empirical_isopach.models.directional.multiscale.conditioning import condition, corrected
from src.empirical_isopach.models.directional.multiscale.geometry import LocalBasis, trend
from src.empirical_isopach.models.directional.multiscale.posterior import Marginals
from src.empirical_isopach.models.directional.multiscale.spread import footprint, attenuation


@pytest.mark.parametrize("lower,upper", [(False, np.nan), (True, np.nan), (True, 0.37)])
def test_event_evidence_and_latent_cdf_against_direct_quadrature(lower, upper):
    # One coefficient, one measurement and one recorded event with separate response errors.
    data = Observations(
        [[0, 0], [2, 1]], [0.20, -0.11], [False, True], [0, 0], [2, 3], [False, lower], [np.nan, upper]
    )
    sd = np.array([0.7, 0.35])
    X = np.array([[0.8], [1.3]])
    s = condition(X, np.zeros(2), data, sd)
    centers, weights, corr, ess = corrected(s, 16, 4141)

    def density(beta):
        ref = norm.pdf(beta) * norm.pdf(0.20, loc=0.8 * beta, scale=sd[0])
        if np.isfinite(upper):
            event = norm.cdf((upper - 1.3 * beta) / sd[1]) - norm.cdf((-0.11 - 1.3 * beta) / sd[1])
        elif lower:
            event = norm.sf((-0.11 - 1.3 * beta) / sd[1])
        else:
            event = norm.cdf((-0.11 - 1.3 * beta) / sd[1])
        return ref * event

    z = quad(density, -9, 9)[0]
    assert 0.5 < ess <= 1.0
    assert abs(s["logz"] + corr - np.log(z)) < 0.0015
    expected = quad(density, -9, 0.2)[0] / z
    actual = weights @ norm.cdf((0.2 - centers[:, 0]) / np.sqrt(s["complete_cov"][0, 0]))
    assert abs(expected - actual) < 0.002


def test_rigid_rotation_of_coherent_covariance_and_training_reach():
    rng = np.random.default_rng(95)
    xy = rng.normal(size=(14, 2)) * 20
    target = np.vstack((xy, [[0, 0], [200, 70], [95, -120]]))
    angle = 0.74
    rotate = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    data = Observations(xy, np.log(np.arange(1, 15)), np.zeros(14, bool), np.zeros(14), np.arange(14))
    alt = Observations(xy @ rotate, data.log_value, data.censored, data.log_sd, data.row_ids)
    for frac in (0.0, 0.5, 1.0):
        a = LocalBasis.fit(xy, 1.0, 0.5, frac).features(target, 1.0)
        b = LocalBasis.fit(xy @ rotate, 1.0, 0.5, frac).features(target @ rotate, 1.0)
        np.testing.assert_allclose(a @ a.T, b @ b.T, atol=2e-9)
    np.testing.assert_allclose(
        attenuation(target, 1.0, footprint(data, 1.0)),
        attenuation(target @ rotate, 1.0, footprint(alt, 1.0)),
        atol=1e-10,
    )
    grouped = LocalBasis.fit(xy, 1.0, 0.5, 0.0).features(target, 1.0)
    shared = LocalBasis.fit(xy, 1.0, 0.5, 0.5).features(target, 1.0)
    full = LocalBasis.fit(xy, 1.0, 0.5, 1.0).features(target, 1.0)
    np.testing.assert_allclose(shared @ shared.T, (grouped @ grouped.T + full @ full.T) / 2, atol=1e-10)


def test_bearing_dependence_disappears_at_vent_and_antipode():
    from src.empirical_isopach.models.mass_mixture.geometry import RADIUS

    theta = np.linspace(0.0, 2 * np.pi, 17)
    for r in (0.0, 2 * RADIUS):
        xy = r * np.column_stack((np.sin(theta), np.cos(theta)))
        X, d, _ = trend(xy, 1.0, 120.0, 1)
        np.testing.assert_allclose(X[:, 3:], 0, atol=1e-7)
        np.testing.assert_allclose(d, d[0], atol=1e-12)


def test_mixture_quantiles_do_not_depend_on_other_points_in_batch():
    means = np.array([[-3, 0.8, 20], [1.4, -0.7, -0.1], [0.2, 9, 1.0]])
    weights = np.array([0.31, 0.48, 0.21])
    var = np.array([[0.2, 0.7, 4], [0.05, 4, 0.5], [0.1, 9, 2]])
    mix = Marginals(weights, means, var)
    for q in (0.025, 0.25, 0.5, 0.75, 0.975):
        x = mix.quantile(q)
        np.testing.assert_allclose(mix.cdf(x), q, atol=3.0e-12)
        for j in range(len(x)):
            solo = Marginals(weights, means[:, j : j + 1], var[:, j : j + 1]).quantile(q)
            np.testing.assert_allclose(x[j], solo[0], atol=2.0e-10)
