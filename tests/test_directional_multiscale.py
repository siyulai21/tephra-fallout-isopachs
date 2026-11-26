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
    actual = weights @ norm.cdf((0.2 - centers[:, 0]) / np.sqrt(max(1e-20, s["complete_cov"][0, 0])))
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


def test_bounded_integration_marginalizes_unused_coefficients_and_coincident_sites():
    # Two independent bounds at the same latent location have rank one, even
    # when the field has many nuisance coefficients. Their probabilities must
    # both enter the likelihood without a jitter-dependent extra direction.
    data = Observations(
        [[0, 0], [2, 1], [2, 1]],
        [0.2, -0.1, -0.3],
        [False, True, True],
        [0, 0, 0],
        [2, 3, 4],
        [False, False, True],
        [np.nan, np.nan, 0.4],
    )
    B = np.array([[0.8], [1.3], [1.3]])
    noise = np.array([0.7, 0.35, 0.4])
    state = condition(B, np.zeros(3), data, noise)
    enlarged = condition(np.pad(B, ((0, 0), (0, 40))), np.zeros(3), data, noise)
    assert state["event_factor"].shape == enlarged["event_factor"].shape == (2, 1)
    centers, weights, correction, _ = corrected(state, 14, 751)
    extra, ew, ec, _ = corrected(enlarged, 14, 751)
    np.testing.assert_allclose(centers[:, 0], extra[:, 0], atol=1e-12)
    np.testing.assert_allclose(weights, ew, atol=1e-12)
    np.testing.assert_allclose(correction, ec, atol=1e-12)

    def density(beta):
        return (
            norm.pdf(beta)
            * norm.pdf(0.2, loc=0.8 * beta, scale=0.7)
            * norm.cdf((-0.1 - 1.3 * beta) / 0.35)
            * (norm.cdf((0.4 - 1.3 * beta) / 0.4) - norm.cdf((-0.3 - 1.3 * beta) / 0.4))
        )

    evidence = quad(density, -9, 9, epsabs=1e-11)[0]
    assert abs(state["logz"] + correction - np.log(evidence)) < 0.001
    actual = weights @ norm.cdf((0.1 - centers[:, 0]) / np.sqrt(max(1e-20, state["complete_cov"][0, 0])))
    assert abs(actual - quad(density, -9, 0.1)[0] / evidence) < 0.002


def test_event_subspace_recovers_the_complete_coefficient_posterior():
    rng = np.random.default_rng(71)
    n, m = 28, 75
    B = rng.normal(size=(n, m)) / np.sqrt(m)
    y = rng.normal(size=n)
    upper = np.full(n, np.nan)
    upper[18:] = y[18:] + 0.6
    censored, lower = np.arange(n) >= 12, np.arange(n) >= 18
    data = Observations(rng.normal(size=(n, 2)), y, censored, np.full(n, 0.1), np.arange(n), lower, upper)
    state = condition(B, np.full(n, 0.3), data, np.full(n, 0.5))
    variance = 0.5**2 + 0.1**2
    exact = B[~censored]
    X = B[censored]
    P = np.eye(m) + exact.T @ exact / variance + (X.T * state["tau"]) @ X
    h = exact.T @ (y[~censored] - 0.3) / variance + X.T @ state["eta"]
    covariance = np.linalg.inv(P)
    np.testing.assert_allclose(state["mean"], covariance @ h, atol=1e-11)
    np.testing.assert_allclose(state["cov"], covariance, atol=1e-11)
    conditional = covariance - covariance @ X.T @ np.linalg.pinv(X @ covariance @ X.T) @ X @ covariance
    np.testing.assert_allclose(state["complete_cov"], conditional, atol=1e-11)
    expected = (
        -0.5 * (len(exact) * np.log(2 * np.pi * variance) + np.sum((y[~censored] - 0.3) ** 2) / variance)
        + 0.5 * h @ covariance @ h
        - 0.5 * np.linalg.slogdet(P)[1]
        + state["cs"].sum()
    )
    np.testing.assert_allclose(state["logz"], expected, atol=1e-11)


def test_posterior_storage_retains_unconstrained_prior_directions():
    from src.empirical_isopach.models.directional.multiscale.posterior import (
        Component,
        pack_gaussians,
        unpack_gaussian,
    )

    rng = np.random.default_rng(351)
    Q = np.linalg.qr(rng.normal(size=(40, 5)))[0]
    parts = []
    for j in range(4):
        small = np.diag(np.linspace(0.1, 0.8, 5))
        covariance = np.eye(40) + Q @ (small - np.eye(5)) @ Q.T
        parts.append(
            Component(
                0.25, 32.0, 1, 0.5, 3.0, 0.4, 0.0, Q @ rng.normal(size=5), np.linalg.cholesky(covariance)
            )
        )
    packet = pack_gaussians(parts, {(32.0, 1, 0.5): Q})
    assert packet["l0"].shape == (5, 5)
    for i, original in enumerate(parts):
        mean, chol = unpack_gaussian(packet, i)
        np.testing.assert_allclose(mean, original.mean, atol=2e-12)
        np.testing.assert_allclose(chol @ chol.T, original.chol @ original.chol.T, atol=2e-12)
        orthogonal = np.eye(40) - Q @ Q.T
        np.testing.assert_allclose(orthogonal @ (chol @ chol.T) @ orthogonal, orthogonal, atol=2e-12)


def test_local_length_does_not_expand_with_distal_range():
    from src.empirical_isopach.models.directional.multiscale.geometry import separated_covariance

    points = np.array([[0.0, 0.0], [4.0, 0.0], [15.0, 0.0]])

    def kernel(p):
        return separated_covariance(np.vstack((p, p)), p, 0.5)

    source, distal = kernel(points), kernel(points + [1000.0, 0.0])
    np.testing.assert_allclose(source[3:], distal[3:], atol=1e-14)
    assert distal[0, 2] > source[0, 2]
    np.testing.assert_allclose(np.diag(source[:3]), 0.8)
    np.testing.assert_allclose(np.diag(source[3:]), 0.2)


def test_dense_surveys_do_not_exhaust_the_local_contrast_budget():
    xy = np.random.default_rng(625).normal(size=(150, 2)) * 70
    grouped = LocalBasis.fit(xy, 1.0, 0.2, 0.0)
    coherent = LocalBasis.fit(xy, 1.0, 0.2, 1.0)
    assert len(coherent.averaging) > 96
    assert len(coherent.averaging) > len(grouped.averaging)
    assert coherent.unresolved(xy, 1.0).max() < 2e-5


def test_directional_source_gate_has_vanishing_cartesian_curvature():
    # The first harmonic previously behaved as x*sqrt(x*x+y*y), with a
    # direction-dependent second derivative. The solid harmonic is cubic.
    theta = np.arange(16) * np.pi / 8

    def magnitude(radius):
        xy = radius * np.column_stack((np.sin(theta), np.cos(theta)))
        return abs(trend(xy, 1.0, 8.0, 1)[0][:, 3:]).max() / radius**2

    assert magnitude(1e-4) < 0.11 * magnitude(1e-3)


def test_compressed_bound_posterior_against_an_independent_integral():
    from src.empirical_isopach.models.directional.multiscale.compression import compression

    data = Observations([[0.0, 0.0]], [-0.2], [True], [0.0], [2])
    state = condition(np.ones((1, 1)), np.zeros(1), data, np.array([0.3]))
    sims = [corrected(state, 14, seed) for seed in (713, 977)]
    parts, error, minimum = compression(state, sims, np.ones((1, 1)), 12)
    density = lambda x: norm.pdf(x) * norm.cdf((-0.2 - x) / 0.3)
    normalizer = quad(density, -10, 10, epsabs=1e-11)[0]
    assert minimum > 0 and error < 0.02
    for x in (-2.0, -1.0, -0.3, 0.2):
        expected = quad(density, -10, x, epsabs=1e-11)[0] / normalizer
        actual = sum(w * norm.cdf((x - m[0]) / np.sqrt(c[0, 0])) for w, m, c in parts)
        assert abs(actual - expected) < 0.025


def test_volume_refinement_targets_unresolved_parameter_uncertainty(monkeypatch, tmp_path):
    from types import SimpleNamespace
    from src.empirical_isopach.models.directional.multiscale import integrals
    from src.empirical_isopach.models.directional.multiscale.geometry import SCHEMA, LEGACY_SCHEMA

    calls = []

    def trials(model, data, parameter_seed, point_seed, draws, order):
        calls.append((draws, order))
        # Perfect spatial integration, but the independent population is biased
        # until 8,192 draws. A subsequent refinement must confirm its change.
        total = 2.0 if parameter_seed == 79031 and draws < 8192 else 1.0
        return (np.full((draws, 8), total / 8), np.zeros((draws, 3)), np.zeros(draws, int))

    monkeypatch.setattr(integrals, "trials", trials)
    component = SimpleNamespace(scale=32.0, evolving=0, length=0.5, spread=0.2, tail=4.5, noise=0.26)
    model = SimpleNamespace(protocol=SCHEMA, components=[component], weights=np.ones(1))
    result = integrals.integrate(model, None, tmp_path / "adaptive.npz")
    assert result["integration_passed"]
    assert calls[-1] == (16384, 12)
    assert all(order <= 12 for _, order in calls)
    calls.clear()
    model.protocol = LEGACY_SCHEMA
    old = integrals.integrate(model, None, tmp_path / "legacy.npz")
    assert not old["integration_passed"]
    assert calls[-1] == (8192, 15)
