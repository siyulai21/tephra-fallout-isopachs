"""Independent numerical identities and likelihood/uncertainty semantics."""
import numpy as np
import pytest
from scipy.integrate import quad
from scipy.special import logsumexp
from scipy.stats import vonmises_fisher

from src.empirical_isopach.core.observations import Observations
from src.empirical_isopach.models.mass_mixture.geometry import (
    AREA, RADIUS, to_sphere, from_sphere, centers, log_kernel,
    log_normalizers, log_field, kernel_samples, exponential_log_normalizer,
)
from src.empirical_isopach.models.mass_mixture.inference import (
    packed_data, observation_log_likelihood, model_function, flatten,
)
from src.empirical_isopach.models.mass_mixture.prior import BASELINE, SENSITIVITIES, Prior
from src.empirical_isopach.models.mass_mixture.posterior import predictions, volume_summary


@pytest.mark.parametrize("latitude", [-75., 0., 36., 90.])
def test_equal_area_sphere_roundtrip(latitude):
    xy = np.array([[0., 0.], [200., -500.], [9000., 3000.], [-12000., 0.]])
    unit = to_sphere(xy, latitude)
    np.testing.assert_allclose(np.linalg.norm(unit, axis=1), 1., atol=1e-14)
    np.testing.assert_allclose(from_sphere(unit, latitude), xy, atol=2e-10)
    with pytest.raises(ValueError):
        to_sphere([[40000., 0.]], latitude)


@pytest.mark.parametrize("scale", [1., 300., 10000.])
@pytest.mark.parametrize("q", [.3, 1., 1.000001, 2.5, 40.])
def test_power_normalization_by_independent_quadrature(scale, q):
    _, log_z = log_normalizers(np.log(scale), q)
    b = q*(scale/RADIUS)**2
    # Integrate log distance to resolve sub-km caps and diffuse global tails.
    value, error = quad(lambda t: np.exp(-q*t-log_z)*b*np.exp(t)*2*np.pi*RADIUS**2,
                        0, np.log1p(2/b), epsabs=1e-10, epsrel=1e-10)
    assert abs(value-1) < 2e-10
    assert error < 1e-8


def test_gaussian_matches_scipy_directional_density():
    center = centers(np.log(900.), np.array([1., 2.]))
    unit = to_sphere([[100., 50.], [-2000., 1000.], [8000., -9000.]], 0)
    actual = log_kernel(unit, np.log(900.), center, 2., 0.)
    expected = vonmises_fisher(center, (RADIUS/900.)**2).logpdf(unit)-2*np.log(RADIUS)
    np.testing.assert_allclose(actual, expected, atol=1e-11)


@pytest.mark.parametrize("scale", [1., 60., 1000., 1e6, 1e9])
def test_exponential_normalization_and_uniform_limit(scale):
    from scipy.special import gammainc
    log_z = exponential_log_normalizer(np.log(scale))
    expected = np.log(2*np.pi)+2*np.log(scale)+np.log(gammainc(2., 2*RADIUS/scale))
    assert log_z == pytest.approx(expected, abs=1e-10)
    if scale == 1e9:
        assert np.exp(log_z)/AREA == pytest.approx(1., abs=1e-5)


def test_exponential_inverse_cdf():
    from scipy.stats import qmc
    unit = kernel_samples(np.log(60.), np.array([0., 0., 1.]), 2., 0.,
        qmc.Sobol(3, scramble=True, seed=91).random_base2(16), exponential_fraction=1.)
    radius = RADIUS*np.sqrt(2*(1-unit[:, 2]))
    for value in (30., 60., 120., 240.):
        expected = 1-(1+value/60)*np.exp(-value/60)
        assert abs(np.mean(radius <= value)-expected) < .00004


@pytest.mark.parametrize("q,eta", [(.5, 1.), (1., 1.), (2.5, 1.), (2., 0.), (40., .5)])
def test_inverse_cdf_mass_samples(q, eta):
    from scipy.stats import qmc
    unit = kernel_samples(np.log(200.), np.array([0., 0., 1.]), q, eta,
                          qmc.Sobol(3, scramble=True, seed=91).random_base2(16))
    np.testing.assert_allclose(np.linalg.norm(unit, axis=1), 1., atol=1e-14)
    u = 1-unit[:, 2]
    l2 = (200./RADIUS)**2
    b = q*l2
    for boundary in (.001, .01, .1, 1.):
        gaussian = -np.expm1(-boundary/l2)/-np.expm1(-2/l2)
        power = (np.log1p(boundary/b)/np.log1p(2/b) if q == 1 else
                 np.expm1((1-q)*np.log1p(boundary/b))/np.expm1((1-q)*np.log1p(2/b)))
        assert abs(np.mean(u <= boundary)-((1-eta)*gaussian+eta*power)) < .00015


def example_samples(n=60):
    return {"log_volume": np.log(np.linspace(80, 150, n)),
        "log_scale": np.tile(np.log([50., 200., 500.]), (n, 1)),
        "offset": np.tile([[0., 1.], [1., -1.], [2., .5]], (n, 1, 1)),
        "weights": np.tile([.2, .3, .5], (n, 1)), "exponent": np.full(n, 2.),
        "tail_fraction": np.full(n, .4), "noise": np.full(n, .3),
        "contamination": np.full(n, .1)}


def test_volume_is_not_integral_of_pointwise_quantiles_or_observation_noise():
    flat = example_samples()
    chain = {k: v.reshape((2, 30)+v.shape[1:]) for k, v in flat.items()}
    before = volume_summary(chain)
    chain["noise"] *= 30
    assert volume_summary(chain) == before
    assert before["median"] == pytest.approx(115.)
    # Displaying any additional remote points cannot change the whole integral.
    a = log_field(flat, to_sphere([[10., 20.]], 0))
    b = log_field(flat, to_sphere([[10., 20.], [10000., 0.]], 0))
    np.testing.assert_allclose(a[:, 0], b[:, 0])


def test_permutation_of_fans_does_not_change_the_field():
    values = example_samples()
    reordered = {k: (v[:, [2, 0, 1]] if k in ("log_scale", "offset", "weights") else v)
                 for k, v in values.items()}
    unit = to_sphere([[0., 0.], [100., 400.], [8000., -2000.]], 40)
    np.testing.assert_allclose(log_field(values, unit), log_field(reordered, unit), atol=1e-12)


@pytest.mark.parametrize("latitude", [0., 65.])
def test_displaced_multifan_total_by_independent_whole_earth_quadrature(latitude):
    from src.probabilistic_isopach.global_volume import EarthDomain, earth_cells
    values = example_samples(1)
    values["log_scale"] = np.log([[500., 1400., 4000.]])
    values["exponent"][:] = .7  # Even a nonintegrable planar tail is finite here.
    values["kernel_weights"] = np.array([[.2, .3, .5]])
    domain = EarthDomain.wgs84(latitude)
    total = 0.
    for xy, area in earth_cells(domain, [0., 100., 500., 2000., 5000., 9000., domain.rho_max], 96, 512, 500):
        total += np.exp(log_field(values, to_sphere(xy, latitude))[0])@area*1e-5
    assert total == pytest.approx(np.exp(values["log_volume"][0]), rel=1e-5)


def example_data():
    return Observations([[0., 0.], [100., 200.], [500., -100.]], np.log([30., 2., 5.]),
                        [False, True, True], [0., .2, .5], [1, 2, 3], [False, False, True])


@pytest.mark.parametrize("prior", [BASELINE, SENSITIVITIES[0], SENSITIVITIES[-2], Prior(exponential_core=False)])
def test_jax_numpy_likelihood_parity_and_padding(prior):
    from numpyro import handlers
    values = {k: v[0] for k, v in example_samples().items()}
    values["exponential_fraction"] = .35 if prior.exponential_core else 0.
    eta, zeta = values["tail_fraction"], values["exponential_fraction"]
    values["kernel_weights"] = np.array([(1-eta)*(1-zeta), (1-eta)*zeta, eta])
    if not prior.exponential_core:
        values["two_kernel_weights"] = values["kernel_weights"][[0, 2]]
    values["tail_log"] = np.log(values["exponent"]-(1 if prior.integrable_plane_tail else 0))
    if not prior.contamination:
        values["contamination"] = 0.
    data = example_data()
    actual = observation_log_likelihood({k: np.asarray(v)[None] for k, v in values.items()}, data, 0).sum()
    unit_volume = {k: np.asarray(v)[None] for k, v in values.items()}
    unit_volume["log_volume"] = np.zeros(1)
    values["field_level"] = values["log_volume"]+log_field(unit_volume, to_sphere(data.xy, 0)).mean()
    parameters = {k: v for k, v in values.items() if k not in
                  ("log_volume", "exponent", "tail_fraction", "exponential_fraction")}
    if not prior.exponential_core:
        parameters.pop("kernel_weights")
    model = handlers.seed(handlers.substitute(model_function(prior), data=parameters), rng_seed=42)
    trace = handlers.trace(model).get_trace(*packed_data(data, 0))
    factor = trace["observations"]
    expected = float(factor["fn"].log_prob(factor["value"]))
    assert actual == pytest.approx(expected, abs=1e-10)
    # 125 padded sites supply no likelihood information.
    assert len(packed_data(data, 0)[0]) == 128


def test_amplitude_centering_preserves_the_complete_physical_joint_density():
    from numpyro.infer.util import log_density
    import numpyro.distributions as dist
    data = example_data()
    values = {k: v[0] for k, v in example_samples().items()}
    values.update(kernel_weights=np.array([.2, .3, .5]), tail_log=0.)
    unit = {k: np.asarray(v)[None] for k, v in values.items()}
    unit["log_volume"] = np.zeros(1)
    parameters = {k: v for k, v in values.items() if k not in ("log_volume", "exponent", "tail_fraction")}
    parameters["field_level"] = values["log_volume"]+log_field(unit, to_sphere(data.xy, 0)).mean()
    for p in (BASELINE, SENSITIVITIES[2]):
        actual, _ = log_density(model_function(p), packed_data(data, 0), {}, parameters)
        expected = (dist.Normal(p.log_volume_mean, p.log_volume_sd).log_prob(values["log_volume"])
            +dist.Normal(p.log_scale_mean, p.log_scale_sd).log_prob(values["log_scale"]).sum()
            +dist.Normal(0., p.offset_sd).log_prob(values["offset"]).sum()
            +dist.Dirichlet(np.ones(3)*p.fan_concentration/3).log_prob(values["weights"])
            +dist.Dirichlet(np.ones(3)*p.kernel_concentration/3).log_prob(values["kernel_weights"])
            +dist.Normal(p.tail_log_mean, p.tail_log_sd).log_prob(values["tail_log"])
            +dist.HalfNormal(p.noise_scale).log_prob(values["noise"])
            +dist.Beta(1., 9.).log_prob(values["contamination"])
            +observation_log_likelihood({k: np.asarray(v)[None] for k, v in values.items()}, data, 0).sum())
        assert float(actual) == pytest.approx(float(expected), abs=1e-10)


def test_predictive_bound_probability_and_quantiles():
    flat = example_samples()
    samples = {k: v.reshape((2, 30)+v.shape[1:]) for k, v in flat.items()}
    result = predictions(samples, example_data(), 0)
    expected = logsumexp(observation_log_likelihood(flat, example_data(), 0), axis=0)-np.log(60)
    np.testing.assert_allclose(result["log_score"], expected, atol=1e-12)
    assert np.all(result["lower"] < result["median"])
    assert np.all(result["median"] < result["upper"])
    assert np.all(result["log_score"][1:] <= 0)


def test_invalid_prior_rejected():
    with pytest.raises(ValueError):
        Prior(components=0)
    with pytest.raises(ValueError):
        Prior(log_scale_sd=-1)


def test_sampling_efficiency_does_not_override_mixing_failures():
    from src.empirical_isopach.models.mass_mixture.study import assess_diagnostics
    d = {"passed": False, "max_label_invariant_rhat": 1.001,
         "min_label_invariant_bulk_ess": 500, "min_label_invariant_tail_ess": 600,
         "divergences": 0, "max_depth_fraction": .3, "min_energy_bfmi": .8}
    assert assess_diagnostics(d)["passed"]
    assert assess_diagnostics(d)["trajectory_efficiency_warning"]
    assert not assess_diagnostics({**d, "divergences": 1})["passed"]
    assert not assess_diagnostics({**d, "max_label_invariant_rhat": 1.1})["passed"]
    assert not assess_diagnostics({**d, "min_energy_bfmi": .2})["passed"]


def test_cached_posteriors_reject_changed_inputs_priors_and_corrupted_samples(tmp_path, monkeypatch):
    from dataclasses import replace
    from src.empirical_isopach.models.mass_mixture import study
    flat = example_samples()
    chain = {k: v.reshape((2, 30)+v.shape[1:]) for k, v in flat.items()}
    diagnostics = {"passed": True, "max_label_invariant_rhat": 1.001,
        "min_label_invariant_bulk_ess": 500, "min_label_invariant_tail_ess": 500,
        "divergences": 0, "max_depth_fraction": 0., "min_energy_bfmi": .8}
    calls = []

    def fake_sample(*args):
        calls.append(True)
        return chain, diagnostics

    monkeypatch.setattr(study, "sample", fake_sample)
    data = example_data()
    first, _ = study.fit_cached(data, 0, BASELINE, tmp_path, attempts=1, progress=None)
    cached, _ = study.fit_cached(data, 0, BASELINE, tmp_path, attempts=1, progress=None)
    np.testing.assert_array_equal(first["log_volume"], cached["log_volume"])
    assert len(calls) == 1
    changed = data.subset([True, True, False])
    with pytest.raises(ValueError, match="Stale posterior cache"):
        study.fit_cached(changed, 0, BASELINE, tmp_path, attempts=1, progress=None)
    with pytest.raises(ValueError, match="Stale posterior cache"):
        study.fit_cached(data, 0, replace(BASELINE, log_volume_mean=1.), tmp_path, attempts=1, progress=None)
    path = tmp_path/"attempt-1.npz"
    path.write_bytes(path.read_bytes()+b"unexpected modification")
    with pytest.raises(ValueError, match="Posterior checksum mismatch"):
        study.fit_cached(data, 0, BASELINE, tmp_path, attempts=1, progress=None)


def test_observation_mixture_nonmixing_is_not_hidden_by_good_field_diagnostics():
    from src.empirical_isopach.models.mass_mixture.study import assess_posterior
    d = {"passed": True, "max_label_invariant_rhat": 1.001,
        "min_label_invariant_bulk_ess": 1000, "min_label_invariant_tail_ess": 1000,
        "divergences": 0, "max_depth_fraction": 0., "min_energy_bfmi": .8}
    rng = np.random.default_rng(67)
    epsilon = rng.normal(np.array([.1, .1, .5, .5])[:, None], .01, (4, 1000))
    assessed = assess_posterior(d, {"contamination": epsilon})
    assert not assessed["passed"]
    assert assessed["observation_mixture"]["rhat"] > 1.01
    assert d["passed"] and d["max_label_invariant_rhat"] == 1.001
