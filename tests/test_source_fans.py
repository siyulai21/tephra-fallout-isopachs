"""Geological shape constraints and independent area/likelihood identities."""
from dataclasses import replace
import numpy as np
import pytest
from scipy.integrate import quad
from scipy.special import logsumexp
from src.empirical_isopach.models.mass_mixture import source_geometry as sg
from src.empirical_isopach.models.mass_mixture.geometry import RADIUS, log_kernel, log_field
from src.empirical_isopach.models.mass_mixture.prior import BASELINE
from src.empirical_isopach.models.mass_mixture.inference import model_function, packed_data, observation_log_likelihood
from tests.test_mass_mixture import example_data, example_samples


@pytest.mark.parametrize('q', [.5, 1., 2.5])
def test_isotropic_limit_matches_existing_analytical_density(q):
    unit = np.array([[0., 0., 1.], [.6, 0., .8], [0., 0., -1.]])
    ls = np.log(np.array([30., 300., 3000.]))
    kw = np.array([.2, .3, .5])
    actual = sg.log_kernel(unit, ls, np.zeros((3, 4)), np.full(3, q), kw)
    expected = log_kernel(unit, ls, np.tile([0., 0., 1.], (3, 1)), np.full(3, q), .5, kernel_weights=kw)
    np.testing.assert_allclose(actual, expected, atol=1e-10)


@pytest.mark.parametrize('q', [.4, 1., 3.])
@pytest.mark.parametrize('angular', [np.array([1.3, -.6, .4, .9]), np.array([2.2, -.9])])
def test_directional_normalization_with_adaptive_angular_integral(q, angular):
    ls = np.log(900.)
    actual = sg.normalizers(np.array(ls), angular, np.array(q))
    for j in range(3):
        integrand = lambda theta: np.exp(sg.radial_normalizers(sg.directional_scales(np.array(ls), angular, np.atleast_1d(theta)), q)[j, 0]-actual[j])/(2*np.pi)
        total, error = quad(integrand, 0, 2*np.pi, epsabs=1e-10, epsrel=1e-10)
        assert total == pytest.approx(1., abs=2e-10)
        assert error < 1e-8


def source_samples():
    s = example_samples(1)
    s.update(angular=np.array([[[1.2, .1, .3, -.4], [-.5, 1., -.3, .6], [.7, -.9, .4, .2]]]),
             kernel_weights=np.array([[.2, .3, .5]]))
    return s


def test_every_fan_and_mixture_peaks_at_source_and_thins_along_rays():
    angle = np.linspace(0, 2*np.pi, 31, endpoint=False)
    radius = np.r_[0., np.geomspace(.01, 2*RADIUS*(1-1e-10), 100)]
    r, a = np.meshgrid(radius, angle)
    u = .5*(r/RADIUS)**2
    unit = np.stack((np.sqrt(u*(2-u))*np.cos(a), np.sqrt(u*(2-u))*np.sin(a), 1-u), axis=-1)
    values = log_field(source_samples(), unit.reshape(-1, 3)).reshape(r.shape)
    assert np.all(np.diff(values, axis=1) <= 1e-10)
    np.testing.assert_allclose(values[:, 0], values[0, 0], atol=1e-12)


@pytest.mark.parametrize('angular', [np.array([1.1, -.3, .2, .4]), np.array([2.2, -.9])])
def test_mass_partition_sampler_recovers_directional_mass_and_radial_cdf(angular):
    from scipy.stats import qmc
    ls, q = np.log(350.), 1.3
    draws = sg.mass_samples(ls, angular, q, np.array([0., 1., 0.]),
                           qmc.Sobol(3, scramble=True, seed=51).random_base2(17))
    np.testing.assert_allclose(np.linalg.norm(draws, axis=1), 1., atol=1e-14)
    angle = np.arctan2(draws[:, 1], draws[:, 0]) % (2*np.pi)
    norm = sg.normalizers(np.array(ls), angular, np.array(q))[1]
    for lower, upper in [(0., np.pi/2), (np.pi/2, np.pi), (np.pi, 2*np.pi)]:
        integrand = lambda theta: np.exp(sg.radial_normalizers(sg.directional_scales(np.array(ls), angular, np.atleast_1d(theta)), q)[1, 0]-norm)/(2*np.pi)
        expected = quad(integrand, lower, upper, epsabs=1e-10)[0]
        assert np.mean((angle >= lower) & (angle < upper)) == pytest.approx(expected, abs=4e-5)
    from scipy.special import gammainc
    radius = RADIUS*np.sqrt(2*(1-draws[:, 2]))
    scale = np.exp(sg.directional_scales(np.array(ls), angular, angle))
    cdf = gammainc(2., radius/scale)/gammainc(2., 2*RADIUS/scale)
    for value in [.1, .5, .9]:
        assert np.mean(cdf < value) == pytest.approx(value, abs=4e-5)


def test_quadrature_refinement_for_extreme_directionality_and_scale():
    angles = np.array([[4., -3., 2., 1.], [-3., 4., -1., 2.], [0., 0., 0., 0.]])
    for scale in [1., 300., 10000., 1e7]:
        for q in [.25, 1., 10.]:
            a = sg.normalizers(np.full(3, np.log(scale)), angles, np.full(3, q), 256)
            b = sg.normalizers(np.full(3, np.log(scale)), angles, np.full(3, q), 1024)
            np.testing.assert_allclose(a, b, atol=1e-7)


def test_elliptic_mixture_volume_by_independent_whole_earth_area_cells():
    from src.probabilistic_isopach.global_volume import EarthDomain, earth_cells
    from src.empirical_isopach.models.mass_mixture.geometry import to_sphere
    values = source_samples()
    values['angular'] = values['angular'][..., :2]
    values['log_scale'] = np.log([[500., 1200., 2400.]])
    domain = EarthDomain.wgs84(60.)
    total = 0.
    for xy, area in earth_cells(domain, [0., 100., 500., 2000., 5000., 9000., domain.rho_max], 96, 512, 512):
        total += np.exp(log_field(values, to_sphere(xy, 60.))[0])@area*1e-5
    assert total == pytest.approx(np.exp(values['log_volume'][0]), rel=2e-5)


@pytest.mark.parametrize('shape', ['harmonic', 'elliptic'])
def test_jax_numpy_source_likelihood_and_complete_joint_match(shape):
    from numpyro.infer.util import log_density
    import numpyro.distributions as dist
    data = example_data()
    p = replace(BASELINE, source_anchored=True, source_shape=shape)
    values = {k: v[0] for k, v in source_samples().items()}
    if shape == 'elliptic':
        values['angular'] = values['angular'][:, :2]
    values.update(tail_log=0., exponent=2., offset=np.zeros((3, 2)))
    unit = {k: np.asarray(v)[None] for k, v in values.items()}
    unit['log_volume'] = np.zeros(1)
    from src.empirical_isopach.models.mass_mixture.geometry import to_sphere
    params = {k: v for k, v in values.items() if k not in ('log_volume','exponent','offset','tail_fraction')}
    params['field_level'] = values['log_volume']+log_field(unit, to_sphere(data.xy, 0)).mean()
    actual, trace = log_density(model_function(p), packed_data(data, 0), {}, params)
    likelihood = observation_log_likelihood({k: np.asarray(v)[None] for k, v in values.items()}, data, 0).sum()
    assert float(trace['observations']['fn'].log_prob(trace['observations']['value'])) == pytest.approx(likelihood, abs=1e-10)
    expected = (dist.Normal(p.log_volume_mean,p.log_volume_sd).log_prob(values['log_volume'])
        +dist.Normal(p.log_scale_mean,p.log_scale_sd).log_prob(values['log_scale']).sum()
        +dist.Normal(0.,p.angular_sd).log_prob(values['angular']).sum()
        +dist.Dirichlet(np.ones(3)*p.fan_concentration/3).log_prob(values['weights'])
        +dist.Dirichlet(np.ones(3)*p.kernel_concentration/3).log_prob(values['kernel_weights'])
        +dist.Normal(p.tail_log_mean,p.tail_log_sd).log_prob(values['tail_log'])
        +dist.HalfNormal(p.noise_scale).log_prob(values['noise'])
        +dist.Beta(1.,9.).log_prob(values['contamination'])+likelihood)
    assert float(actual) == pytest.approx(float(expected), abs=1e-10)


def test_underresolved_heavy_fan_triggers_refit_without_changing_physical_prior(monkeypatch,tmp_path):
    from dataclasses import asdict
    from src.empirical_isopach.models.mass_mixture import study
    from src.empirical_isopach.models.mass_mixture.source_protocol import BASELINE
    prior=replace(BASELINE,components=1,angular_nodes=32,integrable_plane_tail=False)
    seen=[]
    def cached(data,latitude,p,directory,settings,attempts,progress):
        seen.append(p)
        samples={'log_volume':np.full((2,4),np.log(100.)),
            'log_scale':np.full((2,4,1),4.),'offset':np.zeros((2,4,1,2)),
            'angular':np.tile([2.5,0.],(2,4,1,1)),
            'weights':np.ones((2,4,1)),'kernel_weights':np.tile([.2,.3,.5],(2,4,1)),
            'exponent':np.full((2,4),.3),'angular_nodes':np.full((2,4),p.angular_nodes)}
        return samples,{'diagnostics':{'passed':True,'assessment':'test sampler'},
            'posterior_file':'attempt-1.npz','prior':p.describe()}
    monkeypatch.setattr(study,'fit_cached',cached)
    samples,fit=study.fit_checked(None,0,prior,tmp_path,progress=None)
    assert [p.angular_nodes for p in seen]==[32,64,128]
    physical=lambda p:{k:v for k,v in asdict(p).items() if k!='angular_nodes'}
    assert all(physical(p)==physical(prior) for p in seen)
    assert fit['posterior_file']=='quadrature-128/attempt-1.npz'
    assert len(fit['normalization_refinements'])==2
    assert not fit['normalization_refinements'][0]['source_checks']['normalization']['passed']
    assert fit['diagnostics']['passed']
    np.testing.assert_allclose(np.exp(samples['log_volume']),100.)
