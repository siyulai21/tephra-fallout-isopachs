from dataclasses import replace
import numpy as np
import pytest
from src.empirical_isopach.models.mass_mixture import vent_envelope as ve
from src.empirical_isopach.models.mass_mixture.geometry import RADIUS, log_field, to_sphere
from src.empirical_isopach.models.mass_mixture.inference import model_function, packed_data, observation_log_likelihood
from src.empirical_isopach.models.mass_mixture.prior import BASELINE
from tests.test_mass_mixture import example_samples, example_data


def samples():
    s = example_samples(1)
    s.update(core_log_scale=np.log([80.]), envelope_strength=np.array([.3]))
    return s


def test_source_is_global_maximum_but_secondary_maximum_is_allowed():
    s = samples()
    s['core_log_scale'] = np.log([20.])
    s['log_scale'] = np.log([[100., 100., 100.]])
    s['offset'] = np.tile([[6., 0.]], (1, 3, 1))
    x = np.arange(0., 1201., 5.)
    y = log_field(s, to_sphere(np.column_stack((x, x*0)), 0))[0]
    assert y.argmax() == 0
    assert y[120] > y[60] and y[120] > y[180]
    rng = np.random.default_rng(31)
    unit = rng.normal(size=(10000, 3));unit /= np.linalg.norm(unit, axis=1)[:, None]
    assert log_field(s, unit).max() < y[0]


def test_envelope_volume_by_independent_whole_earth_quadrature():
    from src.probabilistic_isopach.global_volume import EarthDomain, earth_cells
    s = samples();s['core_log_scale'] = np.log([300.]);s['log_scale'] = np.log([[500.,1400.,4000.]])
    domain = EarthDomain.wgs84(60.)
    total = 0.
    for xy, area in earth_cells(domain, [0.,100.,500.,2000.,5000.,9000.,domain.rho_max],96,512,512):
        total += np.exp(log_field(s,to_sphere(xy,60.))[0])@area*1e-5
    assert total == pytest.approx(np.exp(s['log_volume'][0]),rel=1e-5)


def test_uniform_limit_and_coincident_or_opposite_components():
    assert np.exp(ve.gaussian_log_z(np.array(0.))) == pytest.approx(4*np.pi*RADIUS**2)
    for center in [np.array([[0.,0.,1.]]),np.array([[0.,0.,-1.]])]:
        z = ve.normalizer(np.log(1000.),np.log([1000.]),center,np.array([1.]),np.array(.5))
        assert np.isfinite(z)


def test_complete_joint_density_and_jax_gradient():
    import jax
    from numpyro.infer.util import log_density
    import numpyro.distributions as dist
    p=replace(BASELINE,vent_envelope=True)
    values={k:v[0] for k,v in samples().items()}
    data=example_data();unit={k:np.asarray(v)[None] for k,v in values.items()};unit['log_volume']=np.zeros(1)
    parameters={k:v for k,v in values.items() if k not in ('log_volume','exponent','tail_fraction')}
    parameters['field_level']=values['log_volume']+log_field(unit,to_sphere(data.xy,0)).mean()
    actual,trace=log_density(model_function(p),packed_data(data,0),{},parameters)
    likelihood=observation_log_likelihood({k:np.asarray(v)[None] for k,v in values.items()},data,0).sum()
    assert float(trace['observations']['fn'].log_prob(trace['observations']['value'])) == pytest.approx(likelihood,abs=1e-10)
    expected=(dist.Normal(p.log_volume_mean,p.log_volume_sd).log_prob(values['log_volume'])
        +dist.Normal(p.log_scale_mean,p.log_scale_sd).log_prob(values['log_scale']).sum()
        +dist.Normal(p.log_scale_mean,p.log_scale_sd).log_prob(values['core_log_scale'])
        +dist.Normal(0.,p.offset_sd).log_prob(values['offset']).sum()
        +dist.Dirichlet(np.ones(3)*p.fan_concentration/3).log_prob(values['weights'])
        +dist.Beta(1.,1.).log_prob(values['envelope_strength'])
        +dist.HalfNormal(p.noise_scale).log_prob(values['noise'])
        +dist.Beta(1.,9.).log_prob(values['contamination'])+likelihood)
    assert float(actual) == pytest.approx(float(expected),abs=1e-10)
    gradient=jax.grad(lambda level:log_density(model_function(p),packed_data(data,0),{},dict(parameters,field_level=level))[0])(parameters['field_level'])
    assert np.isfinite(float(gradient))


def test_supported_design_is_order_independent_and_training_only():
    from src.empirical_isopach.models.mass_mixture.supported_basis import design
    data=example_data()
    a,scales=design(data,0,3)
    b,other=design(data.subset([2,0,1]),0,3)
    np.testing.assert_array_equal(a,b)
    np.testing.assert_array_equal(scales,other)
    train=data.subset([0,1])
    selected,_=design(train,0,3)
    heldout=to_sphere(data.xy[[2]],0)
    assert not np.any(np.all(selected==heldout,axis=1))
    assert all(np.any(np.all(center==to_sphere(train.xy,0),axis=1)) for center in selected)


def test_supported_basis_likelihood_and_joint_density():
    from numpyro.infer.util import log_density
    import numpyro.distributions as dist
    from src.empirical_isopach.models.mass_mixture.supported_basis import design
    p=replace(BASELINE,vent_envelope=True,supported_basis=True)
    data=example_data();centers,scales=design(data,0,3)
    values={k:v[0] for k,v in samples().items()}
    width=-.4
    values.update(fixed_center=centers,log_scale=np.log(scales)+width,offset=np.zeros((3,2)))
    unit={k:np.asarray(v)[None] for k,v in values.items()};unit['log_volume']=np.zeros(1)
    params={k:values[k] for k in ('core_log_scale','envelope_strength','weights','noise','contamination')}
    params.update(basis_width=width,field_level=values['log_volume']+log_field(unit,to_sphere(data.xy,0)).mean())
    actual,trace=log_density(model_function(p),packed_data(data,0),{'basis_centers':centers,'basis_scales':scales},params)
    likelihood=observation_log_likelihood({k:np.asarray(v)[None] for k,v in values.items()},data,0).sum()
    assert float(trace['observations']['fn'].log_prob(trace['observations']['value'])) == pytest.approx(likelihood,abs=1e-10)
    expected=(dist.Normal(p.log_volume_mean,p.log_volume_sd).log_prob(values['log_volume'])
        +dist.Normal(p.basis_width_log_mean,p.basis_width_log_sd).log_prob(width)
        +dist.Normal(p.log_scale_mean,p.log_scale_sd).log_prob(values['core_log_scale'])
        +dist.Dirichlet(np.ones(3)*p.fan_concentration/3).log_prob(values['weights'])
        +dist.Beta(1.,1.).log_prob(values['envelope_strength'])
        +dist.HalfNormal(p.noise_scale).log_prob(values['noise'])
        +dist.Beta(1.,9.).log_prob(values['contamination'])+likelihood)
    assert float(actual) == pytest.approx(float(expected),abs=1e-10)


def test_positive_partition_matches_independent_concentric_cap_integral():
    from src.empirical_isopach.models.mass_mixture.posterior import mass_partition
    from src.empirical_isopach.core.observations import Observations
    s=samples();s['offset'][:]=0
    angles=np.arange(8)*np.pi/4
    xy=300*np.column_stack((np.cos(angles),np.sin(angles)))
    data=Observations(xy,np.zeros(8),np.zeros(8,bool),np.zeros(8),np.arange(8))
    result=mass_partition({k:v[None] for k,v in s.items()},data,0,draws=1,power=14)
    cutoff=1-np.cos(result['farthest_exact_site_km']/RADIUS)
    k0=RADIUS**2*np.exp(-2*s['core_log_scale'][0])
    k=RADIUS**2*np.exp(-2*s['log_scale'][0])
    def outside(concentration):
        return 2*np.pi*RADIUS**2/concentration*(np.exp(-concentration*cutoff)-np.exp(-2*concentration))
    total=np.exp(ve.normalizer(s['core_log_scale'][0],s['log_scale'][0],
        np.tile([0.,0.,1.],(3,1)),s['weights'][0],s['envelope_strength'][0]))
    expected=(outside(k0)+s['envelope_strength'][0]*np.sum(s['weights'][0]*(outside(k)-outside(k+k0))))/total
    actual=result['parts']['beyond_farthest_site']['fraction']['median']
    assert actual == pytest.approx(expected,abs=.001)
    assert result['max_positive_quadrature_mass_error_before_partition_normalization'] < .001
