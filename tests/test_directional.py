"""Check the finite-Earth integral, mesh replacement, and observation likelihood."""
from dataclasses import replace

import numpy as np
import pytest
from scipy.integrate import quad
from scipy.stats import multivariate_normal, norm

from src.empirical_isopach.core.observations import Observations
from src.empirical_isopach.models.directional.geometry import BASE, Geometry, RADIUS
from src.empirical_isopach.models.directional.inference import background, field, flat, training_arrays
from src.empirical_isopach.models.directional.integration import integrate, make_rule, radial_integral
from src.empirical_isopach.support import TriangleRegion


def layout():
    x,y = np.meshgrid([10.,30.,50.,70.],[20.,40.,60.])
    xy = np.c_[x.ravel(), y.ravel()]
    return Observations(xy, .2+.02*xy[:,0], np.zeros(len(xy),bool), np.zeros(len(xy)), np.arange(len(xy)))


@pytest.mark.parametrize("shape", [0., 1e-8, .125, .25, .5-1e-10, .5, .5+1e-10, .85, .99999])
def test_radial_mass_against_independent_finite_earth_integration(shape):
    scale = 37.
    for lo, hi in ((0., 2*RADIUS),(53., 1432.),(12000., 12100.)):
        def f(u):
            radius = scale*np.expm1(u)
            z = radius/scale
            logt = -z if shape == 0 else -np.log1p(shape*z)/shape
            return radius*np.exp(logt)*scale*np.exp(u)
        expected = quad(f, np.log1p(lo/scale), np.log1p(hi/scale), epsabs=1e-14)[0]
        assert np.isclose(radial_integral(lo,hi,scale,shape), expected, rtol=1e-8, atol=1e-12)


def test_positive_polar_partition_against_independent_cartesian_area_integral():
    data = layout()
    g = Geometry(data, 58.)
    draws = {"source":np.array([[2.2]]), "log_range":np.array([[np.log(19.)]]),
             "fan_log_range":np.full((1,1,len(g.centers)), -200.), "fan_logit":np.array([[-1.]]),
             "tail_shape":np.array([[.25]]), "nodes":(.3+.025*g.vertices[:,0])[None,None,:]}
    actual = integrate(draws, g, make_rule(g, 6, 10))[0,0]
    p = flat(draws)
    base_total = 2*np.pi*np.exp(2.2)*radial_integral(0.,2*RADIUS,19.,.25)
    compact = TriangleRegion(g.triangles)
    xy,w = compact.quadrature(16)
    correction = w @ (np.exp(field(p,g,xy)[0])-np.exp(background(p,*g.coords(xy))[0]))
    assert actual.min() >= 0
    assert np.isclose(actual.sum(), (base_total+correction)*1e-5, rtol=1e-5)
    xy,w = g.mesh.quadrature(16)
    assert np.isclose(actual[0].sum(), w@np.exp(field(p,g,xy)[0])*1e-5,rtol=1e-5)
    # Replacing all vertex log thicknesses changes only the local patches.
    far = np.array([[-300.,-100.],[400.,310.]])
    assert np.allclose(field(p,g,far), background(p,*g.coords(far)))


def test_training_geometry_ignores_thickness_order_and_repeated_records():
    d = layout()
    g = Geometry(d,0)
    ix = np.r_[np.arange(len(d.xy))[::-1],0]
    other = d.subset(ix); other.log_value += 10.
    assert Geometry(other,0).describe() == g.describe()
    no_mesh = Geometry(d,0,replace(BASE, spacing_factor=1e-8))
    assert len(no_mesh.vertices) == 0
    assert not no_mesh.locate([[0.,0.]])[2].any()
    assert no_mesh.describe()["support"]["n_retained_triangles"] == 0


def test_repeated_node_and_limit_share_one_latent_thickness():
    import jax
    jax.config.update("jax_enable_x64",True)
    from numpyro.infer.util import log_density
    from src.empirical_isopach.models.directional.inference import _model
    from jax import numpy as jnp
    base = layout(); xy = base.xy[0]
    ix = np.r_[np.arange(len(base.xy)),0,0]
    d = base.subset(ix)
    d.censored[-1] = True; d.lower_censored[-1] = True
    d.log_sd[-2:] = [.2,.35]
    g = Geometry(d, 0)
    a, relevant = training_arrays(g,d)
    noise, ratio = .6, g.policy.local_sd_ratio
    z = {"source":jnp.array(2.),"log_range":jnp.log(35.),"fan_logit":jnp.array(-1.),
         "log_noise":jnp.log(noise),"log_multiplier":jnp.full(a["ang"].shape[1],-1.),
         "tail_aux":jnp.array(.25),"innovation":jnp.zeros(a["relevant"].shape)}
    j = int(a["at"][0]); assert j in relevant
    def expected(y):
        mu = background({"source":np.array([2.]),"log_range":np.log([35.]),
                         "fan_log_range":(np.log(g.reaches)-1)[None,:],"fan_logit":np.array([-1.]),
                         "tail_shape":np.array([.25])}, *g.coords(xy[None,:]))[0,0]
        vals = y[[0,len(d.xy)-2]]
        err = noise**2+np.array([0.,.2**2]); prior = (noise*ratio)**2
        node_var = 1/(1/prior+np.sum(1/err))
        m = mu+node_var*np.sum((vals-mu)/err)
        return multivariate_normal.logpdf(vals,[mu,mu],np.diag(err)+prior)+norm.logsf((y[len(d.xy)-1]-m)/np.hypot(noise,.35))
    first = float(log_density(_model,(),a,z)[0]); old = expected(a["y"])
    a["y"][len(d.xy)-2] += .8
    second = float(log_density(_model,(),a,z)[0])
    assert np.isclose(second-first,expected(a["y"])-old,atol=1e-8)



def test_published_estimates_match_their_draws_and_compared_data(monkeypatch):
    from pathlib import Path
    from benchmarks.verify_directional import main
    monkeypatch.chdir(Path(__file__).resolve().parents[1])
    main()


def test_an_isolated_distal_site_extends_only_its_nearby_bearings():
    from src.empirical_isopach.models.directional.geometry import angular_weights, fan_design
    vertices = np.array([[15.,30.],[25.,50.],[18.,60.],[30.,55.]])
    east = np.array([[700.,0.]])
    centers, reaches, widths, gap = fan_design(vertices, np.vstack((vertices,east)), 1.,np.pi/4,False)
    i = np.argmax(reaches)
    assert np.isclose(reaches[i],700.) and np.isclose(widths[i],np.pi/8)
    bearings = np.deg2rad([90.,105.,125.,180.,270.])
    weights = angular_weights(bearings,centers,widths)
    assert weights[0,i] == 1. and 0. < weights[1,i] < 1.
    assert np.all(weights[2:,i] == 0.)
    assert np.all(weights[-2:] == 0.)
    assert np.all(weights.sum(1) <= 1.+1e-13)
    hidden = fan_design(vertices,np.vstack((vertices,east)),1.,np.pi/4,True)
    angle = np.deg2rad(gap["midpoint_deg"])
    assert angular_weights(np.array([angle]),hidden[0],hidden[2])[0,-1] > 0.


def test_unsampled_proximal_addition_has_zero_likelihood_effect_and_exact_mass():
    from src.empirical_isopach.models.directional.source_cap import cap_kernel, cap_volumes
    # A central cross-site triangle covers the source, but the nearest record is a bound.
    xy = np.array([[-40.,-40.],[40.,-40.],[0.,45.],[-35.,35.],[8.,0.]])
    d = Observations(xy,np.ones(5),np.array([False]*4+[True]),np.zeros(5),np.arange(5))
    g = Geometry(d,0.)
    draws = {"source":np.array([[4.]]),"log_range":np.log([[35.]]),
             "fan_log_range":np.full((1,1,len(g.centers)),np.log(25.)),"fan_logit":np.array([[-1.]]),
             "tail_shape":np.array([[.25]]),"nodes":np.ones((1,1,len(g.vertices)))}
    a, vent, amplitude, extra = cap_volumes(d,g,draws)
    from src.empirical_isopach.models.directional.geometry import polar
    assert a == np.min(polar(xy,g.axis)[0])
    assert cap_kernel(polar(xy,g.axis)[0],a).max() == 0
    assert amplitude[0] > 0 and np.isclose(vent[0]+amplitude[0],np.exp(4.))
    independent = quad(lambda r: 2*np.pi*r*cap_kernel(r,a)*amplitude[0]*1e-5,0,a)[0]
    assert np.isclose(extra[0,0],independent,rtol=1e-12)
    d.xy[-1] = 0.
    nearest,_,amplitude,extra = cap_volumes(d,Geometry(d,0),draws)
    assert nearest == 0 and amplitude[0] == 0 and extra[0,0] == 0
