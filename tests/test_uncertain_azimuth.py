from types import SimpleNamespace

import numpy as np

from src.empirical_isopach.models.directional.azimuth.field import (
    DirectionalSupport, log_shift, stratified_states,
)
from src.empirical_isopach.models.directional.azimuth.calibration import scores_for_member
from src.empirical_isopach.models.directional.azimuth.archive import phase_medians
from src.empirical_isopach.models.directional.ensemble.fields import Marginal
from src.empirical_isopach.core.observations import Observations
from src.empirical_isopach.core.distributions import LogMixture


def test_finite_rays_protect_sites_and_do_not_turn_a_bound_into_a_fan():
    xy = np.array([[0.,10.],[0.,20.],[40.,0.],[-13.,0.],[0.,-9.]])
    data = Observations(xy,np.zeros(5),[False,False,False,True,True],np.zeros(5),np.arange(5),
                        [False,False,False,False,True])
    g = SimpleNamespace(axis=1.,locate=lambda xy:(None,None,np.zeros((len(xy),3))))
    support = DirectionalSupport(data,g)
    np.testing.assert_allclose(support.evaluate(np.vstack((xy,[0.,0.])))[0],0,atol=1e-14)
    h,_ = support.evaluate([[0.,30.],[-30.,0.],[0.,-5.],[30.,0.]])
    assert h[0]>0 and h[1]>0 and h[2]<1e-14 and h[3]<1e-14
    np.testing.assert_allclose(support.evaluate([[-35.,-35.]],np.array([[.5,.25,.25]]))[0],0)
    np.testing.assert_allclose(np.ptp(support.evaluate([[0.,-50.],[0.,-500.]])[0]),0)


def test_angular_mean_is_one_median_can_contract_and_fan_can_extend():
    angles = np.arange(8192)*2*np.pi/8192
    draws = np.exp(log_shift(4.,.8,.7,angles))
    np.testing.assert_allclose(draws.mean(),1.,atol=2e-12)
    assert np.median(draws)<.3 and draws.max()>3
    np.testing.assert_array_equal(log_shift(8.,0.,.7,angles),np.zeros_like(angles))


def test_each_chain_retains_exact_structure_probabilities_and_reproducible_phases():
    member = SimpleNamespace(draws={"source":np.zeros((4,64))})
    p = [.4,.25,.2,.15]
    k,phase,w = stratified_states([member],p)
    k2,phase2,w2 = stratified_states([member],p)
    np.testing.assert_array_equal(k,k2)
    np.testing.assert_array_equal(phase,phase2)
    np.testing.assert_array_equal(w,w2)
    for kc,wc in zip(k.reshape(4,-1),w[0].reshape(4,-1),strict=True):
        np.testing.assert_allclose(np.bincount(kc,weights=wc,minlength=4),np.asarray(p)/4)


def test_zero_exposure_preserves_heldout_density_and_both_censoring_tails():
    data = Observations(np.zeros((3,2)),np.array([.3,.8,-1.]),[False,True,True],
                        np.zeros(3),np.arange(3),[False,False,True])
    mu = np.array([[.1,-.7,.3],[.2,.4,.2]])
    var = np.ones_like(mu)
    tested = scores_for_member(mu,var,data,np.zeros(3),32)
    expected = LogMixture(np.full(2,.5),mu,var).log_score(data.log_value,data.censored,data.lower_censored)
    np.testing.assert_allclose(tested,np.broadcast_to(expected[:,None],tested.shape),atol=1e-14)


def test_analytic_phase_median_for_a_single_component_including_atom():
    fake = SimpleNamespace(base=SimpleNamespace(distribution=lambda xy:Marginal(np.zeros((1,len(xy))),[1.])),
                           support=SimpleNamespace(evaluate=lambda xy:(np.array([0.,1.]),None)),
                           concentration_weights=np.array([0.,1.,0.,0.]))
    from scipy.special import i0
    np.testing.assert_allclose(phase_medians(fake,np.zeros((2,2))),[1.,1/i0(2.)],rtol=1e-12)

def test_nonzero_exposure_angular_marginal_matches_dense_predictive_integration():
    from scipy.special import log_ndtr,logsumexp
    from src.empirical_isopach.models.directional.azimuth.field import CONCENTRATIONS
    data = Observations(np.zeros((3,2)),[.25,2.,-.7],[False,True,True],np.zeros(3),np.arange(3),[False,False,True])
    means = np.array([[0.,.8,-2.],[-1.,3.,.25]])
    variance = np.array([[.42,.52,.62],[.9,.4,.3]])
    exposure = np.array([.8,.64,.9])
    got = scores_for_member(means,variance,data,exposure,128)
    phase = np.arange(8192)*2*np.pi/8192.
    for j,k in enumerate(CONCENTRATIONS):
        shift = log_shift(k,exposure[None,:],0.,phase[:,None])
        z = (data.log_value-means[:,None,:]-shift[None,:,:])/np.sqrt(variance[:,None,:])
        lp = -.5*(np.log(2*np.pi*variance[:,None,:])+z*z)
        lp[:,:,1] = log_ndtr(z[:,:,1])
        lp[:,:,2] = log_ndtr(-z[:,:,2])
        np.testing.assert_allclose(got[:,j],logsumexp(lp,axis=(0,1))-np.log(2*len(phase)),atol=1e-10)


def test_analytic_radial_deposit_checks_global_units_and_distinct_medians():
    from scipy.stats import qmc
    from src.empirical_isopach.models.directional.ensemble.integrals import polar_nodes
    from src.empirical_isopach.models.directional.geometry import RADIUS
    uv = qmc.Sobol(2,scramble=True,seed=41791).random_base2(16)
    _,area,r,theta = polar_nodes(uv,1.)
    scale,peak,k = 80.,120.,4.
    baseline = peak*np.exp(-r/scale)
    exposure = r/(r+scale)
    coherent = baseline*np.exp(log_shift(k,exposure,theta,1.35))
    central = baseline*np.exp(log_shift(k,exposure,theta,theta+np.pi/2))
    expected = 2*np.pi*peak*scale**2*(1-(1+2*RADIUS/scale)*np.exp(-2*RADIUS/scale))*1e-5
    np.testing.assert_allclose(np.mean(area*coherent)*1e-5,expected,rtol=2e-4)
    assert np.mean(area*central)*1e-5 < .6*expected
