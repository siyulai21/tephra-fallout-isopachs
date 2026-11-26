"""Continuous directions preserve areas, topology, and observation probabilities."""
import numpy as np

from src.empirical_isopach.core.observations import Observations
from src.empirical_isopach.models.nested import inference,profile
from src.empirical_isopach.models.nested.footprints import flow,growth,numerics
from src.empirical_isopach.models.nested.geometry import ANGLES


def test_fused_likelihood_matches_scipy_for_points_and_both_interval_tails():
    rng=np.random.default_rng(4142)
    data=Observations(np.zeros((6,2)),np.log([3.,1.,2.,5.,.1,20.]),
                      [False,True,True,True,True,True],np.array([0.,.1,.2,.3,0.,0.]),np.arange(6),
                      [False,False,True,True,True,True],
                      [np.nan,np.nan,np.nan,np.log(7.),np.log(.1001),np.log(25.)])
    z=rng.normal(size=(100,6))
    tail=rng.random(100)<.5
    r=np.exp(rng.normal(4,1.5,(100,6)))
    pars=np.array(profile.budget_parameters(z,tail))
    mu=profile.log_profile(r,z,tail)
    expected=inference.log_events(mu,pars[-1],data).sum(axis=1)
    actual=numerics.log_events(r,pars,tail,data.log_value,data.log_sd,data.censored,data.lower_censored,data.interval_log_upper)
    np.testing.assert_allclose(actual,expected,rtol=1e-9,atol=2e-7)
    assert np.isfinite(numerics.interval_event(1.e15,1.e-12))
    from scipy.integrate import quad
    for lo in (-40.,-2.,0.,1.,32.):
        width=1.e-10
        mid=lo+width/2
        reference=np.log(width)-.5*(mid*mid+np.log(2*np.pi))+np.log(
            quad(lambda u:np.exp(-mid*width*u-.5*(width*u)**2),-.5,.5)[0])
        np.testing.assert_allclose(numerics.interval_event(lo,width),reference,rtol=1e-13)


def test_training_and_prediction_use_the_identical_continuous_geography():
    rng=np.random.default_rng(442)
    r=np.geomspace(.01,12000,31)
    theta=rng.uniform(-np.pi,np.pi,len(r))
    g=rng.integers(0,len(growth.SHAPES)*ANGLES,24)
    g[-1]=(len(growth.SHAPES)-1)*ANGLES+4
    z=rng.normal(size=(len(g),8))
    table=flow.TrainingRanks(r,theta)
    expected=flow.ranks(r,theta,g,z[:,6:])
    np.testing.assert_allclose(table.at(z,g),expected,rtol=1e-11,atol=1e-9)


def test_angular_flow_preserves_area_and_nesting_without_requiring_star_convexity():
    # On each annulus, changing the angle is an exact translation of a periodic
    # integrand. Dense independent angular integration checks the full map.
    r=np.repeat([2.,30.,200.,1500.],4096)
    theta=np.tile(np.arange(4096)*2*np.pi/4096,4)
    geom=np.array([3*ANGLES+11,170*ANGLES+24])
    warp=np.array([[.7,1.4],[-1.1,-1.8]])
    before=growth.ranks(r,theta,geom)
    after=flow.ranks(r,theta,geom,warp)
    for q in (10.,100.,800.):
        area0=(before<=q).reshape(2,4,4096).mean(axis=-1)
        area1=(after<=q).reshape(2,4,4096).mean(axis=-1)
        np.testing.assert_allclose(area1,area0,atol=2/4096,rtol=0)
    assert np.all((after<10.) <= (after<100.))


def test_interrupted_refinement_resumes_only_verified_completed_populations(tmp_path,monkeypatch):
    import json
    import pytest
    from src.empirical_isopach.models.nested.footprints import continuous
    data=Observations(np.array([[1.,2.],[4.,5.]]),np.log([4.,2.]),
                      np.zeros(2,bool),np.zeros(2),np.arange(2))
    meta=dict(input_sha256="test-input",vent_lonlat=[0.,15.])
    def packet(n):
        return continuous.FlowField(np.zeros((4*n,6)),np.zeros(4*n,int),np.zeros(4*n,bool),15.,
                                    np.full(4*n,100.),np.ones(4*n),np.zeros(4*n,int),np.ones(4*n),
                                    4,np.zeros((4*n,2)))
    def attempt(n,passed):
        return dict(particles_per_population=n,intrinsic_passed=passed,
                    continuation=dict(passed=passed),spatial_populations=dict(passed=passed))
    def interrupted(*args,initial,ceiling,seed,on_attempt):
        assert (initial,ceiling)==(4,8)
        on_attempt(packet(4),attempt(4,False))
        raise KeyboardInterrupt
    monkeypatch.setattr(continuous,"fit_field",interrupted)
    with pytest.raises(KeyboardInterrupt):
        continuous.write_fit(tmp_path,"full",data,None,meta,initial=4,ceiling=8)
    assert not (tmp_path/"full.json").exists()
    def resumed(*args,initial,ceiling,seed,on_attempt):
        assert initial==ceiling==8
        model,check=packet(8),attempt(8,True)
        on_attempt(model,check)
        return model,dict(seed=seed,initial=initial,ceiling=ceiling,attempts=[check],passed=True)
    monkeypatch.setattr(continuous,"fit_field",resumed)
    receipt=continuous.write_fit(tmp_path,"full",data,None,meta,initial=4,ceiling=8)
    assert [a["particles_per_population"] for a in receipt["numerics"]["attempts"]]==[4,8]
    assert len(receipt["checkpoints"])==2 and receipt["numerics"]["initial"]==4
    assert json.loads((tmp_path/"full-progress.json").read_text())["identity"]==receipt["identity"]
    with pytest.raises(ValueError,match="differs"):
        continuous.write_fit(tmp_path,"full",data,None,{**meta,"input_sha256":"changed"},initial=4,ceiling=8)
