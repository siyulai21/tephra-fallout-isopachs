"""Independent numerical and design checks on the spherical continuation."""
import numpy as np
import pytest
from numpy.polynomial.legendre import leggauss
from scipy.stats import qmc

from benchmarks.mass_mixture import case
from src.empirical_isopach.models.mass_mixture.bearings.field import BearingSupport, multiplier, phases
from src.empirical_isopach.models.mass_mixture.bearings.volume import integrate_draw
from src.empirical_isopach.models.mass_mixture.geometry import RADIUS, log_field, to_sphere


def test_angular_envelope_matches_direct_rays_and_preserves_training():
    data,*_=case("missing_sector",11)
    g=BearingSupport(data,0.)
    p=data.xy*g.scale
    r=np.linalg.norm(p,axis=1)
    far=r.argmax()
    theta=np.r_[np.linspace(-.02,2*np.pi+.03,371),np.arctan2(p[far,0],p[far,1])]
    dot=np.clip(np.column_stack((np.sin(theta),np.cos(theta)))@(p/r[:,None]).T,0,1)
    np.testing.assert_allclose(g.reach(theta),np.max(r*dot**g.power,axis=1),rtol=1e-12,atol=1e-11)
    np.testing.assert_allclose(g.evaluate(data.xy)[0],0,atol=1e-15)
    np.testing.assert_equal(g.evaluate([[0.,0.]])[0],[0.])
    np.testing.assert_allclose(g.evaluate(g.mesh.triangles.mean(1))[0],0,atol=1e-15)
    assert g.evaluate([[150.,150.]])[0][0]>0


def test_upper_limit_anchors_its_own_location_and_does_not_extend_positive_reach():
    from src.empirical_isopach.core.observations import Observations
    d,*_=case("missing_sector",10)
    xy=np.array([700.,700.])
    def augmented(lower):
        return Observations(np.vstack([d.xy,xy]),np.r_[d.log_value,0.],np.r_[d.censored,True],np.r_[d.log_sd,0.],
                            np.r_[d.row_ids,1000],np.r_[d.lower_censored,lower])
    original,upper,lower=(BearingSupport(x,0.) for x in (d,augmented(False),augmented(True)))
    theta=np.pi/4
    assert upper.reach(theta)==original.reach(theta) and lower.reach(theta)>upper.reach(theta)*3
    assert upper.evaluate([xy])[0][0]==pytest.approx(0,abs=1e-15)
    assert upper.evaluate([xy*1.5])[0][0]>.5


@pytest.mark.parametrize("source",[False,True])
def test_integrated_retention_matches_independent_surface_area_quadrature(source):
    data,*_=case("missing_sector",10)
    support=BearingSupport(data,0.)
    draw={"log_volume":np.log(100.),"exponent":2.3,"log_scale":np.log([70.,190.]),"offset":np.array([[0.,.2],[.8,-.5]]),
          "weights":np.array([.4,.6]),"tail_fraction":.3,"exponential_fraction":.4,
          "kernel_weights":np.array([.42,.28,.3])}
    if source:
        draw["angular"]=np.array([[.55,.2],[-.25,.4]])
    p,w=leggauss(190)
    r=50.*np.expm1((p+1)*np.log1p(2*RADIUS/50.)/2)
    dr=w*(r+50.)*np.log1p(2*RADIUS/50.)/2
    theta=(np.arange(512)+.5)*2*np.pi/512
    rr,tt=np.meshgrid(r,theta)
    xy=np.column_stack((rr.ravel()*np.sin(tt.ravel())*support.axis,rr.ravel()*np.cos(tt.ravel())/support.axis))
    f={k:np.asarray(v)[None,...] for k,v in draw.items()}
    field=np.exp(log_field(f,to_sphere(xy,0.))[0])/1e7
    h,b=support.evaluate(xy)
    area=np.broadcast_to(dr*r*(2*np.pi/512),rr.shape).ravel()
    truth=np.array([area@(field*multiplier(h,b,.59,power)) for power in (0,1,4)])
    u=qmc.Sobol(4,scramble=True,seed=719).random_base2(16)
    estimates,groups,_=integrate_draw(draw,support,.59,u,0,200,True)
    np.testing.assert_allclose(truth[0],1.,atol=2e-5)
    np.testing.assert_allclose(estimates[-1,:3],truth,atol=.003)
    np.testing.assert_allclose(groups.sum(1),estimates[-1,:3],atol=1e-12)
    assert np.all(np.diff(truth)<0)


def test_unknown_fan_one_orientation_per_deposit_and_enhanced_stress():
    phi=phases((4,600))
    np.testing.assert_array_equal(phi,phases((4,600)))
    h=np.full_like(phi,.8)
    np.testing.assert_allclose(multiplier(h,phi,phi,4),1.)
    np.testing.assert_allclose(multiplier(h,phi+np.pi,phi,4),.2)
    assert multiplier(h,.2,phi,4,enhanced=True).mean()==pytest.approx(1.,abs=.005)
