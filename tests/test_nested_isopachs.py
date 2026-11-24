import numpy as np
import pytest
from scipy.integrate import quad
from scipy.special import ndtr

from src.empirical_isopach.core.observations import read_observations
from src.empirical_isopach.models.nested.geometry import ANGLES, DIAMETER, SHAPES, area_radius, ranks
from src.empirical_isopach.models.nested.inference import log_events
from src.empirical_isopach.models.nested.profile import above_slab, contour_radius, integrated, log_profile
from src.empirical_isopach.models.mass_mixture.displacement import prior_ratio


def test_resume_preserves_a_foreign_posterior(tmp_path):
    import json
    from src.empirical_isopach.core.catalog import datasets
    from src.empirical_isopach.models.nested.report import nested_dataset

    path = datasets()[0]["path"]
    receipt = tmp_path/"full.json"
    original = json.dumps({"identity":{"input_sha256":"other eruption","protocol":"earlier run","engine_sha256":"old"}})
    receipt.write_text(original)
    with pytest.raises(ValueError,match="Resume inputs or model differ"):
        nested_dataset(path,tmp_path,resume=True)
    assert receipt.read_text() == original


def test_closed_volume_matches_independent_quadrature_and_layers():
    for tail in (False,True):
        z = np.array([[.7,-.6,1.,-.4,.2]])
        outer = np.array([tail])
        def integrand(radius):
            return 2*np.pi*radius*1e-5*np.exp(log_profile(np.array([[radius]]),z,outer)[0,0])
        reference = quad(integrand,0,DIAMETER,points=[100*np.exp(-.6*1.6)],epsabs=1e-8)[0]
        volume = integrated(z,outer)[0]
        assert volume==pytest.approx(reference,rel=1e-8)
        r = contour_radius(z,outer,1.)[0]
        assert log_profile(np.array([[r]]),z,outer)[0,0]==pytest.approx(0.,abs=1e-12)
        upper = above_slab(z,outer,1.)[0]
        assert 0 <= volume-upper <= np.pi*DIAMETER**2*1e-5
        assert upper==pytest.approx(quad(integrand,0,r,points=[min(r,100*np.exp(-.6*1.6))])[0]-np.pi*r*r*1e-5)


def test_irregular_shapes_normalize_and_remain_nested():
    for shape in SHAPES:
        levels = np.geomspace(1,DIAMETER/shape.floor,60)
        x = area_radius(shape,levels)
        assert np.all(np.diff(x)>=-1e-9)
        assert x[-1]==pytest.approx(DIAMETER,rel=.02)
        assert np.max(abs(x-area_radius(shape,levels,2048))/np.maximum(1,x))<.009
    r = np.array([1.,4.,300.,4000.])
    theta = np.array([0.,1.,2.,3.])
    np.testing.assert_allclose(ranks(r,theta,np.array([0]))[0],r,rtol=1e-12)
    np.testing.assert_allclose(ranks(r,theta,np.array([ANGLES+1])),
                               ranks(r,theta-2*np.pi/ANGLES,np.array([ANGLES])))


def test_two_sided_interval_is_one_event_and_survives_splitting(tmp_path):
    p=tmp_path/"bounds.csv"
    p.write_text("Lon,Lat,Vent Lon,Vent Lat,Thickness,Kind,Lower_cm,Upper_cm\n1,1,0,0,,interval,2,4\n0,0,0,0,1,measurement,,\n")
    with pytest.raises(ValueError,match="Two-sided"):
        read_observations(p,allow_intervals=False)
    rows,info=read_observations(p)
    sub=rows.subset(np.isfinite(rows.interval_log_upper))
    sd=.6
    expected=.95*(ndtr(np.log(4)/sd)-ndtr(np.log(2)/sd))+.05*(ndtr(np.log(4)/(3*sd))-ndtr(np.log(2)/(3*sd)))
    assert np.exp(log_events(np.zeros((1,1)),np.array([sd]),sub)[0,0])==pytest.approx(expected)
    assert info["n_intervals"]==1 and info["n_measurements"]==1 and info["n_lower_limits"]==0


def test_soft_displacement_identity_and_penalty():
    d={"weights":np.array([[1.],[1.]]),"offset":np.array([[[0.,0.]],[[2.,0.]]]),"log_volume":np.zeros(2)}
    np.testing.assert_allclose(prior_ratio(d),[1,.5])
    np.testing.assert_array_equal(prior_ratio({**d,"angular":np.zeros(2)}),[1,1])


def test_smooth_distal_taper_matches_independent_quadrature_and_preserves_sites():
    from scipy.integrate import quad
    from src.empirical_isopach.models.nested.distal import edges, ranked_sites, tapered_volumes
    from src.empirical_isopach.models.nested.geometry import DIAMETER
    from src.empirical_isopach.models.nested.inference import Deposit
    from src.empirical_isopach.models.nested.profile import log_profile, parameters
    z = np.array([[.1,.2,0.,-.5,0.],[-.1,1.2,0.,.2,0.]])
    model = Deposit(z,np.array([0,65]),np.array([True,False]),30.)
    xy = np.array([[0.,0.],[15.,25.],[-30.,60.]])
    edge = edges(model,xy)
    r,ids = ranked_sites(model,xy)
    assert np.all(r[ids]<=edge[:,None])
    values = tapered_volumes(model,edge,1.,64)
    assert np.all(values<tapered_volumes(model,edge,4.,64))
    assert np.all(values<model.volume())
    for j in range(2):
        def radial(x):
            rr = np.array([[x]])
            lt = log_profile(rr,z[j:j+1],model.power_tail[j:j+1])[0,0]
            return 2*np.pi*1e-5*x*np.exp(lt-(max(x/edge[j]-1,0.))**2)
        ref = quad(radial,0,DIAMETER,points=[edge[j],parameters(z)[1][j]],epsabs=1e-9)[0]
        assert values[j] == pytest.approx(ref,rel=1e-8)


@pytest.mark.parametrize("tail",[False,True])
def test_empirical_distal_budget_normalizes_and_inverts(tail):
    from src.empirical_isopach.models.nested.profile import ODDS_CENTRE, ODDS_SPREAD, parameters
    z = np.array([[.3,-1.,-.5,1.1,0.,.8],[-.2,1.,.4,-1.2,0.,-1.]])
    branch = np.full(len(z),tail)
    h,L,b,_,_,_ = parameters(z)
    for j in range(len(z)):
        case, law = z[j:j+1], branch[j:j+1]
        def radial_log(logr):
            r = np.exp(logr)
            return 2*np.pi*1e-5*r*r*np.exp(log_profile(np.array([[r]]),case,law)[0,0])
        reference = quad(radial_log,-45.,np.log(DIAMETER),points=[np.log(L[j])],epsabs=1e-9)[0]
        assert integrated(case,law)[0] == pytest.approx(reference,rel=1e-9)
        core = integrated(case,law,L[j])[0]
        # The reference continuation extends in equivalent-area radius.
        radius = L[j]*1e14
        if not tail:
            odds = integrated(case,law,radius)[0]/core-1
            assert np.log(odds)==pytest.approx(ODDS_CENTRE+ODDS_SPREAD*z[j,3])
        for r in (.7*L[j],2*L[j]):
            cm = np.exp(log_profile(np.array([[r]]),case,law)[0,0])
            assert contour_radius(case,law,cm)[0]==pytest.approx(r)


def test_remaining_donor_mass_and_whole_earth_continuation():
    from src.empirical_isopach.core.observations import Observations
    from src.empirical_isopach.models.nested.inference import Deposit
    from src.empirical_isopach.models.nested.continuation import join,residual_scaled,slab_for
    for u in (.8,1.2,4.):
        curve = lambda x: np.exp(-3*min(x,1.)-.5*max(0.,x-1.))
        inside = quad(lambda x:x*curve(x),0.,u,points=[min(u,1.)])[0]
        outside = quad(lambda x:x*curve(x),u,np.inf)[0]
        assert residual_scaled(-np.log(curve(u)),3.,.5) == pytest.approx(outside/inside/curve(u),rel=1e-8)
    xy = np.array([[4.,2.],[20.,45.],[-70.,-39.],[106.,10.]])
    data = Observations(xy,np.ones(4),np.zeros(4,dtype=bool),np.zeros(4),np.arange(4))
    z = np.zeros((4,6)); z[:,3] = [-1.,0.,.7,1.]
    original = Deposit(z,np.array([0,51,105,300]),np.array([False,True,False,True]),25.)
    m = join(original,data)
    np.testing.assert_allclose(m.field(xy),original.field(xy),rtol=0,atol=1e-12)
    for i in range(4):
        one = m.subset(slice(i,i+1))
        def value(logr):
            r=np.exp(logr)
            return 2*np.pi*1e-5*r*r*np.exp(one.log_at(np.array([[r]]))[0,0])
        reference = quad(value,-40,np.log(DIAMETER),points=sorted([np.log(100),np.log(one.edge[0])]),epsabs=1e-9)[0]
        assert one.volume()[0] == pytest.approx(reference,rel=1e-8)
        r = min(2*one.edge[0],.9*DIAMETER)
        assert one.radius_at(np.exp(one.log_at(np.array([[r]]))[0,0]))[0] == pytest.approx(r)
        assert 0 <= slab_for(one,.1)[0] < one.volume()[0]
