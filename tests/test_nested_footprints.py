"""Physical nesting, correct area ranks, station independence and protection against wrong inputs."""
import numpy as np

from src.empirical_isopach.models.nested.footprints import geometry as geo
from src.empirical_isopach.models.nested.geometry import ANGLES, DIAMETER
from src.empirical_isopach.models.nested.footprints.posterior import FootprintField
from src.empirical_isopach.models.nested.continuation import join
from src.empirical_isopach.models.nested.inference import Deposit
from src.empirical_isopach.core.observations import Observations
from src.probabilistic_isopach.global_volume import EarthDomain


def test_footprints_enclose_their_inner_levels_and_count_clipped_area():
    theta = np.arange(720)*np.pi/360
    levels = np.geomspace(.05,20*DIAMETER,180)[:,None]
    shapes = [geo.SHAPES[i] for i in (0,3,9,11,18,27,34)]
    for shape in shapes:
        reach = geo.main_front(shape,levels,theta)
        assert np.min(np.diff(reach,axis=0)) >= -1e-8
        area = geo.equivalent_radius(shape,levels[:,0],720)
        assert np.min(np.diff(area)) >= -1e-7
        assert area[-1] <= DIAMETER*(1.+1e-12)
    for shape in shapes[:4]:
        if shape.family in ("single","turning"):
            np.testing.assert_allclose(geo.equivalent_radius(shape,np.array([1.,7.,30.,220.])),
                                       [1.,7.,30.,220.],rtol=1e-10)


def test_proximal_round_limit_and_finite_secondary_reach_are_area_preserving():
    from dataclasses import replace
    bearings = 2*np.pi*np.arange(720)/720
    base = next(shape for shape in geo.SHAPES if shape.family == "two phases")
    t = np.array([.00001, 2., 40., 1100.])
    g = t*t/(t+base.core_km)
    for fraction in (0., .3, .8):
        shape = replace(base, distal_ratio=fraction)
        r = geo.main_front(shape, t[0], bearings)
        np.testing.assert_allclose(r, t[0], rtol=1e-9)
        h = fraction*g + (.8-fraction)*4*base.core_km*g/(4*base.core_km+g)
        np.testing.assert_allclose(geo.equivalent_radius(shape, t, 1536), np.hypot(t,h), rtol=2e-10)
    assert geo.main_front(replace(base, distal_ratio=0.), 1100., np.deg2rad(base.second)) < geo.main_front(replace(base, distal_ratio=.8),1100., np.deg2rad(base.second))


def test_rank_coordinates_recover_a_contour_under_rotation_and_sampled_disk_clipping():
    rng = np.random.default_rng(31421)
    theta = rng.uniform(-np.pi,np.pi,40)
    levels = np.geomspace(.07,2.5*DIAMETER,40)
    # Includes both the turning shape and the junction between two phases.
    for shape_id in (0,5,10,17,23):
        shape = geo.SHAPES[shape_id]
        dist = geo.main_front(shape,levels,theta)
        ok = dist < DIAMETER
        expected = geo.equivalent_radius(shape,levels[ok],1536)
        for turn in (0,11,31):
            got = geo.ranks(dist[ok],theta[ok]+turn*2*np.pi/ANGLES,[shape_id*ANGLES+turn])[0]
            np.testing.assert_allclose(got,expected,rtol=.006,atol=1e-5)
    shape_id = next(i for i,s in enumerate(geo.SHAPES) if s.satellite_first)
    shape = geo.SHAPES[shape_id]
    summit = geo.ranks([shape.satellite],[np.deg2rad(shape.second)],[shape_id*ANGLES])
    np.testing.assert_allclose(summit,0.,atol=1e-8)
    assert geo.ranks([0.],[0.],[shape_id*ANGLES])[0,0]>0.


def test_fit_keeps_the_same_profile_along_an_intrinsic_level_without_station_rays():
    axis = EarthDomain.wgs84(0.).axis_scale
    xy = np.array([[0.,5.],[16.,35.],[40.,-22.],[-8.,-20.]])
    data = Observations(xy,np.log([13.,3.,2.,4.]),np.zeros(4,bool),np.zeros(4),np.arange(4))
    shape_id = 8
    raw = Deposit(np.zeros((2,6)),np.full(2,shape_id*ANGLES),np.array([False,True]),0.)
    field = FootprintField(**vars(join(raw,data,ranker=geo.ranks)))
    angles = np.arange(72)*np.pi/36.
    r = geo.main_front(geo.SHAPES[shape_id],28.,angles)
    xy = np.column_stack((axis*r*np.sin(angles),r*np.cos(angles)/axis))
    assert np.ptp(field.field(xy),axis=1).max()<.015
    assert np.isfinite(field.volume()).all()
    np.testing.assert_allclose(geo.prior().sum(),1.)
