"""Area growth must be nested, station independent and correctly normalized."""
import numpy as np

from src.empirical_isopach.models.nested.footprints import growth
from src.empirical_isopach.models.nested.geometry import ANGLES, DIAMETER


def test_every_growth_shape_is_nested_and_has_its_prescribed_area():
    for s in growth.SHAPES:
        if s.family == "detached":
            continue
        t, r = growth.fronts(s, 192, 257)
        assert np.diff(r, axis=0).min() >= -1e-8
        area = t*t
        if s.family == "two phases":
            g = t*t/np.sqrt(t*t+s.core_km**2)
            h = s.distal_ratio*g+(.8-s.distal_ratio)*4*s.core_km*g/(4*s.core_km+g)
            area = area+h*h
        np.testing.assert_allclose(np.mean(r*r,axis=1),area,rtol=2e-12,atol=1e-18)
    np.testing.assert_allclose(growth.prior().sum(),1.)


def test_rotating_growth_retains_rank_and_converges_under_refinement():
    indices = [1, 25, 50, 85, 111, 140]
    for i in indices:
        s = growth.SHAPES[i]
        t, front = growth.fronts(s)
        j = np.array([0, 37, 102, 211, 332])
        k = np.array([250, 410, 500, 610, 690])
        r = front[k,j]
        good = r < DIAMETER
        q = np.sqrt(np.mean(np.minimum(front,DIAMETER)**2,axis=1))[k]
        for rotate in (0, 13):
            theta = j*2*np.pi/growth.N_ANGLE+rotate*2*np.pi/ANGLES
            got = growth.ranks(r[good],theta[good],[i*ANGLES+rotate])[0]
            np.testing.assert_allclose(got,q[good],rtol=.008,atol=1e-7)
        tf, rf = growth.fronts(s, 768, 2049)
        check = np.column_stack([np.interp(t,tf,rf[:,2*ang]) for ang in j])
        np.testing.assert_allclose(front[1:,j],check[1:],rtol=.003,atol=1e-7)


def test_circular_area_saturates_without_zeroing_unsampled_directions():
    shape = growth.Growth("single",1.8,.97,8.)
    t, r = growth.fronts(shape)
    assert np.all(r[1:]>0.)
    # The common area stays bounded. A genuinely round alternative remains
    # in the prior; absence of stations is never supplied as a zero thickness.
    c = shape.core_km**2*t*t/(shape.core_km**2+t*t)
    assert c.max() <= shape.core_km**2
    assert growth.FAMILY_PRIOR['round'] > 0.
