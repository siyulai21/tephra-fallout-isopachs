"""Numerical and cartographic regressions for matched global deposit maps."""
import numpy as np
from scipy.integrate import quad

from src.empirical_isopach.visualization.deposit_field import PolarField,doubling_levels
from src.empirical_isopach.visualization.deposit_integrals import within_cells
from src.empirical_isopach.visualization.deposit_maps import geography


def test_detached_ring_area_and_uncut_volume():
    radii=np.arange(5.)
    log=np.array([-1.,-1.,1.,1.,-1.])
    field=PolarField(radii,np.broadcast_to(log[:,None],(5,24)),1.05)
    volume, areas=within_cells(field,[1.,16.],order=12)
    np.testing.assert_allclose(areas,[np.pi*(3.5**2-1.5**2),0.])
    np.testing.assert_allclose(field.outer_crossings(1.),3.5)
    expected=2*np.pi*1e-5*quad(lambda r:r*10.**np.interp(r,radii,log),0.,4.,points=radii[1:-1])[0]
    np.testing.assert_allclose(volume,expected,rtol=1e-12)
    np.testing.assert_array_equal(doubling_levels(4.),[.5,1.,2.,4.])


def test_whole_earth_exceedance_and_bearing_seam():
    f=PolarField(np.array([0.,1.,2.]),np.ones((3,16)),.98)
    vol,area=within_cells(f,[.5,10.,16.])
    np.testing.assert_allclose(vol,4*np.pi*10.e-5)
    np.testing.assert_allclose(area,[4*np.pi,4*np.pi,0.])
    np.testing.assert_allclose(f.outer_crossings(.5),2.)
    np.testing.assert_allclose(f.on_grid(np.array([-.0001,.0001]),np.array([1.])),1.)


def test_pinned_land_and_lakes_distinguish_source_geography():
    import shapely
    land,lakes=geography()
    assert shapely.contains_xy(land,23.72,37.98)  # Athens
    assert not shapely.contains_xy(land,25.,36.)  # Aegean Sea
    assert shapely.contains_xy(lakes,-87.4,44.)  # Lake Michigan
