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


def test_regional_replay_checks_directional_thinning_in_any_map_orientation():
    import pytest
    from benchmarks.refresh_figures import same_description

    reference = np.array([4., 3., 1., 0.])
    angle = np.deg2rad(57.)
    rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    for axes in (np.eye(2), rotation):
        def coefficients(east_north, *, radial=3., intercept=4.):
            return [intercept, radial, *(axes @ east_north)]

        target = coefficients(reference[2:])
        for field in ("stacked.components[0].model.beta", "spline.trend_beta"):
            same_description(coefficients([1., 2e-7]), target, field)
            with pytest.raises(AssertionError):
                same_description(coefficients([1., 2e-5]), target, field)
            with pytest.raises(AssertionError):
                same_description(coefficients([1., 0.], radial=3.01), target, field)
            with pytest.raises(AssertionError):
                same_description(coefficients([1., 0.], intercept=4.01), target, field)


def test_published_contour_counts_excursion_rows_and_respects_one_cm_floor():
    from types import SimpleNamespace
    from src.empirical_isopach.visualization.deposit_maps import outer_contour
    field = PolarField(np.arange(5.),np.tile(np.log10([.25,.25,4.,4.,.25])[:,None],(1,16)),1.)
    data = SimpleNamespace(xy=np.array([[0.,2.]]*19+[[0.,0.]]),row_ids=np.arange(20))
    record = outer_contour(field,data)
    assert (record["level_cm"],record["inside"],record["next_level_inside"])==(4.,19,0)
    assert record["target_met"]
    data.xy[0] = [0.,4.]
    record = outer_contour(field,data)
    assert (record["level_cm"],record["inside"],record["target_met"])==(1.,18,False)


def test_near_view_uses_distinct_proximal_sites_and_has_a_distal_only_ceiling():
    from types import SimpleNamespace
    from src.empirical_isopach.visualization.deposit_maps import local_half_width
    points = np.column_stack((np.arange(1.,9.),np.zeros(8)))
    sample = SimpleNamespace(xy=points)
    half_width = local_half_width(sample)
    sample.xy = np.concatenate((points,np.repeat(points,20,axis=0),points+1000.))
    assert local_half_width(sample)==half_width
    sample.xy = points+1000.
    assert local_half_width(sample)==100.


def test_integrating_pointwise_quartiles_differs_from_quantiles_of_total_volume():
    from types import SimpleNamespace
    from src.empirical_isopach.models.nested.report import MapView
    from src.empirical_isopach.visualization.thickness_band import field_quantiles
    # A thin western half accompanies a thick eastern half, and conversely.
    # Every draw has the same total, although thickness is uncertain at both sites.
    draws = np.array([[1.,9.],[1.,9.],[9.,1.],[9.,1.]])
    view = MapView(SimpleNamespace(latitude=0.,field=lambda xy:np.log(draws)))
    xy = np.array([[-1.,0.],[1.,0.]])
    q = field_quantiles(view,xy)
    np.testing.assert_allclose(q[1],view.central(xy))
    np.testing.assert_allclose((10.**q).sum(axis=1),[2.,6.,18.])
    np.testing.assert_array_equal(np.quantile(draws.sum(axis=1),[.25,.75]),[10.,10.])
