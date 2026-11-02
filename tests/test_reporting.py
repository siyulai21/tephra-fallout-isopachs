"""Numerical reporting checks independent of the fitted model implementation."""
import numpy as np
import pytest

from src.probabilistic_isopach.contours import extract_levels, closed_domain, split_antimeridian
from src.probabilistic_isopach.data import Observations
from src.probabilistic_isopach.volume import converged_volume


def test_closed_contour_area_matches_analytic_circle_and_rejects_cut_contours():
    x = np.linspace(-4, 4, 401)
    X, Y = np.meshgrid(x, x)
    log_surface = -(X**2+Y**2)
    records, rings = extract_levels(x, x, log_surface, [np.exp(-20), np.exp(-4)])
    assert records[0]["status"] == "extends_beyond_map"
    assert records[0]["area_km2"] is None
    assert records[1]["area_km2"] == pytest.approx(4*np.pi, rel=1e-4)
    for ring in rings[np.exp(-4)]:
        np.testing.assert_allclose(ring[0], ring[-1])


def test_enclosed_hole_is_subtracted_from_isopach_area():
    x = np.linspace(-4, 4, 501)
    X, Y = np.meshgrid(x, x)
    # log T = -(r² - 4)²; T >= exp(-1) is the annulus 3 <= r² <= 5.
    log_surface = -(X**2+Y**2-4)**2
    records, rings = extract_levels(x, x, log_surface, [np.exp(-1)])
    assert len(rings[np.exp(-1)]) == 2
    assert records[0]["area_km2"] == pytest.approx(2*np.pi, rel=.002)


class ConstantSurface:
    def __init__(self):
        self.data = Observations([[-10, -10], [10, -10], [10, 10], [-10, 10]],
                                 [0, 0, 0, 0], [False]*4, [0]*4, [1, 2, 3, 4])

    def predict_log(self, xy):
        return np.full(len(xy), np.log(2.)), np.full(len(xy), .4)


def test_volume_units_and_mean_versus_median_are_distinct():
    model = ConstantSurface()
    result = converged_volume(model, [-10, -10, 10, 10], resolutions=(8, 16))
    assert result["converged"]
    assert result["median_surface_km3"] == pytest.approx(400*2*1e-5)
    assert result["mean_km3"] == pytest.approx(400*2*np.exp(.2)*1e-5)
    assert result["fraction_expected_volume_outside_data_hull"] == 0


def test_domain_search_cannot_fabricate_a_closed_curve():
    with pytest.raises(ValueError, match="No requested contour closes"):
        closed_domain(ConstantSurface(), levels=[.1, 1.], max_extent_km=100)


def test_antimeridian_export_has_no_spurious_world_spanning_segment():
    lines = split_antimeridian(np.array([[179., 10.], [-179., 12.], [-177., 13.]]))
    assert lines[0][-1] == [180., 11.]
    assert lines[1][0] == [-180., 11.]
    for line in lines:
        assert np.all(np.abs(np.diff(np.array(line)[:, 0])) <= 180)
