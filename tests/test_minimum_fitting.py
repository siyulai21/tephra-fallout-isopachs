"""Guard the source, likelihood, held-out and model-plot meanings of the four observations."""
from dataclasses import replace
from hashlib import sha256

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm

from benchmarks.minimum_fit_inputs import verify
from src.empirical_isopach.core.audit import audited_data
from src.empirical_isopach.core.catalog import datasets
from src.empirical_isopach.core.distributions import LogMixture
from src.empirical_isopach.core.fitting import model_input
from src.empirical_isopach.support import SupportMesh
from src.empirical_isopach.visualization.deposit_maps import plot_observations
from src.empirical_isopach.visualization.maps import observations


def test_minimum_has_point_likelihood_and_symbol_while_bounds_keep_their_events(tmp_path):
    src = tmp_path / "field.csv"
    src.write_text("Lon,Lat,Vent Lon,Vent Lat,Kind,Thickness,Limit_cm,Lower_cm,Upper_cm,LogSD\n"
                   "1,1,0,0,measurement,4,,,,0.2\n2,2,,,lower_limit,,4,,,0.2\n"
                   "3,3,,,upper_limit,,4,,,0.2\n4,4,,,interval,,,2,6,0.2\n")
    before = sha256(src.read_bytes()).digest()
    target = model_input(src, tmp_path / "run")
    assert target != src and model_input(target, tmp_path / "again") == target
    assert sha256(src.read_bytes()).digest() == before
    data, meta = audited_data(target)
    order = np.argsort(data.row_ids)
    points = data.subset(order)
    np.testing.assert_array_equal(points.censored, [False, False, True, True])
    np.testing.assert_array_equal(points.reported_bounds().censored, [False, True, True, True])
    np.testing.assert_array_equal(points.reported_minimum, [False, True, False, False])
    np.testing.assert_array_equal(points.log_sd, [0.2] * 4)
    assert meta["minimum_as_measurement_rows"] == [3]
    field = LogMixture([1.], np.zeros((1, 4)), np.ones((1, 4)))
    expected = [norm.logpdf(np.log(4)), norm.logpdf(np.log(4)), norm.logcdf(np.log(4)),
                np.log(norm.cdf(np.log(6)) - norm.cdf(np.log(2)))]
    np.testing.assert_allclose(field.score(points), expected)
    fig, ax = plt.subplots()
    try:
        bounds = [*points.xy.min(axis=0) - 1, *points.xy.max(axis=0) + 1]
        handles, count = plot_observations(ax, points, meta, [1, 5], ["blue", "red"], bounds)
        assert count == 4 and [h.get_marker() for h in handles] == ["o", "v", "s", "*"]
        np.testing.assert_allclose(ax.collections[0].get_offsets(), points.xy[:2])
        ax.clear()
        observations(ax, points)
        np.testing.assert_allclose(ax.collections[0].get_offsets(), points.xy[:2])
    finally:
        plt.close(fig)


def test_all_registered_fitting_rows_recover_the_original_field_evidence():
    for record in datasets():
        verify(record)


def test_geometry_uses_the_same_points_as_independently_reported_measurements():
    mazama = next(r for r in datasets() if r["slug"] == "mazama")
    fit, _ = audited_data(mazama["path"])
    equivalent = replace(fit, reported_minimum=None)
    np.testing.assert_array_equal(SupportMesh(fit).xy, SupportMesh(equivalent).xy)
    # Both the interpolation footprint and the candidate contour levels use all fitted points.
    from src.empirical_isopach.contour_support import candidate_levels
    np.testing.assert_array_equal(candidate_levels(fit), candidate_levels(equivalent))
    np.testing.assert_array_equal(SupportMesh(fit).triangles, SupportMesh(equivalent).triangles)
