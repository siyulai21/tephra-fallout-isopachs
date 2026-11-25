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
    assert meta["reported_source_sha256"] == before.hex()
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


def test_matched_synthetic_benchmark_uses_the_current_observation_rule():
    from benchmarks.regional import synthetic_case
    data, _, _, _, _ = synthetic_case("censored", 0)
    assert data.reported_minimum.any() and data.censored.any()
    assert not np.any(data.lower_censored)
    np.testing.assert_array_equal(data.log_value[data.reported_minimum], 3.)
    np.testing.assert_array_equal(data.log_value[data.censored], 1.)


def test_global_training_uses_only_the_effective_point_or_censored_value():
    from src.empirical_isopach.core.observations import Observations
    from src.empirical_isopach.models.directional.multiscale.fit import fit
    from src.empirical_isopach.models.mass_mixture.inference import packed_data
    from src.empirical_isopach.models.nested.footprints.posterior import fit_field

    xy = np.array([[1, 2], [4, -1], [20, 10], [45, 25], [80, 4], [-30, 15]], float)
    values = np.log([12., 8., 5., 2., .6, 1.])
    reported = Observations(xy, values, [False, False, False, False, True, True],
                            np.zeros(6), np.arange(6), [False, False, False, False, False, True],
                            [np.nan] * 5 + [np.log(3.)], reported_minimum=[True, False, False, True, False, False])
    plain = replace(reported, reported_minimum=None)
    # Compare the arrays passed to the actual spherical sampler, including both interval endpoints.
    for marked, unmarked in zip(packed_data(reported, 15.), packed_data(plain, 15.), strict=True):
        np.testing.assert_array_equal(marked, unmarked)
    first, _ = fit(reported, 15., seed=57, power=6)
    second, _ = fit(plain, 15., seed=57, power=6)
    np.testing.assert_array_equal(first.central(xy), second.central(xy))
    first, _ = fit_field(reported, 15., initial=128, ceiling=128, seed=57)
    second, _ = fit_field(plain, 15., initial=128, ceiling=128, seed=57)
    for key, draws in vars(first).items():
        np.testing.assert_array_equal(draws, getattr(second, key))
