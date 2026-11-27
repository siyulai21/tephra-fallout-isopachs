"""Keep the reviewed B-Tm release covered alongside the frozen comparison studies."""
from pathlib import Path

import pytest

from benchmarks.btm_cryptotephra import NEW, OLD
from benchmarks.publication import BTM_THICKNESS_ONLY
from benchmarks.verify_btm_audit import verify_inputs, verify_results
from benchmarks.verify_btm_two_phase import verify as verify_two_phase
from src.empirical_isopach.core.geometry import survey_bounds
from src.empirical_isopach.core.observations import read_observations
from src.empirical_isopach.core.validation import spatial_folds, training_mask
from src.empirical_isopach.models.cubic_spline.model import SplineCandidate, TrendRegularizedSpline
from src.empirical_isopach.validation import automatic_buffer


def test_two_phase_source_and_fitting_edition():
    verify_two_phase()


def test_reviewed_btm_inputs_and_fallout_results(monkeypatch):
    monkeypatch.chdir(Path(__file__).resolve().parents[1])
    verify_results(verify_inputs())
    from benchmarks.verify_bearing_continuation import verify
    results=BTM_THICKNESS_ONLY/"results"
    verify(results/"directional/bearing",results/"spherical-mixture/posterior.npz")


def test_registered_btm_cryptotephra_release(monkeypatch):
    from benchmarks.verify_btm_cryptotephra import verify_inputs as verify_core_evidence, verify_release

    monkeypatch.chdir(Path(__file__).resolve().parents[1])
    verify_release(verify_core_evidence())


@pytest.mark.parametrize("path", [OLD, NEW], ids=["thickness-only","cryptotephra"])
def test_censored_spline_stationarity_in_kinashibetsu_revision(path):
    data, _ = read_observations(path)
    train, _ = training_mask(data, spatial_folds(data.xy, 4), 2, automatic_buffer(data)[0])
    candidate = SplineCandidate(tail="exponential", knot_intervals=12, tension=0.,
                                smoothing=10000., trend_strength=0., noise_ratio=1.5)
    model = TrendRegularizedSpline(candidate).fit(data.subset(train), bounds=survey_bounds(data.xy))
    assert 0 <= model.newton_decrement <= 1e-7


def test_core_profiles_add_one_upper_likelihood_per_site():
    from benchmarks.btm_cryptotephra import interpreted_limits, rows
    from src.empirical_isopach.core.audit import audited_data
    from src.empirical_isopach.models.directional.archive_readings import load_inventory

    old, _ = audited_data(OLD)
    new, meta = audited_data(NEW)
    assert len(new.xy) == len(old.xy) + 3
    assert rows(NEW)[:len(old.xy)] == rows(OLD)
    assert interpreted_limits() == {"longgang":.06, "quanyang":.64, "old-084":.45}
    for limit in meta["interpreted_upper_limits"]:
        index, = (new.row_ids == limit["row_id"]).nonzero()[0]
        assert new.censored[index] and not new.lower_censored[index]
        assert limit["limit_cm"] == interpreted_limits()[limit["id"]]
    inventory, _ = load_inventory(NEW)
    assert inventory["scenarios"] == [] and not inventory.get("validation_scenario")
