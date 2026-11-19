"""Keep the reviewed B-Tm release covered alongside the frozen comparison studies."""
from pathlib import Path

from benchmarks.verify_btm_audit import verify_inputs, verify_results
from src.empirical_isopach.core.catalog import resolve_dataset
from src.empirical_isopach.core.geometry import survey_bounds
from src.empirical_isopach.core.observations import read_observations
from src.empirical_isopach.core.validation import spatial_folds, training_mask
from src.empirical_isopach.models.cubic_spline.model import SplineCandidate, TrendRegularizedSpline
from src.empirical_isopach.validation import automatic_buffer


def test_reviewed_btm_inputs_and_fallout_results(monkeypatch):
    monkeypatch.chdir(Path(__file__).resolve().parents[1])
    verify_results(verify_inputs())
    from benchmarks.verify_bearing_continuation import verify
    results=Path("eruptions/b-tm/results")
    verify(results/"directional/bearing",results/"spherical-mixture/posterior.npz")


def test_censored_spline_stationarity_in_kinashibetsu_revision():
    data, _ = read_observations(resolve_dataset("B-Tm"))
    train, _ = training_mask(data, spatial_folds(data.xy, 4), 2, automatic_buffer(data)[0])
    candidate = SplineCandidate(tail="exponential", knot_intervals=12, tension=0.,
                                smoothing=10000., trend_strength=0., noise_ratio=1.5)
    model = TrendRegularizedSpline(candidate).fit(data.subset(train), bounds=survey_bounds(data.xy))
    assert 0 <= model.newton_decrement <= 1e-7
