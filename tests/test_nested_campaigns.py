"""Historical campaign CSVs must obey the current observation policy in new fits."""
import numpy as np

from benchmarks.nested_growth_challenges import survey, legacy_posterior
from benchmarks.nested_reach_challenge import EXTRA
from src.empirical_isopach.core.audit import audited_data


def test_historical_pinatubo_transfer_fits_all_minima_as_points(tmp_path, monkeypatch):
    original, source = audited_data(EXTRA)
    minima = set(original.row_ids[original.lower_censored & ~original.interval])
    upper = set(original.row_ids[original.censored & ~original.lower_censored])
    assert len(minima) == 13
    captured = []

    def inspect(folder, key, train, test, meta):
        captured.append((train, test, meta))
        return dict(numerics=dict(passed=True), heldout=[])

    monkeypatch.setattr(legacy_posterior, 'write_fit', inspect)
    survey(('pinatubo-1991', tmp_path, True))
    train, test, meta = captured[0]
    assert len(test.xy) == 10 and not set(train.row_ids) & set(test.row_ids)
    assert set(np.r_[train.row_ids, test.row_ids]) == set(original.row_ids)
    assert not any((part.lower_censored & ~part.interval).any() for part in (train, test))
    fitted_minima = set(np.r_[train.row_ids[train.reported_minimum], test.row_ids[test.reported_minimum]])
    fitted_upper = set(np.r_[train.row_ids[train.censored & ~train.lower_censored], test.row_ids[test.censored & ~test.lower_censored]])
    assert fitted_minima == minima and fitted_upper == upper
    points = {int(i): float(y) for part in (train, test) for i,y in zip(part.row_ids,part.log_value,strict=True)}
    assert all(points[int(i)] == y for i,y in zip(original.row_ids,original.log_value,strict=True))
    assert meta['reported_source_sha256'] == source['input_sha256'] != meta['input_sha256']
