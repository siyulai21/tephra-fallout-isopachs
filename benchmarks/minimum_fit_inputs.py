"""Verify every registered source-to-fit translation, optionally refreshing the declared fitting files."""
import argparse
import json

import numpy as np

from src.empirical_isopach.core.audit import audited_data
from src.empirical_isopach.core.catalog import data_directory, datasets
from src.empirical_isopach.core.fitting import POLICY, fitting_bytes


def verify(record, write=False):
    target = record["path"]
    source = data_directory() / record.get("reported_file", record["file"])
    effective = fitting_bytes(source)
    if source == target and effective != source.read_bytes():
        raise ValueError(f"{record['slug']}: register the reported table and its distinct fitting table")
    if write and source != target and (not target.exists() or target.read_bytes() != effective):
        target.write_bytes(effective)
    if target.read_bytes() != effective:
        raise ValueError(f"{target.name}: regenerate the fitting table from {source.name}")
    data, meta = audited_data(target)
    before, original = audited_data(source)
    ids = np.argsort(data.row_ids)
    restored = data.reported_bounds().subset(ids)
    raw = before.subset(np.argsort(before.row_ids))
    for key in ("xy", "row_ids", "log_value", "lower_censored", "censored", "interval_log_upper", "log_sd"):
        np.testing.assert_array_equal(getattr(raw, key), getattr(restored, key))
    minima = raw.lower_censored & ~raw.interval
    np.testing.assert_array_equal(minima, data.reported_minimum[ids])
    assert not np.any(data.lower_censored & ~data.interval)
    assert meta.get("reported_source_sha256", meta["input_sha256"]) == original["input_sha256"]
    return int(minima.sum())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    assert json.loads((data_directory() / "datasets.json").read_text())["observation_fitting_policy"]["id"] == POLICY
    for item in datasets():
        print(item["slug"], verify(item, args.write), "reported minima fitted as measurements")
