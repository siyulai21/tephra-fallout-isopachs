"""Replay the source classifications, geographic predictions, and before/after median maps."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from benchmarks.minimum_fit_inputs import verify as verify_input
from benchmarks.publication import reported_edition, result_home
from src.empirical_isopach.core.audit import audited_data
from src.empirical_isopach.core.catalog import datasets
from src.empirical_isopach.core.fitting import map_observation_policy
from src.empirical_isopach.visualization.deposit_field import PolarField, digest
from src.empirical_isopach.visualization.principal import volume_rows

REGIONAL = ("stacked", "spline")
GLOBAL = ("directional", "spherical-mixture", "nested-isopachs")


def read(path):
    return json.loads(path.read_text())


def compare_frame(frame, path, write):
    if write:
        frame.to_csv(path, index=False)
    saved = pd.read_csv(path, keep_default_na=False, na_values=[""], dtype={"source_id": str})
    pd.testing.assert_frame_equal(frame, saved, check_dtype=False, rtol=2e-7, atol=1e-9)


def map_at_sites(home, family, data, identity):
    if family in REGIONAL:
        path = home / "map-display.json"
        if not path.exists():
            return None, None
        display = read(path)
        if display["input_sha256"] != identity:
            return None, None
        log = display["inner_contours"][family]["mapped_site_log10_cm"]
        indexed = pd.Series(log, index=display["accepted_row_ids"])
        return np.power(10., indexed.loc[data.row_ids].to_numpy()), None
    path = home / family / "deposit.json"
    if not path.exists():
        return None, None
    display = read(path)
    if display["input_sha256"] != identity:
        return None, None
    grid = home / family / "median-grid.npz"
    assert digest(grid) == display["median_grid_sha256"]
    assert digest(home / family / "deposit.png") == display["figure_sha256"]
    with np.load(grid) as a:
        field = PolarField(a["radii_km"], a["log10_cm"], float(a["axis"]))
    return 10. ** field.at_xy(*data.xy.T), float(10. ** field.at_xy(0., 0.))


def predictions(home, family, data):
    if family in REGIONAL:
        held = pd.read_csv(home / "heldout_predictions.csv").set_index("row_id").loc[data.row_ids]
        scores = held[family + "_log_score"]
        above_min = held[family + "_median"] >= data.log_value
        covered = (held[family + "_lower"] <= data.log_value) & (data.log_value <= held[family + "_upper"])
    else:
        packet = {"directional": "multiscale", "spherical-mixture": "extent",
                  "nested-isopachs": "footprints"}[family]
        held = pd.read_csv(home / family / packet / "heldout.csv.gz")
        if "design" in held:
            held = held[held.design == "geographic"]
        held = held.set_index("row_id").loc[data.row_ids]
        if "evidence" in held:
            np.testing.assert_array_equal(held.evidence == "exact", ~data.censored)
        else:
            np.testing.assert_array_equal(held.exact, ~data.censored)
        scores = held["log_score" if family == "nested-isopachs" else "principal_log_score"]
        cdf = held["cdf" if family == "nested-isopachs" else "principal_cdf"]
        above_min, covered = cdf <= .5, cdf.between(.025, .975)
    assert held.index.is_unique and len(held) == len(data.xy)
    return held.fold.to_numpy(), scores.to_numpy(), above_min.to_numpy(), covered.to_numpy()


def assess(record, write=False):
    verify_input(record)
    data, meta = audited_data(record["path"])
    previous = reported_edition(record)
    home, old_home = result_home(record), result_home(previous)
    source_hash = meta["reported_source_sha256"]
    rows = pd.read_csv(previous["path"], encoding="utf-8-sig", keep_default_na=False)
    rows.index += 2
    reported = rows.loc[data.row_ids]
    kind = np.where(data.reported_minimum, "lower_limit",
                    np.where(data.interval, "interval", np.where(data.censored, "upper_limit", "measurement")))
    tags = pd.Series(reported.get("Kind","measurement"), index=reported.index).replace("", "measurement")
    np.testing.assert_array_equal(tags, kind)
    distances = np.hypot(*data.xy.T)
    points, stats = [], []
    policy = map_observation_policy(data, meta)["observation_policy"]
    families = (*REGIONAL, *GLOBAL)
    pairs = volume_rows(home, input_sha256=meta["input_sha256"])
    assert {r["method"] for r in pairs} == set(families)
    common_folds = None
    for family in families:
        receipt = read(home / "map-display.json" if family in REGIONAL else home / family / "deposit.json")
        assert receipt["observation_policy"] == policy
        new, vent = map_at_sites(home, family, data, meta["input_sha256"])
        old, old_vent = map_at_sites(old_home, family, data, source_hash)
        assert new is not None
        fold, score, above, covered = predictions(home, family, data)
        if common_folds is None:
            common_folds = fold
        np.testing.assert_array_equal(fold, common_folds)
        near = data.reported_minimum & (distances <= 25.)
        stats.append(dict(method=family, reported_measurements=int((~data.censored & ~data.reported_minimum).sum()),
                          reported_minima=int(data.reported_minimum.sum()), minima_within_25_km=int(near.sum()),
                          old_vent_cm=old_vent, new_vent_cm=vent,
                          old_max_at_minima_within_25_km_cm=float(old[near].max()) if old is not None and near.any() else None,
                          new_max_at_minima_within_25_km_cm=float(new[near].max()) if near.any() else None,
                          heldout_coverage95_reported_measurements=float(covered[kind=="measurement"].mean()),
                          heldout_coverage95_minimum_as_point=float(covered[data.reported_minimum].mean()),
                          heldout_fraction_medians_at_or_above_minimum=float(above[data.reported_minimum].mean())))
        for j, (row_id, row) in enumerate(reported.iterrows()):
            points.append(dict(method=family, row_id=row_id, source_id=row.get("EvidenceID") or f"row-{row_id}",
                               latitude_deg=float(row.Lat), longitude_deg=float(row.Lon),
                               projected_distance_km=distances[j], reported_kind=kind[j],
                               fit_kind="measurement" if data.reported_minimum[j] else kind[j],
                               reported_measurement_cm=np.exp(data.log_value[j]) if kind[j] == "measurement" else None,
                               reported_lower_limit_cm=np.exp(data.log_value[j]) if kind[j] == "lower_limit" else None,
                               reported_upper_limit_cm=np.exp(data.log_value[j]) if kind[j] == "upper_limit" else None,
                               interval_lower_cm=np.exp(data.log_value[j]) if data.interval[j] else None,
                               fitted_point_cm=np.exp(data.log_value[j]) if not data.censored[j] else None,
                               interval_upper_cm=np.exp(data.interval_log_upper[j]) if data.interval[j] else None,
                               previous_map_cm=old[j] if old is not None else None, current_map_cm=new[j],
                               geographic_fold=int(fold[j]), heldout_log_score=score[j],
                               heldout_probability_of_upper_limit_or_interval=np.exp(score[j]) if data.censored[j] else None))
    compare_frame(pd.DataFrame(points), home / "reported-and-fitted-sites.csv", write)
    compare_frame(pd.DataFrame(stats), home / "minimum-fit-comparison.csv", write)
    return stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true")
    parser.add_argument("--case", action="append", default=[])
    args = parser.parse_args()
    for r in datasets():
        if r.get("reported_file") and (not args.case or r["slug"] in args.case):
            assess(r, args.write)
            print(r["name"], "five maps, all geographic holdouts and source categories verified")
