"""Replay the source classifications, geographic predictions, and before/after median maps."""
import argparse
from dataclasses import replace
import json

import numpy as np
import pandas as pd

from benchmarks.minimum_fit_inputs import verify as verify_input
from benchmarks.publication import published_methods, reported_edition, result_home
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
    # Entirely missing numeric columns (e.g. an unavailable earlier map) round-trip as NaN.
    frame = frame.where(frame.notna(), np.nan)
    for column in frame.columns[frame.isna().all()]:
        frame[column] = frame[column].astype(float)
    if write:
        frame.to_csv(path, index=False)
    saved = pd.read_csv(path, keep_default_na=False, na_values=[""], dtype={"source_id": str, "site": str, "reference": str})
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
    if display.get("contour_evaluation"):
        from src.empirical_isopach.models.directional.multiscale.posterior import DirectionalField
        model = DirectionalField.load(home / family / "multiscale/posterior.npz")
        return 10. ** model.central(data.xy), float(10. ** model.central(np.zeros((1, 2)))[0])
    with np.load(grid) as a:
        field = PolarField(a["radii_km"], a["log10_cm"], float(a["axis"]))
    return 10. ** field.at_xy(*data.xy.T), float(10. ** field.at_xy(0., 0.))


def predictions(home, family, data):
    if family in REGIONAL:
        held = pd.read_csv(home / "heldout_predictions.csv").set_index("row_id").loc[data.row_ids]
        scores = held[family + "_log_score"]
        median_cm = np.exp(held[family + "_median"].to_numpy())
        cdf_at_value = np.full(len(held), np.nan)
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
        cdf_at_value = cdf.to_numpy()
        median_cm = np.exp(held.median_log_cm.to_numpy()) if "median_log_cm" in held else np.full(len(held), np.nan)
        above_min, covered = cdf <= .5, cdf.between(.025, .975)
    assert held.index.is_unique and len(held) == len(data.xy)
    return held.fold.to_numpy(), scores.to_numpy(), above_min.to_numpy(), covered.to_numpy(), median_cm, cdf_at_value


def check_annotation_parity(home, families, data, meta):
    """Erase the provenance marker and replay each field at all near and distal sites."""
    from benchmarks.refresh_figures import regional_fields
    from src.empirical_isopach.core.distributions import predictive_distribution
    from src.empirical_isopach.contour_support import candidate_levels
    from src.empirical_isopach.support import SupportMesh
    from src.empirical_isopach.visualization.deposit_field import open_field

    plain = replace(data, reported_minimum=None)
    clean_meta = {key:value for key,value in meta.items() if key not in {
        "fitting_policy", "reported_source", "reported_source_sha256", "n_reported_measurements",
        "n_minima_fitted_as_measurements", "minimum_as_measurement_rows"}}
    np.testing.assert_array_equal(candidate_levels(data), candidate_levels(plain))
    np.testing.assert_array_equal(SupportMesh(data).triangles, SupportMesh(plain).triangles)
    xy = np.vstack(([0., 0.], data.xy))
    values = {}
    if any(f in REGIONAL for f in families):
        report = read(home/"report.json")
        first, second = regional_fields(data, report), regional_fields(plain, report)
        for family in set(REGIONAL).intersection(families):
            left = predictive_distribution(first[family], xy).quantile(.5)
            right = predictive_distribution(second[family], xy).quantile(.5)
            values[family] = (left / np.log(10.), right / np.log(10.))
    for family in set(GLOBAL).intersection(families):
        left, _, _ = open_field(home/family, family, data, meta)
        right, _, _ = open_field(home/family, family, plain, clean_meta)
        values[family] = (left.central(xy), right.central(xy))
    maximum = {}
    for family, (left, right) in values.items():
        np.testing.assert_array_equal(left, right)
        maximum[family] = float(np.max(np.abs(left - right)))
    return maximum


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
    tags = pd.Series(reported.get("Kind", "measurement"), index=reported.index)
    tags = tags.str.strip().str.lower().replace("", "measurement")
    np.testing.assert_array_equal(tags, kind)
    distances = np.hypot(*data.xy.T)
    points, stats = [], []
    policy = map_observation_policy(data, meta)["observation_policy"]
    families = tuple(k for k in (*REGIONAL, *GLOBAL) if k in published_methods(record))
    if not families:
        return []
    pairs = volume_rows(home, input_sha256=meta["input_sha256"])
    assert {r["method"] for r in pairs} == set(families)
    parity = check_annotation_parity(home, families, data, meta)
    common_folds = None
    for family in families:
        receipt = read(home / "map-display.json" if family in REGIONAL else home / family / "deposit.json")
        assert receipt["observation_policy"] == policy
        new, vent = map_at_sites(home, family, data, meta["input_sha256"])
        old, old_vent = map_at_sites(old_home, family, data, source_hash)
        assert new is not None
        fold, score, above, covered, held_median, held_cdf = predictions(home, family, data)
        if common_folds is None:
            common_folds = fold
        np.testing.assert_array_equal(fold, common_folds)
        near = data.reported_minimum & (distances <= 25.)
        stats.append(dict(method=family, max_annotation_effect_log10_cm=parity[family], reported_measurements=int((~data.censored & ~data.reported_minimum).sum()),
                          reported_minima=int(data.reported_minimum.sum()), minima_within_25_km=int(near.sum()),
                          old_vent_cm=old_vent, new_vent_cm=vent,
                          old_max_at_minima_within_25_km_cm=float(old[near].max()) if old is not None and near.any() else None,
                          new_max_at_minima_within_25_km_cm=float(new[near].max()) if near.any() else None,
                          heldout_coverage95_reported_measurements=float(covered[kind=="measurement"].mean()),
                          heldout_coverage95_minimum_as_point=float(covered[data.reported_minimum].mean()),
                          heldout_fraction_medians_at_or_above_minimum=float(above[data.reported_minimum].mean())))
        for j, (row_id, row) in enumerate(reported.iterrows()):
            points.append(dict(method=family, row_id=row_id, source_id=row.get("EvidenceID") or f"row-{row_id}",
                               site=row.get("Site") or None, reference=row.get("Reference") or row.get("Refs") or None,
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
                               geographic_fold=int(fold[j]), heldout_median_cm=held_median[j],
                               heldout_cdf_at_value_or_lower_endpoint=held_cdf[j], heldout_log_score=score[j],
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
            print(r["name"], "published maps, geographic holdouts and source categories verified")
