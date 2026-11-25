"""Verify the current Mazama publication against its source audit and matched geographic deletions."""
import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from src.empirical_isopach.core.audit import audited_data
from src.empirical_isopach.core.catalog import dataset_record, data_directory, resolve_dataset
from src.empirical_isopach.visualization.deposit import digest, read_json
from src.empirical_isopach.visualization.principal import volume_rows


def check(home):
    path = resolve_dataset("Mazama")
    record = dataset_record(path)
    data, meta = audited_data(path)
    identity = meta["input_sha256"]
    with path.open(newline="") as stream:
        raw = list(csv.DictReader(stream))
    audit = read_json(data_directory()/record["audit"])["cases"]["principal"]
    assert audit["sha256"] == identity == digest(path)
    assert len({r["EvidenceID"] for r in raw}) == len(data.xy) == len(raw) == 300
    counts = Counter(r["Kind"] for r in raw)
    assert [audit[key] for key in ("n_measurements","n_intervals","n_lower_limits","n_upper_limits")] == [46,29,129,96]
    assert counts == {"measurement":46,"interval":29,"lower_limit":129,"upper_limit":96}
    assert not any(r["Pseudo"].lower() == "yes" for r in raw)

    from PIL import Image
    evidence = data_directory()/"audits/Mazama"
    for name in ("evidence-map","yellowstone-cores","distal-review/distal-thickness"):
        info = read_json(evidence/(name+".json"))
        with Image.open(evidence/(name+".png")) as image:
            assert json.loads(image.info["Description"]) == info
        assert info["input_sha256"] == identity
    source_map = read_json(evidence/"evidence-map.json")
    assert source_map["site_types"] == counts and source_map["accepted_sites"] == len(raw)

    distal = read_json(evidence/"distal-review/distal-thickness.json")
    from pyproj import Geod
    geod = Geod(ellps="WGS84")
    distances = [geod.inv(*meta["vent_lonlat"],float(r["Lon"]),float(r["Lat"]))[2]/1000 for r in raw]
    included = Counter(r["Kind"] for r,km in zip(raw,distances,strict=True) if km>100)
    assert distal["site_types"] == included and distal["plotted_sites"] == sum(included.values())

    regional = read_json(home/"report.json")
    assert regional["input"]["input_sha256"] == identity
    methods = regional["validation"]["methods"]
    assert set(methods) == {"stacked","spline"}
    folds = methods["stacked"]["selections"]
    assert len(folds) == 4
    all_ids = set(data.row_ids)
    held = [row for part in folds for row in part["withheld_row_ids"]]
    assert len(held) == len(all_ids) and set(held) == all_ids
    for a,b in zip(folds,methods["spline"]["selections"],strict=True):
        assert all(a[k] == b[k] for k in ("training_row_ids","withheld_row_ids"))
        assert not set(a["training_row_ids"]).intersection(a["withheld_row_ids"])
    observed = pd.read_csv(home/"accepted_observations.csv")
    assert len(observed) == len(all_ids) and set(observed.row_id) == all_ids
    packet = read_json(home/"thickness-bands.json")
    assert packet["input_sha256"] == identity and packet["regional_report_sha256"] == digest(home/"report.json")
    assert set(packet["methods"]) == set(methods)

    for key in ("directional","spherical-mixture","nested-isopachs"):
        family = home/key
        receipt = read_json(family/"deposit.json")
        assert receipt["input_sha256"] == identity
        assert receipt["median_grid_sha256"] == digest(family/"median-grid.npz")
        assert receipt["figure_sha256"] == digest(family/"deposit.png")
        assert receipt["outer_contour"]["accepted_rows"] == len(data.xy)
        if key == "nested-isopachs":
            detail = family/"footprints"
            report = read_json(detail/"report.json")
            assert report["accepted_rows"] == len(data.xy)
            for name, expected in report["posterior_sha256"].items():
                assert digest(detail/name) == expected
            partitions = [report["fits"][f"fold-{i}"] for i in range(4)]
        elif key == "directional" and (family/"multiscale/report.json").is_file():
            detail = family/"multiscale"
            report = read_json(detail/"report.json")
            for name, checksum in report["artifacts_sha256"].items():
                assert digest(detail/name) == checksum
            partitions = [dict(train=f["train_row_ids"], test=f["test_row_ids"]) for f in report["folds"]
                          if f["design"] == "geographic"]
        else:
            detail = family/"extent"
            report = read_json(detail/"report.json")
            assert digest(detail/"volumes.npz") == report["volume_draws_sha256"]
            assert record["slug"] not in report["calibration"]["donors"]
            partitions = sorted((p for p in report["folds"] if p["design"] == "geographic"),key=lambda p:p["fold"])
        assert report["input_sha256"] == identity
        expected = report["artifacts_sha256"]["heldout.csv.gz"] if detail.name == "multiscale" else report["heldout_sha256"]
        assert digest(detail/"heldout.csv.gz") == expected
        heldout = pd.read_csv(detail/"heldout.csv.gz")
        if "design" in heldout:
            heldout = heldout[heldout.design == "geographic"]
        assert len(heldout) == len(all_ids) and set(heldout.row_id) == all_ids
        for i,(base,other) in enumerate(zip(folds,partitions,strict=True)):
            assert base["training_row_ids"] == other["train"]
            assert base["withheld_row_ids"] == other["test"] == heldout[heldout.fold == i].row_id.tolist()
    priors = read_json(home/"spherical-mixture/extent/prior-sensitivity.json")
    assert len(priors) == 8 and set(priors) == set(read_json(home/"spherical-mixture/report.json")["fits"])
    rows = volume_rows(home,input_sha256=identity)
    assert len(rows) == 5 and {r["method"] for r in rows} == {
        "stacked","spline","directional","spherical-mixture","nested-isopachs"}
    for r in rows:
        assert 0 < r["thickness_q25_volume_km3"] <= r["median_map_volume_km3"] <= r["thickness_q75_volume_km3"]
    study = Path("comparisons/nested-isopachs/geography")
    matched_inputs = pd.read_csv(study/"paired-results.csv")
    assert set(matched_inputs.loc[matched_inputs["case"]=="mazama","input_sha256"]) == {identity}
    benchmark = pd.read_csv(study/"regional-benchmark.csv")
    reference = benchmark[benchmark.slug=="mazama"].set_index("method")
    for r in rows:
        if r["method"] in methods:
            prior = reference.loc[r["method"]]
            np.testing.assert_allclose([r["median_map_volume_km3"],r["heldout_log_score"]],
                                       [prior.median_surface_km3,prior.equal_region_log_score],rtol=1e-10)
    from benchmarks.directional_comparison import case_extremes
    fixed = read_json(home/"directional/ensemble/report.json")
    stages = ["report.json","thickness-bands.json","directional/single-structure/report.json",
              "directional/ensemble/report.json","directional/azimuth/report.json","directional/multiscale/report.json",
              "spherical-mixture/report.json",
              "nested-isopachs/footprints/report.json"]
    for k in ("directional","spherical-mixture","nested-isopachs"):
        stages.extend([f"{k}/deposit.json",f"{k}/median-grid.npz"])
    files = {str(Path("eruptions/mazama/results")/name):digest(home/name) for name in stages}
    files[str(Path("data")/record["file"])] = identity
    files[str(Path("data")/record["audit"])] = digest(data_directory()/record["audit"])
    return dict(input_sha256=identity, observations=len(raw), site_types=counts, matched_geographic_folds=len(folds),
                spherical_prior_fits=len(priors), methods=rows,files=files,
                independent_regional_benchmark_sha256=digest(study/"regional-benchmark.csv"),
                directional_median_map_km3=next(r["median_map_volume_km3"] for r in rows if r["method"]=="directional"),
                fixed_direction_control_km3=fixed["median_surface"]["volume_km3"],
                octant_extremes=case_extremes(record,read_json(home/"directional/single-structure/report.json"),by_octant=True))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--home",type=Path,default=Path("eruptions/mazama/results"))
    parser.add_argument("--output",type=Path)
    args = parser.parse_args()
    result = json.dumps(check(args.home),indent=2,allow_nan=False)+"\n"
    if args.output:
        args.output.write_text(result)
    else:
        published = read_json(args.home/"release.json")
        actual = json.loads(result)
        # Projected distances and bearings may round differently between PROJ builds.
        extrema = pd.DataFrame(actual.pop("octant_extremes")).set_index(["slug","octant","row_id"])
        stored = pd.DataFrame(published.pop("octant_extremes")).set_index(["slug","octant","row_id"])
        pd.testing.assert_frame_equal(extrema,stored,check_dtype=False,rtol=2e-12,atol=1e-9)
        assert published == actual, "Published current-input receipt differs"
    print(result)
