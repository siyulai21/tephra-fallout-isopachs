"""Verify the B-Tm evidence selections, current four-family release and sensitivity mass draws."""
from collections import Counter
import csv
from hashlib import sha256
import gzip
import json
from pathlib import Path
import subprocess
from zipfile import ZipFile

import numpy as np
import pandas as pd

from benchmarks.btm_source_sensitivities import SOURCE_CASES
from benchmarks.btm_thinning_report import verify as verify_thinning
from benchmarks.btm_thinning_volume_report import verify as verify_thinning_volumes
from benchmarks.verify_directional import verify as verify_single
from benchmarks.verify_directional_ensemble import verify_case as verify_ensemble
from benchmarks.verify_visuals import verify_case_map
from src.empirical_isopach.core.catalog import datasets
from src.empirical_isopach.models.directional.ensemble.fields import draw_weights
from src.empirical_isopach.models.directional.ensemble.integrals import interval
from src.empirical_isopach.models.directional.ensemble.members import NAMES
from src.empirical_isopach.models.mass_mixture.source_protocol import protocol

AUDIT = Path("data/audits/B-Tm")
HOME = Path("eruptions/b-tm/results")
STUDY = HOME/"source-audit"


def read(path):
    return json.loads(path.read_text())


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def table(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def verify_inputs():
    manifest = read(AUDIT/"provenance.json")
    original = Path(manifest["original_input"]["file"])
    assert digest(original) == manifest["original_input"]["sha256"]
    assert digest(AUDIT/"evidence.csv") == manifest["ledger_sha256"]
    old = table(original)
    ledger = table(AUDIT/"evidence.csv")
    original_rows = {int(row["original_row"]): row for row in ledger if row["original_row"]}
    assert set(original_rows) == set(range(2, 86))
    assert {i for i, row in enumerate(old, 2) if row["Pseudo"].lower() == "yes"} == set(range(86, 93))
    for i, entry in original_rows.items():
        row = old[i-2]
        assert all(float(row[k]) == float(entry[v]) for k, v in
                   (("Lat", "original_lat"), ("Lon", "original_lon"), ("Thickness", "original_cm")))
    ids = {row["id"] for row in ledger}
    assert len(ids) == len(ledger) == 117
    assert all(row["source"] and row["locator"] and row["decision"] for row in ledger)
    costa = table(AUDIT/"costa-table-crosswalk.csv")
    assert [row["costa_id"] for row in costa] == [f"P{i:02d}" for i in range(1, 84)]
    assert all(row["review"] and (not row["evidence_id"] or row["evidence_id"] in ids) for row in costa)
    root = Path("comparisons/source-revision/input-inventory/b-tm-workbook-review")
    yang = table(root/"source-crosswalk.csv")
    assert {int(row["source_sheet_row"]) for row in yang} == set(range(2, 78))
    assert len({row["original_csv_row"] for row in yang}) == len(yang) == 76
    sources = {r["id"]: r for r in read(AUDIT/"sources.json")}
    public = Path("data/sources/b-tm")
    for name, ref in [("yang2021-original-supplement.zip","Yang21"),
                      ("costa2024-supplement.pdf","Costa24"), ("nakanishi2020-data.xml","Naka20")]:
        assert digest(public/name) == sources[ref]["source_bytes_sha256"]
    with ZipFile(public/"yang2021-original-supplement.zip") as archive:
        filename = next(s for s in archive.namelist() if s.endswith(".xlsx"))
        assert archive.read(filename) == (root/"yang2021-thickness.xlsx").read_bytes()
    all_cases = {}
    for scenario, case in manifest["cases"].items():
        path = Path(case["file"])
        assert digest(path) == case["sha256"]
        selected = table(path)
        actual = {row["EvidenceID"]: row for row in selected}
        assert len(actual) == len(selected) == len({(r["Lat"],r["Lon"]) for r in selected})
        counts = Counter(row["Kind"] for row in selected)
        assert list(map(counts.get, ("measurement","lower_limit","upper_limit"))) == [
            case[k] for k in ("n_measurements","n_lower_limits","n_upper_limits")]
        assert [(r["Vent Lon"], r["Vent Lat"]) for r in selected if r["Vent Lon"] or r["Vent Lat"]] == [("128.057","42.006")]
        for entry in ledger:
            key, kind = entry["id"], entry["evidence_kind"]
            value = entry["thickness_cm"]
            limit = {"upper_limit": entry["upper_cm"], "lower_limit": entry["lower_cm"]}.get(kind, "")
            if scenario in {"range_low","range_high"} and kind == "range":
                kind, value = "measurement", entry["lower_cm" if scenario == "range_low" else "upper_cm"]
            if scenario == "correlations" and key in {"old-002","old-008"}:
                kind, value, limit = "measurement", {"old-002":"150","old-008":"26"}[key], ""
            if kind not in {"measurement","lower_limit","upper_limit"}:
                assert key not in actual
                continue
            row = actual[key]
            assert (row["Lat"],row["Lon"],row["Site"]) == (entry["latitude"],entry["longitude"],entry["site"])
            assert (row["Kind"],row["Thickness"],row["Limit_cm"]) == (kind,value,limit)
        assert set(actual) <= ids
        all_cases[scenario] = actual
    assert len(all_cases["principal"]) == 68
    assert all("tumen" not in observations for observations in all_cases.values())
    assert set(all_cases["range_low"]) == set(all_cases["range_high"])
    assert len(set(all_cases["range_low"]) - set(all_cases["principal"])) == 12
    assert "old-066" not in all_cases["principal"]
    assert all_cases["range_low"]["old-066"]["Thickness"] == "0.5"
    assert all_cases["range_high"]["old-066"]["Thickness"] == "1"
    supplemental = table(AUDIT/"additional-evidence.csv")
    assert len(supplemental) == len({r["id"] for r in supplemental}) == 23
    assert not {r["id"] for r in supplemental} & set(all_cases["principal"])
    source_class = read(AUDIT/"source-class-provenance.json")
    assert source_class["principal_input_sha256"] == manifest["cases"]["principal"]["sha256"]
    assert source_class["ledger_sha256"] == digest(AUDIT/"evidence.csv")
    compiled = {r["id"] for r in ledger if r["decision"] == source_class["principal_field_log_status"]}
    assert len(compiled) == 25 and compiled == set(source_class["compiled_scalar_ids"])
    utasai = source_class["utasai_id"]
    principal_rows = all_cases["principal"]
    assert utasai not in compiled and "Utasai" in principal_rows[utasai]["Site"]
    assert all(principal_rows[k]["Kind"] == "measurement" for k in compiled | {utasai})
    strict = source_class["cases"]["omit_compilation_values"]
    file = Path(strict["file"])
    observations = table(file)
    assert digest(file) == strict["sha256"] and len(observations) == 42
    assert observations == [v for k,v in principal_rows.items() if k not in compiled | {utasai}]
    assert Counter(r["Kind"] for r in observations) == {"measurement":38,"lower_limit":3,"upper_limit":1}
    manifest["cases"].update(source_class["cases"])
    return manifest


def verify_results(manifest):
    release = read(STUDY/"release.json")
    if Path(".git").exists():
        tracked = subprocess.check_output(["git", "ls-files", "-z"]).decode().split("\0")
        missing = sorted(set(release["files"]).difference(tracked))
        assert not missing, f"Release files missing from the Git index: {missing}"
    for name, expected in release["files"].items():
        assert digest(Path(name)) == expected, name
    matched = pd.read_csv(STUDY/"matched-regional.csv").set_index(["slug", "method"])
    historical = pd.read_csv(STUDY/"spline-replay.csv").set_index(["slug", "method"])
    assert set(matched.index) == set(historical.index) == {
        (r["slug"], family) for r in datasets() for family in ("spline", "stacked")}
    for key, row in historical.iterrows():
        assert bool(row.same_input) == (key[0] != "b-tm")
        np.testing.assert_allclose(row.current_score, matched.at[key,"equal_region_log_score"])
        np.testing.assert_allclose(row.current_hull_km3, matched.at[key,"expected_regional_km3"])
    trials = pd.read_csv("comparisons/regional/synthetic/comparison.csv")
    assert len(trials) == 80 and not trials.duplicated(["method", "scenario", "seed"]).any()
    assert trials.groupby(["method","scenario"]).size().eq(10).all()
    if trials.completed.eq(0).any():
        assert trials.loc[trials.completed.eq(0), "failure"].notna().all()
    assert read(STUDY/"regional-benchmark-design.json")["spline_model_sha256"] == digest(
        Path("src/empirical_isopach/models/cubic_spline/model.py"))
    rec = next(r for r in datasets() if r["slug"] == "b-tm")
    principal = manifest["cases"]["principal"]
    reference = read(HOME/"directional/single-structure/report.json")
    ensemble = read(HOME/"directional/ensemble/report.json")
    sphere = read(HOME/"spherical-mixture/report.json")
    regional = read(HOME/"report.json")
    for report in [reference, ensemble, sphere, regional]:
        meta = report["input"]
        assert meta["input_sha256"] == principal["sha256"] == digest(rec["path"])
        assert meta["audit"]["sha256"] == digest(AUDIT/"provenance.json")
        assert (meta["n_measurements"], meta["n_lower_limits"], meta["n_upper_limits"]) == (64, 3, 1)
    extremes = pd.read_csv(STUDY/"octant-extremes.csv")
    verify_single(HOME/"directional/single-structure", rec, principal["sha256"], release["files"][str(HOME/"directional/single-structure/report.json")], extremes)
    verify_ensemble(rec, principal["sha256"], release["directional_median_map_km3"])
    verify_case_map(rec, HOME.resolve())
    rv, sv = regional["validation"], sphere["validation"]
    for key in ("row_ids","folds","buffer_km"):
        assert rv[key] == sv[key]
    for model in rv["methods"].values():
        for fold, peer in zip(sv["fits"],model["selections"],strict=True):
            for key in ("withheld_row_ids","training_row_ids"):
                assert fold[key] == peer[key]
    for f, expected in read(HOME/"spherical-mixture/artifacts.json").items():
        assert digest(HOME/"spherical-mixture"/f) == expected
    prior, alternatives = protocol("source-fans")
    assert set(sphere["fits"]) == {p.name for p in (prior, *alternatives)}
    assert sphere["geometry"] == "source-fans"
    support = read(HOME/"support/report.json")
    assert support["provenance"]["parent_report_sha256"] == digest(HOME/"report.json")
    sfolder = HOME/"spherical-mixture/support"
    with gzip.open(sfolder/"b-tm.json.gz", "rt") as stream:
        screen = json.load(stream)
    assert screen["input_sha256"] == principal["sha256"]
    assert screen["surface_samples_sha256"] == digest(sfolder/"b-tm.npz")
    assert {k:v for k,v in screen.items() if k != "surface_samples_sha256"} == read(HOME/"spherical-mixture/complete-contour-audit.json")
    scenarios = pd.read_csv(STUDY/"source-scenarios.csv").set_index("case")
    assert scenarios.index.is_unique
    assert set(scenarios.index) == set(manifest["cases"]) == {"principal", *SOURCE_CASES}
    assert {p.parent.name for p in STUDY.glob("*/result.json")} == set(SOURCE_CASES)
    for name, row in scenarios.iterrows():
        case = manifest["cases"][name]
        assert row.input_sha256 == case["sha256"] and row.n == sum(case[k] for k in (
            "n_measurements", "n_lower_limits", "n_upper_limits"))
    for name in SOURCE_CASES:
        folder = STUDY/name
        report = read(folder/"result.json")
        assert report["input"] == manifest["cases"][name]
        assert "b-tm" not in report["weighting"]["calibration_cases"]
        assert report["weighting"]["members"] == list(NAMES)
        assert report["weighting"]["weights"] == ensemble["weighting"]["weights"]
        observed = report["coherent_fields"]["volume_km3"]
        with np.load(folder/"volume-draws.npz") as archive:
            assert set(archive.files) == set(NAMES)
            draws = [archive[member] for member in NAMES]
        assert all(a.ndim == 2 and a.shape[0] == 4 and np.isfinite(a).all() and (a > 0).all() for a in draws)
        weights = draw_weights(report["weighting"]["weights"], [a.size for a in draws])
        actual = interval(np.concatenate([a.ravel() for a in draws]), weights)
        np.testing.assert_allclose(list(actual.values()), list(observed.values()), rtol=1e-12)
        centre = report["median_surface"]
        row = scenarios.loc[name]
        np.testing.assert_allclose([centre["volume_km3"], observed["median"], observed["lower95"], observed["upper95"]],
                                   row[["median_map_km3","volume_median_km3","lower95","upper95"]].astype(float))
        assert row.map_integral_passed == centre["passed"]
        assert row.full_members_passed == sum(m["attempts"][-1]["diagnostics"]["passed"]
                                             for m in report["coherent_fields"]["members"].values())
        np.testing.assert_allclose(np.asarray(centre["parts_sectors_km3"]).sum(), centre["volume_km3"])
    verify_thinning()
    verify_thinning_volumes(STUDY/"thinning/volumes")
    variant_fits = read(STUDY/"thinning/fits.json")
    variant_runs = read(STUDY/"thinning/volumes/diagnostics.json")["runs"]
    assert all(run["posterior_sha256"] == variant_fits[k]["full_fit"]["attempts"][-1]["posterior_sha256"]
               for k, run in variant_runs.items())
    with np.load(HOME/"directional/single-structure/directional-mass.npz") as reference_mass, np.load(STUDY/"thinning/volumes/volume-draws.npz") as variants:
        np.testing.assert_allclose(variants["smooth"], reference_mass["mass"].sum(axis=(-1,-2)), rtol=2e-12)
    print("Verified B-Tm sources, segmented thinning, all four fitted methods, geographic folds, contour screens and four source sensitivities.")


if __name__ == "__main__":
    verify_results(verify_inputs())
