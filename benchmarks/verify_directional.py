"""Verify published directional estimates, mass accounting and paired evidence."""
import csv
import gzip
import json
from pathlib import Path
from hashlib import sha256

import numpy as np
import pandas as pd

from src.empirical_isopach.core.catalog import datasets
from src.empirical_isopach.models.directional.estimate import interval
from src.empirical_isopach.models.mass_mixture.study import audited_data
from src.empirical_isopach.models.directional.geometry import COMPASS, POLICIES, polar
from src.empirical_isopach.models.directional.integration import PARTS
from src.empirical_isopach.models.directional.study import digest


def read(path):
    return json.loads(path.read_text())


def check_interval(draws, recorded):
    actual = interval(draws)
    assert all(np.isclose(actual[k], recorded[k], rtol=1e-12, atol=1e-14) for k in actual)


def verify(folder, record, comparison, study, extremes):
    manifest = read(folder/"artifacts.json")
    for name, expected in manifest.items():
        assert digest(folder/name) == expected, f"Changed {folder/name}"
    report = read(folder/"report.json")
    assert study["cases"][record["slug"]] == manifest["report.json"]
    assert report["input"]["input_sha256"] == comparison.input_sha256 == digest(record["path"])
    observations, _ = audited_data(record["path"])
    ids = report["validation"]["geographic"]["row_ids"]
    by_id = {row_id:i for i,row_id in enumerate(observations.row_ids)}
    assert len(ids) == len(by_id) and set(ids) == set(by_id)
    observations = observations.subset([by_id[i] for i in ids])
    # IEEE values from the published round-trip CSV reproduce the signed fitting arrays.
    # Different PROJ/libm builds can round an independently projected coordinate differently.
    with (folder.parent/"accepted_observations.csv").open(newline="") as stream:
        archived = {int(r["row_id"]):r for r in csv.DictReader(stream)}
    assert set(archived) == set(ids)
    rows = [archived[i] for i in ids]
    boolean = {"True":True,"False":False}
    frozen = {"xy":np.array([[float(r["x_km"]),float(r["y_km"])] for r in rows],dtype="<f8"),
              "log_value":np.array([float(r["log_value"]) for r in rows],dtype="<f8"),
              "log_sd":np.array([float(r["known_log_sd"]) for r in rows],dtype="<f8"),
              "row_ids":np.array(ids,dtype="<i8"),
              "censored":np.array([boolean[r["censored"]] for r in rows]),
              "lower_censored":np.array([boolean[r["lower_bound"]] for r in rows])}
    for name, values in frozen.items():
        signed = report["designs"]["reference"]["signature"]["arrays_sha256"][name]
        assert sha256(np.ascontiguousarray(values).tobytes()).hexdigest() == signed, name
        actual = getattr(observations,name)
        if name == "xy":
            np.testing.assert_allclose(actual,values,rtol=1e-12,atol=1e-7)
        elif name in ("log_value","log_sd"):
            np.testing.assert_allclose(actual,values,rtol=1e-13,atol=1e-14)
        else:
            np.testing.assert_array_equal(actual,values)
    axis = report["designs"]["reference"]["geometry"]["authalic_axis_scale"]
    radius, angle = polar(observations.xy,axis)
    compass = np.array(COMPASS)[np.floor((angle+np.pi/8)/(np.pi/4)).astype(int)%8]
    quadrant = report["validation"]["quadrants"]
    assert set(extremes.octant) == set(compass[~observations.censored])
    for row in extremes.itertuples():
        i, = np.flatnonzero(observations.row_ids == row.row_id)
        assert np.isclose(radius[i],radius[(~observations.censored)&(compass == row.octant)].max())
        assert np.isclose(row.observed_cm,np.exp(observations.log_value[i]))
        assert row.quadrant == quadrant["folds"][i]
        value = quadrant["predictions"]["median"][i]
        assert np.isnan(row.predicted_median_cm) if value is None else np.isclose(row.predicted_median_cm,np.exp(value))
    profiles = [p.name for p in POLICIES]
    assert [p["name"] for p in report["policies"]] == profiles
    assert list(report["fits"]) == profiles
    base = report["fits"]["reference"]
    assert base["part_names"] == list(PARTS)
    assert list(base["compass"]) == list(COMPASS)
    assert base["attempts"][-1]["posterior_sha256"] == manifest["posterior.npz"]
    assert base["mass_sha256"] == manifest["directional-mass.npz"]
    with np.load(folder/"directional-mass.npz") as parts, np.load(folder/"volume-draws.npz") as volumes:
        mass = parts["mass"]
        assert mass.shape == (*volumes["reference"].shape, len(PARTS), len(COMPASS))
        assert np.isfinite(mass).all() and (mass >= 0).all()
        assert np.array_equal(parts["draw_index"], np.arange(mass.shape[1]))
        total = mass.sum(axis=(-2,-1))
        np.testing.assert_allclose(total, volumes["reference"],rtol=1e-12,atol=1e-14)
        cap = report["unobserved_proximal_cap"]
        assert cap["observed_thickness_change_cm"] == 0.
        with np.load(folder/"source-cap.npz") as extended:
            assert (extended["extra_km3"] >= 0).all()
            np.testing.assert_allclose(extended["volume_km3"], total+extended["extra_km3"],rtol=1e-12)
            check_interval(extended["volume_km3"],cap["volume_km3"])
            check_interval(extended["extra_km3"],cap["extra_km3"])
        for policy, fit in report["fits"].items():
            values = volumes[policy]
            assert values.shape == (4,fit["attempts"][-1]["draws_per_chain"])
            assert (values > 0).all() and np.isfinite(values).all()
            check_interval(values, fit["volume_km3"])
            assert fit["passed"] == (fit["attempts"][-1]["diagnostics"]["passed"] and
                                     fit["volume_diagnostics"]["passed"] and fit["quadrature"]["passed"])
        for j,s in enumerate(COMPASS):
            check_interval(mass[...,j].sum(-1), base["compass"][s]["volume"])
        for i,r in enumerate(PARTS):
            part = mass[...,i,:].sum(-1)
            check_interval(part, base["parts"][r]["volume_km3"])
            check_interval(part/total, base["parts"][r]["fraction"])
    geo = report["validation"]["geographic"]
    old = read(folder.parent/"spherical-mixture/report.json")["validation"]
    assert (geo["row_ids"],geo["folds"],geo["buffer_km"]) == (old["row_ids"],old["folds"],old["buffer_km"])
    for current, previous in zip(geo["fits"],old["fits"],strict=True):
        for key in ("training_row_ids","withheld_row_ids"):
            assert current[key] == previous[key]
    arcs = pd.read_csv(folder/"isopach-fragments.csv")
    assert len(arcs) == report["contours"]["triangle_level_pairs"]
    assert arcs.displayed.sum() == report["contours"]["displayed_segments"]
    assert (arcs.displayed == (arcs.measured_bracket & arcs.has_segment & (arcs.posterior_crossing_frequency >= .9))).all()
    screen = read(folder/"complete-contour-audit.json")
    for level in screen["refined_levels"]:
        for item in level["policies"].values():
            assert item["eligible"] == (not item["reasons"])
    assert report["complete_bands"] == {p:r["selected_band"] for p,r in screen["policies"].items()}


def main():
    root = Path("comparisons/directional")
    study = read(root/"verification.json")
    cases = pd.read_csv(root/"comparison.csv").set_index("slug")
    extremes = pd.read_csv(root/"octant-extremes.csv")
    for record in datasets():
        slug = record["slug"]
        verify(Path("eruptions")/slug/"results/directional", record, cases.loc[slug], study, extremes[extremes.slug==slug])
        with gzip.open(root/"spherical-support"/(slug+".json.gz"),"rt") as f:
            screen = json.load(f)
        assert digest(root/"spherical-support"/(slug+".npz")) == screen["surface_samples_sha256"]
        assert screen["input_sha256"] == cases.loc[slug].input_sha256
    trials = pd.read_csv(root/"known-volumes.csv")
    with gzip.open(root/"known-volume-fits.json.gz","rt") as f:
        evidence = json.load(f)
    assert len(evidence) == len(trials) == 28
    for row in trials.itertuples():
        result = evidence[f"{row.scenario}-{row.seed}"]
        assert row.passed == result["fit"]["passed"]
        assert np.isclose(row.median, result["fit"]["volume_km3"]["median"])
        assert np.isclose(row.truth_km3, result["truth"]["total_km3"])
    with gzip.open(root/"independent-replications.json.gz","rt") as f:
        replicas = json.load(f)
    assert set(replicas) == set(cases.index)
    for slug, run in replicas.items():
        assert run["design"]["replication"]["input_sha256"] == cases.loc[slug].input_sha256
        assert run["integral"]["passed"] == cases.loc[slug].replication_passed
    print(f"Verified {len(cases)} published eruptions, {len(cases)*len(POLICIES)} policy fits, "
          f"{len(replicas)} independent replications and {len(evidence)} known-volume trials.")


if __name__ == "__main__":
    main()
