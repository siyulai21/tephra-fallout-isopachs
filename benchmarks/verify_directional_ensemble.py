"""Verify published median fields, coherent volumes, exclusions and geographic evidence."""
import csv
import gzip
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .publication import read_hashes, verify_files
from scipy.special import logsumexp

from src.empirical_isopach.core.catalog import datasets
from src.empirical_isopach.models.mass_mixture.study import audited_data
from src.empirical_isopach.models.directional.study import digest
from src.empirical_isopach.models.directional.ensemble.fields import draw_weights, load_packet
from src.empirical_isopach.models.directional.ensemble.integrals import interval
from src.empirical_isopach.models.directional.ensemble.members import MEMBERS, NAMES
from src.empirical_isopach.models.directional.ensemble.weights import PACKET, calibrate


def read(p):
    return json.loads(p.read_text())


def verify_case(rec, expected_input, median_map_km3, *, home=None):
    home = Path("eruptions")/rec["slug"]/"results" if home is None else Path(home)
    folder = home/"directional"/"ensemble"
    report = read(folder/"report.json")
    for f,h in read(folder/"artifacts.json").items():
        assert digest(folder/f)==h,(rec["slug"],f)
    data,meta = audited_data(rec["path"])
    assert digest(rec["path"]) == report["input"]["input_sha256"] == expected_input
    assert digest(PACKET)==report["calibration_sha256"]
    correct = calibrate(meta,rec["slug"])
    np.testing.assert_allclose(correct["weights"],report["weighting"]["weights"],rtol=1e-7)
    assert rec["slug"] not in report["weighting"]["calibration_cases"]
    # Use recorded coordinates to reproduce boundary decisions across PROJ builds.
    with (home/"accepted_observations.csv").open(newline="") as stream:
        sites = {int(row["row_id"]):row for row in csv.DictReader(stream)}
    xy = np.array([[float(sites[i][k]) for k in ("x_km","y_km")] for i in data.row_ids])
    np.testing.assert_allclose(data.xy,xy,rtol=1e-12,atol=1e-7)
    from dataclasses import replace
    data = replace(data, xy=xy)
    model = load_packet(folder,data,meta["vent_lonlat"][1],report)
    centre = report["median_surface"]
    central = centre["volume_km3"]
    assert np.isclose(central,median_map_km3)
    cells = np.array(centre["parts_sectors_km3"])
    assert cells.shape==(5,8) and (cells>=0).all() and np.isclose(cells.sum(),central)
    assert centre["passed"] == all(centre["numerical"][x] for x in ("spatial_passed","draw_split_passed","resolved_level_splits_passed"))
    weighted_means = []
    with np.load(folder/"coherent-mass.npz") as f:
        assert set(f)==set(NAMES)
        arrays = [f[p.name] for p in MEMBERS]
        for m,a in zip(model.members,arrays,strict=True):
            assert a.shape==(4,m.records[-1]["draws_per_chain"],5,8) and np.isfinite(a).all() and (a>=0).all()
            weighted_means.append(a[:,m.index].sum(axis=(2,3)).mean())
        weights = draw_weights(model.model_weights,[a.shape[0]*a.shape[1] for a in arrays])
        all_mass = np.concatenate([a.reshape(-1,5,8) for a in arrays])
        joint = report["coherent_fields"]
        for values, expected in [(all_mass.sum(axis=(1,2)),joint["volume_km3"]),
            *zip(all_mass.sum(axis=2).T,joint["parts"],strict=True),*zip(all_mass.sum(axis=1).T,joint["sectors"],strict=True)]:
            actual = interval(values,weights)
            np.testing.assert_allclose(list(actual.values()),list(expected.values()),rtol=1e-12)
        np.testing.assert_allclose(model.model_weights@weighted_means,joint["field_mean_volume_km3"])
    with np.load(folder/"median-grids.npz") as grids:
        for key in ("local","expanded"):
            xs,ys,z = [grids[key+"_"+v] for v in ("x_km","y_km","log10_cm")]
            idx = np.flatnonzero(np.isfinite(z[1]))[::max(1,np.isfinite(z[1]).sum()//33)]
            iy,ix = np.unravel_index(idx,z[1].shape)
            d = model.distribution(np.column_stack((xs[ix],ys[iy])))
            for i,p in enumerate((.025,.5,.975)):
                np.testing.assert_allclose(d.quantile(p)/np.log(10.),z[i,iy,ix],atol=1e-10)
    assert np.isclose(np.exp(model.distribution([[0.,0.]]).quantile(.5))[0],report["map"]["vent_median_cm"])
    legacy = read(home/"directional/single-structure/report.json")
    for layout,v in report["validation"].items():
        old = legacy["validation"][layout]
        for key in ("folds","row_ids","buffer_km"):
            assert v[key]==old[key],key
        scores = []
        for f,o in zip(v["fits"],old["fits"],strict=True):
            assert f["withheld_row_ids"]==o["withheld_row_ids"]
            assert f["training_row_ids"]==o.get("training_row_ids",[])
            assert not set(f["withheld_row_ids"])&set(f["training_row_ids"])
            if f["status"]=="estimated":
                lp = logsumexp(f["member_log_scores"],b=model.model_weights,axis=1)
                ids = [v["row_ids"].index(i) for i in f["withheld_row_ids"]]
                np.testing.assert_allclose(lp,np.asarray(v["predictions"]["log_score"])[ids].astype(float))
                scores.append(-lp.mean())
        if scores:
            assert np.isclose(np.mean(scores),v["equal_region_log_score"])
            assert np.isclose(v["reference_score"],old["equal_region_log_score"],rtol=1e-10)
    audit = read(folder/"complete-contour-audit.json")
    assert report["complete_bands"]=={k:v["selected_band"] for k,v in audit["policies"].items()}
    for row in audit["refined_levels"]:
        assert all(x["eligible"]==(not x["reasons"]) for x in row["policies"].values())
    table = pd.read_csv(folder/"fragment-decisions.csv")
    np.testing.assert_array_equal(table.displayed,table.measured_bracket&(table.crossing_probability>=.9)&(table.grid_segments>0))


def main():
    root=Path("comparisons/directional/fixed-ensemble")
    tab=pd.read_csv(root/"comparison.csv").set_index("slug")
    verification=read(root/"verification.json")
    for rec in datasets():
        slug = rec["slug"]
        home = Path("eruptions")/slug/"results"
        historic_input = tab.loc[slug].input_sha256 if slug in tab.index else None
        if digest(rec["path"]) == historic_input:
            expected_input = historic_input
            median_map = tab.loc[slug].median_map_km3
            expected_report = verification["case_reports"][slug]
        else:
            release = read(home/rec.get("release_receipt", "source-audit/release.json"))
            expected_input = release["files"][rec["path"].relative_to(Path.cwd()).as_posix()]
            median_map = release["directional_median_map_km3"]
            expected_report = release["files"][str(home/"directional/ensemble/report.json")]
        assert digest(home/"directional/ensemble/report.json") == expected_report
        verify_case(rec, expected_input, median_map)
    assert digest(PACKET)==verification["calibration_sha256"]
    assert all(check["identical"] and check["rerun_sha256"]==check["published_sha256"]
               for check in verification["regional_benchmark"].values())
    # The study retains the hashes of its original publication, including its old text.
    for path, checksum in verification["artifacts"].items():
        if path in {"README.md","volumes.png","comparison.csv","known-volumes.csv"}:
            name = str(root/path)
            verify_files({name: read_hashes()[name]})
        else:
            assert digest(root/path) == checksum, path
    with gzip.open(root/"simulation-records.json.gz","rt") as f:
        stress=json.load(f)
    assert len(stress)==14
    for key,row in stress.items():
        if key.startswith("invisible_sector"):
            assert row["truth"]["same_data_hidden_total_km3"]>row["truth"]["total_km3"]
            assert set(r["truth"] for r in row["contour_accuracy"])=={"hidden_lobe","sampled_equivalent"}
    print(f"Verified {len(datasets())} ensemble median-map packets, all partitioned volume histories, held-out predictions and 14 simulated deposits.")


if __name__=="__main__":
    main()
