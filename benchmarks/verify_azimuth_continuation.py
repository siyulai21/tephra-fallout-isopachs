"""Replay quantitative maps, whole-deposit intervals, calibration and archive evidence."""
import json
from dataclasses import replace
from hashlib import sha256
from pathlib import Path
from benchmarks.publication import previous_btm_file

import numpy as np
import pandas as pd

from src.empirical_isopach.core.catalog import datasets
from src.empirical_isopach.models.directional.azimuth.archive import assess, phase_medians
from src.empirical_isopach.models.directional.azimuth.calibration import calibrate, PACKET
from src.empirical_isopach.models.directional.azimuth.primary import read_field
from src.empirical_isopach.models.directional.archive_readings import load_inventory, positions
from src.empirical_isopach.models.directional.ensemble.fields import inverse_cdf
from src.empirical_isopach.models.directional.ensemble.integrals import interval
from src.empirical_isopach.models.directional.study import digest
from src.empirical_isopach.models.mass_mixture.study import audited_data
from src.empirical_isopach.visualization.comparison import saved_observations


def verify_published_estimates():
    """Check the six original fits against their numerical identities before promotion."""
    baseline = json.loads(Path("comparisons/directional/verification.json").read_text())
    for slug, expected in baseline["cases"].items():
        folder = previous_btm_file(Path("eruptions") / slug / "results/directional/azimuth")
        report = json.loads((folder/"report.json").read_text())
        assert report["protocol"] == expected["protocol"], slug
        assert report["input"]["input_sha256"] == expected["input_sha256"], slug
        assert digest(folder/"fields.npz") == expected["fields_sha256"], slug
        values = {key: report[key] for key in baseline["report_keys"]}
        values["validation"] = {kind: {key: group[key] for key in baseline["validation_keys"]}
                                for kind, group in report["validation"].items()}
        canonical = json.dumps(values, sort_keys=True, separators=(",", ":"), allow_nan=False)
        assert sha256(canonical.encode()).hexdigest() == expected["estimates_sha256"], slug


def verify(rec):
    home = Path("eruptions")/rec["slug"]/"results"
    folder = home/"directional/azimuth"
    r = json.loads((folder/"report.json").read_text())
    for name,value in json.loads((folder/"artifacts.json").read_text()).items():
        assert digest(folder/name)==value,(rec["slug"],name)
    data,meta = audited_data(rec["path"])
    assert r["data_rows"]==data.row_ids.tolist() and meta["input_sha256"]==r["input"]["input_sha256"]
    assert r["calibration_sha256"]==digest(PACKET)
    assert rec["slug"] not in r["weights"]["calibration_cases"]
    w = calibrate(meta,rec["slug"])
    for key in ("structure","concentration"):
        np.testing.assert_allclose(w[key],r["weights"][key],atol=1e-7)
    # Recorded coordinates reproduce triangle edge choices across PROJ versions.
    saved = saved_observations(home/"accepted_observations.csv")
    coordinates = dict(zip(saved.row_ids,saved.xy,strict=True))
    xy = np.array([coordinates[k] for k in data.row_ids])
    np.testing.assert_allclose(xy,data.xy,atol=1e-6)
    data = replace(data,xy=xy)
    report_volume = r["volume"]
    with np.load(folder/"fields.npz") as a:
        model = read_field(folder,data,r)
        q = interval(a["mass_km3"],a["field_weights"])
        np.testing.assert_allclose(list(q.values()),list(report_volume["coherent_volume_km3"].values()))
        np.testing.assert_allclose(model.support.evaluate(data.xy)[0],0,atol=1e-12)
        np.testing.assert_allclose(model.support.evaluate([[0.,0.]])[0],0,atol=1e-12)
        if r["archive"]:
            inventory,file = load_inventory(rec["path"])
            assert r["archive_evidence_sha256"]==digest(file)
            site,ww = assess(model,data,inventory,meta["vent_lonlat"])
            if ww is None:
                assert "archive_weights" not in a
            else:
                np.testing.assert_allclose(a["archive_weights"],ww,rtol=1e-8)
            exact = phase_medians(model,positions(inventory["sites"],meta["vent_lonlat"]))
            np.testing.assert_allclose(exact,list(r["archive"]["analytic_phase_median_cm"].values()),rtol=1e-8)
            for s in site["scenarios"].values():
                assert 1 <= s["effective_draws"] <= len(model.weights)+1e-8
    np.testing.assert_allclose(np.sum(report_volume["median_map_parts_by_sector_km3"]),report_volume["median_map_km3"])
    with np.load(folder/"map-grid.npz") as grid:
        x,y,z = [grid[k] for k in ("x_km","y_km","revised_log10_cm")]
        ix,iy = np.unravel_index(np.linspace(0,z.size-1,31).astype(int),z.shape)
        points = np.column_stack((x[iy],y[ix]))
        old,new,_ = model.components(points)
        np.testing.assert_allclose(inverse_cdf(new.T,model.weights)[0]/np.log(10.),z[ix,iy],atol=1e-8)
        np.testing.assert_allclose(model.weights@(new>=np.log(10.)),grid["probability_ge_10cm"][ix,iy])
        np.testing.assert_allclose(inverse_cdf(old.T,model.base.weights)[0]/np.log(10.),
                                   grid["control_log10_cm"][ix,iy],atol=1e-8)
    display = json.loads((folder/"map-display.json").read_text())
    assert display["fields_sha256"]==digest(folder/"fields.npz")
    assert display["median_map_km3"]==report_volume["median_map_km3"]
    assert display["input_sha256"]==meta["input_sha256"] and display["surface"]=="quantitative pointwise median"
    with np.load(folder/"median-grids.npz") as grid:
        for frame in ("local","expanded"):
            x,y,z = [grid[frame+"_"+k] for k in ("x_km","y_km","log10_cm")]
            valid = np.flatnonzero(np.isfinite(z))
            iy,ix = np.unravel_index(valid[np.linspace(0,len(valid)-1,39).astype(int)],z.shape)
            d = model.distribution(np.column_stack((x[ix],y[iy])))
            np.testing.assert_allclose(d.quantile(.5)/np.log(10.),z[iy,ix],atol=1e-8)
    heldout = pd.read_csv(folder/"heldout.csv")
    for kind, group in r["validation"].items():
        exact = heldout[(heldout.method=="revised")&(heldout.design==kind)&(heldout.reading=="exact")]
        covered = exact.observed_or_limit_cm.between(exact.predictive_lower95_cm,exact.predictive_upper95_cm)
        c = group["observation_interval"]
        assert not exact.row_id.duplicated().any()
        assert c["total_exact"]==int((~data.censored).sum()) and len(exact)==c["scored_exact"]
        assert c["covered_exact"]==int(covered.sum())
        if len(exact):
            np.testing.assert_allclose(c["coverage95_exact"],covered.mean())
    baseline = json.loads((home/"directional/ensemble/report.json").read_text())
    for design,tests in r["validation"].items():
        for fold,control in zip(tests["folds"],baseline["validation"][design]["fits"],strict=True):
            assert fold["train"]==control["training_row_ids"] and fold["test"]==control["withheld_row_ids"]
            assert not set(fold["train"]).intersection(fold["test"])
    decisions = pd.read_csv(folder/"fragment-decisions.csv")
    assert (decisions.loc[decisions.displayed,"crossing_probability"]>=.9).all()
    assert decisions.loc[decisions.displayed,"measured_bracket"].all()
    audit = json.loads((folder/"complete-contour-audit.json").read_text())
    assert audit["draws_per_chain"]==r["draws_per_chain"]
    assert all(len(v)==8 for v in report_volume["median_map_parts_by_sector_km3"])


if __name__=="__main__":
    verify_published_estimates()
    for rec in datasets():
        verify(rec)
    print(f"Verified {len(datasets())} paired directional fields, source and site anchors, archives, folds and complete volume intervals.")
