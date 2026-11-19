"""Replay quantitative maps, whole-deposit intervals, calibration and archive evidence."""
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from src.empirical_isopach.core.catalog import datasets
from src.empirical_isopach.models.directional.azimuth.archive import assess, phase_medians
from src.empirical_isopach.models.directional.azimuth.calibration import calibrate, PACKET
from src.empirical_isopach.models.directional.azimuth.field import AzimuthField
from src.empirical_isopach.models.directional.archive_readings import load_inventory, positions
from src.empirical_isopach.models.directional.ensemble.fields import Ensemble, Member, inverse_cdf
from src.empirical_isopach.models.directional.ensemble.members import MEMBERS
from src.empirical_isopach.models.directional.ensemble.integrals import interval
from src.empirical_isopach.models.directional.geometry import Geometry
from src.empirical_isopach.models.directional.study import digest
from src.empirical_isopach.models.mass_mixture.study import audited_data
from src.empirical_isopach.visualization.comparison import saved_observations


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
        members = []
        for policy in MEMBERS:
            prefix = policy.name+"_"
            draws = {k[len(prefix):]:a[k] for k in a if k.startswith(prefix) and k!=prefix+"indices"}
            members.append(Member(Geometry(data,meta["vent_lonlat"][1],policy),draws,a[prefix+"indices"],[]))
        model = AzimuthField(Ensemble(members,r["weights"]["structure"]),data,r["weights"]["concentration"])
        np.testing.assert_array_equal(model.k,a["concentration_index"])
        np.testing.assert_allclose(model.phase,a["azimuth_rad"],atol=1e-13)
        np.testing.assert_allclose(model.weights,a["field_weights"],atol=1e-13)
        q = interval(a["mass_km3"],a["field_weights"])
        np.testing.assert_allclose(list(q.values()),list(report_volume["coherent_volume_km3"].values()))
        np.testing.assert_allclose(model.support.evaluate(data.xy)[0],0,atol=1e-12)
        np.testing.assert_allclose(model.support.evaluate([[0.,0.]])[0],0,atol=1e-12)
        if r["archive"]:
            inventory,file = load_inventory(rec["path"])
            assert r["archive_evidence_sha256"]==digest(file)
            site,ww = assess(model,data,inventory,meta["vent_lonlat"])
            np.testing.assert_allclose(a["archive_weights"],ww,rtol=1e-8)
            exact = phase_medians(model,positions(inventory["sites"],meta["vent_lonlat"]))
            np.testing.assert_allclose(exact,list(r["archive"]["analytic_phase_median_cm"].values()),rtol=1e-8)
            for s in site["scenarios"].values():
                assert 1 <= s["effective_draws"] <= len(ww)+1e-8
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
    for rec in datasets():
        verify(rec)
    print("Verified six paired directional fields, source and site anchors, archives, folds and complete volume intervals.")
