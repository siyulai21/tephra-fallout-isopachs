"""Replay numerical claims in the published B-Tm directional and archive audit."""
import argparse
import csv
from pathlib import Path
import json

import numpy as np

from src.empirical_isopach.models.directional.archive_readings import load_inventory,site_predictions
from src.empirical_isopach.models.directional.archive_reliability import probabilities
from src.empirical_isopach.models.directional.bearing_calibration import PACKET,calibrate
from src.empirical_isopach.models.directional.bearing_report import load_fields
from src.empirical_isopach.models.directional.ensemble.fields import draw_weights
from src.empirical_isopach.models.directional.ensemble.integrals import interval
from src.empirical_isopach.models.directional.study import digest
from src.empirical_isopach.models.mass_mixture.study import audited_data

read=lambda p:json.loads(Path(p).read_text())


def verify(folder,posterior=None):
    folder=Path(folder)
    path=Path("data/eruptions/B-Tm_reviewed.csv");data,meta=audited_data(path);data=data.canonical()
    inv,src=load_inventory(path,Path("data/audits/B-Tm/archive-observations.json"))
    for f,sha in read(folder/"artifacts.json").items():
        assert digest(folder/f)==sha,f
    expected={"quantitative",*(s["id"] for s in inv["scenarios"])}
    reports={k:read(folder/k/"result.json") for k in expected}
    assert len(expected)==7
    weights,other=calibrate(None,"b-tm",meta["input_sha256"],meta["vent_lonlat"])
    np.testing.assert_allclose(weights,reports["quantitative"]["structural_weights"])
    for k,r in reports.items():
        assert r["input_sha256"]==meta["input_sha256"]
        assert r["archive_inventory_sha256"]==digest(src)
        assert r["calibration_sha256"]==digest(PACKET) and r["calibration_eruptions"]==other
        assert [s["id"] for s in r["sites"]]==[s["id"] for s in inv["sites"]]
        for s in r["sites"]:
            assert 0<=s["mesh_weight"]<=1+1e-12
            assert 0<=s["new_site_cm"]["lower95"]<=s["new_site_cm"]["median"]<=s["new_site_cm"]["upper95"]
        if r["scenario"]:
            a=r["archive_likelihood"]
            z=[b["log_probability"] for b in a["bridges"]]
            total=weights @ np.exp(z)
            np.testing.assert_allclose(np.log(total),a["log_predictive_probability"])
            np.testing.assert_allclose(weights*np.exp(z)/total,r["structural_weights"])
        if "members" not in r:
            assert r["scenario"]["role"]=="cross_prediction"
            continue
        with np.load(folder/k/"sector-draws.npz") as archive:
            arrays=[archive[m["name"]] for m in r["members"]]
        for a,m in zip(arrays,r["members"],strict=True):
            assert a.shape==(m["chains"],m["draws_per_chain"],8) and np.isfinite(a).all() and (a>=0).all()
        w=draw_weights(r["structural_weights"],[a.shape[0]*a.shape[1] for a in arrays])
        sector=np.concatenate([a.reshape(-1,8) for a in arrays])
        total=sector.sum(axis=1);west=sector[:,5:8].sum(axis=1)
        for key,values in (("volume_km3",total),("sw_w_nw_km3",west),("sw_w_nw_fraction",west/total)):
            recalculated=interval(values,w)
            for q,v in recalculated.items():np.testing.assert_allclose(v,r["coherent_fields"][key][q])
        field=load_fields(folder/k,data,meta["vent_lonlat"][1])
        s=site_predictions(field,inv["sites"],meta["vent_lonlat"])
        for observed,stored in zip(s,r["map_sites"],strict=True):
            np.testing.assert_allclose(observed["latent_cm"]["median"],stored["latent_cm"]["median"])
        m=r["median_surface"]
        np.testing.assert_allclose(np.asarray(m["parts_sectors_km3"]).sum(),m["volume_km3"])
        assert m["passed"]==(m["spatial_passed"] and m["draw_split_passed"] and m["level_splits_passed"])
    for k in ("quantitative","visible_analogs"):
        v=read(folder/k/"validation.json");audit=read(folder/k/"complete-contour-audit.json")
        assert set(audit["policies"])=={"strict","standard","permissive"}
        for kind,g in v.items():
            for f in g["folds"]:
                assert not set(f["training_row_ids"]) & set(f["withheld_row_ids"])
                if "archive_training_ids" in f:
                    assert not set(f["archive_training_ids"]) & set(f["withheld_archives"])
            scores=[f["negative_log_score"] for f in g["folds"] if f["status"]=="estimated"]
            np.testing.assert_allclose(np.mean(scores),g["equal_region_negative_log_score"])
    for pair in ((0,0),(1,1),(0,1),(1,0),(.25,.75)):
        w=probabilities([1,.1,.2,.04],*pair)
        assert np.all(w>=0) and np.isclose(w.sum(),1)
    assert (folder/"archive-reliability.csv").exists()
    sphere=read(folder/"spherical-archive-checks.json")
    assert sphere["input_sha256"]==meta["input_sha256"]
    assert digest(folder/"spherical-archive-weights.npz")==sphere["weights_sha256"]
    if posterior:
        assert digest(posterior)==sphere["posterior_sha256"]
        with np.load(posterior) as packet,np.load(folder/"spherical-archive-weights.npz") as archive:
            volume=np.exp(packet["log_volume"].ravel())
            for k,r in sphere["scenarios"].items():
                w=archive[k].ravel()
                np.testing.assert_allclose(w.sum(),1)
                np.testing.assert_allclose(1/(w@w),r["raw_weight_ess"])
                for q,x in interval(volume,w).items():
                    np.testing.assert_allclose(x,r["full_field_volume_km3"][q],rtol=1e-11)
    transfer = folder/"shard-transfer-comparison.csv"
    if transfer.is_file():
        provenance = read(folder/"shard-transfer-provenance.json")
        assert provenance["file_sha256"] == digest(transfer)
        assert provenance["input_sha256"] == meta["input_sha256"]
        with transfer.open() as stream:
            rows = list(csv.DictReader(stream))
        assert len(rows) == 42
        for row in rows:
            value = float(row["transfer_equivalent_cm"])
            if row["shards_per_g_dry"]:
                np.testing.assert_allclose(value,float(row["shards_per_g_dry"])/25_000/10)
            assert 0 <= float(row["p_local_at_or_below_equivalent"]) <= 1
        if posterior:
            from scipy.special import ndtr
            from src.empirical_isopach.models.directional.archive_readings import positions
            from src.empirical_isopach.models.mass_mixture.geometry import log_field,to_sphere
            from src.empirical_isopach.models.mass_mixture.inference import flatten
            ids = sorted({r["archive_id"] for r in rows})
            by_id = {s["id"]:s for s in inv["sites"]}
            xy = positions([by_id[i] for i in ids],meta["vent_lonlat"])
            unit = to_sphere(xy,meta["vent_lonlat"][1])
            with np.load(posterior) as raw:
                samples = flatten(dict(raw))
            mu = np.concatenate([log_field({k:v[i:i+128] for k,v in samples.items()},unit)
                                 for i in range(0,len(samples["noise"]),128)])
            with np.load(folder/"spherical-archive-weights.npz") as saved:
                for row in rows:
                    if row["model"] != "spherical":
                        continue
                    idx = ids.index(row["archive_id"])
                    z = (np.log(float(row["transfer_equivalent_cm"])) - mu[:,idx])/samples["noise"]
                    eps = samples["contamination"]
                    cdf = saved[row["scenario"]].ravel() @ ((1-eps)*ndtr(z) + eps*ndtr(z/4))
                    np.testing.assert_allclose(cdf,float(row["p_local_at_or_below_equivalent"]),rtol=1e-10)
    print("Verified angular calibration, archive holdouts, pointwise median fields, eight-sector mass budgets, preservation sensitivities and the independent spherical update.")


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--folder",type=Path,default=Path("eruptions/b-tm/results/directional/bearing"))
    parser.add_argument("--posterior",type=Path,default=Path("eruptions/b-tm/results/spherical-mixture/posterior.npz"))
    a=parser.parse_args();verify(a.folder,a.posterior)
