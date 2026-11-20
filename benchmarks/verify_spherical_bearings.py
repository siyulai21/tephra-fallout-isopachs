"""Replay principal spherical masses, map checks, geographic partitions and release evidence."""
import gzip
from hashlib import sha256
from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy.stats import qmc

from .publication import SPHERICAL_REFERENCE,read_hashes,preceding_spherical_bytes
from src.empirical_isopach.core.audit import audited_data
from src.empirical_isopach.core.catalog import datasets
from src.empirical_isopach.models.mass_mixture.bearings.field import PROTOCOL,BearingSupport,phases
from src.empirical_isopach.models.mass_mixture.bearings.maps import extent_diagnostics
from src.empirical_isopach.models.mass_mixture.bearings.report import design
from src.empirical_isopach.models.mass_mixture.bearings.study import digest,read
from src.empirical_isopach.models.mass_mixture.bearings.volume import integrate_draw
from src.empirical_isopach.models.mass_mixture.inference import flatten


def verify_case(rec):
    home=Path("eruptions")/rec["slug"]/"results/spherical-mixture"
    r=read(home/"report.json"); parent=read(home/"bearing/parent-report.json")
    data,meta=audited_data(rec["path"])
    assert r["protocol"]==PROTOCOL and r["parent_protocol"]==parent["protocol"]
    assert meta["input_sha256"]==r["input"]["input_sha256"]==parent["input"]["input_sha256"]
    assert r["bearing"]["parent_report_sha256"]==digest(home/"bearing/parent-report.json")
    assert r["bearing"]["model_sha256"]==design()["model_sha256"]
    for file,identity in read(home/"artifacts.json").items():
        assert digest(home/file)==identity,(rec["slug"],file)
    assert r["validation"]["fits"]==parent["validation"]["fits"]
    assert list(r["fits"])==list(parent["fits"]) and len(r["fits"])==8
    held=pd.read_csv(home/"heldout_predictions.csv")
    assert held.row_id.tolist()==r["validation"]["row_ids"] and held.fold.tolist()==r["validation"]["folds"]
    np.testing.assert_allclose(held.log_score_power_0,parent["validation"]["predictions"]["log_score"],rtol=1e-8,atol=1e-8)
    np.testing.assert_allclose(np.logaddexp.reduce(held[["log_score_power_0","log_score_power_1","log_score_power_4"]],axis=1)-np.log(3),held.log_score)
    np.testing.assert_allclose(-held.groupby("fold").log_score.mean().mean(),r["validation"]["metrics"]["equal_region_log_score"])
    mass=pd.read_csv(home/"volume_draws.csv.gz")
    for name,fit in r["fits"].items():
        sub=mass[mass.assumption==name]
        assert fit["posterior_sha256"]==parent["fits"][name]["posterior_sha256"]
        assert fit["parent_volume"]==parent["fits"][name]["volume"]
        q=fit["volume"]
        np.testing.assert_allclose(sub.volume_km3.quantile([.025,.5,.975]),[q[k] for k in ("lower95","median","upper95")],rtol=1e-10)
        assert not sub.duplicated(["parent_flat_index","width_power"]).any()
    lat=meta["vent_lonlat"][1];support=BearingSupport(data,lat)
    np.testing.assert_allclose(support.evaluate(data.xy)[0],0.,atol=1e-12)
    with np.load(home/"posterior.npz") as p, np.load(home/"bearing/baseline.npz") as d:
        q=r["fits"]["baseline"]
        assert q["posterior_sha256"]==digest(home/"posterior.npz")
        ix,values=d["flat_index"],d["volume_km3"]
        phi=phases((p["log_volume"].shape[0],len(ix)//p["log_volume"].shape[0]))
        np.testing.assert_array_equal(phi,d["azimuth_rad"])
        flat=flatten(dict(p))
        np.testing.assert_allclose(values[:,0],np.exp(flat["log_volume"][ix]))
        np.testing.assert_allclose(d["part_sector_km3"].sum((2,3)),values)
        assert np.all(values[:,1:]<=values[:,:1]*(1+1e-12)) and (values>0.).all()
        u=qmc.Sobol(4,scramble=True,seed=831774).random_base2(13)
        rr=np.linalg.norm(support.mesh.xy*support.scale,axis=1)
        for j in np.linspace(0,len(ix)-1,16,dtype=int):
            draw={k:v[ix[j]] for k,v in flat.items()}
            independent,_,_=integrate_draw(draw,support,phi[j],u,rr.min(),rr.max(),False)
            np.testing.assert_allclose(independent[-1,:3],values[j]/values[j,0],atol=.035)
    if "archive_evidence" in rec:
        archives = read(home/"archive-evidence/spherical-archive-checks.json")
        assert archives["bearing_packet_sha256"]==digest(home/"bearing/baseline.npz")
        from src.empirical_isopach.core.catalog import data_directory
        assert archives["archive_inventory_sha256"]==digest(data_directory()/rec["archive_evidence"])
        q=archives["scenarios"]["quantitative"]["full_field_volume_km3"]
        np.testing.assert_allclose([q[k] for k in ("lower95","median","upper95")],
                                   [r["fits"]["baseline"]["volume"][k] for k in ("lower95","median","upper95")])
    extent=pd.read_csv(home/"bearing/sector-areas.csv")
    assert np.all(extent.principal_km2<=extent.reference_km2+1e-8)
    current=extent_diagnostics(extent.to_dict("records"))
    saved=r["bearing"]["map_draw_check"]["history"][-1]
    for s,v in zip(saved["checks"],current,strict=True):
        assert (s["sector"],s["level_cm"],s["resolved"])==(v["sector"],v["level_cm"],v["resolved"])
        if s["resolved"]:
            np.testing.assert_allclose(s["split_symmetric_difference_fraction"],v["split_symmetric_difference_fraction"],atol=1e-9)
    with gzip.open(home/"support"/(rec["slug"]+".json.gz"),"rt") as fp:
        screen=json.load(fp)
    assert screen.pop("surface_samples_sha256")==digest(home/"support"/(rec["slug"]+".npz"))
    assert screen==read(home/"complete-contour-audit.json")
    assert screen.get("field_protocol",PROTOCOL)==PROTOCOL
    assert screen["baseline_posterior_sha256"]==r["fits"]["baseline"]["posterior_sha256"]


def main():
    for name,v in read_hashes(SPHERICAL_REFERENCE/"previous-files.sha256").items():
        assert sha256(preceding_spherical_bytes(name)).hexdigest()==v,name
    for rec in datasets():
        verify_case(rec)
    base=Path("comparisons/spherical-mixture")
    from PIL import Image
    with Image.open(base/"bearings/paired-isopachs.png") as plot:
        sources=json.loads(plot.info["Description"])
    for rec in datasets():
        folder=Path("eruptions")/rec["slug"]/"results/spherical-mixture"
        assert sources[rec["slug"]]["report_sha256"]==digest(folder/"report.json")
        assert sources[rec["slug"]]["parent_posterior_sha256"]==digest(folder/"posterior.npz")
    with gzip.open(base/"bearings/known-deposits.json.gz","rt") as stream:
        packets=json.load(stream)
    refs={}
    for name,study in (("displaced",base/"stress-report.json"),("source",Path("comparisons/source-revision/stress-report.json"))):
        for record in read(study)["runs"]:
            refs[name,record["scenario"],record["seed"]]=record
    assert len(packets)==len(refs)==24
    for packet in packets:
        ref=refs[packet["geometry"],packet["scenario"],packet["seed"]]
        assert packet["posterior_sha256"]==ref["fits"]["baseline"]["posterior_sha256"]
        assert packet["volumes_km3"]["original"]==ref["truth_km3"]
    print("Verified all six spherical reconstructions, 48 prior fits, 24 geographic deletions, 24 known-volume fits and the preceding release.")


if __name__=="__main__":
    main()
