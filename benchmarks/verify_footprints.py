"""Replay all published footprint fits, matched folds, map probes, totals and stress-test identities."""
import gzip
import json

import numpy as np
import pandas as pd

from .footprint_catalogue import published_folder
from .footprint_study import STUDY, paired, structural, controls
from .publication import published_datasets
from .verify_deposit_maps import verify_global
from src.empirical_isopach.core.audit import audited_data
from src.empirical_isopach.models.directional.ensemble.members import splits
from src.empirical_isopach.models.nested.footprints.posterior import FootprintField, PROTOCOL, engine, predictions
from src.empirical_isopach.visualization.deposit import digest, read_json


def main():
    pairs, sensitivity = [], []
    quadrants = json.loads(gzip.decompress((STUDY/"quadrant-fits.json.gz").read_bytes()))
    for rec in published_datasets():
        case = rec["slug"]
        home=published_folder(case,rec)
        packet=home/"footprints"
        report=read_json(packet/"report.json")
        data,meta=audited_data(rec["path"])
        assert report["protocol"]==PROTOCOL and report["engine_sha256"]==engine()
        assert meta["input_sha256"]==report["input_sha256"] and len(data.xy)==report["accepted_rows"]
        assert {k:digest(packet/k) for k in report["posterior_sha256"]}==report["posterior_sha256"]
        posterior=FootprintField.load(packet/"full-selected.npz")
        assert report["heldout_sha256"]==digest(packet/"heldout.csv.gz") and report["support_sha256"]==digest(packet/"support.json.gz")
        probes=read_json(packet/"archive-probes.json")
        assert probes["input_sha256"]==meta["input_sha256"] and probes["posterior_sha256"]==report["posterior_sha256"]["full-selected.npz"]
        q=np.quantile(posterior.volume(),[.025,.5,.975])
        np.testing.assert_allclose(q,[report["conditional_interval95_km3"][0],report["complete_draw_median_km3"],report["conditional_interval95_km3"][1]])
        np.testing.assert_allclose(sum(report["posterior_family"].values()),1.)
        exact=report["independent_draw_integration"][-1]
        assert exact["passed"]==(exact["median_error"]<.01 and exact["p95_error"]<.05)
        assert report["draw_checks_passed"]==(report["fits"]["full"]["passed"] and exact["passed"])
        screen=json.loads(gzip.decompress((packet/"support.json.gz").read_bytes()))
        assert screen["status"]!="completed" or {"strict","standard","permissive"}==set(screen["policies"])
        verify_global(home,"nested-isopachs",data,meta)
        sector = pd.read_csv(packet/"sector-volumes.csv")
        check = read_json(packet/"sector-checks.json")
        assert check["table_sha256"]==digest(packet/"sector-volumes.csv")
        assert check["input_sha256"]==meta["input_sha256"] and check["posterior_sha256"]==probes["posterior_sha256"]
        assert int(sector[["exact","lower","upper","interval"]].sum().sum())+check["unassigned_vent_sites"]==len(data.xy)
        np.testing.assert_allclose(sector.median_map_fraction.sum(),1.)
        np.testing.assert_allclose(sector.median_map_km3.sum(),check["median_map_km3"])
        frame=pd.read_csv(packet/"heldout.csv.gz")
        assert len(frame)==len(data.xy) and set(frame.row_id)==set(data.row_ids)
        for i,train,test in splits(data,meta["vent_lonlat"][1],"geographic"):
            key=f"fold-{i}"
            fit=report["fits"][key]
            assert fit["train"]==train.row_ids.tolist() and fit["test"]==test.row_ids.tolist()
            assert not set(fit["train"]) & set(fit["test"])
            assert fit["attempts"][-1]["posterior_sha256"]==report["posterior_sha256"][key+"-selected.npz"]
            for attempt in fit["attempts"]:
                assert attempt["identity"]["engine_sha256"]==engine()
            if len(fit["attempts"])==2:
                assert not fit["attempts"][0]["numerics"]["passed"]
            pred=predictions(FootprintField.load(packet/(key+"-selected.npz")),test)
            block=frame[frame.fold==i]
            assert block.row_id.tolist()==[p["row_id"] for p in pred]
            assert block.exact.tolist()==[p["exact"] for p in pred]
            for k in ("log_score","cdf"):
                np.testing.assert_allclose(block[k],[p[k] for p in pred],rtol=1e-7,atol=2e-8)
        np.testing.assert_allclose(-frame.groupby("fold").log_score.mean().mean(),report["validation"]["negative_log_score"])
        for i,train,test in splits(data,meta["vent_lonlat"][1],"quadrants"):
            trial=quadrants[f"{case}/{i}"]
            ids=trial["identity"] if trial["status"]=="estimated" else trial
            assert ids["input_sha256"]==meta["input_sha256"]
            assert ids["test"]==test.row_ids.tolist() and ids["train"]==([] if train is None else train.row_ids.tolist())
        pairs.extend(paired(case,rec,home))
        sensitivity.extend(structural(case,controls(case,rec)[2],home))
        print(case,len(data.xy),"observations verified",flush=True)
    for expected,name in ((pairs,"paired-results.csv"),(sensitivity,"geography-prior-sensitivity.csv")):
        pd.testing.assert_frame_equal(pd.DataFrame(expected),pd.read_csv(STUDY/name),check_dtype=False,rtol=3e-8,atol=1e-9)
    known=pd.read_csv(STUDY/"known-deposits.csv.gz")
    unique=known[["scenario","seed","truth"]].drop_duplicates()
    assert len(unique)==22 and len(known)==len(unique)*3*2*4
    scores=pd.read_csv(STUDY/"withheld-surveys.csv.gz")
    assert scores.groupby(["survey","row_id"]).method.nunique().eq(3).all()
    assert scores.groupby("survey").row_id.nunique().to_dict()=={
        "aso-4":28,"b-tm":3,"calbuco-2015":43,"pinatubo-1991":10,"pinatubo-all-offshore":34}
    print("Verified the data identity, full and withheld deposits, three geographic laws, quadrants, known deposits and campaigns.")


if __name__=="__main__":
    main()
