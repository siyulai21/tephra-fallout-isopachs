"""Matched geographic and directional withholding for the gap-continuation ensemble."""
import argparse
import json
from pathlib import Path
import numpy as np
from scipy.special import logsumexp

from src.empirical_isopach.core.catalog import datasets
from src.empirical_isopach.models.directional.bearing_support import RATIOS, with_ratio
from src.empirical_isopach.models.directional.ensemble.fields import Ensemble, load_members
from src.empirical_isopach.models.directional.ensemble.members import NAMES, splits
from src.empirical_isopach.models.directional.ensemble.validation import component_scores, fold_record
from src.empirical_isopach.models.directional.bearing_calibration import CANDIDATES, calibrate
from src.empirical_isopach.models.directional.study import write_json
from src.empirical_isopach.models.mass_mixture.study import audited_data

def one_case(rec, work):
    data, meta=audited_data(rec["path"]); data=data.canonical()
    lat=meta["vent_lonlat"][1]
    out={"slug":rec["slug"],"input_sha256":meta["input_sha256"],"vent":meta["vent_lonlat"],"folds":[]}
    for kind in ("geographic","quadrants"):
        for k,train,test in splits(data,lat,kind):
            state=fold_record(work,kind,k,train,test)
            item={x:state[x] for x in ("fold","training_row_ids","withheld_row_ids","status","passed")}
            item["kind"]=kind
            if state["status"]=="estimated":
                base=load_members(train,lat,work,kind,k,n=None)
                controls=Ensemble(base, np.ones(len(base)))
                observed=controls.distribution(test.xy,observation=True,log_sd=test.log_sd)
                item["control_scores"]=component_scores(observed,controls.sizes,test).tolist()
                columns=[]
                for m in base[:3]:
                    for ratio in RATIOS:
                        member=with_ratio(m,train,ratio)
                        np.testing.assert_allclose(member.geometry.coords(train.xy)[0],m.geometry.coords(train.xy)[0],rtol=2e-14,atol=1e-13)
                        e=Ensemble([member],[1.])
                        dist=e.distribution(test.xy,observation=True,log_sd=test.log_sd)
                        columns.append(dist.log_score(test.log_value,test.censored,test.lower_censored))
                item["scores"]=np.column_stack(columns).tolist()
                np.testing.assert_allclose(np.asarray(item["scores"])[:,::len(RATIOS)],
                                           np.asarray(item["control_scores"])[:,:3],rtol=2e-11,atol=1e-10)
                item["posterior_sha256"]={m.geometry.policy.name:m.records[-1]["posterior_sha256"] for m in base}
            out["folds"].append(item)
    return out


def summarize(packet):
    tables=[]
    for case in packet["cases"]:
        w,external=calibrate(packet,case["slug"],case["input_sha256"],case["vent"])
        old,_=calibrate(packet,case["slug"],case["input_sha256"],case["vent"],"control_scores")
        for kind in ("geographic","quadrants"):
            folds=[f for f in case["folds"] if f["kind"]==kind and f["status"]=="estimated"]
            if not folds: continue
            scores=[-logsumexp(np.array(f["scores"]),b=w,axis=1).mean() for f in folds]
            reference=[-logsumexp(np.array(f["control_scores"]),b=old,axis=1).mean() for f in folds]
            tables.append({"slug":case["slug"],"kind":kind,"new_nll":float(np.mean(scores)),
                "control_nll":float(np.mean(reference)),"scored_rows":sum(len(f["withheld_row_ids"]) for f in folds),
                "flagged_folds":[f["fold"] for f in folds if not f["passed"]],"external_calibration":external,
                "weights":w.tolist(),"control_weights":old.tolist()})
    return tables


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--work",type=Path,required=True)
    p.add_argument("--btm-work",type=Path);p.add_argument("--output",type=Path,required=True);a=p.parse_args()
    packet={"protocol":"bearing-continuation-1","candidates":CANDIDATES,"controls":list(NAMES),"cases":[]}
    for rec in datasets():
        work=a.btm_work if rec["slug"]=="b-tm" and a.btm_work is not None else a.work/rec["slug"]
        packet["cases"].append(one_case(rec,work))
        print("Checked",rec["slug"],flush=True)
    packet["matched_scores"]=summarize(packet)
    write_json(a.output,packet)
