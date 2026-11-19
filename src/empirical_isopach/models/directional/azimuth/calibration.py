"""Blocked predictive calibration. No target or qualitative archive enters its own weights."""
import json
from pathlib import Path

import numpy as np
from scipy.special import log_ndtr, logsumexp

from ..ensemble.fields import Ensemble, load_members
from ..ensemble.members import NAMES, splits
from ..ensemble.validation import fold_record
from ..ensemble.weights import solve
from ..study import write_json
from .field import CONCENTRATIONS, DirectionalSupport, log_normalizer


PACKET = Path(__file__).resolve().parents[3]/"assets"/"azimuth-calibration.json"


def scores_for_member(mu, variance, test, h, phases):
    """Integrate the unknown phase at a single withheld location (Chebyshev rule)."""
    sd = np.sqrt(variance)
    constants = -.5*np.log(2*np.pi)-np.log(sd)
    scores = []
    cosine = np.cos((np.arange(phases)+.5)*np.pi/phases)
    for k in CONCENTRATIONS:
        a = k*h
        base = test.log_value-mu+log_normalizer(a)
        logs = []
        for cos in (cosine if k else [0.]):
            z = (base-a*cos)/sd
            dense = constants-.5*z*z
            if test.censored.any():
                dense[:,test.censored] = log_ndtr(np.where(test.lower_censored,-z,z)[:,test.censored])
            logs.append(logsumexp(dense,axis=0)-np.log(len(mu)))
        scores.append(logsumexp(logs,axis=0)-np.log(len(logs)))
    return np.stack(scores,axis=-1)


def case_scores(rec, data, meta, work):
    lat = meta["vent_lonlat"][1]
    groups = []
    for kind in ("geographic","quadrants"):
        for i,train,test in splits(data,lat,kind):
            f = fold_record(work,kind,i,train,test)
            group = {"kind":kind,"fold":i,"passed":f["passed"],"status":f["status"],
                     "train":f["training_row_ids"],"test":f["withheld_row_ids"],
                     "member_attempts":{n:s.get("attempts",[]) for n,s in f["members"].items()}}
            if f["status"]=="estimated":
                members = load_members(train,lat,work,kind,i,n=None)
                support = DirectionalSupport(train,members[0].geometry)
                h,_=support.evaluate(test.xy)
                group["exposure"]=h.tolist()
                results, worst, max_order = [],0.,32
                for m in members:
                    d=Ensemble([m],[1.]).distribution(test.xy,observation=True,log_sd=test.log_sd)
                    previous=scores_for_member(d.means,d.variances,test,h,16)
                    for order in (32,64,128):
                        score=scores_for_member(d.means,d.variances,test,h,order)
                        delta=float(np.max(np.abs(score-previous)))
                        if delta<.001: break
                        previous=score
                    worst=max(worst,delta)
                    max_order=max(max_order,order)
                    results.append(score)
                group.update(log_scores=np.stack(results,axis=1).tolist(),phase_order=max_order,
                             phase_log_score_change=worst,phase_passed=worst<.001)
            groups.append(group)
    return {"slug":rec["slug"],"input_sha256":meta["input_sha256"],"vent":meta["vent_lonlat"],"folds":groups}


def calibrate(meta, slug=None, packet=None):
    packet=json.loads(PACKET.read_text()) if packet is None else packet
    if tuple(packet["members"])!=NAMES or not np.array_equal(packet["concentrations"],CONCENTRATIONS):
        raise ValueError("Incompatible azimuthal calibration packet")
    rows,mass,included=[],[],[]
    for case in packet["cases"]:
        other=case["vent"]; vent=meta["vent_lonlat"]
        same=(abs((other[0]-vent[0]+180)%360-180)<.01 and abs(other[1]-vent[1])<.01)
        if case["slug"]==slug or case["input_sha256"]==meta["input_sha256"] or same: continue
        groups=[[f for f in case["folds"] if f["kind"]==k and f["passed"] and f.get("phase_passed")] for k in ("geographic","quadrants")]
        groups=[g for g in groups if g]
        if groups: included.append(case["slug"])
        for g in groups:
            for f in g:
                rows.extend(f["log_scores"])
                mass.extend(np.full(len(f["test"]),1/(len(groups)*len(g)*len(f["test"]))))
    if rows:
        lp=np.asarray(rows)
        structure=solve(lp[:,:,0],mass)
        conditional=logsumexp(lp,b=structure[None,:,None],axis=1)
        intensity=solve(conditional,mass)
    else:
        structure=np.full(len(NAMES),1/len(NAMES))
        intensity=np.full(len(CONCENTRATIONS),1/len(CONCENTRATIONS))
    return {"calibration_cases":included,"structure":structure.tolist(),"concentration":intensity.tolist()}


def mixture_scores(fold,weights):
    lp=np.asarray(fold["log_scores"])
    component=logsumexp(lp,b=np.asarray(weights["structure"])[None,:,None],axis=1)
    return component[:,0],logsumexp(component,b=np.asarray(weights["concentration"]),axis=1)
