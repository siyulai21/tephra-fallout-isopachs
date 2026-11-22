"""Matched spatial predictions from signed parent fits, with layout rebuilt inside each fold."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import log_ndtr, logsumexp, ndtr

from .extent_review import ExtentFields, STATES
from .directional.azimuth.report import model_for
from .mass_mixture.bearings.field import BearingField, BearingSupport, phases, selected
from .nested.profile import parameters
from .nested.reach import ReachDraws
from ..visualization.deposit_field import digest


def read(path):
    return json.loads(Path(path).read_text())


def take_rows(data, rows):
    part = data.subset(np.isin(data.row_ids, rows))
    if part.row_ids.tolist()!=list(rows):
        raise ValueError("Training or held-out row identifiers differ from the fit")
    return part


def fold_models(family, folder, data, lat, work, *, geographic_only=False):
    if family=="directional":
        r = read(folder/"azimuth/report.json")
        for kind, design in r["validation"].items():
            if geographic_only and kind!="geographic":
                continue
            for f in design["folds"]:
                if f["status"]!="estimated":
                    yield kind, f["fold"], None, None, None, None, f["status"]
                    continue
                train, test = take_rows(data,f["train"]), take_rows(data,f["test"])
                model = model_for(train,lat,work,r["weights"],64,kind,f["fold"])
                hashes = {}
                for m in model.members:
                    name = m.geometry.policy.name
                    hashes[name] = m.records[-1]["posterior_sha256"]
                    if hashes[name]!=f["member_attempts"][name][-1]["posterior_sha256"]:
                        raise ValueError(f"Directional fold posterior changed for {name}")
                yield kind,f["fold"],train,test,model,hashes,bool(f["passed"] and f["phase_passed"])
    elif family=="spherical-mixture":
        r = read(folder/"report.json")
        for f in r["validation"]["fits"]:
            n = f["fold"]
            train, test = take_rows(data,f["training_row_ids"]), take_rows(data,f["withheld_row_ids"])
            fit = f["fit"]
            path = work/"validation"/f"fold-{n}"/fit["posterior_file"]
            if digest(path)!=fit["posterior_sha256"]:
                raise ValueError("Spherical fold posterior changed")
            with np.load(path) as a:
                draws, _, shape = selected(dict(a),256)
            model = BearingField(draws,BearingSupport(train,lat),phases(shape))
            yield "geographic",n,train,test,model,{"posterior":fit["posterior_sha256"]},bool(fit["diagnostics"]["passed"])
    else:
        r = read(folder/"report.json")
        for f in r["folds"]:
            n = f["fold"]
            train, test = take_rows(data,f["training_rows"]), take_rows(data,f["test_rows"])
            h = read(folder/f"fold-{n}.json")["posterior_sha256"]
            path = folder/f"fold-{n}.npz"
            if digest(path)!=h:
                raise ValueError("Nested fold posterior changed")
            model = ReachDraws.load(path)
            yield "geographic",n,train,test,model,{"posterior":h},bool(f["passed"])


def observation_components(family, parent, test):
    if family=="nested-isopachs":
        noise = parameters(parent.z)[-1][:,None]
        return [(s, np.hypot(k*noise,test.log_sd)) for k,s in ((1.,.95),(3.,.05))]
    if family=="directional":
        _,_,bary = parent.members[0].geometry.locate(test.xy)
        local = np.maximum(0.,1.-(bary*bary).sum(1))
        sd = np.concatenate([np.exp(m.samples["log_noise"])[:,None]*np.sqrt(1.+m.geometry.policy.local_sd_ratio**2*local)
                             for m in parent.members])
        return [(1.,np.hypot(sd,test.log_sd))]
    p = parent.draws
    noise = np.tile(p["noise"],3)[:,None]
    eps = np.tile(p["contamination"],3)[:,None]
    return [(weight,np.hypot(k*noise,test.log_sd)) for k,weight in ((1.,1.-eps),(4.,eps))]


def scores(fields, test):
    components = observation_components(fields.family,fields.parent,test)
    out = {}
    for state, mu in zip(fields.states, fields.variants(test.xy), strict=True):
        logs, probabilities = [], []
        for w,sd in components:
            weight = fields.weights[:,None]*w
            z = (test.log_value-mu)/sd
            event = -.5*z*z - np.log(sd)-.5*np.log(2*np.pi)
            upper = test.censored & ~test.lower_censored
            event[:,upper] = log_ndtr(z[:,upper])
            event[:,test.lower_censored] = log_ndtr(-z[:,test.lower_censored])
            bounds = getattr(test,"interval_log_upper",None)
            if bounds is not None and np.isfinite(bounds).any():
                interval = np.isfinite(bounds)
                lo,hi = z[:,interval],(bounds[interval]-mu[:,interval])/sd[:,interval]
                left,right = np.where(lo>0,-hi,lo),np.where(lo>0,-lo,hi)
                log_right = log_ndtr(right)
                with np.errstate(divide="ignore"):
                    event[:,interval] = log_right+np.log(-np.expm1(log_ndtr(left)-log_right))
            with np.errstate(divide="ignore"):
                logs.append(logsumexp(np.log(weight)+event,axis=0))
            probabilities.append(np.sum(weight*ndtr(z),axis=0))
        out[state+"_log_score"] = logsumexp(logs,axis=0)
        out[state+"_cdf"] = np.sum(probabilities,axis=0)
    return out


def validate(family, folder, data, latitude, work):
    rows, fits = [],[]
    for kind,n,train,test,parent,sources,passed in fold_models(family,folder,data,latitude,work):
        record = dict(design=kind,fold=n,fit_passed=passed if isinstance(passed,bool) else False)
        if parent is None:
            fits.append({**record,"status":passed})
            continue
        fields = ExtentFields(family,parent,train,latitude)
        # Actual locations, including upper limits, have exactly the fitted likelihood.
        at = fields.variants(train.xy)
        residual = max(float(np.max(abs(x-at[0]))) for x in at)
        if residual>1e-8:
            raise ArithmeticError(f"Training field differs by {residual}")
        fits.append({**record,"status":"estimated","train":train.row_ids.tolist(),"test":test.row_ids.tolist(),
                     "source_sha256":sources,"maximum_training_log_cm_change":residual})
        rows.append(pd.DataFrame({**record,"row_id":test.row_ids,"exact":~test.censored,
                                  **scores(fields,test)}))
    if rows:
        return pd.concat(rows, ignore_index=True), fits
    columns = ["design", "fold", "fit_passed", "row_id", "exact"]
    columns.extend(f"{state}_{metric}" for state in STATES for metric in ("log_score", "cdf"))
    return pd.DataFrame(columns=columns), fits
