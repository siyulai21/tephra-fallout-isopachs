"""Host-deposit observations and explicit, conditional bed-visibility likelihoods."""
import json
from pathlib import Path
import re

import numpy as np
from scipy.special import log_ndtr, logsumexp

from src.Laea import make_laea_transform
from ...core.catalog import data_directory, dataset_record
from .bearing_support import RATIOS, include_visibility_readings, with_ratio
from .ensemble.fields import Ensemble, Member, load_members, probabilities
from .study import SEED, fit

ELIGIBLE = {"cryptotephra_only","dispersed_no_continuous_bed","inspected_no_bed"}
READINGS = ELIGIBLE | {"attribution_uncertain","unsurveyed","occurrence_only","unclear_thickness","unresolved_horizon"}


def load_inventory(path, supplied=None):
    rec = dataset_record(path)
    if supplied is None and rec and "archive_evidence" in rec:
        supplied = data_directory()/rec["archive_evidence"]
    if supplied is None:
        return {"schema":"tephra-archive-evidence-v1","sites":[],"scenarios":[]}, None
    file = Path(supplied)
    inventory = json.loads(file.read_text())
    if inventory["schema"] != "tephra-archive-evidence-v1":
        raise ValueError("Unknown archive observation schema")
    sites = {row["id"]:row for row in inventory["sites"]}
    if len(sites) != len(inventory["sites"]):
        raise ValueError("Repeated archive identifier")
    for row in sites.values():
        if row["reading"] not in READINGS or not row.get("reference") or not row.get("inspection"):
            raise ValueError("Archive reading, inspection and source locator required")
        if not (-90 <= row["latitude"] <= 90 and -180 <= row["longitude"] <= 180):
            raise ValueError("Invalid archive coordinates")
    used = set()
    for scenario in inventory["scenarios"]:
        key = scenario["id"]
        if key in used or key == "quantitative" or not re.fullmatch("[a-z][a-z0-9_-]{0,63}",key):
            raise ValueError("Invalid or repeated archive scenario identifier")
        used.add(key)
        limits = scenario["effective_limit_cm"]
        if not limits or not scenario.get("assumption") or scenario["role"] not in ("sensitivity","cross_prediction"):
            raise ValueError("Explicit effective scales and their scientific assumptions required")
        groups = []
        for sid, limit in limits.items():
            row = sites[sid]
            if row["reading"] not in ELIGIBLE or row["confidence"] != "secure":
                raise ValueError("Unresolved, uninspected or presence-only horizons cannot constrain visibility")
            if not np.isfinite(limit) or limit <= 0:
                raise ValueError("Visibility scale must be positive and finite")
            groups.append(row.get("dependency_group",sid))
        if len(groups) != len(set(groups)):
            raise ValueError("The same archive horizon cannot supply two independent visibility terms")
    return inventory, file


def positions(sites, vent):
    if not sites:
        return np.empty((0,2))
    trans = make_laea_transform(*vent)[0]
    longitude,latitude = np.array([[s["longitude"],s["latitude"]] for s in sites]).T
    return np.column_stack(trans.transform(longitude,latitude))


def quantitative_model(data, latitude, work, weights, kind="full", fold=None, count=None):
    bases = load_members(data,latitude,work,kind,fold,n=count)
    return Ensemble([with_ratio(m,data,c) for m in bases[:3] for c in RATIOS],weights)


def log_visibility(member, xy, limits):
    d = Ensemble([member],[1.]).distribution(xy, observation=True)
    values = log_ndtr((np.log(limits)-d.means)/np.sqrt(d.variances)).sum(axis=1)
    return values.reshape(member.draws["source"].shape)


def bridge_log_probability(before, after):
    """Bridge the normalization ratio for likelihood L against the thickness-only posterior."""
    a, b = np.ravel(before), np.ravel(after)
    n0,n1 = len(a),len(b)
    l0,l1 = np.log([n0/(n0+n1),n1/(n0+n1)])
    z = logsumexp(a) - np.log(n0)
    for _ in range(250):
        q0 = np.logaddexp(l0, l1+a-z)
        q1 = np.logaddexp(l0, l1+b-z)
        updated = logsumexp(a-q0)-np.log(n0) - logsumexp(-q1)+np.log(n1)
        if abs(updated-z) < 1e-10:
            return float(updated)
        z = updated
    raise ArithmeticError("Archive bridge sampling failed to converge")


def condition_model(data, model, sites, vent, scenario, work):
    index = {s["id"]:s for s in sites}
    selected = scenario["effective_limit_cm"]
    xy = positions([index[k] for k in selected],vent)
    if np.any(np.linalg.norm(data.xy[:,None,:]-xy,axis=2) < 1e-6):
        raise ValueError("Review possible duplicate quantitative and archive observations at the same site")
    limits = np.array(list(selected.values()))
    augmented = include_visibility_readings(data,xy,limits)
    updated, logz, diagnostics = [], [], []
    for i, member in enumerate(model.members):
        geometry = member.geometry
        folder = Path(work)/scenario["id"]/(geometry.policy.name+"-"+str(geometry.ratio))
        policy_index, ratio_index = divmod(i,len(RATIOS))
        raw, result = fit(augmented,geometry,SEED+33000+policy_index*100+ratio_index,folder)
        candidate = Member(geometry,raw,np.arange(raw["source"].shape[1]),result["attempts"])
        old_likelihood = log_visibility(member,xy,limits)
        new_likelihood = log_visibility(candidate,xy,limits)
        z = bridge_log_probability(old_likelihood,new_likelihood)
        chain = [bridge_log_probability(a,b) for a,b in zip(old_likelihood,new_likelihood,strict=True)]
        direct = float(logsumexp(old_likelihood) - np.log(old_likelihood.size))
        logz.append(z)
        diagnostics.append({"policy":geometry.policy.name,"ratio":geometry.ratio,"log_probability":z,
                            "direct_log_probability":direct,"four_independent_chain_log_ratios":chain,
                            "chain_log_ratio_span":float(np.ptp(chain)),"sampling_passed":result["passed"],
                            "attempts":result["attempts"]})
        updated.append(candidate)
    evidence = logsumexp(logz,b=model.model_weights)
    weights = probabilities(model.model_weights*np.exp(logz-evidence))
    return Ensemble(updated,weights),{"log_predictive_probability":float(evidence),
                                    "weights":weights.tolist(),"bridges":diagnostics}


def site_predictions(model, sites, vent, cutoffs=(.1,1.,10.)):
    if not sites:
        return []
    xy = positions(sites,vent)
    field = model.distribution(xy)
    local = model.distribution(xy, observation=True)
    _,_,bary = model.members[0].geometry.locate(xy)
    results = []
    for i,site in enumerate(sites):
        result = {"id":site["id"],"site":site["name"],"reading":site["reading"],
                  "mesh_weight":float(bary[i].sum()),"latent_cm":{},"new_site_cm":{},
                  "probability_local_below":{}}
        results.append(result)
    for q,label in ((.025,"lower95"),(.05,"lower90"),(.5,"median"),(.975,"upper95")):
        latent, observed = (np.exp(d.quantile(q)) for d in (field,local))
        for i,result in enumerate(results):
            result["latent_cm"][label] = float(latent[i])
            result["new_site_cm"][label] = float(observed[i])
    for limit in cutoffs:
        values = local.cdf(np.log(limit))
        for value,result in zip(values,results,strict=True):
            result["probability_local_below"][str(float(limit))] = float(value)
    return results
