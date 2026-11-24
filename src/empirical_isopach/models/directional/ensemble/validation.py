"""Matched geographic and azimuthal refits and independent calibration evidence."""
import json

import numpy as np
from scipy.special import logsumexp

from ....core.validation import interval_metrics
from ....core.distributions import LogMixture
from ....validation import automatic_buffer
from ..study import digest
from .fields import Ensemble, load_members
from .members import MEMBERS, NAMES, directory, splits


def fold_record(work, kind, i, train, test):
    result = {"fold":i, "withheld_row_ids":test.row_ids.tolist(),
              "training_row_ids":[] if train is None else train.row_ids.tolist()}
    states = {}
    for p in MEMBERS:
        folder = directory(work, p, kind, i)
        status = json.loads((folder/"status.json").read_text())
        if status["status"] == "estimated":
            status["design_sha256"] = digest(folder/"design.json")
        states[p.name] = status
    available = all(s["status"] == "estimated" for s in states.values())
    return {**result, "members":states, "status":"estimated" if available else
            ("empty" if not len(test.xy) else "insufficient support"),
            "passed":bool(available and all(s["passed"] for s in states.values()))}


def component_scores(mixture, sizes, test):
    scores = []
    start = 0
    for n in sizes:
        part = slice(start, start+n)
        d = LogMixture(np.full(n, 1/n), mixture.means[part], mixture.variances[part])
        scores.append(d.score(test))
        start += n
    return np.column_stack(scores)


def calibration_case(rec, data, meta, work):
    groups = []
    latitude = meta["vent_lonlat"][1]
    for i, train, test in splits(data, latitude, "geographic"):
        f = fold_record(work, "geographic", i, train, test)
        entry = {"fold":i, "row_ids":f["withheld_row_ids"], "training_row_ids":f["training_row_ids"],
                 "usable":f["passed"], "status":f["status"],
                 "failed_members":[p for p,s in f["members"].items() if not s.get("passed", False)]}
        if f["status"] == "estimated":
            model = Ensemble(load_members(train, latitude, work, "geographic", i, n=None), np.ones(len(MEMBERS)))
            dist = model.distribution(test.xy, observation=True, log_sd=test.log_sd)
            entry["log_scores"] = component_scores(dist, model.sizes, test).tolist()
        groups.append(entry)
    return {"slug":rec["slug"], "input_sha256":meta["input_sha256"], "vent_lonlat":meta["vent_lonlat"],
            "folds":groups}


def validate(data, latitude, work, weights, kind):
    buffer, rule = automatic_buffer(data)
    fits, region_scores, equal_scores, reference_scores = [], [], [], []
    collected = {k:np.full(len(data.xy), np.nan) for k in ("median", "lower", "upper", "log_score")}
    folds = np.full(len(data.xy), -1)
    index = {j:i for i,j in enumerate(data.row_ids)}
    for i, train, test in splits(data, latitude, kind):
        f = fold_record(work, kind, i, train, test)
        fits.append(f)
        ids = [index[k] for k in test.row_ids]
        folds[ids] = i
        if f["status"] != "estimated":
            continue
        model = Ensemble(load_members(train, latitude, work, kind, i, n=None), weights)
        d = model.distribution(test.xy, observation=True, log_sd=test.log_sd)
        scores = component_scores(d, model.sizes, test)
        lp = logsumexp(scores, b=model.model_weights, axis=1)
        collected["log_score"][ids] = lp
        for k, p in (("median", .5), ("lower", .025), ("upper", .975)):
            collected[k][ids] = d.quantile(p)
        region_scores.append(-float(lp.mean()))
        equal_scores.append(-float((logsumexp(scores,axis=1)-np.log(len(MEMBERS))).mean()))
        reference_scores.append(-float(scores[:,NAMES.index("reference")].mean()))
        f.update(negative_log_score=region_scores[-1], uniform_score=equal_scores[-1],
                 reference_score=reference_scores[-1], member_log_scores=scores.tolist())
    ok = np.isfinite(collected["log_score"])
    def average(x):
        return float(np.mean(x)) if x else None
    return {"buffer_km":buffer,"buffer_rule":rule,"row_ids":data.row_ids.tolist(),"folds":folds.tolist(),
            "fits":fits,"equal_region_log_score":average(region_scores),"uniform_score":average(equal_scores),
            "reference_score":average(reference_scores),"scored_rows":int(ok.sum()),"total_rows":len(ok),
            "all_estimable_passed":all(f["passed"] for f in fits if f["status"]=="estimated"),
            "metrics":interval_metrics(data.subset(ok), *(collected[k][ok] for k in collected)) if ok.any() else {},
            "predictions":{k:[float(a) if np.isfinite(a) else None for a in v] for k,v in collected.items()}}


def support_audit(data, latitude, work, full_model, validation):
    from ....contour_support import audit_contours
    from ....core.geometry import survey_bounds
    from ..geometry import RADIUS, polar
    bounds = survey_bounds(data.xy)
    extremes = np.array([[x,y] for x in bounds[[0,2]] for y in bounds[[1,3]]])
    checks = [{k:f[k] for k in ("fold","status","passed")} for f in validation["fits"]]
    info = {"protocol":"supported-isopachs-v1", "deletion_fit_status":checks, "map_bounds_km":bounds.tolist(),
            "posterior_fields_per_structure":full_model.sizes, "surface":"Weighted pointwise latent median"}
    if polar(extremes, full_model.members[0].geometry.axis)[0].max() > 2*RADIUS:
        return {**info,"status":"audit_window_leaves_physical_chart","policies":{},"refined_levels":[]}
    deletions = []
    for f in validation["fits"]:
        if f["passed"]:
            train = data.subset(np.isin(data.row_ids, f["training_row_ids"]))
            members = load_members(train, latitude, work, "geographic", f["fold"],n=len(full_model.members[0].index))
            deletions.append(Ensemble(members, full_model.model_weights))
    result, *_ = audit_contours(full_model, data, bounds, deletions)
    return {**result, **info, "status":"completed"}
