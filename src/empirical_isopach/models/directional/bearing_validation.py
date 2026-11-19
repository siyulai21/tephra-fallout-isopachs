"""Buffered withholding for directional continuations and archive transfer assumptions."""
from pathlib import Path

import numpy as np
from scipy.special import logsumexp
from scipy.spatial import cKDTree

from ...contour_support import audit_contours
from ...core.geometry import survey_bounds
from ...validation import automatic_buffer
from .archive_readings import condition_model,log_visibility,positions,quantitative_model
from .bearing_estimates import packet
from .ensemble.members import run_members,splits
from .ensemble.validation import fold_record
from .geometry import Geometry,RADIUS,polar


def training_readings(data, test, sites, scenario, vent, kind, fold, geometry, buffer):
    selected = scenario["effective_limit_cm"]
    by_id = {site["id"]:site for site in sites}
    ids = list(selected)
    xy = positions([by_id[sid] for sid in ids],vent)
    if kind == "geographic":
        # Extend the coordinate-only fold assignment to the archive via its nearest quantitative site.
        # With ties at a boundary, remove the archive if any tied position is held out.
        all_dist = cKDTree(data.xy).query(xy)[0]
        test_dist = cKDTree(test.xy).query(xy)[0]
        in_group = test_dist <= all_dist + 1e-8
    else:
        angles = polar(xy,geometry.axis)[1]
        in_group = np.floor(angles/(np.pi/2)).astype(int).clip(0,3) == fold
        test_dist = cKDTree(test.xy).query(xy)[0]
    keep = ~in_group & (test_dist > buffer)
    return {**scenario,"effective_limit_cm":{k:selected[k] for k,ok in zip(ids,keep,strict=True) if ok}}


def heldout_visibility(model, sites, vent, limits):
    by_id = {s["id"]:s for s in sites}
    ids = list(limits)
    xy = positions([by_id[k] for k in ids],vent)
    def probability(points,scales):
        logs = []
        for member in model.members:
            ll = log_visibility(member,points,scales)
            logs.append(logsumexp(ll)-np.log(ll.size))
        return float(np.exp(logsumexp(logs,b=model.model_weights)))
    return {"archive_ids":ids,"assumed_effective_limits_cm":limits,
            "joint_predictive_probability":probability(xy,list(limits.values())),
            "individual_predictive_probabilities":{k:probability(xy[i:i+1],[limits[k]]) for i,k in enumerate(ids)}}


def independent_archive_prediction(training, model, sites, vent, scenario, held, remaining,
                                   folder, buffer, weights, already_conditioned):
    by_id={s["id"]:s for s in sites}
    held_xy=positions([by_id[k] for k in held],vent)
    distances=cKDTree(held_xy)
    keep=distances.query(training.xy)[0]>buffer
    original_remaining=remaining
    if remaining:
        rxy=positions([by_id[k] for k in remaining],vent)
        remaining={k:u for (k,u),d in zip(remaining.items(),distances.query(rxy)[0],strict=True) if d>buffer}
    meta={"archive_ids":list(held),"archive_training_ids":list(remaining),
          "quantitative_training_row_ids":training.row_ids[keep].tolist()}
    if not np.all(keep) or remaining!=original_remaining:
        training=training.subset(keep)
        try:
            Geometry(training,vent[1])
        except ValueError as exc:
            return {**meta,"status":"insufficient buffered support","reason":str(exc)}
        quant_work=Path(folder)/"quantitative"
        run_members(training,vent[1],quant_work,kinds=("full",))
        model=quantitative_model(training,vent[1],quant_work,weights)
        already_conditioned=False
    if remaining and not already_conditioned:
        subset={**scenario,"effective_limit_cm":remaining}
        model,_=condition_model(training,model,sites,vent,subset,Path(folder)/"conditioned")
    return {**meta,**heldout_visibility(model,sites,vent,held),"status":"estimated"}


def validate(data, meta, work, weights, archive, scenario, output, kinds=("geographic","quadrants"), count=64):
    validation,deletions = {},[]
    vent = meta["vent_lonlat"]
    buffer, buffer_policy = automatic_buffer(data)
    for kind in kinds:
        scores, folds = [],[]
        for fold,train,test in splits(data,vent[1],kind):
            state = fold_record(work,kind,fold,train,test)
            row = {k:state[k] for k in ("fold","training_row_ids","withheld_row_ids","status","passed")}
            folds.append(row)
            if state["status"] != "estimated":
                if scenario and kind=="quadrants" and state["status"]=="empty":
                    # An unsampled quantitative quadrant can still contain inspected archives.
                    whole=quantitative_model(data,vent[1],work,weights)
                    ids=scenario["effective_limit_cm"]
                    by_id={s["id"]:s for s in archive["sites"]}
                    xy=positions([by_id[k] for k in ids],vent)
                    quadrants=np.floor(polar(xy,whole.members[0].geometry.axis)[1]/(np.pi/2)).astype(int).clip(0,3)
                    held={k:u for (k,u),q in zip(ids.items(),quadrants,strict=True) if q==fold}
                    remaining={k:u for k,u in ids.items() if k not in held}
                    if held:
                        row["archive_only_holdout"]=independent_archive_prediction(data,whole,archive["sites"],vent,
                            scenario,held,remaining,Path(output)/kind/f"archive-only-{fold}",buffer,weights,False)
                continue
            model = quantitative_model(train,vent[1],work,weights,kind,fold)
            row["quantitative_fit_warnings"] = [m.geometry.policy.name for m in model.members[::4]
                                                 if not m.records[-1]["diagnostics"]["passed"]]
            row["passed"] = not row["quantitative_fit_warnings"]
            if scenario:
                sub = training_readings(data,test,archive["sites"],scenario,vent,kind,fold,
                                        model.members[0].geometry,buffer)
                row["archive_training_ids"] = list(sub["effective_limit_cm"])
                row["withheld_archives"] = [k for k in scenario["effective_limit_cm"] if k not in sub["effective_limit_cm"]]
                if sub["effective_limit_cm"]:
                    model,info = condition_model(train,model,archive["sites"],vent,sub,
                                                 Path(output)/kind/f"fold-{fold}")
                    row["archive_fit_warnings"] = [f"{b['policy']} (ratio {b['ratio']:g})" for b in info["bridges"]
                                                   if not b["sampling_passed"]]
                    row["passed"] &= not row["archive_fit_warnings"]
                held={k:scenario["effective_limit_cm"][k] for k in row["withheld_archives"]}
                if held:
                    row["archive_holdout"]=independent_archive_prediction(train,model,archive["sites"],vent,scenario,held,
                        sub["effective_limit_cm"],Path(output)/kind/f"archive-cross-{fold}",buffer,weights,True)
            dist = model.distribution(test.xy,observation=True,log_sd=test.log_sd)
            lp = dist.log_score(test.log_value,test.censored,test.lower_censored)
            scores.append(-float(lp.mean()))
            row.update(negative_log_score=scores[-1],individual_log_scores=lp.tolist(),
                       median_log_cm=dist.quantile(.5).tolist(),weights=model.model_weights.tolist())
            if kind == "geographic" and row["passed"]:
                deletions.append(packet(model,count))
        validation[kind] = {"folds":folds,"equal_region_negative_log_score":float(np.mean(scores)) if scores else None,
                            "buffer_km":buffer,"buffer_rule":buffer_policy,
                            "scored_rows":sum(len(r["withheld_row_ids"]) for r in folds if r["status"]=="estimated")}
    return validation,deletions


def screen(data, model, deletions, validation):
    bounds = survey_bounds(data.xy)
    corners = np.array([[x,y] for x in bounds[[0,2]] for y in bounds[[1,3]]])
    info = {"protocol":"supported-isopachs-v1","deletion_fit_status":[{k:r[k] for k in ("fold","status","passed")}
                  for r in validation["geographic"]["folds"]],"map_bounds_km":bounds.tolist()}
    if max(polar(corners,model.members[0].geometry.axis)[0]) > 2*RADIUS:
        return {**info,"status":"audit_window_leaves_physical_chart","policies":{},"refined_levels":[]}
    result,*_ = audit_contours(model,data,bounds,deletions)
    return {**result,**info,"status":"completed"}
