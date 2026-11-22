"""Publish conditional volumes, geographical predictions and finite-spread alternatives."""
import json
from pathlib import Path

import numpy as np
from scipy.special import logsumexp

from .extent_integrals import integrate, parent_fields, quantiles
from .extent_review import PROTOCOL, STATES
from .extent_validation import validate
from .extent_weights import ASSET, calibration
from ..visualization.deposit import read_json, write_json, volume_interval
from ..visualization.deposit_field import digest


def mix_volume(vol, weights, cal):
    indices = [STATES.index(k) for k in cal["states"]]
    return vol[indices].ravel(),(np.asarray(cal["weights"])[:,None]*weights).ravel()


def draw_check(vol,weights,cal):
    v,w = mix_volume(vol,weights,cal)
    q = quantiles(v,w)
    parts = []
    for start in (0,1):
        vs = v.reshape(-1,len(weights))[:,start::2].ravel()
        ws = w.reshape(-1,len(weights))[:,start::2].ravel()
        parts.append(quantiles(vs,ws/ws.sum()))
    discrepancy = float(np.max(np.ptp(parts,axis=0)/q))
    return q,discrepancy


def predictive_summary(frame, fits, cal):
    logs = np.array([frame[s+"_log_score"] for s in cal["states"]], dtype=float)
    ranks = np.array([frame[s+"_cdf"] for s in cal["states"]], dtype=float)
    weights = np.asarray(cal["weights"])
    frame = frame.copy()
    frame["principal_log_score"] = logsumexp(np.log(weights)[:,None]+logs,axis=0)
    frame["principal_cdf"] = weights@ranks
    result = {}
    for kind in sorted({f["design"] for f in fits}):
        g = frame.loc[frame.design == kind]
        records = [f for f in fits if f["design"] == kind]
        if g.empty:
            result[kind] = dict(negative_log_score=None, parent_matched_score=None,
                                state_scores={s: None for s in STATES}, fold_scores=[],
                                passed_folds=0, attempted_folds=len(records), all_folds_passed=False,
                                passed_only_score=None, n_rows=0, n_exact=0,
                                coverage95_exact=None, matched_parent_coverage95=None)
            continue
        data = g.loc[g.exact]
        columns = [s+"_log_score" for s in ("principal",*STATES)]
        scores = -g.groupby("fold")[columns].mean()
        passing = [f["fold"] for f in records if f["fit_passed"]]
        result[kind] = dict(negative_log_score=float(scores.principal_log_score.mean()),
                            parent_matched_score=float(scores.parent_log_score.mean()),
                            state_scores={s:float(scores[s+"_log_score"].mean()) for s in STATES},
                            fold_scores=scores.principal_log_score.tolist(),passed_folds=len(passing),attempted_folds=len(records),
                            all_folds_passed=bool(len(passing)==len(records) and len(records)==4),
                            passed_only_score=float(scores.loc[passing,"principal_log_score"].mean()) if passing else None,
                            n_rows=len(g),n_exact=len(data),coverage95_exact=float(data.principal_cdf.between(.025,.975).mean()) if len(data) else None,
                            matched_parent_coverage95=float(data.parent_cdf.between(.025,.975).mean()) if len(data) else None)
    return frame,result


def prepare(folder, family, data, meta, name, out, work, *, slug=None, precomputed=None):
    folder, out = Path(folder),Path(out)
    out.mkdir(parents=True,exist_ok=True)
    lat = meta["vent_lonlat"][1]
    rp = "azimuth/report.json" if family=="directional" else "report.json"
    packet = {"directional":"azimuth/fields.npz","spherical-mixture":"posterior.npz","nested-isopachs":"full.npz"}[family]
    parent = read_json(folder/rp)
    if parent.get("input",parent)["input_sha256"]!=meta["input_sha256"]:
        raise ValueError("Training and published data differ")
    cal = calibration(family,meta["input_sha256"],exclude_slug=slug)
    old_interval,old_ok = volume_interval(family,parent)
    identity = False
    if family=="spherical-mixture":
        with np.load(folder/packet) as a:
            identity = "angular" in a or "core_log_scale" in a
    if precomputed is None:
        frame, fits = validate(family,folder,data,lat,work)
        fields = parent_fields(folder,family,data,lat)
        vol, history = integrate(fields,lat)
        weights = fields.weights
    else:
        frame,fits,vol,weights,history = precomputed
    refinements = []
    for count in (512,1024,None):
        q,gap = draw_check(vol,weights,cal)
        refinements.append(dict(draws_per_state=len(weights),split_relative_difference=gap,
                                integration=history))
        if gap<.15 or count is None or identity or family=="directional":
            break
        if family=="nested-isopachs" and len(weights)>=4096:
            break
        fields = parent_fields(folder,family,data,lat,count=count if family=="spherical-mixture" else 4096)
        vol,history = integrate(fields,lat)
        weights = fields.weights
    np.savez_compressed(out/"volumes.npz",volume_km3=vol,weights=weights)
    frame,metrics = predictive_summary(frame,fits,cal)
    frame.to_csv(out/"heldout.csv.gz",index=False,compression={"method":"gzip","mtime":0},lineterminator="\n")
    if identity:
        interval,passed = old_interval,old_ok
    else:
        interval,passed = q[[0,2]].tolist(),bool(old_ok and history[-1]["passed"] and gap<.15)
    r = dict(protocol=PROTOCOL,family=family,name=name,input_sha256=meta["input_sha256"],
             parent_sha256={rp:digest(folder/rp),packet:digest(folder/packet)},calibration=cal,calibration_sha256=digest(ASSET),
             exact_parent_identity=identity,conditional_interval95_km3=list(interval),draw_checks_passed=bool(passed),
             parent_conditional_interval95_km3=list(old_interval),parent_draw_checks_passed=bool(old_ok),
             state_conditional_intervals95_km3={s:quantiles(v,weights)[[0,2]].tolist() for s,v in zip(STATES,vol,strict=True)},
             state_order=list(STATES),volume_draws_sha256=digest(out/"volumes.npz"),draw_refinements=refinements,
             validation=metrics,folds=fits,heldout_sha256=digest(out/"heldout.csv.gz"),
             interpretation="Conditional on the recorded sites, likelihood, family and continuation choices. Independent whole-Earth integrals of coherent fields give the interval. Complete-contour eligibility is assessed separately.")
    write_json(out/"report.json",r)
    return r


def prior_sensitivity(folder, data, meta, work, key, fit, cal):
    """Independently integrate a signed fixed-prior spherical posterior."""
    path = Path(work)/key/fit["posterior_file"]
    h = digest(path)
    if h!=fit["posterior_sha256"]:
        raise ValueError(f"Spherical prior posterior differs for {key}")
    with np.load(path) as raw:
        identity = "angular" in raw or "core_log_scale" in raw
    if identity:
        v = fit["volume"]
        return dict(posterior_sha256=h,conditional_interval=[v["lower95"],v["upper95"]],
                    passed=bool(fit["diagnostics"]["passed"]),exact_identity="source-centred kernel",
                    parent_fit_passed=fit["diagnostics"]["passed"])
    refines = []
    for count in (128,256,512,1024):
        fields = parent_fields(folder,"spherical-mixture",data,meta["vent_lonlat"][1],
                               states=cal["states"],spherical=path,count=count)
        volumes,history = integrate(fields,meta["vent_lonlat"][1],orders=(10,11,12,13,14))
        weights = np.concatenate([w*fields.weights for w in cal["weights"]])
        q = quantiles(volumes.ravel(),weights)
        parts = []
        for i in (0,1):
            w = weights.reshape(volumes.shape)[:,i::2].ravel()
            parts.append(quantiles(volumes[:,i::2].ravel(),w/w.sum()))
        gap = float(np.max(np.ptp(parts,axis=0)/q))
        refines.append(dict(draws_per_chain=count,conditional_interval=q[[0,2]].tolist(),
                            split_relative_difference=gap,integration=history))
        if gap<.15 and history[-1]["passed"]:
            break
    return dict(posterior_sha256=h,conditional_interval=q[[0,2]].tolist(),draws_per_chain=count,
                split_relative_difference=gap,passed=bool(history[-1]["passed"] and gap<.15 and fit["diagnostics"]["passed"]),
                parent_fit_passed=fit["diagnostics"]["passed"],refinements=refines)


def review_dataset(folder, family, data, meta, name, *, work=None, output=None, input_path=None):
    """Normal entry point after fitting a new eruption through any of the global families."""
    from .extent_support import save_screen, support_screen
    folder = Path(folder)
    out = Path(output or folder/"extent")
    if work is None:
        work = folder/("work/ensemble" if family=="directional" else "posteriors")
    from ..core.catalog import dataset_record
    target = dataset_record(input_path) if input_path is not None else None
    r = prepare(folder,family,data,meta,name,out,Path(work),slug=target["slug"] if target else None)
    cal = r["calibration"]
    save_screen(out/"support.json.gz",support_screen(folder,family,data,meta,Path(work),cal))
    if family=="spherical-mixture":
        parent = read_json(folder/"report.json")
        priors = {k:prior_sensitivity(folder,data,meta,work,k,fit,cal)
                  for k,fit in parent["fits"].items() if k!="baseline"}
        priors["baseline"] = dict(posterior_sha256=parent["fits"]["baseline"]["posterior_sha256"],
                                 conditional_interval=r["conditional_interval95_km3"],passed=r["draw_checks_passed"])
        write_json(out/"prior-sensitivity.json",priors)
    return r
