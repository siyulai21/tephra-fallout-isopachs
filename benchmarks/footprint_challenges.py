"""Independent campaigns, known deposits, and repeated sampling on a circular deposit."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from hashlib import sha256
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import gammainc

from .extent_challenges import paired
from .mass_mixture import SCENARIOS
from .nested_reach_challenge import ONE_SIDE, known_design, matched_models
from .nested_reach_field import FIELD_CASES, partition
from src.empirical_isopach.core.audit import audited_data
from src.empirical_isopach.core.catalog import resolve_dataset
from src.empirical_isopach.core.observations import Observations
from src.empirical_isopach.models.extent_review import ExtentFields
from src.empirical_isopach.models.extent_validation import scores
from src.empirical_isopach.models.extent_weights import calibration
from src.empirical_isopach.models.nested.footprints.posterior import FootprintField, predictions
from src.empirical_isopach.models.nested.footprints.release import fit_part
from src.empirical_isopach.models.nested.geometry import positions, DIAMETER
from src.empirical_isopach.models.nested.reach import ReachDraws
from src.empirical_isopach.models.nested.reach_integrals import nodes
from src.empirical_isopach.models.nested.continuation import load_packet
from src.empirical_isopach.models.nested.study import write_json
from src.empirical_isopach.models.nested.report import MapView
from src.probabilistic_isopach.global_volume import EarthDomain


def rotated_spokes(name):
    # Same true field, radii, and noise in both designs. Only the survey rotates.
    rng = np.random.default_rng(56421)
    radius, theta = np.meshgrid(np.geomspace(8., 550., 9), np.deg2rad([10., 40., 130., 260.]))
    theta = theta.ravel() + np.deg2rad(23. if name.endswith("23") else 0.)
    r = radius.ravel()
    axis = EarthDomain.wgs84(0.).axis_scale
    xy = np.column_stack((axis*r*np.sin(theta), r*np.cos(theta)/axis))
    def truth(xy):
        return np.log(100.) - positions(xy,0.)[0]/80.
    total = 2*np.pi*1e-5*100.*80.**2*gammainc(2.,DIAMETER/80.)
    y = truth(xy) + rng.normal(0., .12, len(xy))
    data = Observations(xy,y,np.zeros(len(y),bool),np.zeros(len(y)),np.arange(len(y))).canonical()
    return data, {"specified": (truth, total)}


def synthetic_id(data):
    h = sha256()
    for a in (data.xy, data.log_value, data.log_sd, data.censored, data.lower_censored, data.interval_log_upper, data.row_ids):
        h.update(a.tobytes())
    return h.hexdigest()


def known(args):
    slug, seed, root = args
    dest = root/f"{slug}-{seed}"
    dest.mkdir(parents=True, exist_ok=True)
    data, truths = rotated_spokes(slug) if slug.startswith("survey_rotation") else known_design(slug,seed)
    meta = {"input_sha256":synthetic_id(data), "vent_lonlat":[0.,0.]}
    receipt = fit_part(dest, "footprints", data, None, meta)
    selected = dest/"refinement" if receipt["identity"]["initial"]==12288 else dest
    new = FootprintField.load(selected/"footprints.npz")
    whole, legacy, diagnostics = matched_models(data, 0., 55031+seed)
    finite = ExtentFields("nested-isopachs", legacy.reduced(512), data, 0.)
    cal = calibration("nested-isopachs", meta["input_sha256"])
    # Keep coherent whole-deposit volume draws independent of the median-map statistic.
    whole_view = MapView(whole.reduced(512))
    new_view = MapView(new.reduced(512))
    volumes = {k:np.quantile(m.volume(),[.025,.5,.975]) for k,m in (("whole area",whole),("footprints",new))}
    rows = []
    for scramble in (42983, 84779):
        xy, weight, _, theta = nodes(14, scramble, 0.)
        weight /= len(weight)
        old = np.concatenate([paired(finite,xy[i:i+128],cal)[1] for i in range(0,len(xy),128)])
        fields = (old, whole_view.central(xy)*np.log(10.), new_view.central(xy)*np.log(10.))
        for method, fitted in zip(("finite corridors","whole area","footprints"), fields, strict=True):
            volume = 1e-5*weight @ np.exp(fitted)
            for key,(surface,total) in truths.items():
                truth = surface(xy)
                for cm in (.5,1.,10.,100.):
                    present, predicted = truth>=np.log(cm), fitted>=np.log(cm)
                    rows.append(dict(scenario=slug,seed=seed,input_sha256=meta["input_sha256"],truth=key,
                                     method=method,scramble=scramble,level_cm=cm,true_km3=total,
                                     median_map_km3=volume,true_area_km2=weight[present].sum(),
                                     false_km2=weight[predicted & ~present].sum(),missed_km2=weight[present & ~predicted].sum(),
                                     west_false_km2=weight[predicted & ~present & (np.sin(theta)<0.)].sum(),
                                     west_missed_km2=weight[present & ~predicted & (np.sin(theta)<0.)].sum(),
                                     lower95=volumes[method][0] if method in volumes else None,
                                     upper95=volumes[method][-1] if method in volumes else None,
                                     fit_passed={"footprints":receipt["numerics"]["passed"],"whole area":diagnostics["reference"]["passed"],
                                                 "finite corridors":diagnostics["source"]["passed"] and diagnostics["importance"][-1]["passed"]}[method]))
    pd.DataFrame(rows).to_csv(dest/"recovery.csv",index=False)
    print(slug,seed,flush=True)


def survey(args):
    slug, root = args
    dest = root/slug
    dest.mkdir(parents=True, exist_ok=True)
    if slug == "pinatubo-all-offshore":
        path = resolve_dataset("pinatubo-1991-layer-c")
        data,meta = audited_data(path)
        # The source audit identifies marine reports. Check the actual accepted row IDs.
        audit = pd.read_csv("data/audits/Pinatubo-1991/full-evidence.csv")
        registered = pd.read_csv(path)
        marine = audit.loc[audit["evidence_id"].str.startswith(("WW-", "W04-", "H01-", "S99-")),"evidence_id"]
        held = registered.EvidenceID.isin(marine).to_numpy()[data.row_ids-2]
        if held.sum()!=34:
            raise ValueError("Pinatubo marine group differs from the audited 34 records")
        train,test = data.subset(~held),data.subset(held)
        whole, legacy, checks = matched_models(train, meta["vent_lonlat"][1], 71183)
        old_ok, whole_ok = checks["source"]["passed"] and checks["importance"][-1]["passed"],checks["reference"]["passed"]
    else:
        train,test,meta,_ = partition(slug,dest)
        old_dir = Path("comparisons/nested-reach/field-surveys")/slug
        import json
        old_info = json.loads((old_dir/"validation.json").read_text())
        if old_info["identity"]["training_rows"]!=train.row_ids.tolist() or old_info["identity"]["test_rows"]!=test.row_ids.tolist():
            raise ValueError("External survey row identities differ")
        # training-parent.npz contains the original continued field.
        whole = load_packet(old_dir/"training-parent.npz")
        legacy = ReachDraws.load(old_dir/"corridor.npz")
        old_ok = old_info["revised_source_passed"]
        whole_ok = old_info["numerics"]["passed"]
    r = fit_part(dest,"footprints",train,test,meta)
    finite = ExtentFields("nested-isopachs",legacy,train,meta["vent_lonlat"][1])
    cal = calibration("nested-isopachs",meta["input_sha256"],exclude_slug="pinatubo-1991-layer-c" if "pinatubo" in slug else slug)
    vals = scores(finite,test)
    from scipy.special import logsumexp
    logp = logsumexp(np.log(cal["weights"])[:,None]+[vals[s+"_log_score"] for s in cal["states"]],axis=0)
    rank = np.asarray(cal["weights"]) @ np.array([vals[s+"_cdf"] for s in cal["states"]])
    rows = []
    for method,preds,passed in (("whole area",predictions(whole,test),whole_ok),
                                ("finite corridors", [dict(row_id=int(i),exact=bool(e),log_score=float(lp),cdf=float(p))
                                                     for i,e,lp,p in zip(test.row_ids,~test.censored,logp,rank,strict=True)],old_ok),
                                ("footprints", r["heldout"], r["numerics"]["passed"])):
        rows.extend(dict(survey=slug,method=method,input_sha256=meta["input_sha256"],fit_passed=passed,**v) for v in preds)
    pd.DataFrame(rows).to_csv(dest/"predictions.csv",index=False)
    print(slug,flush=True)


if __name__ == "__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output",required=True,type=Path)
    p.add_argument("--workers",type=int,default=2)
    p.add_argument("--design",required=True,choices=("known","surveys"))
    a=p.parse_args()
    if a.design=="known":
        jobs=[(s,k,a.output) for k in (90,91) for s in (*SCENARIOS,"invisible_sector")]
        jobs += [(s,k,a.output) for k in (1201,1202) for s in ONE_SIDE]
        jobs += [(s,56421,a.output) for s in ("survey_rotation_0","survey_rotation_23")]
    else:
        jobs=[(s,a.output) for s in (*FIELD_CASES,"pinatubo-all-offshore")]
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        list(pool.map(known if a.design=="known" else survey,jobs))
