"""Matched known-deposit and survey-geometry tests of the published continuation."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import qmc

from .footprint_study import controls
from .nested_reach_challenge import known_design, matched_models
from src.empirical_isopach.core.audit import audited_data
from src.empirical_isopach.core.catalog import datasets
from src.empirical_isopach.models.directional.azimuth.calibration import calibrate
from src.empirical_isopach.models.directional.azimuth.report import model_for
from src.empirical_isopach.models.directional.ensemble.fields import inverse_cdf
from src.empirical_isopach.models.directional.ensemble.integrals import polar_nodes
from src.empirical_isopach.models.extent_integrals import parent_fields
from src.empirical_isopach.models.extent_review import ExtentFields, STATES
from src.empirical_isopach.models.extent_weights import calibration
from src.empirical_isopach.models.mass_mixture.bearings.field import BearingField,BearingSupport,phases,selected
from src.empirical_isopach.visualization.deposit import digest,read_json
from src.probabilistic_isopach.global_volume import EarthDomain

COMPASS = ("N","NE","E","SE","S","SW","W","NW")
LEVELS = (.5,1.,10.,100.)


def paired(fields, xy, cal):
    v = fields.variants(xy)
    w = fields.weights
    a = np.median(v[0],axis=0) if np.ptp(w)<1e-15 else inverse_cdf(v[0].T,w)[0]
    bases = [v[STATES.index(s)] for s in cal["states"]]
    counts = np.concatenate([p*w for p in cal["weights"]])
    b = inverse_cdf(np.concatenate(bases).T,counts)[0]
    return np.stack((a,b))


def compare(fields,lat,cal,truths=None,order=13):
    axis = EarthDomain.wgs84(lat).axis_scale
    totals,errors = [],[]
    for seed in (51071,68389):
        xy,w,r,theta = polar_nodes(qmc.Sobol(2,scramble=True,seed=seed).random_base2(order),axis)
        w = w/len(r)
        sector = ((theta+np.pi/8)//(np.pi/4)).astype(int)%8
        values = np.concatenate([paired(fields,xy[i:i+128],cal) for i in range(0,len(r),128)],axis=1)
        for j,label in enumerate(("parent","principal")):
            for i,region in enumerate(COMPASS):
                use = sector==i
                row = dict(scramble=seed,model=label,sector=region,km3=float(w[use]@np.exp(values[j,use])*1e-5))
                for cm in LEVELS:
                    row[f"area_{cm:g}cm_km2"] = float(w[use & (values[j]>=np.log(cm))].sum())
                totals.append(row)
            for label_truth,(truth,volume) in (truths or {}).items():
                z = truth(xy)
                for cm in LEVELS:
                    actual,pred = z>=np.log(cm),values[j]>=np.log(cm)
                    errors.append(dict(scramble=seed,model=label,truth=label_truth,truth_volume_km3=volume,
                        median_map_km3=float(w@np.exp(values[j])*1e-5),level_cm=cm,
                        false_area_km2=float(w[pred & ~actual].sum()),missed_area_km2=float(w[actual & ~pred].sum())))
    return totals,errors


def run(args):
    slug,family,seed,out,spheric,direction = args
    dest = out/f"{slug}-{family}-{seed}"
    dest.mkdir(parents=True,exist_ok=True)
    if (dest/"sectors.csv").exists():
        return
    if seed:
        data,truths = known_design(slug,seed)
        lat,key = 0.,"synthetic"
        if family=="nested-isopachs":
            _,parent,checks = matched_models(data,lat,55031+seed)
            parent = parent.reduced(512)
            ok = checks["importance"][-1]["passed"] and checks["source"]["passed"]
        elif family=="directional":
            weights = calibrate({"input_sha256":key,"vent_lonlat":[0.,0.]})
            parent = model_for(data,lat,direction/f"{slug}-{seed}",weights,32)
            ok = all(m.records[-1]["diagnostics"]["passed"] for m in parent.members)
        else:
            folder=spheric/f"{slug}-{seed}"
            r=read_json(folder/"spherical.json")
            archive = next(p for p in sorted(folder.glob("attempt-*.npz")) if digest(p)==r["posterior_sha256"])
            with np.load(archive) as raw:
                d,_,shape=selected(dict(raw),96)
            parent=BearingField(d,BearingSupport(data,lat),phases(shape))
            ok=r["parent"]["passed"]
        fields=ExtentFields(family,parent,data,lat)
    else:
        rec=next(r for r in datasets() if r["slug"]==slug)
        data,meta=audited_data(rec["path"],allow_intervals=True)
        lat,key,truths=meta["vent_lonlat"][1],meta["input_sha256"],None
        folder = controls(slug,rec)[0] if family=="nested-isopachs" else Path("eruptions")/slug/"results"/family
        fields=parent_fields(folder,family,data,lat,count=128 if family=="spherical-mixture" else 768)
        ok=None
    cal=calibration(family,key,exclude_slug=slug)
    totals,errors=compare(fields,lat,cal,truths)
    if errors:
        pd.DataFrame(errors).assign(fit_passed=ok).to_csv(dest/"known.csv",index=False)
    pd.DataFrame(totals).to_csv(dest/"sectors.csv",index=False)
    print(slug,family,seed,flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--directional-known-fits",type=Path)
    p.add_argument("--spherical-known-fits",type=Path)
    p.add_argument("--workers",type=int,default=2)
    a=p.parse_args()
    cases=[(r["slug"],f,0) for r in datasets() for f in ("directional","spherical-mixture","nested-isopachs")]
    if (a.directional_known_fits is None)!=(a.spherical_known_fits is None):
        p.error("Supply both known-deposit posterior directories")
    if a.directional_known_fits:
        cases.extend((s,f,seed) for s in ("smooth","missing_sector","invisible_sector") for seed in (90,91)
                     for f in ("directional","spherical-mixture","nested-isopachs"))
    with ProcessPoolExecutor(a.workers) as pool:
        list(pool.map(run,[(*x,a.output,a.spherical_known_fits,a.directional_known_fits) for x in cases]))
