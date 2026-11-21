"""Frozen fallout and local archive-thickening challenges on two actual survey layouts."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from src.empirical_isopach.core.audit import audited_data
from src.empirical_isopach.core.catalog import datasets
from src.empirical_isopach.core.observations import Observations
from src.empirical_isopach.models.nested.continuation import check_and_refine
from src.empirical_isopach.models.nested.geometry import DIAMETER, positions
from src.empirical_isopach.models.nested.inference import fit
from src.empirical_isopach.models.nested.study import write_json


def design(layout, latitude, remobilized):
    radius,angle = positions(layout.xy,latitude)
    scale = np.median(radius)
    bearing = np.angle(np.mean(np.exp(1j*angle[radius>=scale])))
    x,y = radius*np.cos(angle-bearing)/scale,radius*np.sin(angle-bearing)/scale
    # Profiles, adjustments and reporting limits are shared across the layouts.
    # These are entirely simulated thicknesses, using real coordinates only.
    profiles = ((50.,.25,.5,.2),(2.,1.,1.,.4),(.03,0.,1.8,1.8))
    truth = sum(peak*np.exp(-.5*((x-centre)/along)**2-.5*(y/across)**2)
                for peak,centre,along,across in profiles)
    pieces = np.array([2*np.pi*scale**2*peak*along*across*1e-5 for peak,_,along,across in profiles])
    tail_bound = sum(mass*np.exp(-.5*(max(DIAMETER/scale-abs(centre),0.)/max(along,across))**2)
                     for mass,(_,centre,along,across) in zip(pieces,profiles,strict=True))
    addition = 5.*np.exp(-.5*((x-1.9)/.55)**2-.5*(y/.3)**2) if remobilized else np.zeros(len(x))
    observed = (truth+addition)*np.exp(np.random.default_rng(41027).normal(0,.25,len(x)))
    bound = observed<.02
    synthetic = Observations(layout.xy,np.log(np.maximum(observed,.02)),bound,np.zeros(len(x)),layout.row_ids).canonical()
    return synthetic,{"bulk_km3":float(pieces.sum()),"outside_earth_upper_bound_km3":float(tail_bound),
                      "median_site_radius_km":float(scale),"bearing_deg":float(np.rad2deg(bearing)%360),
                      "observations":len(x),"upper_limits":int(bound.sum()),"sites_with_added_signal_above_0_1_cm":int(sum(addition>.1))}


def run(job):
    record,remobilized,output = job
    layout,meta = audited_data(record["path"])
    lat = meta["vent_lonlat"][1]
    data,truth = design(layout,lat,remobilized)
    rows = {"layout":record["slug"],"layout_sha256":meta["input_sha256"],"local_archive_signal":remobilized,
            "truth":truth,"seed":41027,"simulated_log_cm_or_upper_limit":data.log_value.tolist(),
            "row_ids":data.row_ids.tolist(),"upper_limits":data.censored.tolist(),"fits":{}}
    if truth["outside_earth_upper_bound_km3"]>1e-7*truth["bulk_km3"]:
        raise ArithmeticError("Refine the finite-Earth reference volume for this layout")
    for name,budget in (("initial",False),("continued",True)):
        model,sampling = fit(data,lat,budget=budget)
        if budget:
            _,model,sampling=check_and_refine(model,data,sampling)
        q=np.quantile(model.volume(),[.025,.5,.975])
        rows["fits"][name]={"volume_km3":q.tolist(),"numerics":sampling,
                            "covered":bool(q[0]<=truth["bulk_km3"]<=q[2]),
                            "log_error":float(np.log(q[1]/truth["bulk_km3"]))}
    write_json(output/(record["slug"]+("-archive" if remobilized else "-primary")+".json"),rows)
    print(record["slug"],remobilized,truth["bulk_km3"],[v["volume_km3"][1] for v in rows["fits"].values()],flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output",required=True,type=Path)
    p.add_argument("--workers",default=1,type=int)
    a=p.parse_args()
    a.output.mkdir(exist_ok=True,parents=True)
    chosen=[r for r in datasets() if r["slug"] in ("aso-4","ilopango-tbj")]
    with ProcessPoolExecutor(a.workers) as pool:
        list(pool.map(run,[(r,b,a.output) for r in chosen for b in (False,True)]))
