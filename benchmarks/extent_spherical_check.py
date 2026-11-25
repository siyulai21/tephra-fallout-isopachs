"""Independent volume quadrature drawn directly from the normalized spherical kernels."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from .publication import published_datasets, result_home

import numpy as np
from scipy.stats import qmc, spearmanr
from scipy.spatial import cKDTree

from src.empirical_isopach.core.audit import audited_data
from src.empirical_isopach.core.catalog import datasets
from src.empirical_isopach.models.continuation_geometry import SampleLayout
from src.empirical_isopach.models.mass_mixture.bearings.field import BearingSupport, phases, selected, multiplier, POWERS
from src.empirical_isopach.models.mass_mixture.bearings.flank import FlankKernels
from src.empirical_isopach.models.mass_mixture.bearings.volume import mass_nodes
from src.empirical_isopach.models.mass_mixture.geometry import RADIUS, centers, from_sphere, to_sphere
from src.empirical_isopach.models.extent_integrals import quantiles
from src.empirical_isopach.visualization.deposit import read_json,write_json


def run(job):
    rec,out,review_home=job
    slug=rec["slug"]
    family="spherical-mixture"
    home=result_home(rec)/family
    review=(review_home/slug/family if review_home else home)/"extent"
    if (out/(slug+".json")).is_file():
        return
    r=read_json(review/"report.json")
    if r["exact_parent_identity"]:
        write_json(out/(slug+".json"),{"exact_parent_identity":True})
        return
    states=r["state_order"]
    count=r["draw_refinements"][-1]["draws_per_state"]//12
    data,meta=audited_data(rec["path"])
    lat=meta["vent_lonlat"][1]
    layout,support=SampleLayout(data,lat),BearingSupport(data,lat)
    with np.load(home/"posterior.npz") as a:
        ps,_,shape=selected(dict(a),count)
    with np.load(review/"volumes.npz") as a:
        all_volume=a["volume_km3"].reshape((len(states),len(POWERS),-1))
    angles=phases(shape)
    center=centers(ps["log_scale"],ps["offset"])
    pos=to_sphere(data.xy[~data.censored | data.lower_censored],lat)
    distance=cKDTree(pos).query(center.reshape(-1,3))[0].reshape(center.shape[:2])*RADIUS
    nearest=(distance*ps["weights"]).sum(axis=1)
    v=np.median(all_volume[0],axis=0)
    high=v>=np.quantile(v,.975)
    cause=dict(volume_vs_nearest_positive_spearman=float(spearmanr(v,nearest).statistic),
               volume_vs_exponent_spearman=float(spearmanr(v,ps["exponent"]).statistic),
               median_centre_to_positive_km=float(np.median(nearest)),
               upper_tail_centre_to_positive_km=float(np.median(nearest[high])))
    # Paired stratified positions within each actual chain, keeping its published fan phase.
    ids=(np.arange(4)[:,None]*count+np.round(np.linspace(0,count-1,min(count,96))).astype(int)).ravel()
    cal=r["calibration"]
    columns=[states.index(k) for k in cal["states"]]
    weights=np.repeat(cal["weights"],len(ids)*len(POWERS))/(len(ids)*len(POWERS))
    reference=quantiles(all_volume[columns][:,:,ids].ravel(),weights)
    attempts=[]
    for order in (9,11,13):
        nodes=[qmc.Sobol(4,scramble=True,seed=s).random_base2(order) for s in (91823,27239)]
        vols=np.empty((2,len(columns),len(POWERS),len(ids)))
        for j,k in enumerate(ids):
            d={key:v[k] for key,v in ps.items()}
            fk=FlankKernels({key:val[k:k+1] for key,val in ps.items()},data,lat)
            for m,u in enumerate(nodes):
                unit,weight=mass_nodes(d,u)
                xy=from_sphere(unit,lat)
                base,*revised=fk.variants(xy,layout,states)
                h,theta=support.evaluate(xy)
                bearing=np.stack([multiplier(h,theta,angles[k],p) for p in POWERS])
                for opt,field in enumerate(revised):
                    retained=weight*np.exp((field-base)[0])
                    vols[m,opt,:,j]=np.exp(d["log_volume"])*(bearing@retained)/weight.sum()
        q=quantiles(vols.mean(0).ravel(),weights)
        trials=np.array([quantiles(v.ravel(),weights) for v in vols])
        scramble=float(np.max(np.ptp(trials,axis=0)/q))
        diff=float(np.max(np.abs(q/reference-1.)))
        attempts.append(dict(points_per_scramble=2**order,normalized_kernel_q025_q50_q975=q.tolist(),
                             relative_quantile_difference=diff,relative_scramble_spread=scramble))
        if scramble<.05 and diff<.08:
            break
    write_json(out/(slug+".json"),dict(pair_count=len(ids),geographical_q025_q50_q975=reference.tolist(),
               attempts=attempts,passed=bool(scramble<.05 and diff<.08),mechanism=cause))
    print(slug,round(diff,3),round(scramble,3),flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--review-root",type=Path)
    p.add_argument("--workers",type=int,default=2)
    p.add_argument("--case",nargs="+")
    a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=True)
    with ProcessPoolExecutor(a.workers) as pool:
        list(pool.map(run,[(r,a.output,a.review_root) for r in (datasets() if a.review_root else published_datasets()) if not a.case or r["slug"] in a.case]))
