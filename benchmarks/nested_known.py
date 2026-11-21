"""Fixed known-volume and contour challenges on the previously published sampling designs."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from hashlib import sha256
from pathlib import Path
import json

import numpy as np
from scipy.stats import qmc

from .directional import trial
from .mass_mixture import SCENARIOS, case
from src.empirical_isopach.models.directional.azimuth.calibration import calibrate
from src.empirical_isopach.models.directional.azimuth.report import model_for
from src.empirical_isopach.models.directional.ensemble.integrals import polar_nodes
from src.empirical_isopach.models.directional.ensemble.members import NAMES, run_members
from src.empirical_isopach.models.directional.gap_amount import integrated_amount, weight_summary
from src.empirical_isopach.models.directional.ensemble.fields import inverse_cdf
from src.empirical_isopach.models.nested.continuation import check_and_refine
from src.empirical_isopach.models.nested.geometry import positions
from src.empirical_isopach.models.nested.inference import fit
from src.empirical_isopach.models.nested.study import write_json
from src.probabilistic_isopach.global_volume import EarthDomain


def directional_volumes(data,name,seed,output,cache,signatures):
    work = (cache or output/"directional-cache")/f"{name}-{seed}"
    if cache is None:
        run_members(data,0.,work,kinds=("full",))
    for policy in NAMES:
        design = json.loads((work/"fits"/policy/"design.json").read_text())
        if design["signature"]["arrays_sha256"]!=signatures:
            raise ValueError("Cached directional challenge differs from the accepted synthetic rows")
    weights = calibrate({"input_sha256":"synthetic","vent_lonlat":[0.,0.]})
    ref = model_for(data,0.,work,weights,64)
    mass,history = integrated_amount(ref)
    return {"candidate_quantiles_km3":weight_summary(mass,ref.weights).tolist(),
            "quantiles_km3":inverse_cdf(mass[1],ref.weights,(.025,.5,.975)).tolist(),
            "integration":history,"member_passed":[m.records[-1]["diagnostics"]["passed"] for m in ref.members]}


def challenge(job):
    name,seed,output,cache,directional,*options = job
    budget = options[0] if options else True
    output.mkdir(parents=True,exist_ok=True)
    data,totals = trial(name,seed)
    data = data.canonical()
    full,num = fit(data,0.,budget=budget)
    unrestricted = np.quantile(full.volume(),[.025,.5,.975]).tolist()
    if budget:
        raw,full,num = check_and_refine(full,data,num)
        unrestricted = np.quantile(raw.volume(),[.025,.5,.975]).tolist()
    q = np.quantile(full.volume(),[.025,.5,.975])
    model = full.reduced(384)
    axis = EarthDomain.wgs84(0).axis_scale
    if name=="invisible_sector":
        def truth(xy):
            return 5.-positions(xy,0.)[0]/60.
        def hidden(xy):
            d2 = np.sum((xy*[1/axis,axis]-[100.,100.])**2,axis=1)
            return np.logaddexp(truth(xy),np.log(np.maximum(1e-300,2000*np.maximum(1-d2/50**2,0.)**3)))
        fields = [("specified",truth,totals["total_km3"]),("hidden",hidden,totals["same_data_hidden_total_km3"])]
    else:
        _,total,truth = case(name,seed,return_field=True)
        fields = [("specified",truth,total)]
    keys = ("xy","log_value","censored","lower_censored","log_sd","row_ids")
    signatures = {k:sha256(np.ascontiguousarray(getattr(data,k)).tobytes()).hexdigest() for k in keys}
    answer = {"scenario":name,"seed":seed,"quantiles_km3":q.tolist(),"numerics":num,
              "input_hashes":signatures,"truths":{}}
    scrambles,order = ((48529,29111),12) if seed in (10,11) else ((85237,38939),14)
    count = len(scrambles)*2**order
    answer["area_quadrature"] = {"scrambles":list(scrambles),"nodes_per_scramble":2**order}
    answer["unrestricted_volume_km3"] = unrestricted
    for label,field,total in fields:
        areas = np.zeros((3,4))
        for scramble in scrambles:
            uv = qmc.Sobol(2,scramble=True,seed=scramble).random_base2(order)
            for batch in np.array_split(uv,32):
                xy,weights,_,_ = polar_nodes(batch,axis)
                actual = field(xy)
                pred = np.median(model.field(xy),axis=0)
                for j,cm in enumerate((1.,10.,100.)):
                    a,b = actual>=np.log(cm),pred>=np.log(cm)
                    areas[j] += [weights@(b&~a),weights@(a&~b),weights@(a&b),(weights**2)@(a|b)]
        rows = []
        for cm,(fp,fn,hit,squared) in zip((1.,10.,100.),areas,strict=True):
            union = fp+fn+hit
            rows.append({"level_cm":cm,"false_km2":fp/count,"missed_km2":fn/count,
                         "iou":hit/union if union else None,"effective_union_nodes":union**2/squared if squared else 0.})
        answer["truths"][label] = {"volume_km3":total,"covered":bool(q[0]<=total<=q[2]),
                                    "log_volume_error":float(np.log(q[1]/total)),"areas":rows}
    if directional and seed in (90,91):
        answer["directional"] = directional_volumes(data,name,seed,output,cache,signatures)
    write_json(output/f"{name}-{seed}.json",answer)
    print(name,seed,q.tolist(),flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--workers",type=int,default=1)
    p.add_argument("--directional-cache",type=Path)
    p.add_argument("--with-directional",action="store_true")
    p.add_argument("--initial-profile",action="store_true",help="Use the archived prior on slopes")
    a = p.parse_args()
    jobs = [(name,seed,a.output,a.directional_cache,a.with_directional,not a.initial_profile) for seed in (10,11,90,91)
            for name in (*SCENARIOS,"invisible_sector") if name!="invisible_sector" or seed>=90]
    with ProcessPoolExecutor(a.workers) as ex:
        list(ex.map(challenge,jobs))


if __name__=="__main__":
    main()
