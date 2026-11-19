"""Reproduce finite-bearing, uncertain-azimuth maps, cross-validation and synthetic tests."""
import argparse
from pathlib import Path

import numpy as np
from scipy.stats import qmc

from benchmarks.directional import trial
from benchmarks.mass_mixture import SCENARIOS, case
from src.empirical_isopach.core.catalog import datasets
from src.empirical_isopach.models.directional.azimuth import report
from src.empirical_isopach.models.directional.azimuth.calibration import calibrate, case_scores
from src.empirical_isopach.models.directional.azimuth.field import CONCENTRATIONS
from src.empirical_isopach.models.directional.ensemble.fields import inverse_cdf
from src.empirical_isopach.models.directional.ensemble.integrals import polar_nodes
from src.empirical_isopach.models.directional.ensemble.members import NAMES, run_members
from src.empirical_isopach.models.directional.geometry import polar
from src.empirical_isopach.models.directional.study import write_json
from src.empirical_isopach.models.mass_mixture.study import audited_data


def synthetic(work,output,scenario,seed):
    data, volumes = trial(scenario,seed)
    weights = calibrate({"input_sha256":"synthetic","vent_lonlat":[0.,0.]})
    model = report.model_for(data,0.,work,weights,64)
    axis = model.members[0].geometry.axis
    if scenario=="invisible_sector":
        def observed(xy):
            return 5-polar(xy,axis)[0]/60
        def hidden(xy):
            d2 = np.sum((xy*[1/axis,axis]-[100,100])**2,axis=1)
            return np.logaddexp(observed(xy),np.log(2000*np.maximum(0.,1-d2/50**2)**3,
                                where=d2<50**2,out=np.full_like(d2,-np.inf)))
        truths = {"sampled_equivalent":observed,"hidden_lobe":hidden}
    else:
        _,_,known = case(scenario,seed,return_field=True)
        truths = {"specified":known}
    totals = {}
    for scramble in (85237,38939):
        uv = qmc.Sobol(d=2,scramble=True,seed=scramble).random_base2(14)
        for start in range(0,len(uv),128):
            xy,area,_,_ = polar_nodes(uv[start:start+128],axis)
            old,new,bary = model.components(xy)
            h,_ = model.support.evaluate(xy,bary)
            med = inverse_cdf(old.T,model.base.weights)[0]
            lower,revised = inverse_cdf(new.T,model.weights,(.10,.50))
            for label,truth in truths.items():
                z = truth(xy)
                for cm in (1.,10.,100.):
                    obs = z>=np.log(cm)
                    for method,f in (("control_median",med),("revised_median",revised),("revised_90pct_exceedance",lower)):
                        pred = f>=np.log(cm)
                        for region,mask in (("whole_Earth",np.ones(len(h),bool)),("high_exposure",h>=.5)):
                            a = area*mask
                            key = (label,cm,method,region)
                            entry = totals.setdefault(key,np.zeros(4))
                            entry += [a@(pred & ~obs),a@(obs & ~pred),a@(obs & pred),(a*a)@(obs|pred)]
    rows = []
    for (truth,cm,method,region),(fp,fn,hit,squares) in totals.items():
        union = fp+fn+hit
        rows.append({"truth":truth,"cm":cm,"method":method,"region":region,
                "false_positive_km2":float(fp/32768),"false_negative_km2":float(fn/32768),"intersection_km2":float(hit/32768),
                "intersection_over_union":float(hit/union) if union else None,
                "union_effective_nodes":float(union**2/squares) if squares else 0.})
    write_json(output,{"scenario":scenario,"seed":seed,"truth_volume_km3":volumes,
                       "members_passed":{m.geometry.policy.name:m.records[-1]["diagnostics"]["passed"] for m in model.members},
                       "weights":weights,"areas":rows})


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage",choices=("fit","calibrate","numerical","audit","maps","report","synthetic"))
    p.add_argument("--work",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--cases",nargs="+")
    p.add_argument("--scenarios",nargs="+",choices=(*SCENARIOS,"invisible_sector"))
    p.add_argument("--seeds",nargs="+",type=int,default=[90,91])
    args = p.parse_args(argv)
    if args.stage=="synthetic" or args.scenarios:
        for scenario in args.scenarios or (*SCENARIOS,"invisible_sector"):
            for seed in args.seeds:
                key = f"{scenario}-{seed}"
                work = args.work/"synthetic"/key
                if args.stage=="fit":
                    data,_ = trial(scenario,seed)
                    run_members(data,0.,work,("full",))
                elif args.stage=="synthetic":
                    synthetic(work,args.output/(key+".json"),scenario,seed)
                else:
                    p.error("Use fit or synthetic with --scenarios")
        return
    cases = []
    for rec in datasets():
        if args.cases and rec["slug"] not in args.cases:
            continue
        work,output = args.work/rec["slug"],args.output/rec["slug"]
        if args.stage in ("fit","calibrate"):
            data,meta = audited_data(rec["path"])
            if args.stage=="fit":
                run_members(data,meta["vent_lonlat"][1],work)
            else:
                cases.append(case_scores(rec,data,meta,work))
        elif args.stage=="report":
            report.run(rec["path"],work,output,rec["name"])
        else:
            getattr(report,args.stage)(rec["path"],work,output)
    if args.stage=="calibrate":
        write_json(args.output,{"protocol":"azimuthal-predictive-calibration-v1","members":NAMES,
                                "concentrations":CONCENTRATIONS.tolist(),"cases":cases})


if __name__=="__main__":
    main()
