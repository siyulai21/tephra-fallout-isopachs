"""Test directional changes in thinning rate using the audited B-Tm data with Samjiyon withheld."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from benchmarks.btm_source_sensitivities import samjiyon_xy, local_summaries
from benchmarks.btm_thinning_fit import VARIANTS, fit_variant, prediction, point_scores
from src.empirical_isopach.core.catalog import data_directory
from src.empirical_isopach.models.directional.ensemble.members import splits
from src.empirical_isopach.models.directional.geometry import BASE, Geometry, polar
from src.empirical_isopach.models.directional.study import digest, write_json
from src.empirical_isopach.models.mass_mixture.study import audited_data


def audit_input():
    root = data_directory()
    principal = json.loads((root/"audits/B-Tm/provenance.json").read_text())["cases"]["principal"]
    path = root/Path(principal["file"]).relative_to("data")
    if digest(path) != principal["sha256"]:
        raise ValueError("The input differs from the B-Tm field-audit selection")
    return path


def run_variant(work, variant, reference_work=None):
    data, meta = audited_data(audit_input())
    data = data.canonical()
    latitude = meta["vent_lonlat"][1]
    xy = samjiyon_xy(meta)
    ref_geometry = Geometry(data,latitude,BASE)
    radius, angle = polar(data.xy,ref_geometry.axis)
    sam_radius, sam_angle = polar(xy[:1],ref_geometry.axis)
    ray = np.geomspace(1., min(radius.max(),4*sam_radius[0]), 96)
    ray_xy = ray[:,None]*xy[0]/sam_radius[0]
    folder=work/variant
    folder.mkdir(parents=True,exist_ok=True)
    settings={"variant":variant,"n_breaks":VARIANTS[variant][0],"exponential_kernels":VARIANTS[variant][1],
              "slope_ratio_log_sd":float(VARIANTS[variant][2]),"principal_input_sha256":meta["input_sha256"],
              "samjiyon_omitted":True,"full_fit":{},"folds":[]}
    groups=[(None,data,None)] if variant=="wide_prior" else [(None,data,None),*splits(data,latitude,"geographic")]
    rows=[]
    for fold,train,test in groups:
        name="full" if fold is None else f"fold-{fold}"
        reference = None
        if reference_work is not None:
            reference=(reference_work/"fits/reference" if fold is None else
                       reference_work/"geographic"/name/"reference")
        g,draws,fit,bounds=fit_variant(train,latitude,variant,fold,folder/name,reference)
        record={"fold":fold,"training_row_ids":train.row_ids.tolist(),"prior_break_domain_km":bounds.tolist(),**fit}
        if fold is None:
            local=prediction(draws,g,xy)
            record["samjiyon"]=local_summaries(local)
            cols={k:np.quantile(draws[k].reshape(-1,draws[k].shape[-1]),[.025,.5,.975],axis=0).T.tolist()
                  for k in ("break_km","log_slope_change") if k in draws}
            record["parameter_quantiles"]=cols
            settings["full_fit"]=record
            args={"means":local.means.reshape((*draws["source"].shape,3)),
                  "variances":local.variances.reshape((*draws["source"].shape,3))}
            for k in ("break_km","log_slope_change"):
                if k in draws:args[k]=draws[k]
            np.savez_compressed(folder/"local-draws.npz",**args)
            line=prediction(draws,g,ray_xy)
            percentiles=np.exp(np.quantile(line.means,[.025,.5,.975],axis=0))
            pd.DataFrame({"radius_km":ray,"field_lower95_cm":percentiles[0],"field_median_cm":percentiles[1],
                          "field_upper95_cm":percentiles[2]}).to_csv(folder/"ray.csv",index=False)
        else:
            held=test
            record["withheld_row_ids"]=held.row_ids.tolist()
            pdf=prediction(draws,g,held.xy,held.log_sd)
            nll=-point_scores(pdf,held)
            low,med,high=[pdf.quantile(p) for p in (.025,.5,.975)]
            q=np.array([int(np.flatnonzero(data.row_ids==row_id)[0]) for row_id in held.row_ids])
            rows.extend({"variant":variant,"row_id":int(held.row_ids[i]),"fold":fold,"radius_km":radius[q[i]],
                         "bearing_degrees":np.rad2deg(angle[q[i]]),"bounded":bool(held.censored[i]),
                         "lower_bound":bool(held.lower_censored[i]),"withheld_log_cm":float(held.log_value[i]),
                         "negative_log_score":float(nll[i]),"log_lower95":low[i],"log_median":med[i],"log_upper95":high[i]}
                        for i in range(len(q)))
            record["mean_negative_log_score"]=float(nll.mean())
            settings["folds"].append(record)
        write_json(folder/"progress.json",settings)
    if rows:
        pd.DataFrame(rows).to_csv(folder/"heldout.csv",index=False)
    settings["site_radius_km"]=float(sam_radius[0])
    settings["site_bearing_degrees"]=float(np.rad2deg(sam_angle[0]))
    write_json(folder/"result.json",settings)
    print("Completed",variant,flush=True)


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work",required=True,type=Path)
    parser.add_argument("--variant",required=True,choices=VARIANTS)
    parser.add_argument("--reference-work",type=Path)
    args=parser.parse_args()
    run_variant(args.work,args.variant,args.reference_work)
