"""Assemble the principal spherical-mixture result from its verified parent fits."""
from copy import deepcopy
import gzip
from pathlib import Path
import shutil
import numpy as np
import pandas as pd

from ..contours import load_surfaces,screen
from .field import PROTOCOL, HALF_REACH, POWERS, SEED, BearingSupport
from .maps import select_extent, ray_crossings
from .study import digest,fit_volume,read,save,validation,verified
from .volume import checked_volumes


def design():
    here=Path(__file__).parent
    return {"half_reach_degrees":HALF_REACH,"powers":list(POWERS),"probabilities":[1/3]*3,"orientation_seed":SEED,
            "model_sha256":{p.name:digest(p) for p in sorted(here.glob("*.py"))},
            "posterior_interpretation":"Parent amplitude and kernel parameters follow the original posterior. "
            "The principal mass is its integrated retained fraction times parent volume. A single latent orientation applies to a complete deposit."}


def assemble(output,data,meta,parent,cache,work, *, precomputed=False):
    output,cache,work=map(Path,(output,cache,work))
    bearing=output/"bearing"
    bearing.mkdir(parents=True,exist_ok=True)
    work.mkdir(parents=True,exist_ok=True)
    lat=meta["vent_lonlat"][1]
    if parent["input"]["input_sha256"]!=meta["input_sha256"]:
        raise ValueError("Input differs from fitted spherical posterior")
    report=deepcopy(parent)
    report["parent_protocol"],report["protocol"]=parent["protocol"],PROTOCOL
    report["bearing"]=design()
    report["reproducibility"]["normalization"]="Finite-Earth component normalization followed by positive quadrature of retained mass under each normalized parent mixture."
    summaries={}
    frames=[]
    for key,fit in report["fits"].items():
        folder=work/"assumptions"
        if not precomputed:
            fit_volume(data,lat,parent,fit,cache,work,key)
        item=read(folder/(key+".json"))
        packet=folder/(key+".npz")
        if (item["packet_sha256"]!=digest(packet) or item["posterior_sha256"]!=fit["posterior_sha256"]
                or item["input_sha256"]!=meta["input_sha256"]):
            raise ValueError(f"Incompatible integrated field packet for {key}")
        if "source_checks" in fit:
            fit["parent_source_checks"]=fit.pop("source_checks")
        fit["parent_volume"]=fit["volume"]
        fit["volume"]=item["volume"]
        d=fit["diagnostics"]
        d["parent_sampler_passed"]=d["passed"]
        d["bearing_integration_passed"]=item["checks"]["spatial_passed"]
        d["bearing_volume_split_passed"]=item["checks"]["draw_split_passed"]
        d["passed"]=all([d["parent_sampler_passed"],d["bearing_integration_passed"],d["bearing_volume_split_passed"]])
        fit["bearing_draw_refinements"]=item["draw_refinements"]
        fit["bearing_volume_checks"]=item["checks"]
        fit["bearing_width_sensitivities_km3"]=item["by_width"]
        fit["enhanced_hidden_fan_km3"]=item["enhanced_hidden_fan"]
        summaries[key]=item
        with np.load(packet) as a:
            masses=a["volume_km3"]
            flat_index=a["flat_index"]
            n=item["draws_per_chain"]
            frames.append(pd.DataFrame({"assumption":key,"chain":np.repeat(np.arange(item["chain_count"]),n*len(POWERS)),
                "parent_flat_index":np.repeat(flat_index,len(POWERS)),"width_power":np.tile(POWERS,len(flat_index)),
                "volume_km3":masses.ravel()}))
        if key=="baseline":
            shutil.copyfile(packet,bearing/"baseline.npz")
    base=summaries["baseline"]
    report["mass_partition"]=base["mass_partition"]
    report["sensitivity_envelope_km3"]=[min(f["volume"]["lower95"] for f in summaries.values()),
                                        max(f["volume"]["upper95"] for f in summaries.values())]
    report["all_assumption_fits_passed_diagnostics"]=all(f["diagnostics"]["passed"] for f in report["fits"].values())
    samples=verified(cache/"baseline"/parent["fits"]["baseline"]["posterior_file"],parent["fits"]["baseline"])
    if digest(output/"posterior.npz")!=parent["fits"]["baseline"]["posterior_sha256"]:
        raise ValueError("Published spherical parameter file differs from the fitted baseline")
    if precomputed:
        val=read(work/"validation.json")
        table=pd.read_csv(work/"heldout.csv",float_precision="round_trip")
        if val["fits"]!=parent["validation"]["fits"]:
            raise ValueError("Integrated fold predictions refer to different fits")
    else:
        val,table=validation(data,lat,parent,cache)
    report["validation"]=val
    table.to_csv(output/"heldout_predictions.csv",index=False)
    draws=pd.concat(frames,ignore_index=True)
    draws.to_csv(output/"volume_draws.csv.gz",index=False,float_format="%.12g",compression={"method":"gzip","mtime":0})
    pd.DataFrame([{"assumption":k,**{q:f["volume"][q] for q in ("median","lower95","upper95")},
                   "parent_median":f["parent_volume"]["median"],"diagnostics_passed":f["diagnostics"]["passed"]}
                  for k,f in report["fits"].items()]).to_csv(output/"estimates.csv",index=False)
    support=BearingSupport(data,lat)
    if np.max(support.evaluate(data.xy)[0])>1e-12:
        raise ArithmeticError("A fitted thickness or limit changed under bearing continuation")
    report["bearing"]["spacing_km"]=support.spacing
    report["bearing"]["positively_sampled_maximum_chord_km"]=float(support.outer)
    reach_file=work/"reach-sensitivity.json"
    if not reach_file.exists() or not precomputed:
        alternatives={}
        for half in (HALF_REACH/2,2*HALF_REACH):
            result,_=checked_volumes(samples,BearingSupport(data,lat,half),parts=False,initial=256,ceiling=1024)
            alternatives[str(half)]={k:result[k] for k in ("volume","checks","draws_per_chain")}
        save(reach_file,{"input_sha256":meta["input_sha256"],"posterior_sha256":parent["fits"]["baseline"]["posterior_sha256"],"results":alternatives})
    alt=read(reach_file)
    if alt["input_sha256"]!=meta["input_sha256"] or alt["posterior_sha256"]!=parent["fits"]["baseline"]["posterior_sha256"]:
        raise ValueError("Incompatible bearing-reach sensitivity")
    report["bearing"]["reach_sensitivity"]=alt["results"]
    view,scope,report["bearing"]["map_draw_check"]=select_extent(samples,data,lat)
    pd.DataFrame(scope).to_csv(bearing/"sector-areas.csv",index=False)
    save(bearing/"ray-crossings.json",ray_crossings(view))
    fold_paths={f["fold"]:cache/"validation"/f"fold-{f['fold']}"/f["fit"]["posterior_file"] for f in val["fits"]}
    groups=load_surfaces(report,output/"posterior.npz",fold_paths)
    screened=screen(data,lat,report,groups)
    save(output/"complete-contour-audit.json",screened)
    # The same field subsets support future reproductions, including training-only deletion geometry.
    folder=output/"support"
    folder.mkdir(exist_ok=True)
    from ....core.catalog import datasets
    slug=next((r["slug"] for r in datasets() if digest(r["path"])==meta["input_sha256"]),"case")
    packet=folder/(slug+".npz")
    np.savez_compressed(packet,**{key+"__"+k:v for key,draws in groups.items() for k,v in draws.items()})
    import json
    receipt={**screened,"surface_samples_sha256":digest(packet)}
    (folder/(slug+".json.gz")).write_bytes(gzip.compress(json.dumps(receipt,separators=(",",":"),allow_nan=False).encode(),mtime=0))
    report["bearing"]["numerical_interpretation"]="NUTS diagnostics and attempt histories describe parent draws. Each fit separately records spatial integration and split-draw checks on derived volume."
    src=report.get("source_checks",{})
    if src.get("source_anchored"):
        src["parent_nonincreasing_along_every_source_ray"]=src.get("nonincreasing_along_every_source_ray")
        src["nonincreasing_along_every_source_ray"]=False
        src["interpretation"]="Retention is at most one and equals one at the vent. A source-focused parent therefore keeps its maximum at the source. Local bound anchors can interrupt radial monotonicity."
    report["bearing"]["parent_reference_km3"]=parent["fits"]["baseline"]["volume"]
    report["bearing"]["parent_geographic_score"]=parent["validation"]["metrics"]["equal_region_log_score"]
    report["bearing"]["parent_report_sha256"]=digest(bearing/"parent-report.json")
    report["published_artifact_sha256"]={name:digest(output/name) for name in ("posterior.npz","volume_draws.csv.gz",
         "estimates.csv","heldout_predictions.csv","complete-contour-audit.json","bearing/baseline.npz","bearing/sector-areas.csv",
         "bearing/ray-crossings.json","bearing/parent-report.json")}
    save(output/"report.json",report)
    from ....global_reporting import figures
    from ....source_reporting import source_figure
    figures(output,report["name"],data,meta,samples,report)
    source_figure(output,report["name"],data,meta,samples,view=view)
    from .narrative import case_page
    case_page(output,report)
    files=[*report["published_artifact_sha256"],"README.md","report.json","field.png","local.png","world.png",
           "map-display.json","volume.png","sampling.png","source.png","support/"+packet.name,"support/"+slug+".json.gz"]
    save(output/"artifacts.json",{n:digest(output/n) for n in files})
    return report
