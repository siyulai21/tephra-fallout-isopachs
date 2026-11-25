"""Assemble paired numerical evidence for the nested geographic model study."""
import argparse
import gzip
import json
from pathlib import Path
import shutil

import numpy as np
import pandas as pd
from scipy.special import expit, logsumexp

from .footprint_catalogue import published_folder
from .publication import published_datasets, result_home
from src.empirical_isopach.models.nested.continuation import load_packet
from src.empirical_isopach.models.nested.footprints.posterior import FootprintField
from src.empirical_isopach.visualization.deposit import digest, read_json

ARCHIVE = Path("archive/2026-11-23-nested-geography")
STUDY = Path("comparisons/nested-isopachs/geography")


def controls(slug, rec):
    base = result_home(rec)
    root = ARCHIVE/f"{slug}-300-controls"
    if (root/"whole/full.npz").is_file():
        if read_json(root/"corridor/extent/report.json")["input_sha256"] != digest(rec["path"]):
            raise ValueError(f"Archived controls have a different input from {slug}")
        return root/"corridor", root/"corridor", root/"whole"
    return base/"nested-isopachs", ARCHIVE/rec["slug"], base/"nested-whole-area"


def write_csv(frame, path):
    gzip_options = {"method":"gzip","mtime":0} if path.suffix==".gz" else None
    frame.to_csv(path,index=False,compression=gzip_options,lineterminator="\n")


def weighted_interval(volumes, weights):
    order = np.argsort(volumes)
    w = weights[order]
    return np.interp([.025,.5,.975], (np.cumsum(w)-.5*w)/sum(w),volumes[order])


def structural(case, whole, new):
    """Sensitivity to two prior predictive models of area allocation on the same likelihood."""
    ref = read_json(whole/"full.json")
    info = read_json(new/"footprints/report.json")
    current = info["fits"]["full"]["attempts"][-1]
    evidence = [np.array(r["attempts"][-1]["log_evidence_exponential_power"])
                for r in (ref["numerics"],current["numerics"])]
    log_odds = logsumexp(evidence[1])-logsumexp(evidence[0])
    pop = logsumexp(evidence[1],axis=1)-logsumexp(evidence[0],axis=1)
    volumes = (load_packet(whole/"full.npz").volume(), FootprintField.load(new/"footprints/full-selected.npz").volume())
    predictions = pd.read_csv(new/"footprints/heldout.csv.gz").merge(
        pd.read_csv(whole/"heldout.csv")[["fold","row_id","log_score"]],on=["fold","row_id"],
        suffixes=("_new","_original"),validate="one_to_one")
    fold_odds = {}
    for i in range(4):
        part = info["fits"][f"fold-{i}"]["attempts"][-1]["numerics"]
        old_part = read_json(whole/f"fold-{i}.json")["numerics"]
        fold_odds[i] = (logsumexp(part["attempts"][-1]["log_evidence_exponential_power"])
                        -logsumexp(old_part["attempts"][-1]["log_evidence_exponential_power"]))
    rows = []
    for odds in (.25,1.,4.):
        wt = expit(log_odds+np.log(odds))
        w = np.r_[np.full(len(volumes[0]),(1.-wt)/len(volumes[0])),np.full(len(volumes[1]),wt/len(volumes[1]))]
        q = weighted_interval(np.concatenate(volumes),w)
        pp = expit(pop+np.log(odds))
        scores = []
        for i,block in predictions.groupby("fold"):
            logit = np.log(odds)+fold_odds[i]
            logp = np.logaddexp(block.log_score_new-np.logaddexp(0.,-logit),block.log_score_original-np.logaddexp(0.,logit))
            scores.append(float(-logp.mean()))
        rows.append(dict(case=case,input_sha256=info["input_sha256"],prior_odds_new_to_original=odds,
                         training_only_geographic_score=np.mean(scores),
                         posterior_weight_new=wt,population_weight_min=pp.min(),population_weight_max=pp.max(),
                         lower95=q[0],median_draw_km3=q[1],upper95=q[2],
                         both_full_fits_passed=info["fits"]["full"]["passed"] and ref["numerics"]["passed"]))
    return rows


def paired(case, rec, path):
    legacy, old_map, whole = controls(case,rec)
    new = read_json(path/"footprints/report.json")
    dep = read_json(path/"deposit.json")
    old = read_json(legacy/"extent/report.json")
    old_dep = read_json(old_map/"deposit.json")
    original = read_json(whole/"report.json")
    original_map = read_json(whole/"median-field.json")
    geographic = read_json(whole/"geographic.json")
    assert new["input_sha256"]==old["input_sha256"]==original["input"]["input_sha256"]
    prior_parts = {f["fold"]:f for f in old["folds"] if f["design"]=="geographic"}
    for k, part in enumerate(geographic["folds"]):
        ref=part["identity"]
        candidate=new["fits"][f"fold-{k}"]
        assert candidate["train"]==ref["training_rows"]==prior_parts[k]["train"]
        assert candidate["test"]==ref["test_rows"]==prior_parts[k]["test"]
    val=old["validation"]["geographic"]
    rows = []
    for method,median,interval,draw_ok,map_ok,score,coverage,folds in (
        ("finite corridors",old_dep["median_map_volume_km3"],old["conditional_interval95_km3"],old["draw_checks_passed"],
         old_dep["integration_passed"],val["negative_log_score"],val["coverage95_exact"],val["passed_folds"]),
        ("original whole area",original_map["median_map_volume_km3"],original["conditional_volume_interval95_km3"],
         original["population_check_passed"] and original["area_check"][-1]["passed"],original_map["integration_passed"],
         geographic["equal_region_negative_log_score"],original["exact_site_coverage95"]["fraction"],original["geographic_folds_passed"]),
        ("dispersal footprints",dep["median_map_volume_km3"],new["conditional_interval95_km3"],new["draw_checks_passed"],
         dep["integration_passed"] and new["median_draw_refinements"][-1]["passed"],new["validation"]["negative_log_score"],
         new["validation"]["coverage95_exact"],new["validation"]["passed_folds"])):
        rows.append(dict(case=case,name=rec["name"],accepted=new["accepted_rows"],input_sha256=new["input_sha256"],
                         method=method,median_map_km3=median,lower95=interval[0],upper95=interval[-1],score=score,
                         exact_coverage=coverage,passed_folds=folds,map_passed=map_ok,draws_passed=draw_ok))
    return rows


def collect(scratch, regional, out):
    out.mkdir(parents=True,exist_ok=True)
    values, odds, sectors, morphs, rotations, receipt = [], [], [], [], [], {}
    for rec in published_datasets():
        case = rec["slug"]
        path=scratch/"release"/case
        values.extend(paired(case,rec,path))
        odds.extend(structural(case,controls(case,rec)[2],path))
        morphs.append(pd.read_csv(scratch/"morphology"/case/"morphology.csv"))
        rotations.append(pd.read_csv(scratch/"morphology"/case/"station-rotation.csv"))
        for i in range(4):
            f=read_json(scratch/"quadrants"/case/f"quadrant-{i}-outcome.json")
            receipt[f"{case}/{i}"]=f
            valid=f["status"]=="estimated"
            held=f["heldout"] if valid else []
            sectors.append(dict(case=case,quadrant=i,status=f["status"],input_sha256=f["identity"]["input_sha256"] if valid else f["input_sha256"],
                                rows=len(held) if valid else len(f["test"]),score=-np.mean([p["log_score"] for p in held]) if held else None,
                                fit_passed=f["numerics"]["passed"] if valid else None,
                                volume_lower95=f["volume_quantiles_km3"][0] if valid else None,
                                volume_upper95=f["volume_quantiles_km3"][-1] if valid else None))
    write_csv(pd.DataFrame(values),out/"paired-results.csv")
    write_csv(pd.DataFrame(odds),out/"geography-prior-sensitivity.csv")
    write_csv(pd.DataFrame(sectors),out/"quadrants.csv")
    (out/"quadrant-fits.json.gz").write_bytes(gzip.compress(json.dumps(receipt,separators=(",",":"),allow_nan=False).encode(),mtime=0))
    write_csv(pd.concat(morphs),out/"contour-morphology.csv.gz")
    write_csv(pd.concat(rotations),out/"station-rotation.csv.gz")
    for directory,file,output,count in (("truths","recovery.csv","known-deposits.csv.gz",20),("surveys","predictions.csv","withheld-surveys.csv.gz",5)):
        files=sorted((scratch/directory).glob("*/"+file))
        if len(files)!=count:
            raise ValueError(f"{directory}: expected {count} experiments, found {len(files)}")
        write_csv(pd.concat([pd.read_csv(f) for f in files]),out/output)
        trials={}
        for f in files:
            selected=(f.parent/"refinement/footprints.json" if (f.parent/"refinement/footprints.json").is_file()
                      else f.parent/"footprints.json")
            trials[f.parent.name]=[read_json(p) for p in dict.fromkeys([f.parent/"footprints.json", selected])]
        (out/(directory+"-fit-receipts.json.gz")).write_bytes(gzip.compress(json.dumps(trials,separators=(",",":"),allow_nan=False).encode(),mtime=0))
    frame=pd.read_csv(regional/"comparison.csv")
    ids={s:read_json(regional/s/"report.json")["input"]["input_sha256"] for s in frame.slug.unique()}
    frame["input_sha256"]=frame.slug.map(ids)
    write_csv(frame,out/"regional-benchmark.csv")
    shutil.copyfile(regional/"synthetic/comparison.csv",out/"regional-known-deposits.csv")


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--scratch",required=True,type=Path)
    p.add_argument("--regional",required=True,type=Path)
    p.add_argument("--output",required=True,type=Path)
    args=p.parse_args()
    collect(args.scratch,args.regional,args.output)
