"""Collect the matched area-growth study without selecting cases or hiding failed fits."""
import argparse
import gzip
import json
from pathlib import Path
import shutil

import numpy as np
import pandas as pd

from .footprint_study import write_csv
from src.empirical_isopach.core.catalog import datasets
from src.empirical_isopach.models.nested.study import digest


def read(path):
    return json.loads(path.read_text())


def packed(path, value):
    path.write_bytes(gzip.compress(json.dumps(value, separators=(",", ":"), allow_nan=False).encode(), mtime=0))


def collect(scratch, output):
    output.mkdir(parents=True, exist_ok=True)
    paired, predicted, measured, morphology, sectors = [], [], [], [], []
    quadrant_rows, quadrant_receipts = [], {}
    for rec in datasets():
        slug, identity = rec["slug"], digest(rec["path"])
        old = scratch/"baseline"/slug
        new = scratch/"release"/slug
        reports = [read(p/"footprints/report.json") for p in (old,new)]
        assert all(r["input_sha256"] == identity for r in reports), slug
        regional_report=read(scratch/"regional"/slug/"report.json")
        assert regional_report["input"]["input_sha256"]==identity
        regional_folds=dict(zip(regional_report["validation"]["row_ids"],regional_report["validation"]["folds"],strict=True))
        for i in range(4):
            a,b = [r["fits"][f"fold-{i}"] for r in reports]
            assert a["train"] == b["train"] and a["test"] == b["test"], (slug,i)
            assert set(b["test"])=={row for row,fold in regional_folds.items() if fold==i}, (slug,i)
        for method,path,report in zip(("Previous footprints","Continuous area growth"),(old,new),reports,strict=True):
            mapped = read(path/"deposit.json")
            assert mapped["input_sha256"] == identity
            validation = report["validation"]
            band = mapped["thickness_band"]
            paired.append(dict(case=slug,method=method,input_sha256=identity,
                               median_map_km3=mapped["median_map_volume_km3"],
                               q25_map_km3=band["volumes_km3"][0],q75_map_km3=band["volumes_km3"][-1],
                               geographic_negative_log_score=validation["negative_log_score"],
                               predictive_coverage95_exact=validation["coverage95_exact"],
                               passed_folds=validation["passed_folds"],full_passed=report["fits"]["full"]["passed"],
                               draw_checks_passed=report["draw_checks_passed"],map_integration_passed=mapped["integration_passed"],
                               map_draws_passed=report["median_draw_refinements"][-1]["passed"],
                               report_sha256=digest(path/"footprints/report.json"),deposit_sha256=digest(path/"deposit.json")))
            frame = pd.read_csv(path/"footprints/heldout.csv.gz")
            frame.insert(0,"method",method); frame.insert(0,"case",slug)
            frame["input_sha256"] = identity
            predicted.append(frame)
        review = scratch/"review"/slug
        check = read(review/"review.json")
        assert check["input_sha256"] == identity and check["field_mode"] == "published median grid"
        assert check["figure_sha256"] == digest(review/"scales.png")
        for name,frames in (("metrics.csv",measured),("morphology.csv",morphology),("sector-errors.csv",sectors)):
            frame = pd.read_csv(review/name)
            if "case" not in frame: frame.insert(0,"case",slug)
            frames.append(frame)
        shutil.copytree(review,output/"review"/slug,dirs_exist_ok=True)
        for i in range(4):
            r=read(scratch/"quadrants"/slug/f"quadrant-{i}-outcome.json")
            quadrant_receipts[f"{slug}/{i}"]=r
            valid=r["status"]=="estimated"
            assert (r["identity"] if valid else r)["input_sha256"] == identity
            held=r["heldout"] if valid else []
            quadrant_rows.append(dict(case=slug,quadrant=i,status=r["status"],input_sha256=identity,
                                      rows=len(held) if valid else len(r["test"]),
                                      negative_log_score=-np.mean([v["log_score"] for v in held]) if held else None,
                                      fit_passed=r["numerics"]["passed"] if valid else None))
    for name,frames in (("geographic-predictions.csv.gz",predicted),("measured-agreement.csv",measured),
                        ("contour-morphology.csv.gz",morphology),("sector-errors.csv",sectors)):
        write_csv(pd.concat(frames,ignore_index=True),output/name)
    write_csv(pd.DataFrame(paired),output/"paired-results.csv")
    write_csv(pd.DataFrame(quadrant_rows),output/"quadrants.csv")
    packed(output/"quadrant-fit-receipts.json.gz",quadrant_receipts)
    known_files=sorted((scratch/"known").glob("*/recovery.csv"))
    assert len(known_files)==20, f"Expected all 20 known-deposit designs, found {len(known_files)}"
    new_known=pd.concat([pd.read_csv(p) for p in known_files],ignore_index=True)
    old_files=sorted((scratch/"known").glob("*/previous-recovery.csv"))
    assert len(old_files)==20, f"Expected 20 freshly matched previous-model controls, found {len(old_files)}"
    old_known=pd.concat([pd.read_csv(p) for p in old_files],ignore_index=True)
    keys=["scenario","seed","input_sha256","truth","scramble","level_cm"]
    assert set(map(tuple,old_known[keys].to_numpy()))==set(map(tuple,new_known[keys].to_numpy()))
    old_known["method"]="previous footprints"
    write_csv(pd.concat([old_known,new_known],ignore_index=True),output/"known-deposits.csv.gz")
    surveys=sorted((scratch/"surveys").glob("*/predictions.csv"))
    assert len(surveys)==5, f"Expected all five independent survey transfers, found {len(surveys)}"
    survey_frame=pd.concat([pd.read_csv(p) for p in surveys],ignore_index=True)
    for survey,block in survey_frame.groupby("survey"):
        ids=[set(g.row_id) for _,g in block.groupby("method")]
        assert len(ids)==2 and ids[0]==ids[1], survey
        assert block.input_sha256.nunique()==1, survey
    write_csv(survey_frame,output/"withheld-surveys.csv.gz")
    for design in ("known","surveys"):
        receipts={p.relative_to(scratch/design).as_posix():read(p)
                  for p in sorted((scratch/design).rglob("*.json"))}
        packed(output/(design+"-fit-receipts.json.gz"),receipts)
    regional=pd.read_csv(scratch/"regional/comparison.csv")
    identities={r["slug"]:digest(r["path"]) for r in datasets()}
    assert set(regional.slug)==set(identities)
    regional["input_sha256"]=regional.slug.map(identities)
    write_csv(regional,output/"regional-benchmark.csv")
    shutil.copyfile(scratch/"regional/synthetic/comparison.csv",output/"regional-known-deposits.csv")


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--scratch",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    a=p.parse_args()
    collect(a.scratch,a.output)
