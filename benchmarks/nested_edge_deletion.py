"""Refit the nested deposit after withholding the row or local cluster setting its sampled edge."""
import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from src.empirical_isopach.core.audit import audited_data
from src.empirical_isopach.core.catalog import datasets
from src.empirical_isopach.models.nested.continuation import check_and_refine, continuation_digest
from src.empirical_isopach.models.nested.inference import fit
from src.empirical_isopach.models.nested.study import digest, write_json
from src.empirical_isopach.validation import automatic_buffer


def run(record):
    path, source, out = record
    data, meta = audited_data(path, allow_intervals=True)
    receipt = json.loads((source/"full.json").read_text())
    if receipt["identity"]["input_sha256"]!=meta["input_sha256"] or digest(source/"full.npz")!=receipt["posterior_sha256"]:
        raise ValueError("Input CSV or published posterior does not match the selected nested fit")
    edges = pd.read_csv(source/"edge-sites.csv")
    if sorted(edges.row_id) != sorted(data.row_ids):
        raise ValueError("Edge-attribution table and accepted rows differ")
    first = int(edges.iloc[0].row_id)
    location = data.xy[data.row_ids==first][0]
    distance = np.linalg.norm(data.xy-location, axis=1)
    buffer, _ = automatic_buffer(data)
    masks = {"single": data.row_ids==first, "cluster": distance<=buffer}
    out.mkdir(exist_ok=True, parents=True)
    result = {"input_sha256":meta["input_sha256"], "engine_sha256":continuation_digest(),
              "edge_table_sha256":digest(source/"edge-sites.csv"), "full_posterior_sha256":digest(source/"full.npz"),
              "dominant_row":first, "geographic_buffer_km":buffer, "fits":{}}
    samples = {}
    for label, withheld in masks.items():
        if label=="cluster" and np.array_equal(withheld,masks["single"]):
            result["fits"][label] = dict(result["fits"]["single"],same_rows_as="single")
            continue
        train, test = data.subset(~withheld),data.subset(withheld)
        parent, status = fit(train,meta["vent_lonlat"][1],budget=True)
        _, model, status = check_and_refine(parent,train,status)
        values = model.volume()
        q = np.quantile(values,[.025,.5,.975])
        samples[label] = values
        result["fits"][label] = {"training_rows":train.row_ids.tolist(),"test_rows":test.row_ids.tolist(),
                                  "withheld_log_scores":model.score(test).tolist(),"volume_km3":q.tolist(),
                                  "numerical":status}
    np.savez_compressed(out/"volumes.npz",**samples)
    result["draws_sha256"] = digest(out/"volumes.npz")
    write_json(out/"report.json",result)
    lines=["# Sensitivity to the site setting the outermost sampled isopach", "",
           f"CSV row {first} most frequently sets the last measured rank. The first test withholds that observation. The second also withholds every accepted row within the shared geographic buffer of {buffer:.1f} km. Each fit re-estimates shape, thinning and the remaining volume. The principal fit retains all accepted sites.", "",
           "| Withheld | Rows | Conditional 95% volume interval (km³) | Full-fit population check |",
           "|---|---:|---:|---|"]
    for key,label in (("single","One site"),("cluster","Site and local cluster")):
        f=result["fits"][key]
        lo,med,hi=f["volume_km3"]
        state="passed" if f["numerical"]["passed"] else "**warning**"
        lines.append(f"| {label} | {len(f['test_rows'])} | {lo:,.1f}–{hi:,.1f} | {state} |")
    lines.extend(["","These sites were selected from their influence on the fitted geometry. The change measures sensitivity to that part of the survey. The observations retain their published provenance and acceptance status.", "",
                  "[Every site's edge probability](../edge-sites.csv) · [Deleted rows, held-out likelihoods and sampling attempts](report.json) · [Complete-volume draws](volumes.npz)"])
    (out/"README.md").write_text("\n".join(lines)+"\n")
    print(source.parent.name,first,[(k,round(v["volume_km3"][1],2)) for k,v in result["fits"].items()],flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--workers",type=int,default=1)
    p.add_argument("--extra",type=Path,help="Accepted CSV for the separate Pinatubo preview")
    p.add_argument("--case",type=Path,help="Run this diagnostic on a single accepted CSV")
    p.add_argument("--fitted",type=Path,help="Existing principal nested fit for --case")
    a=p.parse_args()
    if bool(a.case)!=bool(a.fitted) or (a.extra and a.case):
        p.error("Supply both --case and --fitted for one deposit, or use the catalogue and optional --extra")
    if a.case:
        run((a.case,a.fitted,a.output))
        return
    records=[(r["path"],Path("eruptions")/r["slug"]/"results/nested-whole-area",a.output/r["slug"]) for r in datasets()]
    if a.extra:
        records.append((a.extra,Path("comparisons/nested-isopachs/pinatubo-preview/nested-whole-area"),a.output/"pinatubo-1991"))
    with ProcessPoolExecutor(a.workers) as pool:
        list(pool.map(run,records))


if __name__=="__main__":
    main()
