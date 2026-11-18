"""Preserve the two exploratory directional protocols and all their attempts."""
import argparse
from hashlib import sha256
import gzip
import json
from pathlib import Path
from zipfile import ZipFile

import pandas as pd

from benchmarks.directional import trial
from src.empirical_isopach.models.directional.study import digest, write_json


def read(p):
    return json.loads(p.read_text())


def collect(root, archive):
    with ZipFile(archive) as z:
        sources = {p:sha256(z.read(p)).hexdigest() for p in z.namelist()}
    fits, rows, validation = {}, [], {}
    for design_path in sorted(root.rglob("design.json")):
        folder = design_path.parent
        key = str(folder.relative_to(root))
        design = read(design_path)
        for name, signed in design["signature"]["source_sha256"].items():
            source = "src/"+name
            if source in sources and signed != sources[source]:
                raise ValueError(f"Frozen fitting source mismatch for {source} in {key}")
        attempts = [read(p) for p in sorted(folder.glob("attempt-*.json"))]
        for attempt in attempts:
            if digest(folder/attempt["file"]) != attempt["posterior_sha256"]:
                raise ValueError(f"Changed posterior at {key}")
        if not attempts:
            raise ValueError(f"Missing attempts at {key}")
        result = {"design":design, "attempts":attempts}
        parts = folder.relative_to(root).parts
        row = {"run":key,"group":folder.parent.name,"attempts":len(attempts),
               "sampling_passed":attempts[-1]["diagnostics"]["passed"]}
        if folder.parent.name == "fits":
            fit = read(folder/"integral.json")
            if digest(folder/"mass.npz") != fit["mass_sha256"]:
                raise ValueError(f"Changed mass draws at {key}")
            result["integral"] = fit
            observed = fit["signature"]["integration_source_sha256"]
            frozen = {n:sources["src/empirical_isopach/models/directional/"+n] for n in observed}
            result["loaded_integration_source_sha256"] = frozen
            result["recorded_on_disk_integrator_hash_differs_from_loaded_source"] = observed != frozen
            row.update(fit["volume_km3"],numerically_passed=fit["passed"])
            if parts[0] == "synthetic":
                scenario, seed = parts[1].rsplit("-",1)
                _, truth = trial(scenario,int(seed))
                result["truth"] = truth
                row.update(truth_km3=truth["total_km3"],
                           truth_in_interval=row["lower95"] <= truth["total_km3"] <= row["upper95"])
        fits[key] = result
        rows.append(row)
    for scheme in ("geographic","quadrants"):
        for path in sorted(root.glob("*/"+scheme+".json")):
            validation[str(path.relative_to(root))] = read(path)
    real = [row for row in rows if row["group"] == "fits" and not row["run"].startswith("synthetic/")]
    assert len(real) == 66 and len(validation) == 12
    return {"snapshot_sha256":digest(archive), "source_sha256":sources,
            "fits":fits,"validation":validation}, rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--initial",type=Path,required=True)
    p.add_argument("--initial-log",type=Path,required=True)
    p.add_argument("--first",type=Path,required=True)
    p.add_argument("--fixed-tail",type=Path,required=True)
    p.add_argument("--reintegrated",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    args = p.parse_args()
    initial = {}
    for path in sorted(args.initial.rglob("design.json")):
        folder = path.parent
        records = [read(p) for p in sorted(folder.glob("attempt-*.json"))]
        for rec in records:
            assert rec["posterior_sha256"] == digest(folder/rec["file"])
        initial[str(folder.relative_to(args.initial))] = {"design":read(path),"attempts":records}
    errors = [s for s in args.initial_log.read_text().splitlines() if s.startswith(("ValueError:", "IndexError:"))]
    initial_record = {"fits":initial,"errors":errors,"initial_log_sha256":digest(args.initial_log),
                      "validations":{str(p.relative_to(args.initial)):read(p) for scheme in ("geographic","quadrants")
                                     for p in args.initial.glob("*/"+scheme+".json")}}
    (args.output/"initial-attempts.json.gz").write_bytes(gzip.compress(
        json.dumps(initial_record,allow_nan=False,separators=(",",":")).encode(),mtime=0))
    evidence, rows = {}, []
    for name, folder, filename in (("single_scale",args.first,"first-protocol.zip"),
                                    ("two_scale_fixed_tail",args.fixed_tail,"fixed-tail-protocol.zip")):
        run, table = collect(folder,args.output/filename)
        evidence[name] = run
        rows.extend({"protocol":name,**r} for r in table)
    raw = json.dumps(evidence,allow_nan=False,separators=(",",":")).encode()
    (args.output/"experiments.json.gz").write_bytes(gzip.compress(raw,mtime=0))
    pd.DataFrame(rows).to_csv(args.output/"experiments.csv",index=False)
    write_json(args.output/"reintegration.json",read(args.reintegrated))
    lines = ["# Development of the directional estimate", "",
        "These two exploratory models were each run on all six accepted eruptions with eleven policies, geographic and quadrant deletions, and known-volume trials. The final model was developed after these experiments. All three protocols use the same accepted eruption measurements.", "",
        "| Exploratory model | Aso-4 median km³ [conditional 95%] | West-sector median fraction | Geographic score | Exact-site coverage |", "|---|---:|---:|---:|---:|"]
    for label, run in evidence.items():
        fit = run["fits"]["aso-4/fits/reference"]["integral"]
        v = fit["volume_km3"]
        val = run["validation"]["aso-4/geographic.json"]
        lines.append(f"| {label.replace('_',' ')} | {v['median']:,.0f} [{v['lower95']:,.0f}–{v['upper95']:,.0f}] | {fit['compass']['W']['fraction']['median']:.1%} | {val['equal_region_log_score']:.3f} | {val['metrics']['coverage95_observations']:.1%} |")
    lines.extend(["", "The first protocol assigned one shared radial scale and allowed directional extensions only from the distal half of the retained triangle vertices. An isolated, thick offshore Aso-4 measurement could increase the scale in every direction. Buffered predictions also deteriorated. The second protocol gave compact proximal and directional components separate shares, let isolated measurements constrain narrow bearings, and restored unresolved site variability in gaps.", "",
        "Both early reference models fixed the distal power at four. Even with the improved direction rule, the second model's intervals missed the true 33.6 km³ smooth exponential deposit on all four development seeds. Its multi-lobe medians also exceeded the known total. This motivated inferring the distal shape on a common prior, with fixed power and exponential thinnings retained as sensitivities. The final evaluation adds fresh noise seeds and preserves all numerical warnings.", "",
        f"Before the first complete protocol, the initial implementation produced {sum(len(r['attempts']) for r in initial.values())} posterior attempt records in four cases. One Aso-4 fit stopped when a vectorized sampler compiled for a different array shape was reused. The implementation now keys its sampler by input shapes. All six cases were restarted in fresh directories for the first complete experiment. [Initial designs, attempt checksums and failure](initial-attempts.json.gz).", "",
        "The compressed record includes every design, attempt seed and posterior checksum, geometry, numerical diagnostic, integrated volume, directional budget and held-out prediction. The lighter table indexes the same experiments. The source archives overlay the unchanged common code from the project immediately before this study, commit 9fdd740b. Run the archived benchmark for each case or scenario and integrate its saved fields with the archived estimate function. The early protocol used seeds 10 and 11. The fixed-tail protocol also used seeds 40 and 41.", "",
        "Some fixed-tail integration workers were already running when the source on disk advanced to the final protocol. The record preserves their original on-disk hashes and separately identifies the frozen implementation loaded by those workers. Reintegrating every draw of the Aso-4 and smooth-10 reference fields with the frozen code reproduces every sector and region bit for bit. The published final protocol has consistent source hashes throughout.", "",
        "[Experiments and outcomes](experiments.csv) · [Full exploratory designs, validations and attempts](experiments.json.gz) · [Frozen first implementation](first-protocol.zip) · [Frozen fixed-tail implementation](fixed-tail-protocol.zip) · [Independent reintegration](reintegration.json) · [Final study](../README.md).",""])
    (args.output/"README.md").write_text("\n".join(lines))


if __name__ == "__main__":
    main()
