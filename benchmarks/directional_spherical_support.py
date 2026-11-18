"""Audit published spherical-mixture fields with the same complete-contour screen."""
import argparse
import gzip
import json
from pathlib import Path

import numpy as np

from src.empirical_isopach.core.catalog import datasets
from src.empirical_isopach.models.mass_mixture.contours import load_surfaces, screen
from src.empirical_isopach.models.mass_mixture.study import audited_data
from src.empirical_isopach.models.directional.study import digest


def run(record, output, runs=None, replay=None):
    slug = record["slug"]
    home = Path("eruptions")/slug/"results/spherical-mixture"
    report = json.loads((home/"report.json").read_text())
    data, meta = audited_data(record["path"])
    assert meta["input_sha256"] == report["input"]["input_sha256"]
    if replay:
        with np.load(replay/(slug+".npz")) as archive:
            groups = {}
            for name in archive.files:
                key, field = name.split("__",1)
                groups.setdefault(key,{})[field] = archive[name]
    else:
        fold_paths = {f["fold"]:runs/slug/"posteriors/validation"/f"fold-{f['fold']}"/f["fit"]["posterior_file"]
                      for f in report["validation"]["fits"]}
        groups = load_surfaces(report,home/"posterior.npz",fold_paths)
    output.mkdir(parents=True,exist_ok=True)
    samples = output/(slug+".npz")
    np.savez_compressed(samples,**{key+"__"+k:v for key,draws in groups.items() for k,v in draws.items()})
    result = screen(data,meta["vent_lonlat"][1],report,groups)
    result["surface_samples_sha256"] = digest(samples)
    raw = json.dumps(result,separators=(",",":"),allow_nan=False).encode()
    (output/(slug+".json.gz")).write_bytes(gzip.compress(raw,mtime=0))
    print(slug,{p:s["selected_band"] for p,s in result["policies"].items()},flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    source = p.add_mutually_exclusive_group(required=True)
    source.add_argument("--runs",type=Path,help="Directory of complete spherical posterior runs by eruption")
    source.add_argument("--replay",type=Path,help="Directory of published deterministic field subsets")
    p.add_argument("--output",required=True,type=Path)
    p.add_argument("--cases",nargs="+",choices=[r["slug"] for r in datasets()])
    args = p.parse_args()
    for record in datasets():
        if not args.cases or record["slug"] in args.cases:
            run(record,args.output,args.runs,args.replay)
