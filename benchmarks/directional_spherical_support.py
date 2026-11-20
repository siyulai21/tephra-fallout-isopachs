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


def run(record, output, runs=None, replay=None, published=False):
    slug = record["slug"]
    home = Path("eruptions")/slug/"results/spherical-mixture"
    report = json.loads((home/"report.json").read_text())
    data, meta = audited_data(record["path"])
    assert meta["input_sha256"] == report["input"]["input_sha256"]
    # The earlier combined directory moved to each case. An existing user-supplied
    # directory still takes precedence, and its receipt must match the fitted input.
    old_collection = Path("comparisons/directional/single-structure/spherical-support")
    if published or (replay is not None and replay.resolve() == old_collection.resolve() and not replay.exists()):
        replay = home/"support"
        published = True
    if replay is not None:
        packet = replay/(slug+".npz")
        receipt = replay/(slug+".json.gz")
        if published or receipt.is_file():
            with gzip.open(receipt, "rt") as stream:
                saved = json.load(stream)
            if (saved["input_sha256"] != meta["input_sha256"]
                    or saved["baseline_posterior_sha256"] != digest(home/"posterior.npz")
                    or saved["surface_samples_sha256"] != digest(packet)):
                raise ValueError(f"{slug}: support packet does not match the accepted input and spherical posterior")
        with np.load(packet) as archive:
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
    source.add_argument("--replay", type=Path, help="Replay a flat directory of saved field subsets")
    source.add_argument("--published", action="store_true", help="Replay the support packet beside each registered spherical fit")
    p.add_argument("--output",required=True,type=Path)
    p.add_argument("--cases",nargs="+",choices=[r["slug"] for r in datasets()])
    args = p.parse_args()
    for record in datasets():
        if not args.cases or record["slug"] in args.cases:
            run(record, args.output, args.runs, args.replay, published=args.published)
