"""Independent, longer-chain replication of each real directional reference."""
import argparse
from pathlib import Path

import numpy as np

from src.empirical_isopach.core.catalog import datasets
from src.empirical_isopach.models.directional.estimate import estimate
from src.empirical_isopach.models.directional.geometry import BASE, Geometry
from src.empirical_isopach.models.directional.inference import sample
from src.empirical_isopach.models.directional.study import SEED, digest, write_json
from src.empirical_isopach.models.mass_mixture.study import audited_data


def replicate(record, runs, output):
    import json

    data, meta = audited_data(record["path"])
    latitude = meta["vent_lonlat"][1]
    g = Geometry(data, latitude)
    original = runs/record["slug"]/"fits"/BASE.name
    folder = output/record["slug"]
    folder.mkdir(parents=True, exist_ok=True)
    design = json.loads((original/"design.json").read_text())
    if design["geometry"] != g.describe():
        raise ValueError("Original geometry and data do not match")
    seed = SEED + 99173
    design["signature"]["seed"] = seed
    design["replication"] = {"input_sha256":meta["input_sha256"], "warmup":1800,
        "draws_per_chain":4000,"chains":4,"target_acceptance":.97,
        "implementation_sha256":digest(__file__)}
    manifest = folder/"design.json"
    if manifest.exists() and json.loads(manifest.read_text()) != design:
        raise ValueError("Changed replication inputs or settings")
    write_json(manifest, design)
    info = folder/"attempt-1.json"
    if not info.exists():
        draws, diagnostics = sample(data, g, seed, attempt=2)
        path = folder/"attempt-1.npz"
        with path.with_suffix(".tmp").open("wb") as stream:
            np.savez_compressed(stream, **draws)
        path.with_suffix(".tmp").replace(path)
        write_json(info, {"attempt":1, "seed":seed, "diagnostics":diagnostics,
            "chains":4, "warmup":1800, "draws_per_chain":4000,
            "posterior_sha256":digest(path), "file":path.name})
    result = estimate(data, latitude, BASE, folder)
    print(record["slug"], result["passed"], result["volume_km3"], flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--cases", nargs="+", choices=[r["slug"] for r in datasets()])
    args = parser.parse_args()
    selected = args.cases or [r["slug"] for r in datasets()]
    for slug in selected:
        replicate(next(r for r in datasets() if r["slug"] == slug), args.runs, args.output)

