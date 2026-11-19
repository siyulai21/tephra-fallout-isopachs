"""Refit the six directional members to alternate source readings for B-Tm."""
import argparse
import json
from pathlib import Path

import numpy as np

from src.empirical_isopach.core.catalog import data_directory
from src.empirical_isopach.models.directional.ensemble.members import NAMES, run_members
from src.empirical_isopach.models.directional.ensemble.report import modeled_mass
from src.empirical_isopach.models.directional.ensemble.weights import calibrate
from src.empirical_isopach.models.directional.study import write_json
from src.empirical_isopach.models.mass_mixture.study import audited_data


SOURCE_CASES = ("range_low", "range_high", "correlations", "omit_compilation_values")


def save_volume_draws(folder):
    totals = {}
    for name in NAMES:
        with np.load(folder / "fits" / name / "mass.npz") as archive:
            totals[name] = archive["mass"].sum(axis=(-2, -1))
    np.savez_compressed(folder / "volume-draws.npz", **totals)


def run(work, cases=SOURCE_CASES):
    root = data_directory()
    audit = json.loads((root / "audits/B-Tm/provenance.json").read_text())
    source_class = json.loads((root / "audits/B-Tm/source-class-provenance.json").read_text())
    inputs = {**audit["cases"], **source_class["cases"]}
    if set(cases).difference(SOURCE_CASES):
        raise ValueError("Unknown B-Tm source sensitivity")
    for name in cases:
        case = inputs[name]
        data, meta = audited_data(root.parent / case["file"])
        if meta["input_sha256"] != case["sha256"]:
            raise ValueError(f"Source scenario {name} has changed")
        weights = calibrate(meta, "b-tm")
        if "b-tm" in weights["calibration_cases"]:
            raise ValueError("Target used to fit its own ensemble weights")
        latitude = meta["vent_lonlat"][1]
        folder = work / name
        run_members(data, latitude, folder, kinds=("full",))
        _, joint, _, centre = modeled_mass(data, latitude, folder, weights)
        report = {"case": name, "definition": case["definition"], "input": case,
                  "weighting": weights, "median_surface": centre, "coherent_fields": joint}
        write_json(folder / "result.json", report)
        save_volume_draws(folder)
        print(name, centre["volume_km3"], joint["volume_km3"], flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--cases", nargs="+", choices=SOURCE_CASES, default=SOURCE_CASES)
    args = parser.parse_args()
    run(args.work, args.cases)
