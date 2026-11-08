"""Repeat both Biwa primary-table endpoints on the corrected-coordinate input."""
import argparse
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
import pandas as pd

from benchmarks.aso_evidence import dump
from src.empirical_isopach.models.mass_mixture.inference import Sampling
from src.empirical_isopach.models.mass_mixture.prior import BASELINE
from src.empirical_isopach.models.mass_mixture.study import audited_data, fit_checked


def run(input_path, output):
    input_path, output = Path(input_path), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    protocol = {"input_sha256":sha256(input_path.read_bytes()).hexdigest(),
        "cases":{"primary_6p5":6.5, "primary_7":7.},
        "definition":"Change only original CSV row 53 from the S2 lower-bound event to an exact endpoint of Danhara Table 4's 6.5–7 cm range. Keep both corrected coordinates, the other three bounds and all observations.",
        "interpretation":"Endpoint source sensitivities, not a distribution for the range or bounds on total volume. No selection by volume.",
        "timing":"Declared after the source revision, before either corrected-coordinate endpoint fit. Earlier endpoint experiments used the original coordinates and are retained separately.",
        "sampling":Sampling(seed=20261108).__dict__, "prior":BASELINE.describe(),
        "runner_sha256":sha256(Path(__file__).read_bytes()).hexdigest()}
    p = output/"protocol.json"
    if p.exists() and json.loads(p.read_text()) != protocol:
        raise ValueError("Protocol changed: use a new destination.")
    dump(p, protocol)
    results = {}
    for name, value in protocol["cases"].items():
        folder = output/name; folder.mkdir(exist_ok=True)
        table = pd.read_csv(input_path)
        table.loc[51,["Kind","Limit_cm","Thickness"]] = ["measurement",np.nan,value]
        path = folder/"input.csv"; table.to_csv(path,index=False)
        data, meta = audited_data(path)
        _, fit = fit_checked(data,meta["vent_lonlat"][1],BASELINE,folder/"posteriors",
            Sampling(seed=20261108),progress=lambda t:print(t,flush=True))
        results[name] = {"input_sha256":meta["input_sha256"],"fit":fit}
        dump(output/"report.json",{"protocol":protocol,"results":results})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    run(args.input,args.output)
