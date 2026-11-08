"""Predeclared setting-removal stress test; not a marine transport correction."""
import argparse
from dataclasses import replace
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
import pandas as pd

from benchmarks.aso_evidence import dump
from src.empirical_isopach.core.validation import interval_metrics
from src.empirical_isopach.models.mass_mixture.inference import Sampling
from src.empirical_isopach.models.mass_mixture.posterior import predictions
from src.empirical_isopach.models.mass_mixture.prior import BASELINE
from src.empirical_isopach.models.mass_mixture.study import audited_data, fit_checked


def marine_rows(ledger):
    """Classify only the source's explicit setting, without using thickness."""
    return ledger.loc[ledger.setting.fillna("").str.strip().str.startswith("Submarine"), "csv_row"].astype(int).tolist()


def run(input_path, ledger_path, output):
    input_path, ledger_path, output = map(Path, (input_path, ledger_path, output))
    output.mkdir(parents=True, exist_ok=True)
    data, meta = audited_data(input_path)
    rows = marine_rows(pd.read_csv(ledger_path))
    protocol = {
        "input_sha256": meta["input_sha256"], "ledger_sha256": sha256(ledger_path.read_bytes()).hexdigest(),
        "omitted_csv_rows": rows, "displacement_prior_sd": [3., 1.],
        "sampling": Sampling(seed=20261108).__dict__,
        "timing": "Declared after the user identified Machida & Arai (1988) and before either setting-removal fit.",
        "definition": "Remove every explicitly submarine record in Table S2, irrespective of thickness or residual. Retain the other records, including 35 whose setting is not independently classified by this audit. This is not a certified terrestrial-only sample.",
        "interpretation": "Structural stress test, not evidence that marine records are wrong, a Kuroshio correction, or a probability-weighted uncertainty bound. Environment and geography are confounded. Predictions at omitted marine sites are unbuffered transfer diagnostics, not the geographic model-comparison score.",
        "runner_sha256": sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    path = output/"protocol.json"
    if path.exists() and json.loads(path.read_text()) != protocol:
        raise ValueError("Protocol changed: use a new destination.")
    dump(path, protocol)
    withheld = np.isin(data.row_ids, rows)
    assert withheld.sum() == len(rows) == 28
    results = {}
    for sd in protocol["displacement_prior_sd"]:
        key = f"offset_sd{int(sd)}"
        samples, fit = fit_checked(data.subset(~withheld), meta["vent_lonlat"][1],
            replace(BASELINE, offset_sd=sd), output/key, Sampling(seed=20261108), progress=lambda t:print(t,flush=True))
        p = predictions(samples, data.subset(withheld), meta["vent_lonlat"][1])
        results[key] = {"fit": fit, "training_row_ids": data.row_ids[~withheld].tolist(),
            "withheld_row_ids": data.row_ids[withheld].tolist(),
            "withheld_metrics": interval_metrics(data.subset(withheld),p["median"],p["lower"],p["upper"],p["log_score"]),
            "predictions": {k:v.tolist() for k,v in p.items()}}
        dump(output/"report.json", {"protocol": protocol, "results": results})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--ledger", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    run(args.input, args.ledger, args.output)
