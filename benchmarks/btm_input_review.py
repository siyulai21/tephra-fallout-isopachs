"""Isolate two source-backed distal observation meanings; preserve accepted data."""
import argparse
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.empirical_isopach.core.catalog import resolve_dataset
from src.empirical_isopach.models.mass_mixture.source_protocol import BASELINE
from src.empirical_isopach.models.mass_mixture.study import audited_data, fit_checked


INPUT_SHA256 = "56d82584b2c45fd378d639248416159ffea9350099d1d2009576551d1729cf8d"
SOURCE = "https://doi.org/10.1007/s00445-021-01487-8"


def run(output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    original = resolve_dataset("B-Tm")
    if sha256(original.read_bytes()).hexdigest() != INPUT_SHA256:
        raise ValueError("Re-audit this sensitivity after any accepted-input revision.")
    table = pd.read_csv(original)
    for row_number, coordinates in [(82, [35.583, 135.884, .1]),
                                    (84, [45.431, 141.038, .1])]:
        np.testing.assert_allclose(table.iloc[row_number-2][["Lat", "Lon", "Thickness"]].to_numpy(float),
                                   coordinates, rtol=0, atol=1e-10)
    original_data, meta = audited_data(original)
    data = original_data.subset(original_data.row_ids != 84)
    suigetsu = data.row_ids == 82
    assert suigetsu.sum() == 1 and len(data.row_ids) == 83
    data.censored[suigetsu] = True
    # The recorded value is already 0.1 cm: only its likelihood meaning changes.
    assert np.all(~data.lower_censored) and data.censored.sum() == 1
    selected = table.iloc[data.row_ids-2].copy()
    selected.insert(0, "OriginalCSVRow", data.row_ids)
    selected["Kind"] = np.where(data.censored, "upper_limit", "measurement")
    selected["Limit_cm"] = np.where(data.censored, .1, np.nan)
    selected = selected[["OriginalCSVRow", "Lon", "Lat", "Thickness", "Kind", "Limit_cm"]]
    selected.to_csv(output/"used-observations.csv", index=False)
    meta.update(n_measurements=82, n_upper_limits=1, n_lower_limits=0,
                excluded_nonquantitative_original_rows=[84],
                input_transformation="Original CSV row 82 becomes an upper bound; row 84 supplies no quantitative thickness likelihood. All other accepted rows and arrays are unchanged.")
    samples, fit = fit_checked(data, meta["vent_lonlat"][1], BASELINE,
                              output/"posteriors", progress=lambda text:print(text, flush=True))
    result = {"source": SOURCE, "source_table":"Yang et al. (2021), Supplementing dataset 1, thickness_data!A48:F49",
        "source_evidence": {"82":"Lake Suigetsu is reported as <0.1 cm.",
                            "84":"Lake Kushu is reported as no visible ash layer, with cryptotephra presence discussed in the supporting text; no 0.1 cm bulk thickness is supplied."},
        "original_input_sha256": INPUT_SHA256,
        "sensitivity_assumption":"Use the documented 0.1 cm upper bound for Suigetsu. Remove Kushu from the quantitative thickness likelihood without interpreting it as zero or denying its reported presence. No detection threshold is invented.",
        "scope":"Two-observation sensitivity with source-focused geometry and unchanged priors. Other source ranges, spots, and unverified proximal observations remain unresolved. No new geographic evaluation or full input replacement.",
        "input":{k:v for k,v in meta.items() if k!="transform_inv"},
        "retained_original_row_ids": data.row_ids.tolist(), "fit":fit,
        "runner_sha256":sha256(Path(__file__).read_bytes()).hexdigest()}
    (output/"report.json").write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
    assert sha256(original.read_bytes()).hexdigest() == INPUT_SHA256
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args().output)
