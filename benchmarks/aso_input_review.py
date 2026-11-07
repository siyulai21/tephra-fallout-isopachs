"""One source-backed Aso-4 erosion sensitivity; never rewrite the accepted input."""
import argparse
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.empirical_isopach.core.catalog import resolve_dataset
from src.empirical_isopach.models.mass_mixture.prior import BASELINE
from src.empirical_isopach.models.mass_mixture.study import audited_data, fit_checked


INPUT_SHA256 = "802f22266d905b8d10c0535a7767e7bd83fb7f797692ea869095a3fa410360d8"
SOURCE = "https://doi.org/10.1186/s40623-025-02230-9"


def run(output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    original = resolve_dataset("Aso-4")
    if sha256(original.read_bytes()).hexdigest() != INPUT_SHA256:
        raise ValueError("Re-audit this sensitivity after any accepted-input revision.")
    table = pd.read_csv(original)
    row = table.iloc[53]  # original CSV row 55, including its header
    np.testing.assert_allclose([row["Lat"], row["Lon"], row["Thickness"]],
        [43+45/60+33/3600, 144+18/60+45/3600, 7], rtol=0, atol=1e-6)
    assert row["Refs"] == "Shibata2025"
    table["Kind"] = "measurement"
    table["Limit_cm"] = np.nan
    table.loc[53, ["Kind", "Limit_cm"]] = ["lower_limit", 7.]
    path = output/"aso-4-row55-lower-bound.csv"
    table.to_csv(path, index=False)
    data, meta = audited_data(path)
    assert meta["n_measurements"] == 86 and meta["n_lower_limits"] == 1
    samples, fit = fit_checked(data, meta["vent_lonlat"][1], BASELINE,
        output/"posteriors", progress=lambda text:print(text, flush=True))
    result = {"source":SOURCE, "sections":"3.2 sampling coordinates; 4.1 field descriptions",
        "original_input_sha256":INPUT_SHA256, "original_csv_row":55,
        "source_site":"Loc.1: 43°45′33″ N, 144°18′45″ E",
        "source_evidence":"The paper describes a partly eroded ash layer 4–7 cm thick at this site.",
        "sensitivity_assumption":"Use the observed maximum 7 cm as a lower bound on original deposit thickness. This does not infer the amount eroded or turn the reported range into a measurement standard deviation.",
        "scope":"One likelihood sensitivity under the preceding displaced geometry; no new geographic evaluation or complete source-table audit. It does not certify the remaining records or select a global model.",
        "input":{k:v for k,v in meta.items() if k!="transform_inv"}, "fit":fit,
        "runner_sha256":sha256(Path(__file__).read_bytes()).hexdigest()}
    (output/"report.json").write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.output)
