"""Compare fitted thickness at nine marine ranges and at the two land-map positions."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd

from benchmarks.pinatubo import INPUT, ROOT, src
from benchmarks.refresh_figures import regional_models
from src.Laea import make_laea_transform
from src.empirical_isopach.core.audit import audited_data
from src.empirical_isopach.core.distributions import predictive_distribution
from src.empirical_isopach.models.nested.reach import ReachDraws
from src.empirical_isopach.models.nested.report import MapView
from src.empirical_isopach.models.nested.study import digest
from src.empirical_isopach.visualization.deposit_field import FAMILIES, open_field


def replay(home, output, paired_output=None):
    data, meta = audited_data(INPUT)
    accepted = pd.read_csv(INPUT)
    sample = {v: i for i, v in enumerate(data.row_ids)}
    ranges = []
    for item in src("hess-table-1.csv"):
        ident = "H01-" + item["site"].zfill(2)
        matching = accepted.index[accepted.EvidenceID.eq(ident)]
        if len(matching) and "–" in item["thickness_mm"]:
            lo, hi = (float(v) / 10 for v in item["thickness_mm"].split("–"))
            row = int(matching[0]) + 2
            if accepted.loc[row-2, "Kind"] != "lower_limit" or accepted.loc[row-2, "Limit_cm"] != lo:
                raise ValueError(f"Unexpected common input for {ident}")
            ranges.append({"evidence_id": ident, "station": item["stations"], "csv_row": row,
                           "reported_low_cm": lo, "reported_high_cm": hi})
    if len(ranges) != 9:
        raise ValueError("Expected the nine independent Hess replicate ranges")
    xy = data.xy[[sample[r["csv_row"]] for r in ranges]]
    paired = None
    query = xy
    if paired_output is not None:
        paired = pd.read_csv(home/"source-checks/paired-map-coordinates/paired-coordinates.csv")
        if len(paired) != 51 or not paired.evidence_id.is_unique:
            raise ValueError("Expected every accepted paired sublayer site once")
        project, _, _ = make_laea_transform(*meta["vent_lonlat"])
        points = [np.column_stack(project.transform(paired[p+"_lon"],paired[p+"_lat"])) for p in ("c1","c2")]
        np.testing.assert_allclose(points[0], data.xy[[sample[r] for r in paired.csv_row]], atol=1e-7, rtol=0)
        query = np.vstack([xy, *points])

    def save_principal(key, values):
        predictions["principal_"+key+"_cm"] = values[:len(xy)]
        if paired is not None:
            first, second = np.split(values[len(xy):], 2)
            paired[key+"_at_c1_cm"] = first
            paired[key+"_at_c2_cm"] = second

    predictions = {}
    for name, folder, source in [
        ("principal", home, INPUT), ("upper_endpoint", home/"source-checks/regional-upper-endpoints/regional",
                                            home/"source-checks/regional-upper-endpoints/upper-endpoints.csv")]:
        evidence, local_meta = audited_data(source)
        regional = json.loads((folder/"report.json").read_text())
        if regional["input"]["input_sha256"] != local_meta["input_sha256"]:
            raise ValueError(f"Regional report and input differ for {name}")
        models, _, _ = regional_models(evidence.canonical(), regional)
        for key, model in models.items():
            if name == "principal":
                save_principal(key, np.exp(predictive_distribution(model, query).quantile(.5)))
            else:
                predictions[name+"_"+key+"_cm"] = np.exp(predictive_distribution(model, xy).quantile(.5))
    for family in FAMILIES:
        field, _, _ = open_field(home/family, family, data.canonical(), meta)
        save_principal(family, 10**field.central(query))
    folder = home/"source-checks/marine-intervals"
    receipt = json.loads((folder/"result.json").read_text())
    if receipt["corridor_packet_sha256"] != digest(folder/"corridor.npz"):
        raise ValueError("Interval-sensitivity packet differs from the receipt")
    view = MapView(ReachDraws.load(folder/"corridor.npz").reduced(receipt["volume"]["quadrature"][-1]["draws"]))
    predictions["two_sided_nested_cm"] = 10**view.central(xy)
    for field, values in predictions.items():
        for row, value in zip(ranges, values, strict=True):
            row[field] = value
    with output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=ranges[0].keys(), lineterminator="\n")
        writer.writeheader()
        writer.writerows(ranges)
    if paired is not None:
        paired.to_csv(paired_output, index=False, lineterminator="\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-results", type=Path, default=ROOT/"eruptions/pinatubo-1991-layer-c/results")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--paired-output", type=Path, help="Also compare the fixed principal fields at the original C1 and C2 map positions.")
    args = parser.parse_args()
    replay(args.case_results, args.output, args.paired_output)
