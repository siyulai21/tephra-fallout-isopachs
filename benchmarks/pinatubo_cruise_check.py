"""Evaluate the newly recovered SO-114-23 bed against the preceding 193-site field estimates."""
import argparse
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
from src.Laea import make_laea_transform

from .pinatubo import AUDIT, ROOT
from .pinatubo_evidence import read
from .refresh_figures import recorded_observations, regional_models
from src.empirical_isopach.core.distributions import predictive_distribution
from src.empirical_isopach.visualization.deposit_field import open_field

PREVIOUS_INPUT = "d0db42a7c01582285a7b5a4317798bbfa9d5efb4b5955727afddedb0062352a8"
CRUISE_INTAKE = ROOT / "archive/2026-11-22-pinatubo-intake/pre-visual-review/input.csv"


def check(training, home, output):
    if sha256(training.read_bytes()).hexdigest() != PREVIOUS_INPUT:
        raise ValueError("This test uses the archived 193-observation intake.")
    previous, current = read(training), read(CRUISE_INTAKE)
    if previous != current[:-1] or current[-1]["EvidenceID"] != "S99-SO-114-23":
        raise ValueError("Expected the new core to be the only additional fitted observation.")
    data, meta = recorded_observations({"path":training}, home)
    regional = json.loads((home/"report.json").read_text())
    if regional["input"]["input_sha256"] != PREVIOUS_INPUT:
        raise ValueError("Regional fits do not correspond to the training input")
    models, _, _ = regional_models(data, regional)
    core = current[-1]
    lon, lat = float(core["Lon"]), float(core["Lat"])
    shifts = read(AUDIT/"survey-coverage/suedmeer-registration.csv")
    lons = [lon, *(lon+float(r["station_longitude"])-float(r["figure_longitude"]) for r in shifts)]
    lats = [lat, *(lat+float(r["station_latitude"])-float(r["figure_latitude"]) for r in shifts)]
    project, _, _ = make_laea_transform(*meta["vent_lonlat"])
    xy = np.column_stack(project.transform(lons, lats))
    fields = {k:np.exp(predictive_distribution(model, xy).quantile(.5)) for k, model in models.items()}
    med, _, _ = open_field(home/"nested-isopachs", "nested-isopachs", data, meta)
    fields["nested-isopachs"] = 10**med.central(xy)
    receipt = {"case":"Pinatubo Layer C", "training_input_sha256":PREVIOUS_INPUT, "training_rows":len(previous),
               "training_revision":"9d47ac038b930869dff6628b23b9ccd64681803d", "held_out_evidence_id":core["EvidenceID"],
               "observed_cm":float(core["Thickness"]), "longitude":lon, "latitude":lat,
               "registered_coordinate_shifts":len(shifts), "predictions":{}}
    for key, values in fields.items():
        receipt["predictions"][key] = {"median_field_cm":float(values[0]),
            "min_across_15_registration_offsets_cm":float(min(values[1:])),
            "max_across_15_registration_offsets_cm":float(max(values[1:]))}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(receipt, indent=2, allow_nan=False)+"\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-csv", type=Path, required=True)
    parser.add_argument("--training-results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    check(args.training_csv, args.training_results, args.output)
