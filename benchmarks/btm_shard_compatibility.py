"""Evaluate published shard-count transfers against local predictive thickness distributions."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np

from src.empirical_isopach.models.directional.archive_readings import condition_model, site_predictions
from src.empirical_isopach.models.directional.bearing_report import configuration
from src.empirical_isopach.models.directional.study import digest, write_json
from src.empirical_isopach.models.mass_mixture.archive_check import archive_checks

# Equivalents in cm are from Costa et al., Table S1. Primary counts refer to the
# named size fraction, colour or interval in the audit and are never summed.
PER_MM = 25_000
PROBES = (
    ("longgang", "Costa P36", .00256, None),
    ("longgang", "Sun T2 colourless peak, 53 cm", None, 5314),
    ("longgang", "Sun T2 brown peak, 52 cm", None, 1338),
    ("old-084", "Costa P35", .00144, None),
    ("old-084", "Chen RK12-0151 peak", None, 30671),
    ("old-085", "Costa P74", .038, None),
    ("old-085", "Chen core 13PT-P4_026, 25–27 cm", None, 9529),
)
KEYS = ("quantitative", "both_10", "visible_analogs")


def compare(work, archive_work, posterior, output):
    folder = Path(output)
    site_ids = {key for key, *_ in PROBES}
    data, meta, inventory, source, base, _ = configuration(
        Path("data/eruptions/B-Tm_reviewed.csv"), work, Path("data/audits/B-Tm/archive-observations.json"))
    probes = [p for p in inventory["sites"] if p["id"] in site_ids]
    scenarios = [s for s in inventory["scenarios"] if s["id"] in KEYS]
    definitions = [(key, label, fixed if fixed is not None else count / PER_MM / 10, count)
                   for key, label, fixed, count in PROBES]
    cutoffs = sorted({1., *(v for _, _, v, _ in definitions)})
    with np.load(posterior) as raw:
        sphere, _ = archive_checks(dict(raw), data, meta["vent_lonlat"], {**inventory, "scenarios": scenarios}, cutoffs)
    reference = json.loads((folder/"spherical-archive-checks.json").read_text())
    assert digest(posterior) == reference["posterior_sha256"]
    identities, rows = {}, []
    for key in KEYS:
        scenario = next((s for s in scenarios if s["id"] == key), None)
        model = base if scenario is None else condition_model(data, base, inventory["sites"],
                        meta["vent_lonlat"], scenario, archive_work)[0]
        directional = site_predictions(model, probes, meta["vent_lonlat"], cutoffs)
        prior = json.loads((folder/key/"result.json").read_text())
        np.testing.assert_allclose(model.model_weights, prior["structural_weights"])
        identities[key] = [m.records[-1]["posterior_sha256"] for m in model.members]
        for family, actual, published, field in (
            ("directional", directional, prior["sites"], "probability_local_below"),
            ("spherical", sphere[key]["sites"], reference["scenarios"][key]["sites"], "p_local_below_scale"),
        ):
            indexed = {s["id"]: s for s in actual}
            original = {s["id"]: s for s in published}
            for sid, label, value, count in definitions:
                np.testing.assert_allclose(indexed[sid][field]["1.0"], original[sid][field]["1.0"], rtol=1e-10)
                rows.append({"model": family, "scenario": key, "archive_id": sid,
                             "site": next(s["name"] for s in probes if s["id"] == sid),
                             "source_statistic": label, "shards_per_g_dry": count,
                             "transfer_equivalent_cm": value,
                             "p_local_at_or_below_equivalent": indexed[sid][field][str(value)],
                             "archive_used_for_visibility": sid in (scenario or {}).get("effective_limit_cm", {})})
    with (folder/"shard-transfer-comparison.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0], lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    write_json(folder/"shard-transfer-provenance.json", {
        "input_sha256": meta["input_sha256"], "archive_inventory_sha256": digest(source),
        "spherical_posterior_sha256": digest(posterior), "directional_posterior_sha256": identities,
        "shards_per_g_per_mm": PER_MM, "observed_statistic": "different selected peaks or published equivalent",
        "method": "CDF of a new local thickness including site scatter, using complete posterior samples; the shard observations were never used in these likelihoods.",
        "file_sha256": digest(folder/"shard-transfer-comparison.csv")})
    return rows


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--work", type=Path, required=True)
    p.add_argument("--archive-work", type=Path, required=True)
    p.add_argument("--spherical", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    compare(a.work, a.archive_work, a.spherical, a.output)
