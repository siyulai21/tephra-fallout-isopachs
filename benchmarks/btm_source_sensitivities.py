"""Refit the six directional members to alternate source readings for B-Tm."""
import argparse
import json
from pathlib import Path

import numpy as np

from src.Laea import make_laea_transform
from src.empirical_isopach.core.catalog import data_directory, resolve_dataset
from src.empirical_isopach.core.distributions import LogMixture
from src.empirical_isopach.models.directional.ensemble.fields import (
    Ensemble, Marginal, draw_weights, load_members, load_packet,
)
from src.empirical_isopach.models.directional.ensemble.members import NAMES, run_members
from src.empirical_isopach.models.directional.ensemble.report import modeled_mass
from src.empirical_isopach.models.directional.ensemble.weights import calibrate
from src.empirical_isopach.models.directional.study import write_json
from src.empirical_isopach.models.mass_mixture.study import audited_data


def save_volume_draws(folder):
    totals = {}
    for name in NAMES:
        with np.load(folder / "fits" / name / "mass.npz") as archive:
            totals[name] = archive["mass"].sum(axis=(-2, -1))
    np.savez_compressed(folder / "volume-draws.npz", **totals)


BASE_CASES = ("range_low", "range_high", "correlations")
SAMJIYON_CASES = ("samjiyon_scalar", "samjiyon_component", "samjiyon_complete_stack", "samjiyon_160")
SOURCE_CASES = BASE_CASES + SAMJIYON_CASES + ("omit_compilation_values",)


SAMJIYON_POSITIONS = (
    ("Workbook position", 128.29277, 41.82291),
    ("Samjiyŏn-ŭp, GeoNames 2041442", 128.31886, 41.80356),
    ("Samji-yŏn pond, GeoNames 2041444", 128.33380, 41.83084),
)


def samjiyon_xy(meta):
    transform, _, _ = make_laea_transform(*meta["vent_lonlat"])
    return np.array([transform.transform(lon, lat) for _, lon, lat in SAMJIYON_POSITIONS])


def save_samjiyon_draws(data, meta, fit_work, weights, path):
    """Save the conditional mean and variance of log thickness (thickness in cm) by chain, draw and position."""
    members = load_members(data.canonical(), meta["vent_lonlat"][1], fit_work, n=None)
    model = Ensemble(members, weights)
    predicted = model.distribution(samjiyon_xy(meta), observation=True)
    arrays = {}
    start = 0
    for name, member in zip(NAMES, members, strict=True):
        chains, draws = member.draws["source"].shape
        end = start + chains*draws
        arrays[name+"__mean_log_thickness"] = predicted.means[start:end].reshape(chains, draws, -1)
        arrays[name+"__variance_log_thickness"] = predicted.variances[start:end].reshape(chains, draws, -1)
        start = end
    np.savez_compressed(path, **arrays)
    return local_summaries(predicted)


def local_summaries(prediction):
    field = Marginal(prediction.means, prediction.weights)
    rows = [{"position": site, "longitude": lon, "latitude": lat}
            for site, lon, lat in SAMJIYON_POSITIONS]
    for label, distribution in (("field", field), ("local_prediction", prediction)):
        quantiles = {k: np.exp(distribution.quantile(p)) for k, p in
                     (("lower95", .025), ("median", .5), ("upper95", .975))}
        for i, row in enumerate(rows):
            row[label+"_cm"] = {k: float(v[i]) for k, v in quantiles.items()}
    for threshold in (100, 700):
        log_limit = np.log(threshold)
        field_above = field.weights @ (field.values > log_limit)
        prediction_above = 1 - prediction.cdf(log_limit)
        for i, row in enumerate(rows):
            row[f"field_draw_fraction_above_{threshold}cm"] = float(field_above[i])
            row[f"local_prediction_probability_above_{threshold}cm"] = float(prediction_above[i])
    return rows


def predict_samjiyon(data, meta, folder, full_draws):
    """Use all principal draws and replay the map subset independently."""
    report = json.loads((folder/"report.json").read_text())
    if report["input"]["input_sha256"] != meta["input_sha256"]:
        raise ValueError("Principal prediction packet and source table differ")
    model = load_packet(folder, data, meta["vent_lonlat"][1], report)
    map_prediction = model.distribution(samjiyon_xy(meta), observation=True)
    counts, means, variances = {}, [], []
    start = 0
    with np.load(full_draws) as archive:
        for name, member in zip(NAMES, model.members, strict=True):
            mu = archive[name+"__mean_log_thickness"]
            var = archive[name+"__variance_log_thickness"]
            draws = member.records[-1]["draws_per_chain"]
            chains = member.draws["source"].shape[0]
            shape = (chains, draws, len(SAMJIYON_POSITIONS))
            if mu.shape != shape or var.shape != shape:
                raise ValueError(f"Incomplete local prediction draws for {name}")
            end = start + chains*len(member.index)
            for array, reference in ((mu, map_prediction.means), (var, map_prediction.variances)):
                np.testing.assert_allclose(array[:, member.index].reshape(end-start, -1),
                                           reference[start:end], rtol=1e-8, atol=1e-10)
            counts[name] = chains*draws
            means.append(mu.reshape(chains*draws, -1))
            variances.append(var.reshape(chains*draws, -1))
            start = end
    weights = draw_weights(model.model_weights, list(counts.values()))
    full = LogMixture(weights, np.concatenate(means), np.concatenate(variances))
    return {
        "principal_input_sha256": meta["input_sha256"],
        "map_draws_per_chain_per_structure": report["map_draws_per_chain"],
        "full_draws_by_structure": counts,
        "interpretation": "All retained final-chain draws at the three named positions, conditional on the principal directional ensemble. The local prediction includes estimated residual variation. Gazetteer coordinates illustrate registration sensitivity. The field exposure has no surveyed position.",
        "positions": local_summaries(full),
        "map_packet_positions": local_summaries(map_prediction),
    }


def write_local_predictions(directional, output):
    """Publish the local predictions from a completed principal directional run."""
    data, meta = audited_data(resolve_dataset("B-Tm"))
    ensemble = directional/"ensemble"
    report = json.loads((ensemble/"report.json").read_text())
    if report["input"]["input_sha256"] != meta["input_sha256"]:
        raise ValueError("The directional run must use the accepted B-Tm input")
    output.mkdir(exist_ok=True, parents=True)
    packet = output/"samjiyon-local-draws.npz"
    save_samjiyon_draws(data, meta, directional/"work/ensemble", report["weighting"]["weights"], packet)
    prediction = predict_samjiyon(data.canonical(), meta, ensemble, packet)
    write_json(output/"samjiyon-prediction.json", prediction)


def write_source_class_prediction(data, meta, folder, weights):
    """Check the withheld Samjiyon prediction when compilation-carried thicknesses are omitted."""
    samples = folder/"samjiyon-local-draws.npz"
    positions = save_samjiyon_draws(data, meta, folder, weights, samples)
    write_json(folder/"samjiyon-prediction.json", {"input_sha256":meta["input_sha256"],
               "ensemble_weights":weights, "positions":positions})


def run(work, cases=SOURCE_CASES):
    root = data_directory()
    audit = json.loads((root / "audits/B-Tm/provenance.json").read_text())
    samjiyon = json.loads((root / "audits/B-Tm/samjiyon-provenance.json").read_text())
    source_class = json.loads((root / "audits/B-Tm/source-class-provenance.json").read_text())
    inputs = {**audit["cases"], **samjiyon["cases"], **source_class["cases"]}
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
        if name == "omit_compilation_values":
            write_source_class_prediction(data, meta, folder, weights["weights"])
        print(name, centre["volume_km3"], joint["volume_km3"], flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--principal-directional", type=Path,
                        help="Completed principal directional directory with its work/ensemble fits")
    parser.add_argument("--cases", nargs="*", choices=SOURCE_CASES, default=SOURCE_CASES)
    args = parser.parse_args()
    if not args.cases and not args.principal_directional:
        parser.error("Select a source case or supply --principal-directional for local predictions")
    run(args.work, args.cases)
    if args.principal_directional:
        write_local_predictions(args.principal_directional, args.work)
