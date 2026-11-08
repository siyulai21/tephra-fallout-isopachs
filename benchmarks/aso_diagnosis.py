"""Diagnose Aso-4 extrapolation without changing data or selecting low volumes.

Single-observation deletion is a PSIS approximation, explicitly screened for
unstable importance ratios. Geographic deletion results are independent saved
refits. Neither is an automatic observation-exclusion rule.
"""
import argparse
from hashlib import sha256
import json
from pathlib import Path

import arviz as az
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from src.empirical_isopach.models.mass_mixture.inference import flatten, observation_log_likelihood
from src.empirical_isopach.models.mass_mixture.geometry import centers, to_sphere, log_kernel, RADIUS
from src.empirical_isopach.models.mass_mixture.posterior import quantiles
from src.empirical_isopach.models.mass_mixture.study import audited_data


def weighted_quantiles(values, weights, probabilities=(.025, .5, .975)):
    """Inverse weighted empirical CDF, without interpolation across absent mass."""
    values, weights = np.asarray(values), np.asarray(weights)
    if values.ndim != 1 or weights.shape != values.shape or np.any(weights < 0):
        raise ValueError("Require matching one-dimensional values and nonnegative weights.")
    if not np.isfinite(values).all() or not np.isfinite(weights).all() or weights.sum() <= 0:
        raise ValueError("Require finite values and positive total weight.")
    probabilities = np.asarray(probabilities)
    if probabilities.ndim != 1 or not np.isfinite(probabilities).all() or np.any((probabilities < 0) | (probabilities > 1)):
        raise ValueError("Quantile probabilities must be a finite vector in [0,1].")
    positive = weights > 0
    values, weights = values[positive], weights[positive]
    order = np.argsort(values, kind="stable")
    cdf = np.cumsum(weights[order])/weights.sum()
    return values[order[np.minimum(np.searchsorted(cdf, probabilities), len(values)-1)]]


def deletion_summary(log_volume, log_likelihood, relative_efficiency):
    log_weights, shape = az.psislw(-np.asarray(log_likelihood), reff=relative_efficiency)
    weights = np.exp(log_weights)
    q = weighted_quantiles(np.exp(log_volume), weights)
    n = len(weights)
    threshold = min(.7, 1-1/np.log10(n))
    return {"lower95_km3": float(q[0]), "median_km3": float(q[1]), "upper95_km3": float(q[2]),
        "pareto_k": float(shape), "pareto_threshold": threshold,
        "psis_passed": bool(shape < threshold),
        "importance_ess": float(1/np.sum(weights**2)),
        "relative_efficiency": relative_efficiency,
        "d_mean_log_volume_d_likelihood_power": float(np.cov(log_volume, log_likelihood, ddof=0)[0, 1])}


ORIGINAL_GLOBAL = Path("archive/2026-11-08-aso-4/results/global")


def run(output, original_global=ORIGINAL_GLOBAL):
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    input_path = Path("data/eruptions/Aso-4_tephra.csv")
    data, meta = audited_data(input_path); data = data.canonical()
    table = pd.read_csv(input_path); table["Refs"] = table["Refs"].fillna("")
    table.index = np.arange(len(table))+2
    report_path = Path(original_global)/"report.json"
    report = json.loads(report_path.read_text())
    assert meta["input_sha256"] == report["input"]["input_sha256"]
    posterior_path = report_path.parent/"posterior.npz"
    digest = sha256(posterior_path.read_bytes()).hexdigest()
    assert digest == report["fits"]["baseline"]["posterior_sha256"]
    with np.load(posterior_path) as a:
        samples = {k:a[k] for k in a.files}
    flat = flatten(samples)
    likelihood = observation_log_likelihood(flat, data, meta["vent_lonlat"][1])
    ess = float(az.ess(az.from_dict(posterior={"log_volume":samples["log_volume"]}), method="bulk").log_volume)
    reff = min(1., ess/len(flat["log_volume"]))
    rows = []
    for i, row_id in enumerate(data.row_ids):
        source = table.loc[row_id]
        rows.append({"csv_row": int(row_id), "latitude": float(source.Lat), "longitude": float(source.Lon),
            "thickness_cm": float(source.Thickness), "supplied_reference": source.Refs,
            **deletion_summary(flat["log_volume"], likelihood[:, i], reff)})
    pd.DataFrame(rows).sort_values("csv_row").to_csv(output/"single-row-deletion.csv", index=False)
    volume = np.exp(flat["log_volume"])
    c = centers(flat["log_scale"], flat["offset"])
    unit = to_sphere(data.xy, meta["vent_lonlat"][1])
    nearest = RADIUS*np.arccos(np.clip(np.max(np.einsum("dki,ni->dkn", c, unit), axis=-1), -1, 1))
    source_distance = RADIUS*np.arccos(np.clip(c[...,2], -1, 1))
    kernel = log_kernel(unit, flat["log_scale"], c, flat["exponent"][:,None],
        flat["tail_fraction"][:,None], kernel_weights=flat["kernel_weights"][:,None,:])
    from scipy.special import logsumexp
    contribution = np.log(flat["weights"])[...,None]+kernel
    share = np.mean(np.exp(contribution-logsumexp(contribution, axis=1)[:,None,:]), axis=-1)
    features = {"power_exponent":flat["exponent"], "residual_log_sd":flat["noise"],
        "observation_contamination":flat["contamination"],
        "mass_weighted_scale_km":np.sum(flat["weights"]*np.exp(flat["log_scale"]), axis=1),
        "mass_weighted_center_to_source_km":np.sum(flat["weights"]*source_distance, axis=1),
        "mass_weighted_center_to_nearest_site_km":np.sum(flat["weights"]*nearest, axis=1),
        "mass_weighted_offset_in_scale_units":np.sum(flat["weights"]*np.linalg.norm(flat["offset"],axis=-1), axis=1),
        "mass_vs_site_contribution_total_variation":.5*np.sum(abs(flat["weights"]-share), axis=-1),
        **{name+"_mass_fraction":flat["kernel_weights"][:,i] for i,name in enumerate(("gaussian", "exponential", "power"))}}
    diagnostics = {name:{"posterior":quantiles(value),
        "spearman_with_log_volume":float(spearmanr(flat["log_volume"], value).statistic),
        "volume_bottom_quartile":quantiles(value[volume <= np.quantile(volume,.25)]),
        "volume_top_quartile":quantiles(value[volume >= np.quantile(volume,.75)])}
        for name,value in features.items()}
    groups = []
    for f in report["validation"]["fits"]:
        groups.append({"fold":f["fold"], "withheld_rows":f["withheld_row_ids"],
            "training_rows":f["training_row_ids"], "volume":f["fit"]["volume"],
            "numerical_diagnostics":f["fit"]["diagnostics"],
            "interpretation":"Full refit after the same buffered geographic deletion used in the published validation."})
    result = {"input_sha256":meta["input_sha256"], "posterior_sha256":digest,
        "original_report_sha256":sha256(report_path.read_bytes()).hexdigest(),
        "baseline_volume":report["fits"]["baseline"]["volume"], "features":diagnostics,
        "geographic_deletion_refits":groups, "single_row_deletion":{
            "rows":len(rows), "unstable_csv_rows":[r["csv_row"] for r in rows if not r["psis_passed"]],
            "reference":"https://jmlr.org/papers/v25/19-556.html",
            "interpretation":"PSIS deletion summaries are approximations conditional on the original field family. Pareto-k failures require refitting; importance ESS is not an independent-chain ESS. Influence is not evidence that a record is erroneous. The covariance is a local derivative of posterior mean log volume with respect to that observation's likelihood power."},
        "runner_sha256":sha256(Path(__file__).read_bytes()).hexdigest()}
    (output/"report.json").write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
    print("Completed Aso-4 influence and tail diagnosis.", flush=True)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--original-global", type=Path, default=ORIGINAL_GLOBAL)
    args = parser.parse_args()
    run(args.output, args.original_global)
