"""Reproducible fits, sensitivity refits, and genuinely withheld predictions."""
from dataclasses import asdict, replace
from hashlib import sha256
from pathlib import Path
from importlib.metadata import version
import json
import platform
import numpy as np

from ...core.observations import read_observations
from ...core.catalog import dataset_record, data_directory
from ...core.validation import spatial_folds, training_mask, interval_metrics
from ...validation import automatic_buffer
from .prior import BASELINE, SENSITIVITIES
from .inference import Sampling, sample
from .posterior import volume_summary, predictions


def _model_hashes():
    root = Path(__file__).parent
    hashes = {name: sha256((root/name).read_bytes()).hexdigest()
              for name in ("geometry.py", "prior.py", "inference.py", "source_geometry.py", "vent_envelope.py", "supported_basis.py")}
    hashes["WGS84_EarthDomain"] = sha256((root.parents[2]/"probabilistic_isopach/global_volume.py").read_bytes()).hexdigest()
    return hashes


# Freeze the source/runtime identity when this worker imports its model. A
# concurrent source edit must never relabel already-loaded code as the new code.
_LOADED_MODEL_HASHES = _model_hashes()
_RUNTIME = {"python": platform.python_version(),
    "packages": {k: version(k) for k in ("numpy", "scipy", "numpyro", "jax", "jaxlib", "arviz")}}


def model_hashes():
    return dict(_LOADED_MODEL_HASHES)


def audited_data(path):
    data, meta = read_observations(path)
    record = dataset_record(path)
    if record and record.get("audit"):
        audit_path = data_directory()/record["audit"]
        audit = json.loads(audit_path.read_text())
        case = audit["cases"][record["audit_case"]]
        if meta["input_sha256"] != case["sha256"]:
            raise ValueError("Accepted input differs from its recorded audit.")
        meta["audit"] = {"file": record["audit"], "sha256": sha256(audit_path.read_bytes()).hexdigest(),
                         "case": record["audit_case"], "definition": case["definition"]}
    meta["source"] = "data/"+record["file"] if record else Path(path).name
    return data, meta


def assess_diagnostics(diagnostics):
    """Keep mixing validity separate from NUTS trajectory-length efficiency.

    The initial development sampler bundled a 1% depth-saturation threshold
    into its pass flag. Preserve that raw flag in each attempt, while judging
    inference by R-hat, ESS and divergences. Depth alone does not imply bias:
    https://mc-stan.org/learn-stan/diagnostics-warnings.html#maximum-treedepth
    """
    d = dict(diagnostics)
    d["raw_sampler_passed"] = d["passed"]
    d["passed"] = bool(d["max_label_invariant_rhat"] < 1.01
        and d["min_label_invariant_bulk_ess"] >= 400
        and d["min_label_invariant_tail_ess"] >= 400 and d["divergences"] == 0
        and d["min_energy_bfmi"] >= .3)
    d["trajectory_efficiency_warning"] = d["max_depth_fraction"] >= .01
    d["assessment"] = "R-hat <1.01, bulk/tail ESS >=400, zero divergences, every chain's energy BFMI >=0.3. Maximum-depth saturation is a separate efficiency warning."
    return d


def assess_posterior(diagnostics, samples):
    """Also check the observation-mixture probability, without changing draws.

    The sampler's original diagnostics remain in each immutable attempt file.
    This posterior assessment is repeated when loading a verified cache.
    """
    d = dict(diagnostics)
    epsilon = np.asarray(samples["contamination"])
    if np.ptp(epsilon) > 1e-12:
        import arviz as az
        data = az.from_dict(posterior={"contamination": epsilon})
        rhat = float(az.rhat(data, method="rank").contamination)
        bulk = float(az.ess(data, method="bulk").contamination)
        tail = float(az.ess(data, method="tail").contamination)
        d["observation_mixture"] = {"rhat": rhat, "bulk_ess": bulk, "tail_ess": tail}
        d["max_label_invariant_rhat"] = max(d["max_label_invariant_rhat"], rhat)
        d["min_label_invariant_bulk_ess"] = min(d["min_label_invariant_bulk_ess"], bulk)
        d["min_label_invariant_tail_ess"] = min(d["min_label_invariant_tail_ess"], tail)
        d["label_invariant_variables"] = [*d.get("label_invariant_variables", []), "contamination"]
    else:
        d["observation_mixture"] = {"fixed_probability": float(epsilon.flat[0])}
    if "fixed_center" in samples:
        import arviz as az
        weights = np.asarray(samples["weights"])
        variables = {f"basis_weight_{j}":weights[...,j] for j in range(weights.shape[-1])
                     if np.ptp(weights[...,j]) > 1e-12}
        if variables:
            checked = az.from_dict(posterior=variables)
            rhat = float(az.rhat(checked,method="rank").to_array().max())
            bulk = float(az.ess(checked,method="bulk").to_array().min())
            tail = float(az.ess(checked,method="tail").to_array().min())
            d["identified_basis_weights"] = {"max_rhat":rhat,"min_bulk_ess":bulk,"min_tail_ess":tail}
            d["max_label_invariant_rhat"] = max(d["max_label_invariant_rhat"],rhat)
            d["min_label_invariant_bulk_ess"] = min(d["min_label_invariant_bulk_ess"],bulk)
            d["min_label_invariant_tail_ess"] = min(d["min_label_invariant_tail_ess"],tail)
            d["label_invariant_variables"] = list(dict.fromkeys([*d.get("label_invariant_variables",[]),*variables]))
    return assess_diagnostics(d)


def fit_cached(data, latitude, prior, directory, settings=Sampling(), attempts=3, progress=print):
    """Retry longer, independently initialized runs; never discard failures.

    Retries affect numerical effort, not priors or scientific selection. Use
    the final run, not a convergence-selected subset of chains or pooled runs.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    data = data.canonical()
    digest = sha256()
    for field in (data.xy, data.log_value, data.log_sd, data.censored, data.lower_censored, data.row_ids):
        digest.update(np.asarray(field).tobytes())
    signature = {"input_array_sha256": digest.hexdigest(), "latitude": float(latitude),
                 "prior": asdict(prior), "model_source_sha256": model_hashes(), "runtime": _RUNTIME}
    records = []
    for attempt in range(attempts):
        setting = settings if attempt == 0 else replace(settings,
            warmup=settings.warmup*2, draws=settings.draws*(2**attempt),
            target_accept=max(settings.target_accept, .98), seed=settings.seed+1009*attempt)
        key = {**signature, "sampling": asdict(setting)}
        path = directory/f"attempt-{attempt+1}"
        manifest = path.with_suffix(".json")
        if manifest.exists():
            record = json.loads(manifest.read_text())
            if record["signature"] != key:
                raise ValueError(f"Stale posterior cache: {path}. Choose a fresh output directory.")
            with np.load(path.with_suffix(".npz")) as archive:
                samples = {k: archive[k] for k in archive.files}
            if sha256(path.with_suffix(".npz").read_bytes()).hexdigest() != record["posterior_sha256"]:
                raise ValueError(f"Posterior checksum mismatch: {path}.")
        else:
            if progress:
                progress(f"{directory}: posterior attempt {attempt+1}")
            samples, diagnostics = sample(data, latitude, prior, setting, progress)
            np.savez_compressed(path.with_suffix(".npz"), **samples)
            record = {"signature": key, "diagnostics": diagnostics, "volume": volume_summary(samples),
                      "posterior_sha256": sha256(path.with_suffix(".npz").read_bytes()).hexdigest()}
            manifest.write_text(json.dumps(record, indent=2, allow_nan=False)+"\n")
        assessment = assess_posterior(record["diagnostics"], samples)
        records.append({"attempt": attempt+1, "file": manifest.name, **record,
                        "posterior_assessment": assessment})
        # Retain and use any later already-completed attempt even if an earlier
        # run now passes the separated mixing/efficiency assessment.
        later_exists = (directory/f"attempt-{attempt+2}.json").exists()
        if assessment["passed"] and not later_exists:
            break
    return samples, {"prior": prior.describe(), "volume": record["volume"],
        "diagnostics": assessment, "attempts": records,
        "posterior_file": f"attempt-{len(records)}.npz", "posterior_sha256": record["posterior_sha256"]}


def fit_checked(data, latitude, prior, directory, settings=Sampling(), attempts=3, progress=print):
    """Refine inadequate angular integration without changing physical priors.

    Every rejected resolution retains its complete independent posterior. A
    refinement reruns the posterior at twice the angular resolution; draws
    are never renormalized or reweighted after fitting. There is no change
    to the volume, shape, noise or tail priors, and no volume-based stopping.
    """
    from .source_checks import source_checks
    directory = Path(directory)
    history = []
    for refinement in range(4):
        chosen = replace(prior, angular_nodes=prior.angular_nodes*2**refinement)
        folder = directory if refinement == 0 else directory/f"quadrature-{chosen.angular_nodes}"
        samples, fit = fit_cached(data, latitude, chosen, folder, settings, attempts, progress)
        if not prior.source_anchored:
            return samples, fit
        check = source_checks(samples)
        if check["normalization"]["passed"] or refinement == 3:
            fit["posterior_file"] = str((folder/fit["posterior_file"]).relative_to(directory))
            fit["source_checks"] = check
            fit["normalization_refinements"] = history
            fit["diagnostics"]["sampling_only_passed"] = fit["diagnostics"]["passed"]
            fit["diagnostics"]["normalization_passed"] = check["normalization"]["passed"]
            fit["diagnostics"]["passed"] &= check["normalization"]["passed"]
            fit["diagnostics"]["assessment"] += " Source-fan normalization must also pass angular refinement at relative tolerance 1e-6."
            return samples, fit
        history.append({"angular_nodes":chosen.angular_nodes,"directory":str(folder.relative_to(directory)),
                        "fit":fit,"source_checks":check})
        if progress:
            progress(f"{directory}: angular normalization needs refinement to {2*chosen.angular_nodes} nodes")


def fit_study(path, output, settings=Sampling(), sensitivity=True, validation=True, progress=print,
              baseline_prior=BASELINE, sensitivity_priors=SENSITIVITIES):
    data, meta = audited_data(path)
    if validation:
        automatic_buffer(data)  # Fail insufficient-data holdouts before lengthy fits.
    output = Path(output)
    latitude = meta["vent_lonlat"][1]
    result = {"input": {k: v for k, v in meta.items() if k != "transform_inv"},
              "protocol": ("vent-bounded-training-basis-v1" if baseline_prior.supported_basis else
                           "vent-bounded-free-mixture-v1" if baseline_prior.vent_envelope else
                           "source-focused-elliptic-mixture-v2" if baseline_prior.source_anchored
                           else "mass-normalized-spherical-mixture-v1"), "fits": {}}
    baseline = None
    for prior in (baseline_prior, *sensitivity_priors) if sensitivity else (baseline_prior,):
        samples, fit = fit_checked(data, latitude, prior, output/"posteriors"/prior.name, settings, progress=progress)
        result["fits"][prior.name] = fit
        if prior.name == "baseline":
            baseline = samples
        (output/"study.json").write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
    if validation:
        result["validation"] = cross_validate(data, latitude, output/"posteriors"/"validation", settings, progress, baseline_prior)
    (output/"study.json").write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
    return data, meta, baseline, result


def cross_validate(data, latitude, output, settings=Sampling(), progress=print, baseline_prior=BASELINE):
    data = data.canonical()
    labels = spatial_folds(data.xy, 4)
    buffer, rule = automatic_buffer(data)
    arrays = {k: np.zeros(len(data.xy)) for k in ("median", "lower", "upper", "log_score")}
    folds = []
    for fold in range(4):
        train, test = training_mask(data, labels, fold, buffer)
        samples, fit = fit_checked(data.subset(train), latitude, baseline_prior, Path(output)/f"fold-{fold}",
                                  replace(settings, seed=settings.seed+fold+1), progress=progress)
        for key, value in predictions(samples, data.subset(test), latitude).items():
            arrays[key][test] = value
        folds.append({"fold": fold, "training_row_ids": data.row_ids[train].tolist(),
                      "withheld_row_ids": data.row_ids[test].tolist(), "fit": fit})
    metrics = interval_metrics(data, arrays["median"], arrays["lower"], arrays["upper"], arrays["log_score"])
    metrics["fold_log_scores"] = [-float(np.mean(arrays["log_score"][labels == k])) for k in range(4)]
    metrics["equal_region_log_score"] = float(np.mean(metrics["fold_log_scores"]))
    return {"row_ids": data.row_ids.tolist(), "folds": labels.tolist(), "buffer_km": buffer,
        "buffer_rule": rule, "metrics": metrics, "predictions": {k: v.tolist() for k, v in arrays.items()},
        "fits": folds, "all_fits_passed_diagnostics": all(f["fit"]["diagnostics"]["passed"] for f in folds),
        "interpretation": "Same four geographic outer folds and buffer as the regional peers. Entire posterior is refitted with fixed generic priors using training observations only. No response-based inner tuning is used. Heldout observation prediction does not validate total volume in unobserved regions."}
