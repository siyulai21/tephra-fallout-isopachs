"""Readable global estimates with explicit assumptions and numerical evidence."""
import gzip
from hashlib import sha256
from importlib.metadata import version
import json
from pathlib import Path
import platform
import shutil

import numpy as np
import pandas as pd

from .core.catalog import dataset_record
from .core.validation import interval_metrics
from .models.mass_mixture.study import fit_study
from .models.mass_mixture.inference import Sampling
from .models.mass_mixture.posterior import mass_partition, predictions


LABELS = {"baseline": "Three fans · baseline", "diffuse_tail": "Allow diffuse global tails",
    "wide_geography": "Broader scales and fan offsets", "volume_prior_shift": "Shift volume prior 100-fold",
    "one_fan": "One fan", "five_fans": "Five fans", "gaussian_errors": "Gaussian observation errors",
    "diffuse_mixture_weights": "Diffuse fan and kernel weights",
    "six_bases": "Six measured-location bases", "three_bases": "Three measured-location bases",
    "source_heavy_tails": "Source-focused fans with heavy tails"}
PARTS = {"local_footprint": "Local sampling footprint", "unsampled_proximal": "Unsampled proximal cap",
    "beyond_farthest_site": "Beyond farthest exact site", "other_gaps_and_sectors": "Other gaps and sectors"}


def assumption_label(key, report):
    if key == "baseline":
        return {"source-fans": "Source-focused fans · baseline",
                "source-envelope": "Vent-bounded bases · baseline"}.get(report.get("geometry"), LABELS[key])
    if key == "wide_geography" and report.get("geometry") != "displaced":
        return "Broader scale and shape priors"
    return LABELS[key]



def prior_intervals(output, name, report):
    import matplotlib.pyplot as plt
    from .visualization.style import STYLE
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(1, 2, figsize=(13, 6), layout="constrained")
        for i, (key, fit) in enumerate(report["fits"].items()):
            v = fit["volume"]
            color = "#157F88" if key == "baseline" else "#AD633E"
            axes[0].plot([v["lower95"], v["upper95"]], [i, i], c=color, lw=2, ls="-" if fit["diagnostics"]["passed"] else "--")
        axes[0].set(yticks=range(len(report["fits"])), yticklabels=[assumption_label(k, report) for k in report["fits"]],
            xscale="log", xlabel="Global bulk fallout (km³; log scale)", title="Assumption sensitivity\nConditional 95% intervals · dashed = numerical warning")
        axes[0].invert_yaxis()
        for i, (key, entry) in enumerate(report["mass_partition"]["parts"].items()):
            v = entry["fraction"]
            axes[1].plot([v["lower95"], v["upper95"]], [i, i], c="#157F88", lw=2)
            axes[1].scatter(v["median"], i, c="#157F88", s=35)
        axes[1].set(yticks=range(4), yticklabels=list(PARTS.values()), xlim=(0, 1),
            xlabel="Fraction of total fallout", title="Where the inferred mass lies\nGeometry alone does not validate a deposit")
        axes[1].invert_yaxis()
        for ax in axes:
            ax.grid(axis="x", alpha=.2)
        fig.suptitle(f"{name} · conditional estimates and unobserved mass", fontsize=15)
        fig.savefig(output/"volume.png", dpi=160)
        plt.close(fig)



def figures(output, name, data, meta, samples, report):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from src.empirical_isopach.visualization.style import STYLE
    from .visualization.global_maps import global_maps
    global_maps(output, name, data, meta, samples, report)
    prior_intervals(output, name, report)
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.7), layout="constrained")
        from .models.mass_mixture.bearings.field import PROTOCOL
        bearing = report.get("protocol")==PROTOCOL
        if bearing:
            with np.load(output/"bearing/baseline.npz") as packet:
                masses = packet["volume_km3"].reshape(len(samples["log_volume"]),-1,3)
            values = masses[:,:,1]
        else:
            values = np.exp(samples["log_volume"])
        for chain, value in enumerate(values):
            index = np.linspace(0, len(value)-1, min(2000, len(value))).astype(int)
            axes[0].plot(index, value[index], lw=.55, alpha=.65, label=f"Chain {chain+1}")
            distribution = masses[chain].ravel() if bearing else value
            axes[1].plot(np.sort(distribution), (np.arange(len(distribution))+.5)/len(distribution), lw=1.2)
        axes[0].set(yscale="log", xlabel="Retained iteration", ylabel="Global bulk fallout (km³)",
                    title="Broad-fan volume within each chain" if bearing else "Independent chains · log volume scale")
        axes[1].set(xscale="log", xlabel="Global bulk fallout (km³)", ylabel="Cumulative posterior fraction",
                    title="Per-chain volume distributions")
        for ax in axes:
            ax.grid(alpha=.2)
        axes[0].legend(fontsize=8)
        fig.suptitle(f"{name} · whole-deposit volume by chain", fontsize=15)
        fig.savefig(output/"sampling.png", dpi=160)
        plt.close(fig)


def global_dataset(path, output, name=None, settings=Sampling(), resume=False, progress=print, geometry=None, archives=None):
    output = Path(output)
    if output.exists() and any(output.iterdir()) and not resume:
        raise ValueError("Use a fresh output directory, or --resume for verified posterior caches.")
    record = dataset_record(path)
    name = name or (record["name"] if record else Path(path).stem)
    from .models.mass_mixture.source_protocol import protocol
    from .models.mass_mixture.source_checks import source_checks
    geometry = geometry or (record or {}).get("global_geometry", "displaced")
    prior, alternatives = protocol(geometry)
    data, meta, samples, report = fit_study(path, output, settings, progress=progress,
        baseline_prior=prior, sensitivity_priors=alternatives)
    report["geometry"] = geometry
    report["source_checks"] = source_checks(samples)
    report["name"] = name
    report["mass_partition"] = mass_partition(samples, data, meta["vent_lonlat"][1])
    fits = report["fits"]
    report["sensitivity_envelope_km3"] = [min(f["volume"]["lower95"] for f in fits.values()),
                                        max(f["volume"]["upper95"] for f in fits.values())]
    report["all_assumption_fits_passed_diagnostics"] = all(f["diagnostics"]["passed"] for f in fits.values())
    report["reproducibility"] = {"python": platform.python_version(),
        "packages": {k: version(k) for k in ("numpy", "scipy", "pandas", "matplotlib", "pyproj", "shapely", "numpyro", "jax", "jaxlib", "arviz")},
        "report_source_sha256": sha256(Path(__file__).read_bytes()).hexdigest(),
        "study_source_sha256": sha256((Path(__file__).parent/"models/mass_mixture/study.py").read_bytes()).hexdigest(),
        "normalization": "Whole WGS84 Earth; total mass is the volume parameter. Source fans use analytical radial and checked periodic angular integrals; displaced/envelope models use analytical integrals. No map bounds or isopach cutoff enter inference."}
    root = Path(__file__).resolve().parents[2]
    report["reproducibility"]["source_sha256"] = {str(p.relative_to(root)): sha256(p.read_bytes()).hexdigest()
        for p in sorted((root/"src").rglob("*.py"))}
    base_path = output/"posteriors"/"baseline"/fits["baseline"]["posterior_file"]
    shutil.copyfile(base_path, output/"posterior.npz")
    frames, table = [], []
    for key, fit in fits.items():
        with np.load(output/"posteriors"/key/fit["posterior_file"]) as archive:
            posterior = {k: archive[k] for k in archive.files}
        fit["source_checks"] = source_checks(posterior)
        values = np.exp(posterior["log_volume"])
        checked_data = data.canonical()
        predictive = predictions(posterior, checked_data, meta["vent_lonlat"][1])
        fit["training_checks"] = {**interval_metrics(checked_data, predictive["median"],
            predictive["lower"], predictive["upper"], predictive["log_score"]),
            "interpretation": "In-sample posterior predictive checks. Optimistic; not heldout validation, model weights, or a selection criterion."}
        chain, draw = np.indices(values.shape)
        frames.append(pd.DataFrame({"assumption": key, "chain": chain.ravel(), "draw": draw.ravel(), "volume_km3": values.ravel()}))
        table.append({"assumption": key, **{k: fit["volume"][k] for k in ("lower95", "upper95")},
                      "training_rmse_log": fit["training_checks"]["rmse_log"],
                      "diagnostics_passed": fit["diagnostics"]["passed"]})
    csv = pd.concat(frames).to_csv(index=False).encode()
    (output/"volume_draws.csv.gz").write_bytes(gzip.compress(csv, mtime=0))
    pd.DataFrame(table).to_csv(output/"estimates.csv", index=False)
    if "validation" in report:
        validation = report["validation"]
        pd.DataFrame({"row_id": data.row_ids, "fold": validation["folds"],
            "observed_log_value": data.log_value, "censored": data.censored, "lower_censored": data.lower_censored,
            **validation["predictions"]}).to_csv(output/"heldout_predictions.csv", index=False)
    report["published_artifact_sha256"] = {file: sha256((output/file).read_bytes()).hexdigest()
        for file in ("posterior.npz", "volume_draws.csv.gz", "estimates.csv", "heldout_predictions.csv")}
    from .models.mass_mixture.bearings.report import assemble
    parent = output/"bearing"/"parent-report.json"
    parent.parent.mkdir(parents=True,exist_ok=True)
    parent.write_text(json.dumps(report,indent=2,allow_nan=False)+"\n")
    principal = assemble(output,data,meta,report,output/"posteriors",output/"posteriors"/"bearing-integration")
    from .models.mass_mixture.archive_check import case_archives
    case_archives(path,output,data,meta,archives)
    from .visualization.deposit import render_deposit
    render_deposit(output,"spherical-mixture",data,meta,principal["name"])
    return principal
