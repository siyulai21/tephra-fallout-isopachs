"""Readable global estimates with explicit assumptions and numerical evidence."""
import gzip
from hashlib import sha256
from importlib.metadata import version
import json
from pathlib import Path
import platform
import shutil
import shlex

import numpy as np
import pandas as pd

from .core.catalog import dataset_record
from .core.geometry import survey_bounds
from .core.validation import interval_metrics
from .models.mass_mixture.study import fit_study
from .models.mass_mixture.inference import Sampling, flatten
from .models.mass_mixture.posterior import mass_partition, predictions
from .models.mass_mixture.geometry import log_field, to_sphere
from .support import SupportMesh


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


def figures(output, name, data, meta, samples, report):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from src.probabilistic_isopach.figures import STYLE, _coastlines
    flat = flatten(samples)
    index = np.unique(np.linspace(0, len(flat["log_volume"])-1, min(768, len(flat["log_volume"]))).astype(int))
    flat = {k: v[index] for k, v in flat.items()}
    bounds = survey_bounds(np.vstack([data.xy, [0., 0.]]))
    # Include the source exactly: a coarse grid can otherwise move the drawn
    # innermost contour even when the field itself peaks at the volcano.
    x = np.unique(np.r_[np.linspace(bounds[0], bounds[2], 121), 0.])
    y = np.unique(np.r_[np.linspace(bounds[1], bounds[3], 121), 0.])
    xx, yy = np.meshgrid(x, y)
    xy = np.column_stack((xx.ravel(), yy.ravel()))
    from src.probabilistic_isopach.global_volume import EarthDomain
    domain = EarthDomain.wgs84(meta["vent_lonlat"][1])
    physical = domain.rho(xy) < domain.rho_max
    unit = to_sphere(xy[physical], meta["vent_lonlat"][1])
    fields = np.full((3, len(xy)), np.nan)
    fields[:, physical] = np.concatenate([np.quantile(log_field(flat, unit[i:i+256]), [.025, .5, .975], axis=0)
                                         for i in range(0, len(unit), 256)], axis=1)
    mesh = SupportMesh(data)
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(1, 2, figsize=(13, 6.5), layout="constrained")
        median = fields[1].reshape(xx.shape)/np.log(10)
        lo, hi = np.quantile(data.log_value[~data.censored]/np.log(10), [.02, .98])+[-1., 1.]
        plot = axes[0].pcolormesh(x, y, median, shading="auto", cmap="viridis", vmin=lo, vmax=hi)
        fig.colorbar(plot, ax=axes[0], shrink=.7, label="log₁₀ median thickness (cm)")
        contours = np.arange(np.ceil(lo), np.floor(hi)+.1, .5)
        lines = axes[0].contour(x, y, median, levels=contours, colors="#233440", linewidths=.55)
        axes[0].clabel(lines, fmt=lambda v: f"{10**v:.2g}", fontsize=7)
        plot = axes[1].pcolormesh(x, y, ((fields[2]-fields[0])/np.log(10)).reshape(xx.shape),
                                  shading="auto", cmap="magma", vmin=0)
        fig.colorbar(plot, ax=axes[1], shrink=.7, label="log₁₀(upper / lower 95% latent thickness)")
        for ax in axes:
            _coastlines(ax, meta["vent_lonlat"])
            for polygon in getattr(mesh.geometry, "geoms", [mesh.geometry]):
                if polygon.is_empty:
                    continue
                ax.plot(*polygon.exterior.xy, color="#00BAC6", lw=1.3, ls="--")
                for ring in polygon.interiors:
                    ax.plot(*ring.xy, color="#00BAC6", lw=1.3, ls="--")
            for mask, marker in ((~data.censored, "o"), (data.lower_censored, "^"),
                                 (data.censored & ~data.lower_censored, "v")):
                if mask.any():
                    ax.scatter(*data.xy[mask].T, s=15, marker=marker, color="white", edgecolors="#233440", lw=.6)
            ax.scatter(0, 0, marker="*", s=100, c="#E47445", edgecolors="white")
            ax.set(xlim=(bounds[0], bounds[2]), ylim=(bounds[1], bounds[3]), aspect="equal",
                   xlabel="Easting from source reference (km)", ylabel="Northing (km)")
        axes[0].set_title("Joint field across all fallout fans\nContours shown in cm; cyan = sampling footprint")
        axes[1].set_title("Uncertainty in the latent deposit\nObservation scatter is additional")
        fig.suptitle(f"{name} · global mass model\nThis map window has no role in calculating the total volume.", fontsize=15)
        fig.savefig(output/"field.png", dpi=160)
        plt.close(fig)

        fig, axes = plt.subplots(1, 2, figsize=(13, 6), layout="constrained")
        for i, (key, fit) in enumerate(report["fits"].items()):
            v = fit["volume"]
            color = "#157F88" if key == "baseline" else "#AD633E"
            axes[0].plot([v["lower95"], v["upper95"]], [i, i], c=color, lw=2)
            axes[0].scatter(v["median"], i, c=color, marker="o" if fit["diagnostics"]["passed"] else "x", s=35)
        axes[0].set(yticks=range(len(report["fits"])), yticklabels=[assumption_label(k, report) for k in report["fits"]],
            xscale="log", xlabel="Global bulk fallout (km³; log scale)", title="Assumption sensitivity\nEach line is its own conditional 95% interval")
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

        fig, axes = plt.subplots(1, 2, figsize=(12, 4.7), layout="constrained")
        values = np.exp(samples["log_volume"])
        for chain, value in enumerate(values):
            index = np.linspace(0, len(value)-1, min(2000, len(value))).astype(int)
            axes[0].plot(index, value[index], lw=.55, alpha=.65, label=f"Chain {chain+1}")
            axes[1].plot(np.sort(value), (np.arange(len(value))+.5)/len(value), lw=1.2)
        axes[0].set(yscale="log", xlabel="Retained iteration", ylabel="Global bulk fallout (km³)",
                    title="Independent chains · log volume scale")
        axes[1].set(xscale="log", xlabel="Global bulk fallout (km³)", ylabel="Cumulative posterior fraction",
                    title="Per-chain volume distributions")
        for ax in axes:
            ax.grid(alpha=.2)
        axes[0].legend(fontsize=8)
        fig.suptitle(f"{name} · baseline sampling diagnostics", fontsize=15)
        fig.savefig(output/"sampling.png", dpi=160)
        plt.close(fig)


def write_readme(output, name, report):
    fit = report["fits"]["baseline"]
    v, d = fit["volume"], fit["diagnostics"]
    input_argument = shlex.quote(report["input"]["source"])
    rows = []
    for key, item in report["fits"].items():
        q = item["volume"]
        rows.append(f"| {assumption_label(key, report)} | {q['median']:.3g} | {q['lower95']:.3g}–{q['upper95']:.3g} | {item['training_checks']['rmse_log']:.3f} | {'pass' if item['diagnostics']['passed'] else '**not passed**'} |")
    validation = report.get("validation")
    validation_text = "Geographic validation was not requested for this run."
    if validation:
        m = validation["metrics"]
        validation_text = (f"Four buffered geographic holdouts: equal-region negative log score **{m['equal_region_log_score']:.3f}** "
            f"(lower is better), log-thickness RMSE **{m['rmse_log']:.3f}**, and heldout observation coverage **{m['coverage95_observations']:.1%}**. "
            f"All four posterior fits passed numerical diagnostics: **{validation['all_fits_passed_diagnostics']}**. "
            "These predictions test transfer to withheld observations, not the unobserved whole-deposit volume.")
    text = f"""# {name} · global fallout estimate

[Eruption home](../../README.md) · [Regional GP/spline comparison](../README.md) · [Method and assumptions](../../../../docs/methods/probabilistic-global.md) · [All six eruptions](../../../../comparisons/global/README.md)

The baseline posterior median is **{v['median']:.3g} km³**, with a **{v['lower95']:.3g}–{v['upper95']:.3g} km³ conditional 95% credible interval**. This is bulk fallout across all jointly fitted fans, including co-ignimbrite ash where present in the accepted data. It excludes ignimbrite deposits and is not dense-rock equivalent. Numerical diagnostics: **{'passed' if d['passed'] else 'not passed — provisional numerical summary'}**.

Every posterior field is positive and has total volume exactly equal to its fitted volume parameter on the whole Earth. No closing pseudo-observations, map rectangle, contour cutoff or arbitrary distal radius sets this total. The previous contour screen remains a diagnostic of local support; its rejection does not veto this conditional global estimate.

![Global fallout field and uncertainty](field.png)

## What the interval means

Uncertainty includes fan positions, sizes, relative masses, tail shape, total volume and observation scatter parameters. The latent volume excludes observation noise. It remains conditional on general modelling assumptions: a finite mixture of smooth, source-related, radially decreasing components, a proper volume/scale prior and a stated observation model. Several components can form elongated, separated or overlapping fans; they are mathematical components, not independently measured eruption phases.

No finite dataset alone can exclude an arbitrarily large deposit in an unsampled region. The proper prior makes inference possible; it does not create geological evidence. Unrecorded erosion, reworking, preferential sampling and source misattribution are not automatically covered by this interval. If residual scatter is unresolved areal deposition variability rather than observation/preservation error, central-field volume can differ from area-mean physical mass; that structural correction is not automatically included. Bounds retain their documented likelihood meaning.

## Sensitivity to those assumptions

| Assumption | Median km³ | Conditional 95% interval km³ | Fitted-site log-RMSE | Numerical checks |
|---|---:|---:|---:|---|
{chr(10).join(rows)}

The range across the lower and upper interval endpoints is **{report['sensitivity_envelope_km3'][0]:.3g}–{report['sensitivity_envelope_km3'][1]:.3g} km³**. This is an assumption-sensitivity envelope, **not** a 95% interval, a probability-weighted model average, or an exhaustive bound. Every alternative remains visible, including those that differ substantially; none was selected because its volume matched a published target.

Fitted-site log-RMSE compares the posterior predictive median with the exact measurements used for fitting; bounds contribute their likelihood events instead. These training checks can expose a restrictive alternative, but are optimistic and **not validation or a model-selection rule**. Only the baseline receives the full matched geographic holdout analysis below. Alternatives have not been given probability weights or declared equally plausible.

![Volume sensitivity and mass partition](volume.png)

## Validation and numerical precision

{validation_text}

Baseline chains: maximum rank-normalized split R-hat **{d['max_label_invariant_rhat']:.4f}**, minimum bulk/tail effective sample sizes **{d['min_label_invariant_bulk_ess']:.0f}/{d['min_label_invariant_tail_ess']:.0f}**, minimum energy BFMI **{d['min_energy_bfmi']:.3f}**, divergences **{d['divergences']}**. Diagnostics cover label-invariant global, observed-site and unsampled-field summaries. Passing them is evidence of numerical mixing, not proof of complete mode exploration. All shorter failed attempts are retained in the report. Monte Carlo standard errors of volume log-quantiles are recorded separately; rounded interval endpoints are estimates. [Inspect the chain traces and distributions](sampling.png).

Mass partition uses {report['mass_partition']['n_posterior_fields']} whole posterior fields and {report['mass_partition']['quadrature_points_per_fan']} equal-mass quadrature points per fan. The largest fraction change on doubling that integration rule is **{report['mass_partition']['max_fraction_change_on_doubling']:.4f}**. This numerical partition never changes the analytically normalized total. Its local footprint is a sampling screen, not a confidence region.

## Reproduce and inspect

Use `tephra-isopachs global {input_argument} --output results/new-global-run`. Use `--resume` only for the same run; caches verify inputs, priors, numerical settings, source hashes, runtime versions and posterior checksums. The ordinary `compare` command runs the regional comparison followed by this global analysis.

- [Full report, priors, diagnostics, all attempts and provenance](report.json)
- [Baseline whole-field posterior samples](posterior.npz)
- [Volume samples for every assumption and chain](volume_draws.csv.gz)
- [Assumption summary](estimates.csv)
- [Geographic predictions and fold membership](heldout_predictions.csv)

Input SHA-256: `{report['input']['input_sha256']}`. No accepted observations or source audits were changed for this calculation.
"""
    if not validation:
        text = text.replace("- [Geographic predictions and fold membership](heldout_predictions.csv)\n", "")
    geometry = report.get("geometry", "displaced")
    if geometry != "displaced":
        text = text.replace("exactly equal to", "equal (to the reported normalization tolerance) to")
        text = text.replace("Uncertainty includes fan positions, sizes, relative masses, tail shape, total volume and observation scatter parameters.",
            "Uncertainty includes the fitted field geometry, relative strengths, total volume and observation scatter parameters. The source-focused alternative also varies its tail shape; the bounded-envelope alternative has Gaussian tails.")
        text = text.replace("Several components can form elongated, separated or overlapping fans;", "Several components form overlapping fallout fans;")
        text = text.replace("analytically normalized total", "separately normalized total")
        text = text.replace("equal-mass quadrature points per fan", "positive quadrature points per component")
        text = text.replace("--output results/new-global-run", f"--geometry {geometry} --output results/new-global-run")
        check = report["source_checks"]
        normalization = check["normalization"]
        numerical = (f"The maximum relative change in any posterior component's integral on quadrupling angular resolution was **{normalization['max_relative_component_mass_error']:.3g}** (tolerance 10⁻⁶)."
            if "max_relative_component_mass_error" in normalization else "The envelope and its Gaussian-product overlap terms have analytical whole-Earth integrals.")
        source_text = ("## Source constraint\n\nEvery fitted field has its global thickness maximum at the supplied volcano. "
            "This is an explicit shape assumption, not an invented vent-thickness observation. "
            + ("Thickness decreases along every source ray, so this family cannot represent a secondary radial maximum. " if geometry == "source-fans" else "Smaller secondary maxima are allowed; Gaussian bases are placed at training sites using coordinates only. ")
            + "Real deposits can violate a point-source maximum, particularly with distributed sources or secondary deposition. "
            + numerical+" [Derivation, rejected alternatives and geological limits](../../../../docs/methods/source-anchored-fans.md).\n\n"
            + "![Source-region thickness contours](source.png)\n\n")
        text = text.replace("## What the interval means", source_text+"## What the interval means")
    (output/"README.md").write_text(text)


def global_dataset(path, output, name=None, settings=Sampling(), resume=False, progress=print, geometry=None):
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
        table.append({"assumption": key, **{k: fit["volume"][k] for k in ("median", "lower95", "upper95")},
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
    (output/"report.json").write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
    figures(output, name, data, meta, samples, report)
    from .source_reporting import source_figure
    source_figure(output, name, data, meta, samples)
    write_readme(output, name, report)
    artifacts = ("README.md", "report.json", "posterior.npz", "volume_draws.csv.gz", "estimates.csv",
                 "heldout_predictions.csv", "field.png", "volume.png", "sampling.png", "source.png")
    (output/"artifacts.json").write_text(json.dumps({name: sha256((output/name).read_bytes()).hexdigest()
        for name in artifacts}, indent=2)+"\n")
    return report
