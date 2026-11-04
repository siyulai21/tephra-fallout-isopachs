"""GitHub-readable regional comparison reports with equal method prominence."""
import json
from hashlib import sha256
from importlib.metadata import version
from pathlib import Path
import platform
import shlex

import numpy as np
import pandas as pd

from src.probabilistic_isopach.catalog import dataset_record, data_directory
from src.probabilistic_isopach.data import read_observations
from src.probabilistic_isopach.distributions import predictive_distribution
from .region import SurveyRegion, integrate_region, regional_draws
from .spline import survey_bounds, TrendRegularizedSpline
from .validation import LABELS, METHODS, nested_comparison, fit_method


def provenance():
    root = Path(__file__).resolve().parents[2]
    return {"protocol": "regional-spline-v1", "python": platform.python_version(),
            "packages": {k: version(k) for k in ("numpy", "scipy", "pandas", "matplotlib", "pyproj")},
            "source_sha256": {str(p.relative_to(root)): sha256(p.read_bytes()).hexdigest()
                              for p in sorted((root/"src").rglob("*.py"))}}


def save_selection(output, relative, selection):
    """Keep every searched candidate, but avoid repeating it in the main JSON."""
    records = selection.pop("candidates")
    rows = []
    for index, record in enumerate(records):
        row = {"index": index, **record.pop("candidate", {})}
        scores = record.pop("fold_scores", [])
        row.update({f"fold_{i}_negative_log_score": s for i, s in enumerate(scores)})
        record["failures"] = json.dumps(record.get("failures", []), separators=(",", ":"))
        row.update(record)
        rows.append(row)
    target = output/relative
    target.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(target, index=False)
    selection["candidate_table"] = relative
    selection["n_evaluated"] = len(rows)
    selection["candidate_table_sha256"] = sha256(target.read_bytes()).hexdigest()


def save_figures(output, name, data, metadata, region, models, report):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from src.probabilistic_isopach.figures import STYLE, CMAP, INK, MUTED, TEAL, AMBER, _coastlines
    bounds = np.array(report["fit_bounds_km"])
    x, y = np.linspace(bounds[0], bounds[2], 181), np.linspace(bounds[1], bounds[3], 181)
    xx, yy = np.meshgrid(x, y)
    xy = np.column_stack((xx.ravel(), yy.ravel()))
    inside = region.contains(xy)
    fields, grid = {}, {"x_km": xy[:, 0], "y_km": xy[:, 1], "inside_observation_hull": inside}
    for method, model in models.items():
        distribution = predictive_distribution(model, xy)
        median = distribution.quantile(.5)
        fields[method] = np.ma.array(median.reshape(xx.shape), mask=(~inside).reshape(xx.shape))
        grid[method+"_median_cm"] = np.exp(median)
        grid[method+"_conditional_mean_cm"] = np.exp(distribution.log_expected_thickness())
        grid[method+"_conditional_lower95_cm"] = np.exp(distribution.quantile(.025))
        grid[method+"_conditional_upper95_cm"] = np.exp(distribution.quantile(.975))
    pd.DataFrame(grid).to_csv(output/"predictions.csv", index=False)
    values = np.concatenate([v.compressed() for v in fields.values()])
    low, high = float(values.min()), float(values.max())
    if high <= low:
        high = low+1
    levels = np.linspace(low, high, 24)
    contour_values = np.array([m*10.**k for k in range(-6, 9) for m in (1, 3)])
    contours = np.log(contour_values)
    contours = contours[(contours > low) & (contours < high)]
    boundary = np.vstack((region.vertices, region.vertices[0]))
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(1, 2, figsize=(13, 6.6), layout="constrained", sharex=True, sharey=True)
        for ax, method in zip(axes, METHODS):
            fill = ax.contourf(x, y, fields[method], levels=levels, cmap=CMAP)
            _coastlines(ax, metadata["vent_lonlat"])
            if len(contours):
                lines = ax.contour(x, y, fields[method], levels=contours, colors=INK, linewidths=.65, linestyles="solid")
                ax.clabel(lines, fmt=lambda v: f"{np.exp(v):g}", fontsize=7)
            ax.plot(*boundary.T, color=TEAL, ls="--", lw=1.)
            for mask, marker, color in ((~data.censored, "o", INK),
                    (data.censored & ~data.lower_censored, "v", AMBER), (data.lower_censored, "^", AMBER)):
                if mask.any():
                    ax.scatter(*data.xy[mask].T, s=18, marker=marker, edgecolors=color, facecolors="white", linewidths=.7, zorder=5)
            ax.scatter([0], [0], marker="*", color="#B9573C", edgecolors="white", s=90, zorder=6)
            volume = report["methods"][method]["regional_volume"]
            ax.set(title=f"{LABELS[method]}\nExpected regional volume: {volume['expected_km3']:.1f} km³",
                   xlim=(bounds[0], bounds[2]), ylim=(bounds[1], bounds[3]), aspect="equal",
                   xlabel="Easting from source reference (km)")
            ax.grid(alpha=.15)
        axes[0].set_ylabel("Northing from source reference (km)")
        handles = [Line2D([], [], marker="o", color=INK, markerfacecolor="white", ls="", label="Measurement"),
                   Line2D([], [], marker="*", color="#B9573C", ls="", label="Source reference")]
        if data.lower_censored.any():
            handles.append(Line2D([], [], marker="^", color=AMBER, markerfacecolor="white", ls="", label="Lower bound"))
        if (data.censored & ~data.lower_censored).any():
            handles.append(Line2D([], [], marker="v", color=AMBER, markerfacecolor="white", ls="", label="Upper limit"))
        axes[0].legend(handles=handles, loc="upper left", fontsize=8, framealpha=.95)
        cbar = fig.colorbar(fill, ax=axes, shrink=.72, pad=.02)
        ticks = contours[::max(1, len(contours)//8)] if len(contours) else [low, high]
        cbar.set_ticks(ticks, labels=[f"{np.exp(t):g}" for t in ticks])
        cbar.set_label("Median thickness (cm); shared scale")
        fig.suptitle(f"{name} · same data, same regional footprint\n"
                     "Isopachs stop at the observation hull; gaps inside remain model-dependent.", fontsize=15)
        for suffix in ("png", "svg"):
            fig.savefig(output/f"isopachs.{suffix}", dpi=170)
        plt.close(fig)

        fig, axes = plt.subplots(1, 3, figsize=(13, 4.6), layout="constrained")
        det = ~data.censored
        limits = [float(data.log_value[det].min())-.3, float(data.log_value[det].max())+.3]
        colors = (TEAL, AMBER)
        for ax, method, color in zip(axes[:2], METHODS, colors):
            validation = report["validation"]["methods"][method]
            pred, metrics = validation["predictions"], validation["metrics"]
            median = np.array(pred["median"])[det]
            lower, upper = np.array(pred["lower"])[det], np.array(pred["upper"])[det]
            ax.vlines(data.log_value[det], lower, upper, color=color, alpha=.3, lw=.6)
            ax.scatter(data.log_value[det], median, s=15, color=color)
            ax.plot(limits, limits, "--", color=MUTED, lw=1)
            ax.set(xlabel="Observed log thickness", ylabel="Withheld prediction (log thickness)",
                   title=f"{LABELS[method]}\nRMSE {metrics['rmse_log']:.2f} · coverage {metrics['coverage95_observations']:.0%}")
            ax.grid(alpha=.15)
        prediction_limits = [min(a.get_ylim()[0] for a in axes[:2]), max(a.get_ylim()[1] for a in axes[:2])]
        for ax in axes[:2]:
            ax.set(xlim=limits, ylim=prediction_limits)
        for index, (method, color) in enumerate(zip(METHODS, colors)):
            result = report["methods"][method]
            q = result["joint_volume"]["conditional_quantiles_km3"]
            axes[2].errorbar(q["0.5"], index, xerr=[[q["0.5"]-q["0.025"]], [q["0.975"]-q["0.5"]]],
                            fmt="o", color=color, capsize=4)
            axes[2].scatter(result["regional_volume"]["expected_km3"], index, marker="D", color=color, s=30)
            deletion = [v["regional_deletion_volume"]["expected_km3"]
                        for v in report["validation"]["methods"][method]["selections"]]
            axes[2].plot([min(deletion), max(deletion)], [index+.12, index+.12], ":", color=color, lw=2)
        axes[2].set(yticks=[0, 1], yticklabels=["GP ensemble", "Cubic spline"], ylim=(-.6, 1.6),
                    xlabel="Bulk fallout within observation hull (km³)",
                    title="Regional volume\nDotted: geographic-deletion range")
        axes[2].grid(axis="x", alpha=.15)
        fig.suptitle(f"{name} · geographic holdouts and conditional 95% intervals\n"
                     "Intervals omit parameter/selection uncertainty; bounds contribute scores, not RMSE.", fontsize=13)
        for suffix in ("png", "svg"):
            fig.savefig(output/f"diagnostics.{suffix}", dpi=170)
        plt.close(fig)


def write_readme(output, name, report):
    region, meta = report["region"], report["input"]
    lines = [f"# {name}: two empirical methods, one regional footprint", "",
             "Both methods use the same accepted observations, geographic holdouts and integration area. Neither is designated the preferred method.", "",
             f"**{meta['n_measurements']} exact measurements, {meta['n_lower_limits']} lower bounds, {meta['n_upper_limits']} upper limits.** "
             f"The observation hull covers {region['area_km2']:,.0f} km². The automatic validation buffer is {report['validation']['buffer_km']:.2f} km.", "",
             "![Matched median isopachs](isopachs.png)", "",
             "| Quantity | GP ensemble | Trend-regularized cubic spline |", "|---|---:|---:|"]
    def pair(label, values, fmt=".2f"):
        lines.append("| "+label+" | "+" | ".join(format(x, fmt) for x in values)+" |")
    rows = [report["methods"][m] for m in METHODS]
    metrics = [report["validation"]["methods"][m]["metrics"] for m in METHODS]
    pair("Expected regional bulk fallout (km³)", [r["regional_volume"]["expected_km3"] for r in rows])
    pair("Integral of median thickness surface (km³)", [r["regional_volume"]["median_surface_km3"] for r in rows])
    interval = [r["joint_volume"]["conditional_quantiles_km3"] for r in rows]
    lines.append("| Conditional 95% volume interval (km³) | "+" | ".join(f"{q['0.025']:.2f}–{q['0.975']:.2f}" for q in interval)+" |")
    pair("Outer equal-region negative log score ↓", [r["equal_region_log_score"] for r in metrics], ".3f")
    pair("Outer log-thickness RMSE ↓", [r["rmse_log"] for r in metrics], ".3f")
    pair("Outer 95% observation-interval coverage", [r["coverage95_observations"] for r in metrics], ".1%")
    lines += ["", "Lower predictive score and RMSE are better. Coverage must be read with interval width and predictive score; wider intervals alone are not improvement.", "",
              "![Withheld predictions and regional volumes](diagnostics.png)", "", "## Scope", "",
              "The footprint is the convex hull of measured/bounded locations, with no source point or invented observations added. It can contain large unsampled gaps and can exclude proximal deposits. These are **regional bulk fallout volumes, not total eruption volumes or dense-rock equivalents**. No global extrapolation enters this comparison.", "",
              "The mean volume integrates expected thickness. The median-surface integral is a separate quantity; it is not the median volume. Volume intervals come from complete correlated surface draws, not sums of pointwise interval endpoints.", "",
              "## Automatic spline fit", "", "```json", json.dumps(report["methods"]["spline"]["model"]["candidate"], indent=2), "```", "",
              "Cubic basis size, tension, derivative penalty, trend strength, trend family and observation noise are selected by geographic predictive scores. Every explored candidate is retained in the selection tables. Numerical quadrature points approximate an area-normalized penalty and never become observations.", "",
              "## Reliability checks", ""]
    lines += ["- "+warning for warning in report["notes"]]
    command_input = name if meta["source"].startswith("data/") else meta["source"]
    lines += ["", "## Reproduce and inspect", "", "```sh",
              f"tephra-isopachs compare {shlex.quote(command_input)} --output results/new-regional-run", "```", "",
              "[Complete numerical report](report.json) · [Withheld predictions](heldout_predictions.csv) · [Accepted observation semantics](accepted_observations.csv) · [Selection tables](selection/README.md)", "",
              "The JSON records input/source hashes, package versions, all fold memberships, regional integration refinements, conditional-volume sampling precision and geographic-deletion sensitivity. Historical GP global calculations remain separate."]
    (output/"README.md").write_text("\n".join(lines)+"\n")


def compare_dataset(path, output, name=None, draws=2000, progress=print):
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError("Output directory is not empty; use a fresh destination.")
    data, meta = read_observations(path)
    record = dataset_record(path)
    name = name or (record["name"] if record else Path(path).stem)
    if record and record.get("audit"):
        audit_path = data_directory()/record["audit"]
        audit = json.loads(audit_path.read_text())
        case = audit["cases"][record["audit_case"]]
        if meta["input_sha256"] != case["sha256"]:
            raise ValueError("Accepted input differs from its recorded audit.")
        meta["audit"] = {"file": record["audit"], "sha256": sha256(audit_path.read_bytes()).hexdigest(),
                         "case": record["audit_case"], "definition": case["definition"]}
    meta.pop("transform_inv")
    meta["source"] = "data/"+record["file"] if record else Path(path).name
    region, bounds = SurveyRegion(data.xy), survey_bounds(data.xy)
    validation = nested_comparison(data, region, bounds, progress=lambda s: progress(f"{name}: {s}"))
    models, methods = {}, {}
    notes = ["All probability intervals condition on fitted parameters and selected settings; they omit parameter/selection and preservation/sampling uncertainty.",
             "Four geographic outer folds evaluate the entire tuning procedure; this is development evidence, not independent external validation.",
             "Geographic-deletion volume ranges show refitting sensitivity and are not confidence intervals."]
    for method in METHODS:
        model, selection = fit_method(method, data, bounds, 4, validation["buffer_km"])
        volume = integrate_region(model, region)
        joint = regional_draws(model, region, volume["expected_km3"], draws=draws)
        models[method] = model
        methods[method] = {"model": model.describe(), "selection": selection,
                           "regional_volume": volume, "joint_volume": joint}
        if not volume["converged"]:
            notes.append(f"{LABELS[method]} regional integration did not reach the stated refinement tolerance.")
        if not joint["expected_grid_converged"]:
            notes.append(f"{LABELS[method]} joint-draw quadrature needs further refinement; inspect its grid checks before using intervals.")
        coverage = validation["methods"][method]["metrics"]["coverage95_observations"]
        if coverage < .9:
            notes.append(f"{LABELS[method]}: held-out 95% observation coverage is only {coverage:.1%}; uncertainty is undercovered.")
        progress(f"{name}: completed {LABELS[method]} regional fit and joint draws")
    spline = models["spline"]
    refined = TrendRegularizedSpline(spline.candidate, quadrature_order=12).fit(data, bounds=bounds)
    qxy, _ = region.quadrature(16)
    prediction_change = float(np.max(abs(spline.predict_log(qxy)[0]-refined.predict_log(qxy)[0])))
    fine_volume = integrate_region(refined, region)
    relative_volume_change = abs(fine_volume["expected_km3"]/methods["spline"]["regional_volume"]["expected_km3"]-1)
    methods["spline"]["penalty_quadrature_check"] = {"orders": [6, 12], "max_log_prediction_change": prediction_change,
        "relative_expected_volume_change": relative_volume_change,
        "passed": prediction_change < .02 and relative_volume_change < .005}
    if not methods["spline"]["penalty_quadrature_check"]["passed"]:
        notes.append("Spline trend-penalty quadrature sensitivity exceeds the declared tolerance; inspect its numerical check.")
    edges = methods["spline"]["selection"]["selected_at_explored_boundary"]
    if edges:
        notes.append("Spline selection reaches an explored search boundary: "+", ".join(edges)+". This is a bounded-grid result, not proof of a continuous optimum.")
    if spline.candidate.trend_strength >= 10000:
        notes.append("The selected spline strongly follows its estimated trend. Its conditional volume interval can be extremely narrow because trend-estimation uncertainty is omitted; inspect the geographic-deletion range rather than interpreting narrowness as established precision.")
    report = {"dataset": name, "input": meta, "region": region.describe(), "fit_bounds_km": bounds.tolist(),
              "methods": methods, "validation": validation, "notes": notes, "reproducibility": provenance()}
    output.mkdir(parents=True, exist_ok=True)
    selection_links = []
    for method in METHODS:
        relative = f"selection/{method}-all-data.csv"
        save_selection(output, relative, report["methods"][method]["selection"])
        selection_links.append(f"- [{LABELS[method]}: all-data tuning]({Path(relative).name})")
        for split in report["validation"]["methods"][method]["selections"]:
            relative = f"selection/{method}-outer-{split['fold']}.csv"
            save_selection(output, relative, split["selection"])
            selection_links.append(f"- [{LABELS[method]}: outer fold {split['fold']+1}]({Path(relative).name})")
    (output/"selection/README.md").write_text("# Geographic tuning tables\n\nEach row is one candidate; lower equal-region selection score is better. These inner scores tune models and do not estimate performance. See the outer scores in the [report](../README.md).\n\n"+"\n".join(selection_links)+"\n")
    observations = {"row_id": data.row_ids, "x_km": data.xy[:, 0], "y_km": data.xy[:, 1],
                    "log_value": data.log_value, "censored": data.censored,
                    "lower_bound": data.lower_censored, "known_log_sd": data.log_sd}
    pd.DataFrame(observations).to_csv(output/"accepted_observations.csv", index=False)
    withheld = {**observations, "fold": validation["folds"]}
    for method in METHODS:
        withheld.update({method+"_"+k: v for k, v in validation["methods"][method]["predictions"].items()})
    pd.DataFrame(withheld).to_csv(output/"heldout_predictions.csv", index=False)
    (output/"report.json").write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
    save_figures(output, name, data, meta, region, models, report)
    write_readme(output, name, report)
    return report
