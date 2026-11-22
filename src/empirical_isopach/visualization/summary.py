"""Reader-facing comparisons from recorded estimates, never new inference."""
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, NullFormatter, StrMethodFormatter
from src.empirical_isopach.visualization.style import STYLE

COLORS = {"GP ensemble":"#197C83", "Cubic spline":"#B7682A",
          "Directional":"#6657A1", "Spherical mixture":"#346DA4", "Nested corridor":"#933F66"}


def saved_case_overview(output, name=None):
    """Assemble the common overview only after all four reports are written."""
    import json
    output = Path(output)
    # Old standalone archives can still reproduce their own historical overview.
    directional = next((p for p in ("directional/azimuth/report.json","directional/ensemble/report.json",
                                    "directional/single-structure/report.json","directional/report.json") if (output/p).is_file()),"directional/azimuth/report.json")
    reports = [json.loads((output/p).read_text()) for p in ("report.json",directional,"spherical-mixture/report.json")]
    case_overview(output, name or reports[0]["dataset"], *reports, nested=principal_report(output))


def principal_report(home):
    path = Path(home)/"nested-isopachs/report.json"
    if path.is_file():
        result = json.loads(path.read_text())
        if result.get("protocol") == "nested-corridor-reach-v4":
            return result
    return None


def metrics(regional, directional, spherical, nested=None):
    records = []
    for key, label in (("stacked", "GP ensemble"), ("spline", "Cubic spline")):
        m = regional["validation"]["methods"][key]["metrics"]
        records.append((label, m["equal_region_log_score"], m["coverage95_observations"], True))
    for label, val in (("Directional", directional["validation"]["geographic"]),
                       ("Spherical mixture", spherical["validation"])):
        if label=="Directional" and directional.get("protocol")=="uncertain-azimuth-v1":
            scored = val["all_estimated"]
            passed = bool(scored and scored["rows"]==len(directional["data_rows"]) and
                          all(f["passed"] and f.get("phase_passed") for f in val["folds"] if f["test"]))
            score = scored["revised_score"] if scored else None
            records.append((label,score,val["observation_interval"]["coverage95_exact"],passed))
            continue
        m = val["metrics"]
        passed = (val["all_fits_passed_diagnostics"] if "all_fits_passed_diagnostics" in val
                  else val["all_estimable_passed"] and val["scored_rows"] == val["total_rows"])
        score = val["equal_region_log_score"] if "equal_region_log_score" in val else m["equal_region_log_score"]
        records.append((label, score, m["coverage95_observations"], passed))
    if nested is not None:
        if nested["input_sha256"] != regional["input"]["input_sha256"]:
            raise ValueError("Corridor and regional inputs differ")
        records.append(("Nested corridor", nested["geographic_score"], nested["exact_site_coverage95"]["fraction"],
                        all(f["passed"] for f in nested["folds"])))
    return records


def case_overview(output, name, regional, directional, spherical, nested=None, *, map_home=None):
    with plt.rc_context(STYLE):
        fig, axs = plt.subplots(2, 2, figsize=(12.6, 8.5), layout="constrained")
        for i, (key, label) in enumerate((("stacked", "GP ensemble"), ("spline", "Cubic spline"))):
            item = regional["methods"][key]
            q = item["joint_volume"]["conditional_quantiles_km3"]
            mapped = item["regional_volume"]["median_surface_km3"]
            axs[0, 0].hlines(i, q["0.025"], q["0.975"], color=COLORS[label], linewidth=2)
            axs[0, 0].scatter(mapped, i, color=COLORS[label], marker="D", zorder=4)
        axs[0, 0].set(yticks=[0, 1], yticklabels=["GP ensemble", "Cubic spline"], ylim=(1.6, -.6),
                      xscale="log", xlabel="Hull bulk volume (km³)", title="Inside the observation hull")
        folder = Path(map_home or output)
        families = [("Directional", "directional"), ("Spherical mixture", "spherical-mixture")]
        if nested is not None:
            families.append(("Nested corridor", "nested-isopachs"))
        for i, (label, family) in enumerate(families):
            receipt = folder/family/"deposit.json"
            if receipt.is_file():
                record = json.loads(receipt.read_text())
                lo, hi = record["conditional_volume_interval95_km3"]
                ax = axs[0, 1]
                ax.hlines(i, lo, hi, color=COLORS[label], linewidth=2,
                          linestyle="-" if record["complete_field_checks_passed"] else "--")
                ax.scatter(record["median_map_volume_km3"], i, marker="D", color=COLORS[label], zorder=4)
        axs[0, 1].set(yticks=range(len(families)), yticklabels=[r[0] for r in families],
                      ylim=(len(families)-.4, -.6), xscale="log", xlabel="Whole-Earth bulk volume (km³)",
                      title="Whole-Earth deposits")
        for ax in axs[0]:
            ax.xaxis.set_major_locator(LogLocator(subs=(1, 2, 5)))
            ax.xaxis.set_major_formatter(StrMethodFormatter("{x:,.3g}"))
            ax.xaxis.set_minor_formatter(NullFormatter())
        records = metrics(regional, directional, spherical, nested)
        for i, (label, score, coverage, passed) in enumerate(records):
            if coverage is not None:
                axs[1, 1].barh(i, coverage*100, color=COLORS[label], height=.48, alpha=.85)
                axs[1, 1].text(coverage*100-1, i, f"{coverage:.0%}", ha="right", va="center", color="white", fontsize=9)
            if score is None:
                axs[1, 0].text(.02,i,"Unscored",transform=axs[1, 0].get_yaxis_transform())
                continue
            axs[1, 0].scatter(score, i, c=COLORS[label], s=60, marker="o" if passed else "x")
            axs[1, 0].annotate(f"{score:.3f}", (score, i), xytext=(8, 0), textcoords="offset points", va="center", fontsize=9)
        labels = [r[0] for r in records]
        for ax in axs[1]:
            ax.set(yticks=range(len(records)), yticklabels=labels, ylim=(len(records)-.4, -.6))
        scores = [r[1] for r in records if r[1] is not None] or [0.]
        span = max(max(scores)-min(scores), .25)
        axs[1, 0].set(xlim=(min(scores)-.2*span, max(scores)+.65*span),
                      xlabel="Equal-region negative log predictive score ↓",
                      title="Matched geographic holdouts\nAll accepted measurements and likelihood bounds")
        axs[1, 1].axvline(95, color="#566C72", ls="--", lw=1)
        axs[1, 1].set(xlim=(0, 103), xlabel="Exact-site observation interval coverage (%)",
                      title="Coverage at withheld measurement sites\nDashed line marks nominal 95% coverage")
        for ax in axs.flat:
            ax.grid(axis="x", alpha=.18)
            ax.set_axisbelow(True)
            ax.title.set_fontsize(10.5)
        fig.suptitle(f"{name} · regional and whole-deposit fallout estimates", fontsize=16)
        fig.supxlabel("Diamonds give median-thickness map integrals. Lines span the conditional 95% volume interval. Dashed intervals have numerical warnings.", fontsize=9)
        fig.savefig(Path(output)/"overview.png", dpi=190, bbox_inches="tight", pad_inches=.16)
        plt.close(fig)


def all_case_scores(output, cases):
    from matplotlib.lines import Line2D
    with plt.rc_context(STYLE):
        rows_needed = (len(cases) + 2) // 3
        fig, axes = plt.subplots(rows_needed, 3, figsize=(13.4, 4.15 * rows_needed),
                                 squeeze=False, layout="constrained")
        for ax in axes.flat[len(cases):]:
            ax.set_visible(False)
        for ax, (name, *reports) in zip(axes.flat, cases):
            rows = metrics(*reports)
            for i, (label, score, coverage, passed) in enumerate(rows):
                if score is None:
                    ax.text(.02,i,"Unscored",transform=ax.get_yaxis_transform())
                    continue
                coverage_label = "—" if coverage is None else f"{coverage:.0%}"
                ax.scatter(score, i, s=65, c=COLORS[label], marker="o" if passed else "x")
                ax.annotate(f"{score:.3f}  ·  {coverage_label}", (score, i), xytext=(7, 0),
                            textcoords="offset points", va="center", fontsize=8)
            scores = [r[1] for r in rows if r[1] is not None] or [0.]
            span = max(max(scores)-min(scores), .3)
            ax.set(yticks=range(len(rows)), yticklabels=[r[0] for r in rows], ylim=(len(rows)-.5, -.6),
                   xlim=(min(scores)-.2*span, max(scores)+1.2*span), title=name,
                   xlabel="Geographic prediction score ↓")
            ax.grid(axis="x", alpha=.18)
            ax.tick_params(labelsize=8)
        fig.suptitle(f"{len(cases)} eruptions · geographically withheld fallout measurements\nLabels show score and exact-site 95% observation coverage; each case keeps its own score axis.", fontsize=14)
        fig.legend(handles=[Line2D([], [], c="#4B6068", marker="o", ls="", label="Completed geographic fits"),
                            Line2D([], [], c="#4B6068", marker="x", ls="", label="At least one global geographic fit has a numerical warning")],
                   loc="outside lower center", ncols=2, frameon=False, fontsize=9)
        panels = [ax.get_title() for ax in axes.flat if ax.get_visible()]
        fig.savefig(Path(output)/"method-scores.png", dpi=190, bbox_inches="tight", pad_inches=.16,
                    metadata={"Description": json.dumps({"eruption_panels": panels, "methods_by_case": {case[0]: [r[0] for r in metrics(*case[1:])] for case in cases}})})
        plt.close(fig)
