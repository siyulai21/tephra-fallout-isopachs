"""Reader-facing comparisons from recorded estimates, never new inference."""
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, NullFormatter, StrMethodFormatter
from src.empirical_isopach.visualization.style import STYLE

COLORS = {"GP ensemble":"#197C83", "Cubic spline":"#B7682A",
          "Directional":"#6657A1", "Spherical mixture":"#346DA4"}


def saved_case_overview(output, name=None):
    """Assemble the common overview only after all four reports are written."""
    import json
    output = Path(output)
    # Old standalone archives can still reproduce their own historical overview.
    directional = next((p for p in ("directional/azimuth/report.json","directional/ensemble/report.json",
                                    "directional/single-structure/report.json","directional/report.json") if (output/p).is_file()),"directional/azimuth/report.json")
    reports = [json.loads((output/p).read_text()) for p in ("report.json",directional,"spherical-mixture/report.json")]
    case_overview(output, name or reports[0]["dataset"], *reports)


def metrics(regional, directional, spherical):
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
    return records


def case_overview(output, name, regional, directional, spherical):
    with plt.rc_context(STYLE):
        fig, axs = plt.subplots(2, 2, figsize=(12.6, 8.5), layout="constrained")
        for i, (key, label) in enumerate((("stacked", "GP ensemble"), ("spline", "Cubic spline"))):
            item = regional["methods"][key]
            q = item["joint_volume"]["conditional_quantiles_km3"]
            axs[0, 0].errorbar(q["0.5"], i, xerr=[[q["0.5"]-q["0.025"]], [q["0.975"]-q["0.5"]]],
                              fmt="o", color=COLORS[label], capsize=4, markersize=7)
            expectation = item["regional_volume"]["expected_km3"]
            if expectation <= q["0.975"]:
                axs[0, 0].scatter(expectation, i, marker="D", s=34, color=COLORS[label], zorder=4)
            else:
                axs[0, 0].annotate(f"Unstable mean {expectation:.2g} km³", xy=(.98, i),
                                   xycoords=("axes fraction", "data"), xytext=(0, 10),
                                   textcoords="offset points", ha="right", fontsize=8, color=COLORS[label])
        axs[0, 0].set(yticks=[0, 1], yticklabels=["GP ensemble", "Cubic spline"], ylim=(1.6, -.6),
                      xscale="log", xlabel="Hull bulk volume (km³, logarithmic scale)",
                      title="Regional volume inside the same hull\nCircle = median · diamond = mean")
        ensemble = directional.get("median_surface")
        if directional.get("protocol")=="uncertain-azimuth-v1":
            volume = directional["volume"]
            centre = volume["median_map_km3"]
            dq = volume["coherent_volume_km3"]
            passed = (all(v["diagnostics"]["passed"] for v in directional["members"].values()) and
                      volume["numerical"]["spatial_passed"] and volume["numerical"]["joint_quantile_split_passed"])
        elif ensemble:
            centre = ensemble["volume_km3"]
            dq = directional["coherent_fields"]["volume_km3"]
            passed = directional["coherent_fields"]["all_members_passed"]
        else:
            centre = None
            dq = directional["fits"]["reference"]["volume_km3"]
            passed = directional["fits"]["reference"]["passed"]
        for i, (label, q, passed) in enumerate((
                ("Directional", dq, passed),
                ("Spherical mixture", spherical["fits"]["baseline"]["volume"], spherical["fits"]["baseline"]["diagnostics"]["passed"]))):
            axs[0, 1].errorbar(q["median"], i, xerr=[[q["median"]-q["lower95"]], [q["upper95"]-q["median"]]],
                              fmt="o" if passed else "x", color=COLORS[label], capsize=4, markersize=7)
        if centre is not None:
            axs[0, 1].scatter(centre, -.17, marker="D", s=45, color=COLORS["Directional"], zorder=4)
        axs[0, 1].set(yticks=[0, 1], yticklabels=["Directional", "Spherical mixture"], ylim=(1.6, -.6),
                      xscale="log", xlabel="Whole-Earth bulk volume (km³, logarithmic scale)",
                      title=("Whole-deposit volume · diamond = directional median map\nComplete-field median and 95% interval · × = numerical warning"
                             if centre is not None else "Conditional whole-deposit volume\nMedian and 95% interval · × = numerical warning"))
        for ax in axs[0]:
            ax.xaxis.set_major_locator(LogLocator(subs=(1, 2, 5)))
            ax.xaxis.set_major_formatter(StrMethodFormatter("{x:,.3g}"))
            ax.xaxis.set_minor_formatter(NullFormatter())
        records = metrics(regional, directional, spherical)
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
            ax.set(yticks=range(4), yticklabels=labels, ylim=(3.6, -.6))
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
        fig.suptitle(f"{name} · four methods, two different volume questions", fontsize=16)
        fig.supxlabel("Intervals are conditional on the fitted model. The two volume panels integrate different geographic areas.", fontsize=9)
        fig.savefig(Path(output)/"overview.png", dpi=190, bbox_inches="tight", pad_inches=.16)
        plt.close(fig)


def all_case_scores(output, cases):
    from matplotlib.lines import Line2D
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(2, 3, figsize=(13.4, 8.3), layout="constrained")
        for ax, (name, regional, directional, spherical) in zip(axes.flat, cases):
            rows = metrics(regional, directional, spherical)
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
            ax.set(yticks=range(4), yticklabels=[r[0] for r in rows], ylim=(3.5, -.6),
                   xlim=(min(scores)-.2*span, max(scores)+1.2*span), title=name,
                   xlabel="Geographic prediction score ↓")
            ax.grid(axis="x", alpha=.18)
            ax.tick_params(labelsize=8)
        fig.suptitle("Six eruptions · matched predictive evidence across all four families\nLabels show score and exact-site 95% observation coverage; each case keeps its own score axis.", fontsize=14)
        fig.legend(handles=[Line2D([], [], c="#4B6068", marker="o", ls="", label="Completed geographic fits"),
                            Line2D([], [], c="#4B6068", marker="x", ls="", label="At least one global geographic fit has a numerical warning")],
                   loc="outside lower center", ncols=2, frameon=False, fontsize=9)
        fig.savefig(Path(output)/"method-scores.png", dpi=190, bbox_inches="tight", pad_inches=.16)
        plt.close(fig)
