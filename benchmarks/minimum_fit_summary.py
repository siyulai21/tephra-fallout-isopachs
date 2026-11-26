"""Compare fitted thickness at every reported minimum, including distal sites."""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from benchmarks.publication import result_home, published_methods
from src.empirical_isopach.core.catalog import datasets


def summarize(output):
    output.mkdir(exist_ok=True, parents=True)
    styles = {
        "directional": ("Directional", "#276578", "o"),
        "spherical-mixture": ("Spherical mixture", "#a96329", "s"),
        "nested-isopachs": ("Nested footprints", "#43755a", "^"),
        "stacked": ("GP ensemble", "#745090", "x"),
        "spline": ("Cubic spline", "#a23e56", "+"),
    }
    names = {"mazama":"Mazama", "ilopango-tbj":"Ilopango TBJ", "minoan":"Minoan",
             "pinatubo-1991-layer-c":"Pinatubo Layer C", "sakurajima-satsuma":"Sakurajima–Satsuma",
             "aso-4":"Aso-4", "b-tm":"B-Tm"}
    records = {r["slug"]: r for r in datasets() if r.get("reported_file")}
    cases = [slug for slug in names if slug in records] + sorted(set(records)-set(names))
    rows = []
    fig, axes = plt.subplots(int(np.ceil((len(cases)+1)/2)), 2, figsize=(11.2, 2.35*np.ceil((len(cases)+1)/2)+.9),
                             layout="constrained", squeeze=False)
    for ax, slug in zip(axes.flat, cases):
        record = records[slug]
        table = pd.read_csv(result_home(record)/"reported-and-fitted-sites.csv")
        minima = table[table.reported_kind == "lower_limit"]
        sites = minima.drop_duplicates("row_id")
        ax.axhline(1, color="#495056", linewidth=.85)
        for method, (_, color, marker) in styles.items():
            if method not in published_methods(record):
                continue
            x = minima[(minima.method == method) & minima.previous_map_cm.notna()]
            ratio = x.current_map_cm/x.previous_map_cm
            ax.scatter(x.projected_distance_km, ratio, color=color, marker=marker, alpha=.65, s=16, linewidths=.65)
            rows.append(dict(slug=slug, method=method, minima=len(sites), paired_minima=len(x),
                             within_25_km=int(sites.projected_distance_km.le(25).sum()),
                             beyond_25_km=int(sites.projected_distance_km.gt(25).sum()),
                             median_new_to_old_at_minima=ratio.median() if len(x) else None))
        ax.set(xscale="symlog",yscale="log")
        ax.set_xscale("symlog", linthresh=1.)
        ax.set_title(f"{names.get(slug,record['name'])}  ·  {len(sites)} reported minima", loc="left", fontsize=11)
        ax.grid(alpha=.16, linewidth=.55)
        ax.tick_params(labelsize=8)
    for ax in axes.flat[len(cases):]:
        ax.axis("off")
    key = axes.flat[len(cases)]
    handles = [plt.Line2D([], [], marker=m, color=c, linestyle="", markersize=5, label=label) for label,c,m in styles.values()]
    key.legend(handles=handles, loc="center", ncols=2, frameon=False, fontsize=10)
    fig.suptitle("Change in modeled thickness at reported minima", fontsize=16)
    fig.supxlabel("Projected distance from the vent (km)")
    fig.supylabel("Revised / previous median thickness")
    fig.savefig(output/"site-ratios.png", dpi=155)
    plt.close(fig)
    pd.DataFrame(rows).to_csv(output/"all-distances.csv", index=False)

    lines = ["# Fitting reported lower limits as point thicknesses", "",
             f"These {len(cases)} eruptions have accepted fallout sites with a numeric lower limit. Each fit uses that value "
             "as a point thickness at its actual location. Upper limits retain a one-sided probability and ranges "
             "use the probability between both endpoints. The source tables retain the original qualifiers. "
             "[Fitting and plotting convention](../../docs/methods/reported-minimum-fitting.md)", "",
             "![Ratio of revised to previous median thickness at reported minima versus distance from the vent](site-ratios.png)", "",
             "Each symbol compares two map estimates at the same reported minimum. A ratio below one indicates a "
             "thinner revised estimate. The 25 km division below describes where the sites lie. All distances enter "
             "the same likelihood. A blank comparison means the previous edition had no map on the same audited input. "
             "The current nested maps also use the continuous-area-growth method, so their ratios include that "
             "methodological revision as well as the fitting convention. [Matched nested-method comparison](../nested-isopachs/adaptive-growth/README.md). "
             "[Values by case and method](all-distances.csv)", "",
             "| Eruption | Source minima | Within 25 km | Beyond 25 km | Directional vent, previous → revised (cm) | Spherical vent, previous → revised (cm) | Nested vent, previous → revised (cm) |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for slug in cases:
        home = result_home(records[slug])
        stats = pd.read_csv(home / "minimum-fit-comparison.csv").set_index("method")
        n = int(stats.reported_minima.iloc[0])
        near = int(stats.minima_within_25_km.iloc[0])
        name = names.get(slug, records[slug]["name"])
        def vent(method):
            if method not in stats.index:
                return "—"
            pair = stats.loc[method, ["old_vent_cm", "new_vent_cm"]]
            return " → ".join("—" if pd.isna(x) else f"{x:,.1f}" for x in pair)
        lines.append(f"| [{name}](../../eruptions/{slug}/README.md) | {n} | {near} | {n-near} | "
                     f"{vent('directional')} | {vent('spherical-mixture')} | {vent('nested-isopachs')} |")
    lines += ["", "Vent values in the table are modeled median thicknesses at the vent coordinates. A model can place its thickest deposit elsewhere. "
              "Observed sections can preserve only part of the deposit. Fitting their minima as points can therefore "
              "underestimate total local fallout. Each case links its original sources and separate validation coverage "
              "for directly measured thicknesses and minima fitted as points.", "",
              "The methods published on each input use the same accepted observations and four geographic holdouts. "
              "An annotation-removal check compares the median field at the vent and every accepted site with and without "
              "the provenance marker. The case comparison tables record the maximum absolute difference. "
              "The global estimates include extrapolation into unsampled regions. Numerical warnings and unsupported "
              "contour bands are retained in the individual case reports. [Method scores and volumes](../principal-volumes.csv) · [Matched regional benchmark](regional-benchmark/README.md)", ""]
    (output / "README.md").write_text("\n".join(lines))



if __name__ == "__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    summarize(p.parse_args().output)
