"""Compare fitted thickness at every reported minimum, including distal sites."""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from benchmarks.publication import result_home
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


if __name__ == "__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    summarize(p.parse_args().output)
