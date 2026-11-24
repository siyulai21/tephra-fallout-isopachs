"""Build the matched map-integral table from the registered eruption cases."""
import argparse
from pathlib import Path
import re

import matplotlib.pyplot as plt
import numpy as np

from .nested_reach import case_home
from src.empirical_isopach.core.catalog import datasets
from src.empirical_isopach.core.audit import audited_data
from src.empirical_isopach.models.registry import active_methods
from src.empirical_isopach.visualization.principal import FAMILIES, number, volume_rows, write_rows
from src.empirical_isopach.visualization.summary import COLORS, all_case_scores


from src.empirical_isopach.visualization.deposit_field import digest


def principal_home(record):
    return case_home(record["slug"])


def catalogue_rows():
    rows = []
    for record in sorted(datasets(), key=lambda r:r["slug"]):
        slug = record["slug"]
        home = principal_home(record)
        accepted = len(audited_data(record["path"])[0].xy)
        fitted = volume_rows(home,input_sha256=digest(record["path"]))
        expected = {*active_methods(),*FAMILIES}
        found = {r["method"] for r in fitted}
        if found != expected or len(fitted) != len(expected):
            raise ValueError(f"{slug}: published families {sorted(found)} differ from registered families {sorted(expected)}")
        for row in fitted:
            rows.append(dict(eruption=record["name"],slug=slug,accepted_observations=accepted,**row))
    return rows


LABELS = {"stacked":"GP ensemble","spline":"Cubic spline","directional":"Directional",
          "spherical-mixture":"Spherical mixture","nested-isopachs":"Nested footprints"}


def volume_plot(rows,path):
    names = list(dict.fromkeys(r["eruption"] for r in rows))
    fig,ax = plt.subplots(figsize=(9,5.6),layout="constrained")
    for offset,family in zip((-.24,0.,.24),FAMILIES,strict=True):
        block = [r for r in rows if r["method"]==family]
        y = [names.index(r["eruption"])+offset for r in block]
        color = COLORS[LABELS[family]]
        ax.hlines(y,[r["thickness_q25_volume_km3"] for r in block],[r["thickness_q75_volume_km3"] for r in block],
                  color=color,linewidth=1.7)
        ax.scatter([r["median_map_volume_km3"] for r in block],y,label=LABELS[family],color=color,marker="D",s=23)
    ax.set(yticks=np.arange(len(names)),yticklabels=names,xscale="log",xlabel="Whole-Earth bulk fallout (km³)")
    ax.invert_yaxis()
    ax.grid(axis="x",alpha=.22)
    ax.legend(loc="upper center",bbox_to_anchor=(.5,-.13),ncols=3,frameon=False)
    ax.set_title("Global median maps and integrated Q25–Q75 thickness bands",loc="left")
    fig.savefig(path,dpi=160)
    plt.close(fig)


def family_page(family, rows, records):
    introductions = {
        "directional":("Directional ensemble","The ensemble averages over uncertain fan directions and thinning curves.",
                       "[Paired continuations and validation](../global-extent/README.md) · [Training and fixed-direction controls](fixed-ensemble/README.md)"),
        "spherical-mixture":("Spherical mixture","A mass-normalized mixture reconstructs the deposit on Earth. Eight fixed prior fits test sensitivity to unsampled directions.",
                            "[Prior and geometry diagnostics on their study inputs](comparison.csv) · [Known-deposit tests](bearings/known-volumes.csv)"),
        "nested-isopachs":("Bayesian nested dispersal footprints","The deposit footprint expands as thickness decreases. The fit estimates footprint geometry and the thickness–area law jointly.",
                          "[Geographic experiments and independent surveys](geography/README.md) · [Original whole-area sensitivity](whole-area-study.md)")}
    title,intro,links = introductions[family]
    lines = [f"# {title}", "", intro+" "+links,"",
             "| Eruption | Localities | Median map (km³) | Q25–Q75 thickness maps (km³) | Held-out score ↓ | 95% predictive coverage |",
             "|---|---:|---:|---:|---:|---:|"]
    for row in rows:
        if row["method"]!=family:
            continue
        rec = records[row["slug"]]
        uri = (Path("../..")/principal_home(rec)/family/"README.md").as_posix()
        passed = row["thickness_band_integration_passed"] and row["posterior_checks_passed"]
        score = row["heldout_log_score"]
        coverage = row["heldout_coverage95"]
        lines.append(f"| [{row['eruption']}]({uri}) | {row['accepted_observations']} | **{number(row['median_map_volume_km3'])}** "
                     f"| {number(row['thickness_q25_volume_km3'])}–{number(row['thickness_q75_volume_km3'])}{'' if passed else ' †'} "
                     f"| {'—' if score is None else f'{score:.3f}'}{'' if row['heldout_checks_passed'] else ' †'} "
                     f"| {'—' if coverage is None else f'{coverage:.1%}'} |")
    lines += ["","Each endpoint integrates a pointwise thickness quartile over Earth. A dagger marks a numerical warning. "
              "[All current volumes and numerical flags](../principal-volumes.csv)",""]
    if family=="nested-isopachs":
        lines += ["![Global median maps and integrated pointwise thickness bands on each current dataset](volumes.png)",""]
    return "\n".join(lines)


def refresh_catalogue(output):
    output = Path(output)
    records = {r["slug"]:r for r in datasets()}
    rows = catalogue_rows()
    dest = output/"comparisons"
    dest.mkdir(parents=True,exist_ok=True)
    write_rows(dest/"principal-volumes.csv",rows)
    scores = {}
    for r in rows:
        scores.setdefault(r["eruption"],[]).append((LABELS[r["method"]],r["heldout_log_score"],r["heldout_coverage95"],r["heldout_checks_passed"]))
    all_case_scores(dest,[(n,) for n in scores],scores_by_case=scores)
    for family in FAMILIES:
        (dest/family).mkdir(exist_ok=True)
        (dest/family/"README.md").write_text(family_page(family,rows,records))
    from .azimuth_summary import current_control_summary
    current_control_summary(dest/"directional/fixed-ensemble",records.values())
    volume_plot(rows,dest/"nested-isopachs/volumes.png")
    listing = Path("eruptions/README.md")
    body = listing.read_text()
    for slug in records:
        values = {r["method"]:number(r["median_map_volume_km3"]) for r in rows if r["slug"]==slug}
        pattern = r"(\| \[[^\n]*?\]\("+re.escape(slug)+r"/README\.md\) \|) [^|]* \| [^|]* \| [^|]* \|"
        body,n = re.subn(pattern,lambda m:m[1]+" "+" | ".join(values.get(f,"—") for f in FAMILIES)+" |",body)
        if n!=1:
            raise ValueError(slug)
    (output/"eruptions").mkdir(exist_ok=True)
    (output/listing).write_text(body)


if __name__=="__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--catalogue",action="store_true",help="Write the catalogue, family indexes and plots under this output directory")
    args = parser.parse_args()
    if args.catalogue:
        refresh_catalogue(args.output)
    else:
        args.output.parent.mkdir(parents=True,exist_ok=True)
        write_rows(args.output,catalogue_rows())
