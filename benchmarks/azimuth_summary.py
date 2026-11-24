"""Summarize paired held-out scores and known-field footprint tests for publication."""
import argparse
import json
from pathlib import Path

import shutil
import pandas as pd

from .publication import published_datasets
from src.empirical_isopach.core.catalog import datasets
from src.empirical_isopach.models.directional.study import digest, write_json


PUBLISHED = Path("comparisons/directional")


def case_table(cases=None, published_root=Path("eruptions")):
    """Read the paired controls and both holdout designs from the same fitted report."""
    records = []
    for rec in (published_datasets() if cases is None else datasets()):
        slug = rec["slug"]
        path = (cases/slug/"report.json" if cases is not None else
                published_root/slug/"results/directional/azimuth/report.json")
        r = json.loads(path.read_text())
        if cases is None and r["input"]["input_sha256"] != digest(rec["path"]):
            raise ValueError(f"{slug}: published directional fit and accepted input differ")
        v = r["volume"]
        row = dict(eruption=r["dataset"],slug=slug,input_sha256=r["input"]["input_sha256"],accepted=len(r["data_rows"]),
                   fixed_map_km3=v["matched_control_map_km3"],uncertain_map_km3=v["median_map_km3"],
                   **{"joint_"+k:v["coherent_volume_km3"][k] for k in ("lower95","upper95")})
        for design in ("geographic","quadrants"):
            for subset in ("all_estimated","passed_only"):
                score = r["validation"][design][subset]
                for key in ("folds","rows","control_score","revised_score"):
                    row[design+"_"+subset+"_"+key] = score[key] if score else None
        if cases is None and (path.parent.parent/"extent/report.json").is_file():
            home=path.parent.parent
            current=json.loads((home/"extent/report.json").read_text())
            row["uncertain_map_km3"]=json.loads((home/"deposit.json").read_text())["median_map_volume_km3"]
            row["joint_lower95"],row["joint_upper95"]=current["conditional_interval95_km3"]
            for kind,detail in current["validation"].items():
                row[kind+"_all_estimated_revised_score"]=detail["negative_log_score"]
                row[kind+"_passed_only_revised_score"]=detail["passed_only_score"]
        records.append(row)
    return pd.DataFrame(records).sort_values("slug").reset_index(drop=True)


def synthetic_areas(synthetic, output):
    if synthetic is None:
        for name in ("synthetic-areas.csv", "synthetic-trials.json"):
            if (output/name).resolve() != (PUBLISHED/name).resolve():
                shutil.copyfile(PUBLISHED/name, output/name)
        return pd.read_csv(output/"synthetic-areas.csv")
    frames, trials = [],[]
    for f in sorted(synthetic.glob("*.json")):
        s = json.loads(f.read_text())
        frames.append(pd.DataFrame(s.pop("areas")).assign(design=f.stem))
        trials.append(s)
    if len(trials)!=14:
        raise ValueError("Expected both pre-existing seeds of all seven synthetic designs")
    write_json(output/"synthetic-trials.json",trials)
    areas = pd.concat(frames)
    areas.to_csv(output/"synthetic-areas.csv",index=False)
    return areas


def summarize(cases, synthetic, output):
    output.mkdir(exist_ok=True, parents=True)
    table = case_table(cases)
    table.to_csv(output/"case-comparison.csv", index=False)
    areas = synthetic_areas(synthetic, output)
    rows = ["# How much does uncertain fan direction change the fit?", "",
        "Each pair uses the same observations and structure weights. The control fixes fan direction. "
        "The principal fit lets it vary across possible deposits. The uncertain-direction median-thickness map gives "
        "the principal estimate. The fixed-direction control uses direct posterior quadrature. The interval describes uncertainty in total volume across complete draws.", "",
        "| Eruption | Accepted observations | Median-thickness map (km³) | Fixed-direction control (km³) | Conditional 95% interval (km³) |",
        "|---|---:|---:|---:|---:|"]
    for a in table.itertuples():
        path = Path("eruptions")/a.slug/"results/directional/deposit.json"
        mapped = json.loads(path.read_text())
        rows.append(f"| [{a.eruption}](../../eruptions/{a.slug}/results/directional/README.md) | {a.accepted} | **{mapped['median_map_volume_km3']:,.3g}** | {a.fixed_map_km3:,.1f} | {a.joint_lower95:,.1f}–{a.joint_upper95:,.1f} |")
    rows += ["", "At B-Tm the source review adds two lake limits and one peat limit to the directly measured thicknesses and bounds. "
        "The before-and-after study isolates how those interpretations move the western contours. Each eruption report above gives "
        "all eight compass bearings, supported contour fragments and numerical warnings. "
        "[B-Tm occurrence study](../../eruptions/b-tm/results/cryptotephra/README.md)", "",
        "## Predictions in withheld regions", "",
        "Each score gives equal weight to available folds within an eruption, then equal weight to eruptions. Lower is better. "
        "The weights for a target exclude that eruption. The table uses each published fit’s recorded calibration. "
        "The passing-only rows show how sampling warnings affect the comparison.", "",
        "| Deletion design | Eruptions contributing | Fixed | Uncertain azimuth |", "|---|---:|---:|---:|"]
    for design in ("geographic","quadrants"):
        for subset,name in (("all_estimated","all estimable folds"),("passed_only","passing folds only")):
            d = table[[design+"_"+subset+"_"+k+"_score" for k in ("control","revised")]].dropna()
            rows.append(f"| {design.title()}, {name} | {len(d)} | {d.iloc[:,0].mean():.3f} | {d.iloc[:,1].mean():.3f} |")
    rows += ["", "A quadrant with no test observations has no score. The download gives input hashes, fold and row counts, "
        "and both scores for every eruption. In the original 68-site B-Tm analysis, only two quadrants could be scored. "
        "Adding observations can change which sectors contribute to a pooled comparison. The original six eruptions "
        "informed model development. [Paired scores and volumes](case-comparison.csv) · "
        "[Original 68-site comparison](https://github.com/siyulai21/tephra-fallout-isopachs/tree/f155753dad2dd4d06e4fa48d3d877242ae8539aa/comparisons/directional)", "",
        "## Known deposits with a deliberate sampling gap", "",
        "Fourteen previously specified simulated datasets cover seven designs and two seeds. Two compatible truths on each invisible-sector dataset share all observations but differ by a hidden lobe. The table evaluates false extension and missed fallout on area-weighted whole-Earth nodes, separately for locations with directional exposure of at least 0.5. Each design and seed receives equal weight; compatible truths share that weight. Cells require at least 64 effective union nodes for each of the three mapped decisions.", "",
        "| Region | Thickness | Resolved datasets | False extension, fixed → uncertain (km²) | Missed area, fixed → uncertain (km²) | Overlap, fixed → uncertain |",
        "|---|---:|---:|---:|---:|---:|"]
    for region in ("whole_Earth","high_exposure"):
        for level in (1.,10.,100.):
            pivot = areas[(areas.region==region)&(areas.cm==level)].pivot(index=["design","truth"],columns="method",
                     values=["union_effective_nodes","false_positive_km2","false_negative_km2","intersection_over_union"])
            pivot = pivot[pivot["union_effective_nodes"].min(axis=1)>=64]
            means = pivot.groupby("design").mean()
            def pair(col,digits):
                a = means[col].mean()
                return f"{a.control_median:,.{digits}f} → {a.revised_median:,.{digits}f}"
            rows.append(f"| {region.replace('_',' ')} | {level:g} cm | {len(means)} | {pair('false_positive_km2',0)} | {pair('false_negative_km2',0)} | {pair('intersection_over_union',3)} |")
    rows += ["","Across the resolved high-exposure tests, the revised 1 cm contour cuts false extension at the cost of some missed area. For 10 cm there was already no false extension in those tests, and shortening the contour misses more true fallout. The stricter 90% exceedance outline misses still more. Nine of the fourteen synthetic datasets had a sampling warning in at least one base structure. All four datasets with resolved high-exposure 1 cm and 10 cm comparisons had such a warning. The passing-only subset therefore provides no resolved test of those high-exposure comparisons. These patterns describe a trade-off under the simulated assumptions. They cannot establish the unseen shape of a real deposit. [False-positive and false-negative areas for every trial and threshold](synthetic-areas.csv) · [True volumes and base-fit warnings](synthetic-trials.json)", "",
        "## Methods and earlier controls", "",
        "[Equations and calibration](../../docs/methods/uncertain-azimuth.md) · "
        "[Original six-fit identities before the B-Tm revision](verification.json)", "",
        "- [Fixed-direction ensemble](fixed-ensemble/README.md) examines the earlier weight calibration, geographic prediction and volume coverage on known deposits. Its fixed-direction estimates use different weights from the paired control above.",
        "- [Measured fans and inspected sites](measured-fans/README.md) tests continuation along sampled bearings and links the B-Tm evidence for recognizable beds.",
        "- [Single-structure sensitivities](single-structure/README.md) tests 13 fixed policies, an unresolved proximal cap and independent replications.", ""]
    (output/"README.md").write_text("\n".join(rows))


if __name__=="__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, help="Use new per-case report directories. Defaults to the registered publications.")
    parser.add_argument("--synthetic", type=Path, help="Use new individual trial reports. Defaults to the published fourteen trials.")
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    summarize(args.cases,args.synthetic,args.output)
