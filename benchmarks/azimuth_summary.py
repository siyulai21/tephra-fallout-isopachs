"""Summarize paired held-out scores and known-field footprint tests for publication."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.empirical_isopach.core.catalog import datasets
from src.empirical_isopach.models.directional.study import write_json


def summarize(cases,synthetic,output):
    output.mkdir(exist_ok=True,parents=True)
    records = []
    for rec in datasets():
        slug = rec["slug"]
        r = json.loads((cases/slug/"report.json").read_text())
        v = r["volume"]
        row = dict(eruption=r["dataset"],slug=slug,input_sha256=r["input"]["input_sha256"],accepted=len(r["data_rows"]),
                   fixed_map_km3=v["matched_control_map_km3"],uncertain_map_km3=v["median_map_km3"],
                   **{"joint_"+k:x for k,x in v["coherent_volume_km3"].items()})
        for design in ("geographic","quadrants"):
            for subset in ("all_estimated","passed_only"):
                score = r["validation"][design][subset]
                for key in ("folds","rows","control_score","revised_score"):
                    row[design+"_"+subset+"_"+key] = score[key] if score else None
        records.append(row)
    table = pd.DataFrame(records).sort_values("slug")
    table.to_csv(output/"case-comparison.csv",index=False)
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
    rows = ["# Directional fallout across six eruptions", "",
        "Six eruptions test how much the assumed fan direction changes a contour, a bulk volume and predictions at withheld sites. The principal fit allows extra fallout to point in different directions across possible deposits. The control fixes the direction and uses the same observations and structure weights.", "",
        "Solid pieces of a median contour cross triangles where measured thicknesses bracket the level and the fitted crossing probability reaches 90%. Dashes trace continuation through gaps. A fan can extend into an unsampled direction in an individual simulated deposit even where the median contour retreats.", "",
        "## Western Changbaishan and southwestern Mazama", "",
        "**Western Changbaishan.** At Lake Sihailongwan the preceding fixed-direction map gave 15.8 cm. The paired fixed-direction control gives 16.4 cm and the uncertain-azimuth median gives 8.0 cm. The principal quantitative-only 95% latent range is 0.00015–101 cm. A cryptotephra occurrence without a measured fallout bed still challenges this central prediction. The core description supplies no numerical upper limit. [Primary core study](https://doi.org/10.1016/j.gr.2015.01.013) · [Source and site audit](../../eruptions/b-tm/results/directional/bearing/field-evidence.md)", "",
        "Conditioning only on an assumed 1 cm recognizable bed at Quanyang gives a Sihailongwan median of 0.0084 cm without conditioning on the lake. Assuming recognizable beds at 1 cm at Sihailongwan and 2 cm at Quanyang gives 0.0052 cm at the lake and 0.11 cm at the peat. Those estimates depend on preservation and detection. The second estimate uses both sites in its likelihood. The fixed record keeps transferred shard-to-thickness conversions outside the quantitative table. [B-Tm maps and all archive sensitivities](../../eruptions/b-tm/results/directional/README.md)", "",
        "**Southwestern Mazama.** Along the 225° ray the 10 cm median crossing moves from 199 to 135 km. The ≥10 cm area across the entire southwest octant falls from 13,559 to 6,635 km². The accepted observations do not locate an actual southwest deposit edge. [Mazama maps and all eight bearings](../../eruptions/mazama/results/directional/README.md)", "",
        "## Global bulk volume", "",
        "The two median maps use identical structure weights and training observations. Their difference measures the effect of uncertain fan direction. Earlier case headlines used their original calibration. Every figure below integrates to the antipode.", "",
        "| Eruption | Principal uncertain-direction median map (km³) | Fixed-direction control (km³) | Uncertain-azimuth complete deposits, median [95%] (km³) |",
        "|---|---:|---:|---:|"]
    for a in table.itertuples():
        rows.append(f"| [{a.eruption}](../../eruptions/{a.slug}/results/directional/README.md) | **{a.uncertain_map_km3:,.1f}** | {a.fixed_map_km3:,.1f} | {a.joint_median:,.1f} [{a.joint_lower95:,.1f}–{a.joint_upper95:,.1f}] |")
    rows += ["","## Do the shorter contours predict withheld observations?","",
        "Each eruption learns mixture weights from the other eruptions. Both alternatives are scored on exactly the same withheld rows and limits. Lower negative log predictive scores are better. Averages give each available fold equal weight, then each eruption equal weight.", "",
        "| Deletion design | Eruptions contributing | Fixed | Uncertain azimuth |", "|---|---:|---:|---:|"]
    for design in ("geographic","quadrants"):
        for subset,name in (("all_estimated","all estimable folds"),("passed_only","passing folds only")):
            d = table[[design+"_"+subset+"_"+k+"_score" for k in ("control","revised")]].dropna()
            rows.append(f"| {design.title()}, {name} | {len(d)} | {d.iloc[:,0].mean():.3f} | {d.iloc[:,1].mean():.3f} |")
    rows += ["","The uncertain-azimuth model improves the pooled geographic score slightly. Its quadrant score worsens, including in the passing-only comparison. The primary reporting choice favors less extension of the median contour in unsupported directions. YTT benefits in both designs. Holding out the target protects calibration weights. The method itself was developed with these six eruptions in view. Tambora has no passing quadrant refit. Some other individual fold fits also carry numerical warnings. The fixed-direction results remain available as sensitivities. [Case scores, contributing folds and volumes](case-comparison.csv)", "",
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
        "## Numerical checks and regional comparison", "",
        "The spatial quadrature check passed in all six eruptions. For B-Tm the 100 cm area differs by 24% between split posterior packets at the 512-draw-per-chain ceiling, although the integrated median-map volume differs by about 1%. The thickest contour therefore remains numerically sensitive. Passing integration checks do not identify ash volume outside the measurements.", "",
        "The matched regional benchmark was rerun on the same six audited inputs. Its twelve empirical rows and its eighty synthetic rows reproduced the published CSVs byte for byte, including the four unsuccessful synthetic fits. [Matched regional results](../../eruptions/b-tm/results/source-audit/matched-regional.csv) · [Regional synthetic results](../regional/synthetic/comparison.csv)", "",
        "[Statistical protocol and reproduction commands](../../docs/methods/uncertain-azimuth.md) · [Numerical identities of the six original fits](verification.json)", "",
        "## Related directional studies", "",
        "- [Fixed-direction ensemble](fixed-ensemble/README.md) examines the earlier weight calibration, geographic prediction and volume coverage on known deposits. Its fixed-direction estimates use different weights from the paired control above.",
        "- [Measured fans and inspected sites](measured-fans/README.md) tests continuation along sampled bearings and links the B-Tm evidence for recognizable beds.",
        "- [Single-structure sensitivities](single-structure/README.md) tests 13 fixed policies, an unresolved proximal cap and independent replications.", ""]
    (output/"README.md").write_text("\n".join(rows))


if __name__=="__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases",type=Path,required=True)
    parser.add_argument("--synthetic",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    summarize(args.cases,args.synthetic,args.output)
