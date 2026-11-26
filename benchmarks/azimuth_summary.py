"""Summarize paired held-out scores and known-field footprint tests for publication."""
import argparse
import json
from pathlib import Path

import shutil
import pandas as pd

from .publication import published_methods, reported_edition, result_home
from src.empirical_isopach.core.catalog import datasets
from src.empirical_isopach.models.directional.study import digest, write_json


PUBLISHED = Path("comparisons/directional")


def control_edition(record, packet, root=Path(".")):
    """Use a preceding control only under its own explicit input identity."""
    if (root / result_home(record) / "directional" / packet).is_file():
        return record, "current"
    previous = reported_edition(record)
    if (root / result_home(previous) / "directional" / packet).is_file():
        return previous, "preceding reported-input edition"
    raise FileNotFoundError(f"No published {packet} control for {record['slug']}")


def current_control_summary(output, records):
    """Summarize available controls and declare each control's actual input edition."""
    import matplotlib.pyplot as plt

    output.mkdir(exist_ok=True,parents=True)
    rows = []
    for rec in sorted(records,key=lambda r:r["name"]):
        if "directional" not in published_methods(rec):
            continue
        source, edition = control_edition(rec, "ensemble/report.json")
        home = result_home(source)/"directional"
        ensemble = json.loads((home/"ensemble/report.json").read_text())
        single = json.loads((home/"single-structure/report.json").read_text())
        identity = digest(source["path"])
        assert single["input"]["input_sha256"] == ensemble["input"]["input_sha256"] == identity
        geo = ensemble["validation"]["geographic"]
        assert geo["row_ids"] == single["validation"]["geographic"]["row_ids"]
        measured = home/"bearing/quantitative/validation.json"
        score = None
        if measured.exists():
            validation = json.loads(measured.read_text())["geographic"]
            for observed,base in zip(validation["folds"],geo["fits"],strict=True):
                assert observed["withheld_row_ids"] == base["withheld_row_ids"]
                assert observed["training_row_ids"] == base["training_row_ids"]
            score = validation["equal_region_negative_log_score"]
        rows.append(dict(name=rec["name"],slug=rec["slug"],n=len(geo["row_ids"]),home=home,
                         edition=edition,input_sha256=identity,
                         single=single["validation"]["geographic"]["equal_region_log_score"],
                         fixed=geo["equal_region_log_score"],measured=score,passed=geo["all_estimable_passed"],
                         median=ensemble["median_surface"]["volume_km3"],median_passed=ensemble["median_surface"]["passed"]))
    lines = ["# Directional reference controls","",
             "Each row declares the input edition of its saved controls. Where the current edition has no fixed-direction packet, the preceding reported-input edition remains a historical reference; it is not a matched comparison with the current principal fit. The single-structure fit uses the reference fan geometry. The fixed-direction ensemble averages six structures. The measured-fan column gives the quantitative-only bearing fit where published. Lower scores indicate better predictions. [Principal directional fits](../README.md) · [Matched v5/v6 comparison](../geographic-regularity/README.md) · [Control definitions](../../../docs/methods/directional-ensemble.md)","",
             "| Eruption | Input edition | Sites | Single-structure score | Fixed-direction score | Measured-fan score | Fixed median map (km³) |",
             "|---|---|---:|---:|---:|---:|---:|"]
    for r in rows:
        link=f"../../../{r['home']}/ensemble/README.md"
        score="—" if r["measured"] is None else f"{r['measured']:.3f}"
        lines.append(f"| [{r['name']}]({link}) | {r['edition']} | {r['n']} | {r['single']:.3f} | {r['fixed']:.3f}{'' if r['passed'] else ' †'} | {score} | {r['median']:,.2f}{'' if r['median_passed'] else ' †'} |")
    lines += ["","A dagger marks a numerical warning on the fixed ensemble’s geographic fits or its median-map integral. Each case links the individual policies, holdouts and sampling diagnostics. The [uncertain-azimuth control table](../case-comparison.csv) likewise reads each paired report at its declared input edition; it does not substitute the later finite-spread or principal multiscale estimates.","",
              "![Whole-Earth volumes of the fixed-direction median maps](volumes.png)","",
              "## Known deposits and calibration","",
              "The 14 fixed-design trials compare fitted isopach area with known deposits at 1, 10 and 100 cm. Two trials have identical observed thicknesses under two different hidden deposits. Their fitted intervals covered the larger hidden deposit in neither trial. [Known volumes](known-volumes.csv) · [Contour overlap](contour-accuracy.csv) · [Simulation details](simulation-records.json.gz)","",
              "The original six-eruption training packet fixes the structure weights for these experiments. A target eruption contributes no scores to its own calibration. [Training inputs and weights](../../../src/empirical_isopach/assets/directional-calibration.json) · [Independent numerical checks](verification.json)",""]
    (output/"README.md").write_text("\n".join(lines))
    fig,ax=plt.subplots(figsize=(8.4,4.8),layout="constrained")
    for i,r in enumerate(rows):
        ax.scatter(r["median"],i,s=35,color="#176c76" if r["median_passed"] else "#b7612a")
        ax.annotate(f"{r['median']:,.2f}",(r["median"],i),xytext=(6,0),textcoords="offset points",va="center",fontsize=8)
    ax.set(xscale="log",yticks=range(len(rows)),yticklabels=[r["name"] for r in rows],
           xlabel="Integral of median thickness over Earth (km³)",title="Fixed-direction ensemble")
    ax.set_xlim(min(r["median"] for r in rows)/1.5,max(r["median"] for r in rows)*2)
    ax.grid(axis="x",alpha=.2)
    ax.invert_yaxis()
    fig.savefig(output/"volumes.png",dpi=155)
    plt.close(fig)



def case_table(cases=None, published_root=Path("eruptions")):
    """Read the paired controls and both holdout designs from the same fitted report."""
    records = []
    for rec in datasets():
        if cases is None and "directional" not in published_methods(rec):
            continue
        slug = rec["slug"]
        edition = "provided study"
        if cases is None:
            rec, edition = control_edition(rec, "azimuth/report.json", published_root.parent)
        path = (cases/slug/"report.json" if cases is not None else
                published_root.parent/result_home(rec)/"directional/azimuth/report.json")
        r = json.loads(path.read_text())
        if cases is None and r["input"]["input_sha256"] != digest(rec["path"]):
            raise ValueError(f"{slug}: published directional fit and accepted input differ")
        v = r["volume"]
        row = dict(eruption=r["dataset"],slug=slug,input_sha256=r["input"]["input_sha256"],input_edition=edition,accepted=len(r["data_rows"]),
                   fixed_map_km3=v["matched_control_map_km3"],uncertain_map_km3=v["median_map_km3"],
                   **{"joint_"+k:v["coherent_volume_km3"][k] for k in ("lower95","upper95")})
        for design in ("geographic","quadrants"):
            for subset in ("all_estimated","passed_only"):
                score = r["validation"][design][subset]
                for key in ("folds","rows","control_score","revised_score"):
                    row[design+"_"+subset+"_"+key] = score[key] if score else None
        # These are the paired uncertain-azimuth controls, all from this report.
        # The principal multiscale maps and finite-spread sensitivities have their
        # own tables; borrowing a newer deposit receipt mixes model/input editions.
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
        current = result_home(next(r for r in datasets() if r["slug"] == a.slug))
        path = current/"directional/deposit.json"
        mapped = json.loads(path.read_text())
        rows.append(f"| [{a.eruption}](../../{current}/directional/README.md) | {a.accepted} | **{mapped['median_map_volume_km3']:,.3g}** | {a.fixed_map_km3:,.1f} | {a.joint_lower95:,.1f}–{a.joint_upper95:,.1f} |")
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
