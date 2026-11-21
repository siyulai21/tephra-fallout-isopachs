"""Publish matched empirical, known-volume, and hidden-fan comparisons."""
import argparse
import gzip
import json
from pathlib import Path
from shutil import copyfile

import numpy as np
import pandas as pd

from src.empirical_isopach.core.catalog import datasets
from src.empirical_isopach.models.mass_mixture.bearings.study import read


def build_validation(trials, output, case_dirs, link_prefix=""):
    """Write paired data and the map; return their findings for the family report."""
    output.mkdir(parents=True,exist_ok=True)
    packet_file = output / "known-deposits.json.gz"
    published = Path(__file__).resolve().parents[1] / "comparisons/spherical-mixture/bearings/known-deposits.json.gz"
    if trials is None:
        with gzip.open(published, "rt") as stream:
            packets = json.load(stream)
        if packet_file.resolve() != published:
            copyfile(published, packet_file)
    else:
        from .mass_mixture import SCENARIOS
        expected = {(geometry, scenario, seed)
                    for geometry, seeds in (("displaced", (10, 11)), ("source", (20, 21)))
                    for scenario in SCENARIOS for seed in seeds}
        packets = [{**read(path), "geometry": path.parent.name} for path in sorted(trials.glob("*/*.json"))]
        observed = {(row["geometry"], row["scenario"], row["seed"]) for row in packets}
        if observed != expected or len(packets) != len(expected):
            raise ValueError("The comparison requires all 24 registered geometry, scenario and seed combinations exactly once")
        raw = json.dumps(packets, separators=(",", ":"), allow_nan=False).encode()
        packet_file.write_bytes(gzip.compress(raw, mtime=0))
    estimates,area_rows=[],[]
    for d in packets:
        for name,total in d["volumes_km3"].items():
            for model,q in (("reference",d["reference"]),("principal",d["principal"]["volume"])):
                estimates.append({"geometry":d["geometry"],"scenario":d["scenario"],"seed":d["seed"],"truth":name,
                    "method":model,"truth_km3":total,"median_km3":q["median"],"lower95":q["lower95"],"upper95":q["upper95"],
                    "absolute_log_volume_error":abs(np.log(q["median"]/total)),"covers_truth":q["lower95"]<=total<=q["upper95"],
                    "sampler_passed":d["sampler_passed"],"spatial_passed":d["principal"]["checks"]["spatial_passed"],
                    "draw_split_passed":d["principal"]["checks"]["draw_split_passed"]})
        area_rows += [dict(a,geometry=d["geometry"],scenario=d["scenario"],seed=d["seed"]) for a in d["areas"]]
    t=pd.DataFrame(estimates)
    t.to_csv(output/"known-volumes.csv",index=False)
    areas=pd.DataFrame(area_rows)
    keys=["geometry","scenario","seed","truth","region","level_cm","model"]
    grouped=areas.groupby(keys).agg({"false_area_km2":"mean","missed_area_km2":"mean","intersection_km2":"mean",
        "union_km2":"mean","union_area_squared_weights":"mean"}).reset_index()
    grouped["effective_union_nodes"]=np.divide(2*grouped.union_km2**2,grouped.union_area_squared_weights,
            out=np.zeros(len(grouped)),where=grouped.union_area_squared_weights>0)
    grouped["pair_resolved"]=grouped.groupby(keys[:-1]).effective_union_nodes.transform("min")>=64
    grouped.to_csv(output/"known-areas.csv",index=False)
    real=[]
    for rec in sorted(datasets(),key=lambda r:r["slug"]):
        home=case_dirs[rec["slug"]]
        r=read(home/"report.json")
        v=r["fits"]["baseline"]["volume"]
        old=r["bearing"]["parent_reference_km3"]
        held=pd.read_csv(home/"heldout_predictions.csv")
        hard=held[held.gap_exposure>=.5]
        real.append({"eruption":rec["name"],"slug":rec["slug"],"parent_volume_km3":old["median"],
            "principal_volume_km3":v["median"],"lower95":v["lower95"],"upper95":v["upper95"],
            "parent_score":r["bearing"]["parent_geographic_score"],"principal_score":r["validation"]["metrics"]["equal_region_log_score"],
            "n_high_exposure":len(hard),"high_exposure_parent_score":-hard.log_score_power_0.mean(),
            "high_exposure_principal_score":-hard.log_score.mean(),"coverage":r["validation"]["metrics"]["coverage95_observations"]})
    real=pd.DataFrame(real)
    real.to_csv(output/"paired-cases.csv",index=False)
    sparse=real[real.n_high_exposure>0]
    sparse_text=""
    if len(sparse):
        columns=["high_exposure_parent_score","high_exposure_principal_score"]
        pooled=np.average(sparse[columns],axis=0,weights=sparse.n_high_exposure)
        balanced=sparse[columns].mean().to_numpy()
        sparse_text=(f" Within those readings the site-weighted score was {pooled[0]:.3f} → {pooled[1]:.3f}. "
                     f"Giving each represented eruption equal weight gives {balanced[0]:.3f} → {balanced[1]:.3f}.")
    original=t[t.truth=="original"].groupby("method").agg({"absolute_log_volume_error":"mean","covers_truth":"sum"})
    score_text = (
        f"The geographic score averaged equally across eruptions changed from {real.parent_score.mean():.3f} "
        f"to {real.principal_score.mean():.3f}. Of the withheld readings, {real.n_high_exposure.sum()} "
        f"had gap exposure of at least 0.5.{sparse_text} "
        f"[Paired predictions and gap-exposure scores]({link_prefix}paired-cases.csv)"
    )
    rows=[
       "## Where the map changes","",
       f"![Broad and principal spherical 10 cm isopachs by eruption]({link_prefix}paired-isopachs.png)","",
       "Each panel compares the same 10 cm threshold. Tan shading identifies ground where only the earlier median reaches 10 cm. Blue shading identifies ground where both medians reach it. Only sites inside each frame are visible. The area calculations below cover the entire Earth.","",
       "| Eruption and compass sector | Area ≥10 cm, earlier → principal (km²) | Split-draw area disagreement |","|---|---:|---:|"]
    for slug,name,sector in (("aso-4","Aso-4","S"),("aso-4","Aso-4","SE"),("aso-4","Aso-4","W"),
                             ("b-tm","B-Tm","W"),("mazama","Mazama","SW")):
        if slug not in case_dirs:
            continue
        a=pd.read_csv(case_dirs[slug]/"bearing/sector-areas.csv")
        group=a[(a.sector==sector)&(a.level_cm==10)]
        union=group.split_union_km2.sum()
        effective=union**2/group.split_union_area_squared_weights.sum() if union else 0.
        split=(f"{group.split_disagreement_km2.sum()/union:.1%}" if effective>=64 else "Unresolved")
        rows.append(f"| {name}, {sector} | {group.reference_km2.mean():,.0f} → {group.principal_km2.mean():,.0f} | {split} |")
    rows+=["","These are modelled areas in 45° sectors, averaged over two scrambled whole-Earth integration grids. The sites do not establish the actual deposit edge. The downloadable tables give both grid results, contour-area split checks and all eight bearings at 1, 10 and 100 cm.","",
    ]
    map_section = "\n".join(rows)
    rows = ["## Deposits with known volumes","",
       f"The existing spherical tests contain 24 fits across two kernel geometries, six deposit designs and two seeds per geometry. On the 24 original truths, the earlier and principal conditional intervals contained the true volume in **{int(original.loc['reference','covers_truth'])}/24 and {int(original.loc['principal','covers_truth'])}/24** fits, respectively. Mean absolute log volume error was **{original.loc['reference','absolute_log_volume_error']:.3f} → {original.loc['principal','absolute_log_volume_error']:.3f}**. At 10 cm the principal fit reduced false extension and increased missed fallout in the simulations.","",
       "| Area tested | Threshold | Resolved pairs / 24 | False-positive area, earlier → principal (km²) | Missed fallout area, earlier → principal (km²) |",
       "|---|---:|---:|---:|---:|"]
    for scope in ("Earth","gap exposure >= 0.5"):
        for level in (1.,10.,100.):
            sel=grouped[(grouped.truth=="original")&(grouped.region==scope)&(grouped.level_cm==level)&grouped.pair_resolved]
            label="Whole Earth" if scope=="Earth" else "Gap exposure ≥0.5"
            if sel.empty:
                rows.append(f"| {label} | {level:g} cm | 0 | — | — |")
                continue
            z=sel.groupby("model")[["false_area_km2","missed_area_km2"]].mean()
            pairs=len(sel)//2
            rows.append(f"| {label} | {level:g} cm | {pairs} | {z.at['reference','false_area_km2']:,.0f} → {z.at['principal','false_area_km2']:,.0f} | {z.at['reference','missed_area_km2']:,.0f} → {z.at['principal','missed_area_km2']:,.0f} |")
    warnings=sum(not d["sampler_passed"] for d in packets)
    split=sum(not d["principal"]["checks"]["draw_split_passed"] for d in packets)
    rows+=["",f"Each resolved fit receives equal weight. Both model–truth unions must have at least 64 effective area nodes. Empty or inadequately resolved unions are listed in the CSV. {warnings} parent fits had sampling warnings and {split} bearing-volume estimates retained split-draw warnings. The scores describe these development simulations.","",
       f"Four missing-quadrant fits also have a second compatible truth. Within an 80 km disk centred 110 km east and north of the source, a weaker hidden fan gives identical thickness at every sampled site. The tables retain both truths. [Volume errors and failures]({link_prefix}known-volumes.csv) · [Contour errors by threshold and truth]({link_prefix}known-areas.csv) · [Quadrature replicates and numerical checks]({link_prefix}known-deposits.json.gz)",""]
    from .spherical_bearing_figure import draw
    draw(output, case_dirs)
    return real.set_index("slug"), score_text, map_section, "\n".join(rows)


def summarize(trials, output, cases=Path("eruptions")):
    """Replay the paired validation files separately in a research output directory."""
    case_dirs = {r["slug"]: cases / r["slug"] / "results/spherical-mixture" for r in datasets()}
    _, scores, maps, trials_text = build_validation(trials, output, case_dirs)
    (output / "README.md").write_text(f"# Spherical bearing validation\n\n{scores}\n\n{maps}\n{trials_text}")


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--trials",type=Path,help="Directory holding displaced and source trial folders. By default replay the published gzip packet.")
    p.add_argument("--output",type=Path,default=Path("results/spherical-bearing-comparison"))
    args=p.parse_args()
    summarize(args.trials,args.output)
