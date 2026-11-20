"""Publish matched empirical, known-volume, and hidden-fan comparisons."""
import argparse
import gzip
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.empirical_isopach.core.catalog import datasets
from src.empirical_isopach.models.mass_mixture.bearings.study import read


def summarize(trials,output,cases=Path("eruptions")):
    output.mkdir(parents=True,exist_ok=True)
    packets=[]
    if trials:
        for p in sorted(trials.glob("*/*.json")):
            packets.append({**read(p),"geometry":p.parent.name})
        if len(packets)!=24:
            raise ValueError("The two registered six-design, two-seed suites require 24 fitted deposits")
        (output/"known-deposits.json.gz").write_bytes(gzip.compress(json.dumps(packets,separators=(",",":"),allow_nan=False).encode(),mtime=0))
    else:
        archive=output/"known-deposits.json.gz"
        if not archive.exists():
            from shutil import copyfile
            published=Path(__file__).resolve().parents[1]/"comparisons/spherical-mixture/bearings/known-deposits.json.gz"
            copyfile(published,archive)
        with gzip.open(archive,"rt") as stream:
            packets=json.load(stream)
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
        home=cases/rec["slug"]/"results/spherical-mixture"
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
    rows=["# Spherical isopachs in unsampled directions","",
      "Every eruption uses the same continuation of its fitted spherical kernels. The source and measured thicknesses retain their likelihood. The central map allows an unobserved fan to extend in different directions across simulated deposits. A broad continuation remains possible in one third of the mixture. The first column below is the earlier spherical median under the same observations.","",
      "| Eruption | Earlier → principal volume (km³) | Principal conditional 95% (km³) | Geographic score, earlier → principal | Exact-site 95% coverage |",
      "|---|---:|---:|---:|---:|"]
    for r in real.itertuples():
        rows.append(f"| [{r.eruption}](../../../eruptions/{r.slug}/results/spherical-mixture/README.md) | {r.parent_volume_km3:,.1f} → **{r.principal_volume_km3:,.1f}** | {r.lower95:,.1f}–{r.upper95:,.1f} | {r.parent_score:.3f} → {r.principal_score:.3f} | {r.coverage:.1%} |")
    rows+=["",f"Averaging geographic scores equally across eruptions gives {real.parent_score.mean():.3f} → {real.principal_score.mean():.3f}. Lower is better. The matched predictions retain all limit events and numerical warnings. {real.n_high_exposure.sum()} withheld readings had gap exposure of at least 0.5.{sparse_text} [Paired results, including poorly supported held-out sites](paired-cases.csv)","",
       "## Where the map changes","",
       "![Earlier and principal spherical ten-centimetre isopachs for all six eruptions](paired-isopachs.png)","",
       "Each panel zooms to the same thickness threshold. Tan shading identifies ground where only the earlier median reaches 10 cm. Blue shading identifies ground where both medians reach it. Only sites inside each frame are visible. The area calculations below cover the entire Earth.","",
       "| Eruption and compass sector | Area ≥10 cm, earlier → principal (km²) | Split-draw area disagreement |","|---|---:|---:|"]
    for slug,name,sector in (("aso-4","Aso-4","S"),("aso-4","Aso-4","SE"),("aso-4","Aso-4","W"),
                             ("b-tm","B-Tm","W"),("mazama","Mazama","SW")):
        a=pd.read_csv(cases/slug/"results/spherical-mixture/bearing/sector-areas.csv")
        group=a[(a.sector==sector)&(a.level_cm==10)]
        union=group.split_union_km2.sum()
        effective=union**2/group.split_union_area_squared_weights.sum() if union else 0.
        split=(f"{group.split_disagreement_km2.sum()/union:.1%}" if effective>=64 else "Unresolved")
        rows.append(f"| {name}, {sector} | {group.reference_km2.mean():,.0f} → {group.principal_km2.mean():,.0f} | {split} |")
    rows+=["","These are modelled areas in 45° sectors, averaged over two scrambled whole-Earth integration grids. The sites do not establish the actual deposit edge. The downloadable tables give both grid results, contour-area split checks and all eight bearings at 1, 10 and 100 cm.","",
       "## Deposits with known volumes","",
       f"The existing spherical tests contain 24 fits across two kernel geometries, six deposit designs and two seeds per geometry. On the 24 original truths, the earlier and principal conditional intervals contained the true volume in **{int(original.loc['reference','covers_truth'])}/24 and {int(original.loc['principal','covers_truth'])}/24** fits, respectively. Mean absolute log volume error was **{original.loc['reference','absolute_log_volume_error']:.3f} → {original.loc['principal','absolute_log_volume_error']:.3f}**. A smaller footprint does not guarantee a better total.","",
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
       "Four missing-quadrant fits also have a second compatible truth. Within an 80 km disk centred 110 km east and north of the source, the hidden fan can be substantially weaker while every sampled thickness is identical. Both truths and both changes in error are retained. Diffuse tails and genuine missing lobes remain consequential for a global estimate. [Known volumes and failures](known-volumes.csv) · [Contour errors for every threshold and truth](known-areas.csv) · [Both quadrature replicates and numerical refinements](known-deposits.json.gz)","",
       "[Equations, model assumptions and reproduction](../../../docs/methods/spherical-bearings.md) · [All prior sensitivities](../README.md) · [Preceding release](../../../archive/2026-11-20-spherical-continuation/README.md)",""]
    (output/"README.md").write_text("\n".join(rows))
    from .spherical_bearing_figure import draw
    draw(output,cases)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--trials",type=Path,help="Directory holding displaced and source trial folders. By default replay the published gzip packet.")
    p.add_argument("--output",type=Path,default=Path("comparisons/spherical-mixture/bearings"))
    args=p.parse_args()
    summarize(args.trials,args.output)
