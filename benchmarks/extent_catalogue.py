"""Refresh the three family indexes and matched catalogue table from the principal case reports."""
import argparse
from pathlib import Path
import re

import pandas as pd

from .azimuth_summary import case_table
from .nested_summary import comparison_plot
from .principal_volumes import catalogue_rows
from src.empirical_isopach.core.catalog import datasets
from src.empirical_isopach.visualization.deposit import FAMILIES,digest,read_json,write_json
from src.empirical_isopach.visualization.principal import write_rows


def number(v):
    return f"{v:,.0f}" if v>=1000 else f"{v:,.3g}"


def summary(output):
    def target(path):
        file=output/path
        file.parent.mkdir(parents=True,exist_ok=True)
        return file
    rows=catalogue_rows()
    write_rows(target("comparisons/principal-volumes.csv"),rows)
    table=pd.DataFrame(rows).set_index(["slug","method"])
    case_table().to_csv(target("comparisons/directional/case-comparison.csv"),index=False)
    nd=[]
    sph=pd.read_csv("comparisons/spherical-mixture/comparison.csv").set_index("slug")
    sph=sph.rename(columns={"median_fraction_outside_local_footprint":"parent_median_fraction_outside_local_footprint"})
    src=read_json("comparisons/spherical-mixture/report.json")
    intro={
        "spherical-mixture":"The spherical mixture limits how much a displaced component can rise above its sampled flank. The two continuations retain different crosswind spreads. A separate interval for each of eight prior fits shows how much uncertainty comes from the assumptions about the kernels.",
        "directional":"The uncertain-azimuth ensemble smooths the fitted fan ranges between sampled bearings. Its six structures retain coherent fan azimuths in each draw. The original fixed-direction and uncertain-direction fits are available in each case’s diagnostic sections.",
        "nested-isopachs":"Finite corridors widen more slowly along an isolated transect, with two allowances for an unsampled turning ridge. The fitted whole-area alternative remains part of this family’s uncertainty."}
    extra={
        "spherical-mixture":"[Prior and geometry comparison](comparison.csv) · [Earlier bearing tests on known deposits](bearings/known-volumes.csv) · [Underlying spherical likelihood](../../docs/methods/probabilistic-global.md)",
        "directional":"[Geographic and quadrant predictions](case-comparison.csv) · [Fixed-direction ensemble](fixed-ensemble/README.md) · [Measured fan study](measured-fans/README.md)",
        "nested-isopachs":"[Three-family volume table](cases.csv) · [Original geographic reach experiment](../nested-reach/README.md) · [Independent field surveys](../nested-reach/field-surveys/README.md)"}
    for fam,title in FAMILIES.items():
        lines=[f"# {title} across the eruption catalogue","",intro[fam],"",
               "[Model and uncertainty](../../docs/methods/finite-spread.md) · [Paired maps and withheld-site tests](../global-extent/README.md)","",
               "| Eruption | Median-map bulk volume (km³) | Conditional 95% interval (km³) | Held-out score ↓ | Exact coverage |",
               "|---|---:|---:|---:|---:|"]
        for rec in sorted(datasets(),key=lambda r:r["slug"]):
            slug=rec["slug"]; row=table.loc[(slug,fam)];home=Path("eruptions")/slug/"results"/fam
            report=read_json(home/"extent/report.json")
            lo,hi=report["conditional_interval95_km3"]
            flag="" if report["draw_checks_passed"] else " · provisional"
            val=report["validation"]["geographic"]
            warning="" if val["all_folds_passed"] else " †"
            lines.append(f"| [{rec['name']}](../../eruptions/{slug}/results/{fam}/README.md) | **{number(row.median_map_volume_km3)}** | {number(lo)}–{number(hi)}{flag} | {row.heldout_log_score:.3f}{warning} | {row.heldout_coverage95:.1%} |")
            nd.append(dict(eruption=rec["name"],slug=slug,method={"directional":"Directional","spherical-mixture":"Spherical","nested-isopachs":"Nested corridor"}[fam],
                           lower95_km3=lo,median_map_km3=row.median_map_volume_km3,upper95_km3=hi,
                           equal_region_log_score=row.heldout_log_score,input_sha256=report["input_sha256"]))
            if fam=="spherical-mixture":
                priors=read_json(home/"extent/prior-sensitivity.json")
                updates={"median_map_volume_km3":row.median_map_volume_km3,"lower95":lo,"upper95":hi,
                         "sensitivity_lower":min(v["conditional_interval"][0] for v in priors.values()),
                         "sensitivity_upper":max(v["conditional_interval"][-1] for v in priors.values()),
                         "baseline_sampling_passed":report["draw_checks_passed"],"all_sensitivity_sampling_passed":all(v["passed"] for v in priors.values()),
                         "all_outer_sampling_passed":val["all_folds_passed"],"global_equal_region_negative_log_score":row.heldout_log_score,
                         "global_observation_coverage95":row.heldout_coverage95}
                for k,v in updates.items():sph.loc[slug,k]=v
                next(a for a in src["cases"] if a["slug"]==slug)["extent_report_sha256"]=digest(home/"extent/report.json")
        lines+=["","A dagger marks a geographic fit warning. A provisional interval retains a full-data sampling or integration warning.","",extra[fam],""]
        if fam=="nested-isopachs":
            lines+=["![Three global families with conditional intervals](volumes.png)",""]
        target("comparisons/"+fam+"/README.md").write_text("\n".join(lines))
    nd.sort(key=lambda r:(r["slug"],r["method"]))
    write_rows(target("comparisons/nested-isopachs/cases.csv"),nd)
    comparison_plot(nd,target("comparisons/nested-isopachs/volumes.png"),nested_family="Nested corridor")
    sph.reset_index().to_csv(target("comparisons/spherical-mixture/comparison.csv"),index=False)
    src["estimates"]=sph.reset_index().to_dict(orient="records")
    src["summary_source_sha256"]=digest(Path(__file__))
    src["scope"]="Conditional bulk fallout volume under finite spread. Each prior interval and every geographic fit keeps its own numerical status."
    for m in src["matched_validation"]:
        match=next(row for row in src["estimates"] if row["eruption"]==m["eruption"])
        current=read_json(Path("eruptions")/match["slug"]/"results/spherical-mixture/extent/report.json")
        m["global"]={"equal_region_log_score":match["global_equal_region_negative_log_score"],
                     "coverage95_observations":match["global_observation_coverage95"],
                     "fold_log_scores":current["validation"]["geographic"]["fold_scores"]}
        m["global_outer_sampling_passed"]=match["all_outer_sampling_passed"]
    src["protocol"]="finite-spread-and-angular-resolution-v1"
    from .global_posterior import figures
    figures(target("comparisons/spherical-mixture/comparison.png").parent,src["estimates"])
    write_json(target("comparisons/spherical-mixture/report.json"),src)
    home=Path("eruptions/README.md").read_text()
    for rec in datasets():
        slug=rec["slug"]
        pattern=r"(\| \[[^\n]*?\]\("+re.escape(slug)+r"/README\.md\) \|) [^|]* \| [^|]* \| [^|]* \|"
        v=[number(table.loc[(slug,f),"median_map_volume_km3"]) for f in FAMILIES]
        home,n=re.subn(pattern,lambda m:m[1]+" "+" | ".join(v)+" |",home)
        if n!=1:raise ValueError(f"Expected one catalogue row for {slug}")
    target("eruptions/README.md").write_text(home)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output",required=True,type=Path)
    summary(p.parse_args().output)
