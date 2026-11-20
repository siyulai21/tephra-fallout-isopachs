"""Reader-facing principal spherical results with numerical and geological limits."""
from pathlib import Path
import numpy as np
import pandas as pd

from .study import read


def show(q):
    return f"{q['median']:,.1f} [{q['lower95']:,.1f}–{q['upper95']:,.1f}]"


def case_page(output,r):
    output=Path(output)
    fit=r["fits"]["baseline"]
    vol=fit["volume"]
    ref=fit["parent_volume"]
    val=r["validation"]
    b=r["bearing"]
    rows=[f"# {r['name']} spherical-mixture fallout","",
          f"**{vol['median']:,.1f} km³ of bulk fallout** is the principal whole-Earth volume. The conditional 95% interval is **{vol['lower95']:,.1f}–{vol['upper95']:,.1f} km³**. Each simulated deposit carries one fan direction through the unsampled ground. The map shows median thickness at each location. The volume and its interval summarize the integrals of complete deposits.","",
          "![Principal spherical mixture with uncertain continuation across unsampled bearings](field.png)","",
          f"The underlying spherical mixture gave {show(ref)} km³. The extension leaves thickness at each accepted site and inside supported interpolation triangles unchanged. Beyond finite measured bearings it averages three assumptions with equal weight. A deposit can retain the original broad field, or continue preferentially along one broad or narrower fan. A site carrying only an upper limit constrains its own location and supplies no positive ray.","",
          "[Near-observation field](local.png) · [Source region](source.png) · [Whole-Earth map](world.png) · [Sector-by-sector isopach areas](bearing/sector-areas.csv) · [All crossings along eight bearings](bearing/ray-crossings.json)","",
          "## What the unobserved directions contribute","",
          "| Continuation assumption | Whole-deposit median [conditional 95%], km³ |","|---|---:|"]
    labels=("Broad parent continuation","Unknown fan, 90° half-thickness angle","Unknown fan, 47° half-thickness angle")
    for p,label in enumerate(labels):
        power=(0,1,4)[p]
        rows.append(f"| {label} | {show(fit['bearing_width_sensitivities_km3'][str(power)])} |")
    rows += [f"| Enhanced narrow hidden fan, separate sensitivity | {show(fit['enhanced_hidden_fan_km3'])} |","",
        "The three principal assumptions allow local thickness up to the parent prediction in the same draw. The enhanced-fan sensitivity allows extra ash along a hidden axis. Changing the angular spread of measured reach also changes the inferred mass.","",
        "| Angle at which measured reach halves | Whole-deposit median [conditional 95%], km³ | Numerical checks |","|---|---:|---|"]
    for half,alt in sorted(b["reach_sensitivity"].items(),key=lambda t:float(t[0])):
        ok=alt["checks"]["spatial_passed"] and alt["checks"]["draw_split_passed"]
        rows.append(f"| {float(half):g}° | {show(alt['volume'])} | {'Passed' if ok else 'Warning'} |")
    part=r["mass_partition"]["parts"]
    rows+=["",f"The median fraction inside the screened measurement footprint is **{part['local_footprint']['fraction']['median']:.1%}**. The median fraction beyond the farthest exact site is **{part['beyond_farthest_site']['fraction']['median']:.1%}**. Other mass lies in the intervening gaps and the unmeasured proximal cap. These are sampling classes.","",
           "## Prediction and uncertainty","",
           f"The equal-region held-out log score is **{val['metrics']['equal_region_log_score']:.3f}**, compared with {b['parent_geographic_score']:.3f} for the original geometry. Lower scores indicate better predictions. The 95% predictive interval covered **{val['metrics']['coverage95_observations']:.1%}** of exact held-out thicknesses. Every buffered training fold builds its own bearing support. Censored sites contribute their reported limit events to the score.",""]
    df=pd.read_csv(output/"heldout_predictions.csv")
    far=df[df.gap_exposure>=.5]
    if len(far):
        rows.append(f"Across {len(far)} held-out readings with gap exposure of at least 0.5, mean negative log score was {-far.log_score.mean():.3f} for the principal field and {-far.log_score_power_0.mean():.3f} for its parent. This subset weights individual sites equally.")
    else:
        rows.append("No held-out reading had gap exposure of at least 0.5. These folds supply no direct test in that stratum.")
    screen=read(output/"complete-contour-audit.json")
    standard=screen["policies"].get("standard",{})
    rows+=["","The shared contour audit "+("identified a qualifying standard-policy band." if standard.get("selected_band") else
            "identified no qualifying standard-policy band." if screen["status"]=="completed" else "could not evaluate the entire audit window within the physical chart.")+
          " [Complete decisions and geographic deletions](complete-contour-audit.json)","",
          "| Spherical prior | Whole-deposit median [conditional 95%], km³ | Numerical checks |","|---|---:|---|"]
    for name,item in r["fits"].items():
        d=item["diagnostics"]
        warn=[msg for key,msg in (("parent_sampler_passed","parent sampler"),("bearing_integration_passed","integration"),
              ("bearing_volume_split_passed","volume draw split")) if not d[key]]
        rows.append(f"| {name.replace('_',' ')} | {show(item['volume'])} | {', '.join(warn)+' warning' if warn else 'Passed'} |")
    check=b["map_draw_check"]
    warnings=check["history"][-1]["warnings"]
    if warnings:
        details=", ".join(f"{v['sector']} {v['level_cm']:g} cm ({v['split_symmetric_difference_fraction']:.0%})" for v in warnings)
        rows += ["",f"At {check['draws_per_chain']} parent draws per chain, the following resolved contour areas still differed by at least 10% between draw halves: {details}."]
    else:
        rows += ["",f"The resolved 1, 10 and 100 cm contour areas passed the split-draw test at {check['draws_per_chain']} parent draws per chain. Sparsely represented or empty contour unions remain unresolved."]
    failures=[str(f["fold"]+1) for f in val["fits"] if not f["fit"]["diagnostics"]["passed"]]
    rows+=["",("Geographic folds with sampling warnings (numbered from one) include "+", ".join(failures)+"." if failures
                else "All four geographic training fits passed their sampling checks."),
          "The tail and overall scale can remain weakly identified even when numerical checks pass. The reference-amplitude prior induces a prior on retained volume. Reworking, incomplete sampling and source attribution can add uncertainty beyond these conditional intervals.","",
          "[Prior sensitivity and mass partition](volume.png) · [Derived-volume chains](sampling.png) · [Numerical report](report.json) · [Held-out sites](heldout_predictions.csv) · [Whole-deposit draws](volume_draws.csv.gz) · [Joint sector and support-class volumes](bearing/baseline.npz) · [Original spherical fit and all attempts](bearing/parent-report.json)",""]
    (output/"README.md").write_text("\n".join(rows))
