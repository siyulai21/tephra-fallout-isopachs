"""Publish and verify the held-out B-Tm thinning-rate comparison."""
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from benchmarks.btm_source_sensitivities import local_summaries, samjiyon_xy
from benchmarks.btm_thinning_fit import VARIANTS, warp
from benchmarks.btm_thinning import audit_input
from src.empirical_isopach.core.distributions import LogMixture
from src.empirical_isopach.models.directional.geometry import Geometry, polar
from src.empirical_isopach.models.directional.study import digest, write_json
from src.empirical_isopach.models.mass_mixture.study import audited_data

LABELS={"smooth":"Published smooth reference","one_break":"One shared break","two_breaks":"Two shared breaks",
        "exponential":"Exponential kernels","broken_exponential":"Exponential kernels, one break","wide_prior":"One break, wider slope prior"}
HOME=Path("eruptions/b-tm/results")
PUBLISHED=HOME/"source-audit/thinning"


def read(path):
    return json.loads(path.read_text())


def near_sites():
    path=audit_input()
    frame=pd.read_csv(path)
    data,meta=audited_data(path)
    g=Geometry(data,meta["vent_lonlat"][1])
    xy=samjiyon_xy(meta)[:1]
    radius,bearing=polar(data.xy,g.axis)
    r0,a0=polar(xy,g.axis)
    _,_,w=g.locate(xy)
    rows=[]
    for i in np.flatnonzero(radius<=2*r0[0]):
        entry=frame.iloc[int(data.row_ids[i])-2]
        rows.append({"row_id":int(data.row_ids[i]),"evidence_id":entry.EvidenceID,"site":entry.Site,
                     "kind":entry.Kind,"thickness_or_bound_cm":float(np.exp(data.log_value[i])),
                     "east_km":data.xy[i,0],"north_km":data.xy[i,1],"radius_km":radius[i],
                     "bearing_degrees":np.rad2deg(bearing[i]),"distance_from_workbook_km":np.linalg.norm(data.xy[i]-xy[0])})
    return pd.DataFrame(rows).sort_values("distance_from_workbook_km"),xy[0],float(r0[0]),float(np.rad2deg(a0[0])),float(w.sum())


def plot(output, rays, sites, xy, radius):
    fig,(ax,plan)=plt.subplots(1,2,figsize=(10.8,4.5),gridspec_kw={"width_ratios":[1.3,1]})
    colors={"smooth":"#35424c","one_break":"#007872","two_breaks":"#a46330"}
    for k,color in colors.items():
        a=rays[k]
        ax.plot(a.radius_km,a.field_median_cm,color=color,label=LABELS[k],lw=2)
        if k=="one_break":
            ax.fill_between(a.radius_km,a.field_lower95_cm,a.field_upper95_cm,color=color,alpha=.12,
                            label="One-break 95% field interval")
    ax.axvline(radius,color="#7a7471",lw=.7,ls=":")
    ax.scatter([radius,radius],[160,700],s=34,facecolors="white",edgecolors="#902954",zorder=3)
    for y,label in ((160,"1.6 m"),(700,"7 m")):
        ax.annotate(label,(radius,y),xytext=(6,1),textcoords="offset points",color="#89294f",fontsize=8)
    ax.set(yscale="log",xlabel="Distance from the vent along the Samjiyon bearing (km)",
           ylabel="Thickness (cm, log scale)",title="Thickness along the southeast ray")
    ax.grid(axis="y",alpha=.16);ax.legend(fontsize=8,loc="upper right",frameon=False)
    for kind,marker,label in (("measurement","o","Measured thickness"),("lower_limit","^","Component lower bound")):
        p=sites[sites.kind==kind]
        plan.scatter(p.east_km,p.north_km,marker=marker,s=40,label=label,color="#19736f")
    plan.plot([0,xy[0]*2],[0,xy[1]*2],color="#8a3754",ls="--",lw=1)
    plan.scatter([0],[0],s=72,marker="*",color="#27333b",label="Vent")
    plan.scatter(*xy,s=66,marker="x",lw=2,color="#8a3754",label="Workbook location")
    plan.axhline(0,color="#BBB",lw=.5);plan.axvline(0,color="#BBB",lw=.5)
    plan.set(aspect="equal",xlabel="Easting from vent (km)",ylabel="Northing from vent (km)",
             title="Accepted sites within twice the target radius")
    plan.legend(fontsize=8,loc="lower left",frameon=False)
    fig.tight_layout();fig.savefig(output/"thinning.png",dpi=165,bbox_inches="tight");plt.close(fig)


def publish(work, output):
    from importlib.metadata import version
    output.mkdir(parents=True,exist_ok=True)
    sites,xy,radius,angle,mesh_weight=near_sites()
    sites.to_csv(output/"nearby-sites.csv",index=False)
    runs={k:read(work/k/"result.json") for k in VARIANTS}
    rays={k:pd.read_csv(work/k/"ray.csv") for k in VARIANTS}
    pd.concat([v.assign(variant=k) for k,v in rays.items()],ignore_index=True).to_csv(output/"rays.csv",index=False)
    held=pd.concat([pd.read_csv(work/k/"heldout.csv") for k,v in runs.items() if v["folds"]],ignore_index=True)
    held.to_csv(output/"heldout.csv",index=False)
    scores=held.groupby(["variant","fold"]).negative_log_score.mean().unstack()
    summaries=[]
    draws={}
    for k,run in runs.items():
        info=run["full_fit"]
        v=info["samjiyon"][0]
        proximal=held[(held.variant==k)&(held.radius_km<=2*radius)]
        row={"variant":k,"full_sampling_pass":info["passed"],"field_median_cm":v["field_cm"]["median"],
             **{tag+"_cm":v["local_prediction_cm"][tag] for tag in ("lower95","median","upper95")},
             "p_above_700cm":v["local_prediction_probability_above_700cm"],
             "equal_fold_nll":float(scores.loc[k].mean()) if k in scores.index else None,
             "proximal_nll":float(proximal.negative_log_score.mean()) if len(proximal) else None,
             "flagged_folds":[f["fold"] for f in run["folds"] if not f["passed"]]}
        summaries.append(row)
        with np.load(work/k/"local-draws.npz") as archive:
            draws.update({k+"__"+name:archive[name] for name in archive})
    np.savez_compressed(output/"local-draws.npz",**draws)
    pd.DataFrame(summaries).to_csv(output/"summary.csv",index=False)
    write_json(output/"fits.json",runs)
    plot(output,rays,sites,xy,radius)
    lines=["# Does segmented thinning explain the Samjiyon discrepancy?","",
           "The principal fit and every comparison here use the same 68 audited B-Tm observations. All four disputed Samjiyon thickness interpretations and the other DPRK locality summaries are withheld. The experiment predicts at the workbook position before considering either the 1.6 m or the 7 m account. [Input](../../../../../data/eruptions/B-Tm_reviewed.csv) · [Field evidence](../../../../../data/audits/B-Tm/samjiyon.md)","",
           "## What the published curve allows","",
           "In the directional model, a short-range component and a longer, bearing-dependent component contribute to thickness. Their relative contributions change with distance. Even when both kernels are exponential, their sum can change semi-log slope smoothly. The usual kernels also allow the distal slope to flatten. The standard model has no explicit change point. Inside supported triangles its field can depart from this background through the measured nodes. [Model equations](../../../../../docs/methods/directional-continuation.md)","",
           f"The workbook position is {radius:.1f} km from the vent at {angle:.1f}°. It lies in the interpolation transition zone. Local nodes contribute {mesh_weight:.2%} of the log-field weight and the radial background contributes {1-mesh_weight:.2%}. There are {len(sites)} accepted sites within {2*radius:.1f} km of the vent. Their bearings span {sites.bearing_degrees.min():.0f}–{sites.bearing_degrees.max():.0f}°. The nearest, {sites.iloc[0].site}, lies {sites.iloc[0].distance_from_workbook_km:.1f} km from the workbook location. [Nearby sections, measurements and bounds](nearby-sites.csv)","",
           "The classical segmented-exponential relation uses log thickness and the square root of enclosed isopach area. A source-to-site radial transect is a different quantity. This deposit has no accepted complete directional isopach band from which to identify an observed area-space breakpoint. The comparison here tests the consequence a radial slope change would have at the disputed site. [Pyle (1989)](https://doi.org/10.1007/BF01086757) · [Fierstein and Nathenson (1992)](https://pubs.usgs.gov/publication/70017275) · [Bonadonna et al. (1998)](https://doi.org/10.1016/S0377-0273(98)00007-9)","",
           "Yang and colleagues tested this distinction on B-Tm itself. With the Horn and Schmincke isopachs, they obtained 32.99 km³ using one exponential segment and 26.69 km³ using two. Using their contour-by-contour integration, the Horn and Schmincke areas gave 26.66 km³ and the second set of reconstructed areas gave 62.48 km³. The change of mapped footprint produced the larger difference in that comparison. These bulk ash calculations include a proximal contribution inferred from distal contours. The authors estimated proximal tephra separately when assembling the total eruption volume. Their selected contour sets share source evidence with this audit and do not provide independent validation of the 54.0 km³ directional median-map estimate. [Yang et al. (2021), pp. 4–6 and Table 2](https://doi.org/10.1007/s00445-021-01487-8)","",
           "## Predictions and geographic validation","",
           "| Radial model | Local median, cm | Conditional 95% local interval, cm | Probability above 700 cm | Equal-fold negative log score | Proximal negative log score | Folds with sampling warnings |",
           "|---|---:|---:|---:|---:|---:|---|"]
    for v in summaries:
        score="—" if v["equal_fold_nll"] is None else f"{v['equal_fold_nll']:.3f}"
        prox="—" if v["proximal_nll"] is None else f"{v['proximal_nll']:.3f}"
        flags="Full fit only" if v["equal_fold_nll"] is None else ", ".join(map(str,v["flagged_folds"])) or "None"
        lines.append(f"| {LABELS[v['variant']]} | {v['median_cm']:.1f} | {v['lower95_cm']:.1f}–{v['upper95_cm']:.1f} | {v['p_above_700cm']:.2%} | {score} | {prox} | {flags} |")
    d={r["variant"]:r for r in summaries}
    ps=[d[k]["p_above_700cm"] for k in ("one_break","two_breaks","broken_exponential","wide_prior")]
    knot=runs["one_break"]["full_fit"]["parameter_quantiles"]["break_km"][0]
    q=np.exp(runs["one_break"]["full_fit"]["parameter_quantiles"]["log_slope_change"][0])
    deltas=scores.loc["one_break"]-scores.loc["smooth"]
    complete=sum(v["full_fit"]["passed"] for v in runs.values())
    full_status = "All six full-data fits passed" if complete==6 else f"{complete} of six full-data fits passed"
    lines += ["",f"{full_status} the sampling screen. Across the explicit-break alternatives, the probability of a new local observation above 7 m is {min(ps):.2%}–{max(ps):.2%}. The one-break radius has median {knot[1]:.1f} km and conditional 95% interval {knot[0]:.1f}–{knot[2]:,.0f} km. Its slope multiplier has median {q[1]:.2f} and interval {q[0]:.2f}–{q[2]:.2f}. The observations allow both steeper and gentler post-break thinning. These shared radial changes leave the 7 m account well into the predictive tail at the workbook position. A localized southern lobe or changes in phase geometry are outside this test. Local accumulation, an uncertain outcrop position or a different stratigraphic unit also remain possible.","",
           f"The one-break model’s change in negative log score relative to the published smooth reference was {', '.join(f'{v:+.3f}' for v in deltas)} in geographic folds 0–3. A negative change favours the new curve. All eight proximal observations lie in one held-out region. Four regional scores and a single proximal group give limited evidence for choosing a thinning law. The principal six-structure ensemble has its own calibrated weights and separate estimate. [All held-out scores](heldout.csv) · [Full and map-packet predictions from the principal ensemble](../samjiyon-prediction.json)","",
           "![Radial comparison and locations of nearby accepted measurements](thinning.png)","",
           "## Effect on global volume","",
           "The same full-data draws were integrated over Earth for all six curves. The results give the integral of each pointwise median map, the median and conditional interval of whole-deposit volumes, and independent integration checks. [Global volume sensitivity to the thinning law](volumes/README.md)","",
           "## Experimental design","",
           "The comparison preserves the reference fan width, training-derived directions, two deposition scales, mesh, Gaussian local effects and censored observation likelihood. Distance along each ray uses the source-centred authalic chord from the published geometry. A continuous change in effective radial distance gives each segment a positive slope. For one breakpoint at distance b, the effective distance is min(r,b) + q max(r−b,0). Thus q=1 reproduces the original radial curve. The slope can steepen or flatten. With exponential kernels each component is exponential within each segment. Both break distances are shared across bearings in the two-break model. These are tests of one directional structure, with all other assumptions held fixed. No experimental volume is substituted for the published ensemble.","",
           "Ordered breaks have a uniform prior on log radius between the nearest and farthest positive training-site radii. These endpoints are recomputed in each holdout. Independent increments to log slope have zero-mean normal priors with standard deviation log(2), or log(4) in the wider full-data sensitivity. Posterior break positions and slope changes are integrated over. Four chains and the reference retry thresholds are used throughout. Local prediction intervals include residual site scatter. A field interval in the figure describes only smooth thickness at that location. [Posterior diagnostics, break ranges and training IDs](fits.json) · [Local posterior draws](local-draws.npz) · [Transect values](rays.csv)","",
           "From the repository root, repeat the experiment in a fresh directory.","","\x60\x60\x60sh",
           "for v in smooth one_break two_breaks exponential broken_exponential wide_prior; do",
           "  python -m benchmarks.btm_thinning --work results/btm-break-check --variant \"$v\"",
           "done","python -m benchmarks.btm_thinning_report --work results/btm-break-check --output results/btm-break-report","\x60\x60\x60",""]
    (output/"README.md").write_text("\n".join(lines))
    write_json(output/"design.json",{"input_sha256":runs["smooth"]["principal_input_sha256"],
        "nearby_radius_km":2*radius,"workbook_mesh_weight":mesh_weight,
        "versions":{k:version(k) for k in ("numpy","numpyro","jax","jaxlib","scipy","arviz","shapely")},
        "fit_code_sha256":digest(Path(__file__).with_name("btm_thinning_fit.py")),
        "all_files":{p.name:digest(p) for p in sorted(output.iterdir()) if p.is_file() and p.name!="design.json"}})


def verify(folder=PUBLISHED, reference=HOME/"directional/report.json"):
    design=read(folder/"design.json")
    assert design["input_sha256"]==digest(audit_input())
    assert design["fit_code_sha256"]==digest(Path(__file__).with_name("btm_thinning_fit.py"))
    for name,h in design["all_files"].items():
        assert digest(folder/name)==h,name
    scores=pd.read_csv(folder/"heldout.csv")
    fits=read(folder/"fits.json")
    ref=read(reference)["validation"]["geographic"]
    published=pd.DataFrame({"row_id":ref["row_ids"],"fold":ref["folds"],"log_score":ref["predictions"]["log_score"]}).set_index("row_id")
    smooth=scores[scores.variant=="smooth"].set_index("row_id").loc[published.index]
    np.testing.assert_allclose(smooth.negative_log_score,-published.log_score,rtol=1e-8,atol=1e-10)
    np.testing.assert_array_equal(smooth.fold,published.fold)
    assert set(fits)==set(VARIANTS)
    near,_,_,_,_=near_sites()
    assert set(near.evidence_id)=={"old-002","old-008",*(f"costa-P{i}" for i in range(78,84))}
    assert set(smooth.loc[near.row_id,"fold"])=={0}
    with np.load(folder/"local-draws.npz") as archive:
        for k,v in fits.items():
            assert v["principal_input_sha256"]==design["input_sha256"]
            group=scores[scores.variant==k]
            if k!="wide_prior":
                assert set(group.row_id)==set(published.index) and len(group)==68 and len(v["folds"])==4
                for current,frozen in zip(v["folds"],ref["fits"],strict=True):
                    for field in ("training_row_ids","withheld_row_ids"):
                        assert current[field]==frozen[field]
            mean=archive[k+"__means"]; var=archive[k+"__variances"]
            assert mean.shape==var.shape==(4,v["full_fit"]["attempts"][-1]["draws_per_chain"],3)
            n=mean.shape[0]*mean.shape[1]
            summary=local_summaries(LogMixture(np.full(n,1/n),mean.reshape(n,3),var.reshape(n,3)))
            for row,saved in zip(summary,v["full_fit"]["samjiyon"],strict=True):
                for stat in ("lower95","median","upper95"):
                    np.testing.assert_allclose(row["local_prediction_cm"][stat],saved["local_prediction_cm"][stat])
                np.testing.assert_allclose(row["local_prediction_probability_above_700cm"],saved["local_prediction_probability_above_700cm"],atol=1e-11)
    radii=np.array([0.,9.999999,10.,10.000001,20.,25.])
    np.testing.assert_allclose(warp(radii,np.array([10.,20.]),np.zeros(2)),radii)
    np.testing.assert_allclose(warp(radii,np.array([10.,20.]),np.log([2.,.25])),[0.,9.999999,10.,10.000002,30.,32.5])


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--reference",type=Path,default=HOME/"directional/report.json",
                        help="Published smooth single-structure report for the same 68 observations")
    args=parser.parse_args()
    publish(args.work,args.output)
    verify(args.output,args.reference)
