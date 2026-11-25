"""Assemble the cross-eruption comparison from completed, published ensemble reports."""
import argparse
import gzip
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .publication import published_datasets as datasets, result_home
from src.empirical_isopach.models.directional.study import digest, write_json
from src.empirical_isopach.models.directional.ensemble.weights import PACKET


def volume_figure(table, output):
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from src.empirical_isopach.visualization.style import STYLE
    tab=table.sort_values("median_map_km3")
    with plt.rc_context(STYLE):
        fig,(volume,support)=plt.subplots(1,2,figsize=(11.8,5.5),layout="constrained",
                                         gridspec_kw={"width_ratios":[2.4,1]},sharey=True)
        for i,r in enumerate(tab.itertuples()):
            volume.plot([r.lower95_km3,r.upper95_km3],[i,i],color="#7966A4",lw=2)
            volume.scatter(r.median_map_km3,i,marker="D",s=42,c="#087D80",zorder=4)
            support.barh(i,100*r.footprint_fraction,height=.44,color="#58A8A3")
            support.text(100*r.footprint_fraction+.8,i,f"{r.footprint_fraction:.1%}",va="center",fontsize=9)
        volume.set(yticks=np.arange(len(tab)),yticklabels=tab.dataset,xscale="log",
                   xlabel="Whole-Earth bulk fallout (km³, logarithmic scale)",title="Median-thickness map integral and uncertainty")
        support.set(xlim=(0,max(15,100*tab.footprint_fraction.max()+11)),xlabel="Share of central volume (%)",
                    title="Inside measured triangles")
        volume.set_ylim(len(tab)-.35,-.65)
        for ax in (volume,support):
            ax.grid(axis="x",alpha=.18)
        fig.suptitle("Directional ensemble across six eruptions",fontsize=15)
        handles=[Line2D([],[],marker="D",color="#087D80",ls="",label="Median-map volume"),
                 Line2D([],[],color="#7966A4",label="Conditional 95% volume interval")]
        fig.legend(handles=handles,loc="outside lower center",ncols=2,frameon=False)
        fig.savefig(output/"volumes.png",dpi=170,bbox_inches="tight")
        plt.close(fig)


def summarize(known, regional, output, cli_replay=None):
    output.mkdir(parents=True,exist_ok=True)
    tables = ("comparison.csv","synthetic/comparison.csv")
    regional_check = {}
    for key in tables:
        actual = digest(regional/key)
        expected = digest(Path("comparisons/regional")/key)
        regional_check[key] = {"rerun_sha256":actual,"published_sha256":expected,"identical":actual==expected}
    if not all(v["identical"] for v in regional_check.values()):
        raise ValueError("Review discrepancies in the matched regional benchmark before publishing")
    cases, trials, contours, full = [], [], [], {}
    for rec in datasets():
        home = result_home(rec)
        folder = home/"directional"/"ensemble"
        r = json.loads((folder/"report.json").read_text())
        c, j = r["median_surface"], r["coherent_fields"]
        geo, q = r["validation"]["geographic"], r["validation"]["quadrants"]
        old = json.loads((home/"directional/single-structure/report.json").read_text())
        m = np.array(c["parts_sectors_km3"])
        v = j["volume_km3"]
        cases.append(dict(slug=rec["slug"], dataset=r["dataset"],input_sha256=r["input"]["input_sha256"],
            median_map_km3=c["volume_km3"],lower95_km3=v["lower95"],upper95_km3=v["upper95"],
            uniform_map_km3=c["weight_sensitivity_km3"]["uniform"],
            footprint_km3=m[0].sum(),footprint_fraction=m[0].sum()/m.sum(),
            geographic_score=geo["equal_region_log_score"],uniform_score=geo["uniform_score"],reference_score=geo["reference_score"],
            geographic_coverage=geo["metrics"].get("coverage95_observations"),quadrant_score=q["equal_region_log_score"],
            geographic_all_estimable_passed=geo["all_estimable_passed"],geographic_scored=geo["scored_rows"],
            rows=geo["total_rows"],quadrant_scored=q["scored_rows"],quadrant_all_estimable_passed=q["all_estimable_passed"],
            full_members_passed=sum(v["passed"] for v in j["members"].values()),median_map_numerical_passed=c["passed"],
            map_draws_per_chain=r["map_draws_per_chain"],supported_bands=sum(b is not None for b in r["complete_bands"].values())))
    for f in sorted(known.glob("*.json")):
        a = json.loads(f.read_text())
        scenario, sd = f.stem.rsplit("-",1)
        full[f.stem] = a
        truth = a["truth"]["total_km3"]
        c, joint = a["median_surface"],a["coherent_fields"]
        v=joint["volume_km3"]
        trials.append(dict(scenario=scenario,seed=int(sd),truth_km3=truth,median_map_km3=c["volume_km3"],
                           relative_error=c["volume_km3"]/truth-1,
                           uniform_map_km3=c["weight_sensitivity_km3"]["uniform"],
                           lower95=v["lower95"],upper95=v["upper95"],covers_truth=v["lower95"]<=truth<=v["upper95"],
                           median_map_numerical_passed=c["passed"],full_members_passed=sum(x["passed"] for x in joint["members"].values()),
                           alternate_truth_km3=a["truth"].get("same_data_hidden_total_km3")))
        contours.extend({"scenario":scenario,"seed":int(sd),**v} for v in a["contour_accuracy"])
    if len(trials)!=14:
        raise ValueError("Expected seven stress designs and two independent seeds per design")
    tab,stress,shapes = map(pd.DataFrame,(cases,trials,contours))
    volume_figure(tab,output)
    tab.to_csv(output/"comparison.csv",index=False)
    stress.to_csv(output/"known-volumes.csv",index=False)
    shapes.to_csv(output/"contour-accuracy.csv",index=False)
    (output/"simulation-records.json.gz").write_bytes(gzip.compress(json.dumps(full,allow_nan=False).encode(),mtime=0))
    signed = int((tab.geographic_score<tab.reference_score).sum())
    wins = int((tab.geographic_score<tab.uniform_score).sum())
    lines = ["# Fixed-direction ensemble calibration","",
        "This study uses the earlier six-structure calibration with fixed fan directions. Its central volume integrates pointwise median thickness over the Earth. The matched control in the principal study instead uses the same structure weights as its uncertain-direction fit. "
        "The 95% volume interval comes from integrating spatially correlated draws. Each of the six combinations of fan width and gap treatment "
        "is fitted to the same accepted observations. The reported eruption contributes no holdout scores to its own predictive weights.", "",
        "![Central isopach volumes, complete-field intervals and measured shares](volumes.png)","",
        "| Eruption | Median-map km³ | Conditional 95% volume interval (km³) | Median map inside measured triangles | Full structures passing | Map checks |",
        "|---|---:|---:|---:|---:|---|"]
    for a in tab.itertuples():
        lines.append(f"| [{a.dataset}](../../../eruptions/{a.slug}/results/directional/ensemble/README.md) | **{a.median_map_km3:,.1f}** | "
                     f"{a.lower95_km3:,.1f}–{a.upper95_km3:,.1f} | {a.footprint_fraction:.1%} | "
                     f"{a.full_members_passed}/6 | {'passed' if a.median_map_numerical_passed else '**review**'} |")
    lines += ["","These are model-conditional bulk fallout estimates. A median map can combine common features from incompatible deposits. The joint-field interval "
              "describes plausible complete-deposit volumes. Thick unmeasured source caps, hidden lobes, distal tails, reworking and preservation can sit outside it.", "",
              "## Geographic thickness prediction","",
              "| Eruption | Ensemble score ↓ | Equal weights ↓ | Single reference ↓ | Exact-site 95% coverage | Geographic rows scored | All fitted geographic members passed |",
              "|---|---:|---:|---:|---:|---:|---|"]
    for a in tab.itertuples():
        lines.append(f"| {a.dataset} | {a.geographic_score:.3f} | {a.uniform_score:.3f} | {a.reference_score:.3f} | {a.geographic_coverage:.1%} | "
                     f"{a.geographic_scored}/{a.rows} | {'yes' if a.geographic_all_estimable_passed else '**review**'} |")
    lines += ["",f"The ensemble improved the buffered geographic score over the single reference in **{signed}/6** cases and over equal weights in **{wins}/6**. "
              f"The equal-eruption mean scores were {tab.geographic_score.mean():.3f}, {tab.uniform_score.mean():.3f} and {tab.reference_score.mean():.3f} for the weighted ensemble, "
              "equal weights and single reference respectively. These are retrospective comparisons of thickness predictions. The real eruptions have no known whole-deposit truth.", "",
              f"The three shared support policies produced {int(tab.supported_bands.sum())} qualifying complete median-map bands across all six cases. "
              "Each case links its full rejection log and both geographic and quadrant deletion tests. "
              "The matched GP/spline real and synthetic benchmark tables reproduced byte for byte.", "",
              "## Known deposits and unseen fans","",
              "| Scenario | True km³ | Median-map km³ by seed (90 / 91) | Conditional interval contains truth | All 6 fitted members passed |",
              "|---|---:|---:|---:|---:|"]
    for scenario,g in stress.groupby("scenario",sort=True):
        g=g.sort_values("seed")
        values=" / ".join(f"{x:,.1f}" for x in g.median_map_km3)
        lines.append(f"| {scenario.replace('_',' ')} | {g.truth_km3.iloc[0]:,.1f} | {values} | {g.covers_truth.sum()}/2 | {(g.full_members_passed==6).sum()}/2 |")
    invisible=stress[stress.scenario=="invisible_sector"]
    other=invisible.alternate_truth_km3
    hidden_ok=((invisible.lower95<=other)&(invisible.upper95>=other)).sum()
    lines += ["",f"The primary simulated total fell within the conditional interval in **{int(stress.covers_truth.sum())}/14** trials. "
              f"The two invisible-sector trials also admit a **{other.iloc[0]:.1f} km³** deposit with identical observations. "
              f"The same intervals covered this alternative **{hidden_ok}/2** times. These fixed scenarios challenge the extrapolation assumptions and cannot establish nominal coverage.", "",
              "Isopach accuracy was measured by the fraction of the union area shared by true and estimated exceedance sets. Each area uses full-Earth quadrature at 1, 10 and 100 cm. "
              "The table gives the median overlap over primary scenario/seed/level combinations with effective union area sample size at least 64 for both methods.", "",
              "| Level | Resolved comparisons | Median ensemble overlap | Previous mean-log reference overlap |","|---|---:|---:|---:|"]
    eligible=shapes[(shapes.truth!="hidden_lobe")&(shapes.union_area_ess>=64)]
    for level in (1.,10.,100.):
        sub=eligible[eligible.level_cm==level].pivot(index=["scenario","seed"],columns="method",values="area_intersection_over_union").dropna()
        if len(sub):
            lines.append(f"| {level:g} cm | {len(sub)} | {sub.ensemble_median.median():.1%} | {sub.single_mean_log.median():.1%} |")
    lines += ["","The machine-readable records include both indistinguishable truths, all unresolved levels and the sampler and map-integration warnings.", "",
              "[Case comparison](comparison.csv) · [Known volumes](known-volumes.csv) · [Contour overlap by scenario and level](contour-accuracy.csv) · "
              "[Complete simulation records](simulation-records.json.gz) · [Verification](verification.json)", "",
              "[Method and equations](../../../docs/methods/directional-ensemble.md) · [Single-structure studies](../single-structure/README.md) · "
              "[Principal uncertain-direction comparison](../README.md)",""]
    (output/"README.md").write_text("\n".join(lines))
    artifacts = {f:digest(output/f) for f in ("README.md","volumes.png","comparison.csv","known-volumes.csv","contour-accuracy.csv","simulation-records.json.gz")}
    verification = {"regional_benchmark":regional_check,"calibration_sha256":digest(PACKET),"artifacts":artifacts,
                    "case_reports":{r["slug"]:digest(result_home(r)/"directional/ensemble/report.json") for r in datasets()}}
    if cli_replay is not None:
        run = json.loads((cli_replay/"ensemble/report.json").read_text())
        case, = (r for r in datasets() if digest(r["path"]) == run["input"]["input_sha256"])
        home = result_home(case)/"directional/ensemble"
        hashes = json.loads((home/"artifacts.json").read_text())
        for f, expected in hashes.items():
            if digest(cli_replay/"ensemble"/f) != expected:
                raise ValueError("Normal command replay differs from the study artifact "+f)
        verification["normal_command_replay"] = {"dataset":case["name"],"input_sha256":run["input"]["input_sha256"],
            "matched_files":hashes, "mode":"global --geometry directional --resume, using verified scientific fit caches"}
    write_json(output/"verification.json",verification)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    for flag in ("known","regional","output"):
        parser.add_argument("--"+flag,type=Path,required=True)
    parser.add_argument("--cli-replay",type=Path,help="Optional verified output from the normal directional command")
    args=parser.parse_args()
    summarize(args.known,args.regional,args.output,args.cli_replay)
