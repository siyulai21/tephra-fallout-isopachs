"""Publish paired six-eruption comparisons and known-volume stress tests."""
import argparse
import gzip
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from benchmarks.directional import SCENARIOS, trial
from .publication import published_datasets as datasets, result_home
from src.empirical_isopach.models.directional.estimate import interval
from src.empirical_isopach.models.directional.geometry import COMPASS, Geometry, polar
from src.empirical_isopach.models.directional.source_cap import cap_volumes
from src.empirical_isopach.models.mass_mixture.study import audited_data
from src.empirical_isopach.models.directional.study import digest, write_json


def read(path):
    return json.loads(path.read_text())


def matched_rows(record, run, regional, output):
    slug = record["slug"]
    home = result_home(record)
    old, published = read(home/"spherical-mixture/report.json"), read(home/"report.json")
    fresh = read(regional/slug/"report.json")
    new = read(output/slug/"report.json")
    val, v0 = new["validation"]["geographic"], old["validation"]
    assert len({r["input"]["input_sha256"] for r in (old,published,fresh,new)}) == 1
    for key in ("row_ids","folds","buffer_km"):
        assert val[key] == v0[key] == fresh["validation"][key] == published["validation"][key]
    for fold,a in enumerate(val["fits"]):
        for key in ("training_row_ids","withheld_row_ids"):
            assert a[key] == v0["fits"][fold][key]
            for m in ("stacked","spline"):
                assert a[key] == fresh["validation"]["methods"][m]["selections"][fold][key]
    for m in ("stacked","spline"):
        a,b = fresh["methods"][m]["regional_volume"],published["methods"][m]["regional_volume"]
        assert np.isclose(a["median_surface_km3"], b["median_surface_km3"],rtol=1e-6)
        assert np.isclose(fresh["validation"]["methods"][m]["metrics"]["equal_region_log_score"],
                          published["validation"]["methods"][m]["metrics"]["equal_region_log_score"],rtol=1e-6)
    for k in ("stacked","spline"):
        new_support = read(regional/slug/"support/report.json")["methods"][k]["contours"]
        old_support = read(home/"support/report.json")["methods"][k]["contours"]
        assert all(new_support["policies"][p]["selected_band"] == v["selected_band"]
                   for p,v in old_support["policies"].items())
        for key in ("levels", "refined_levels"):
            assert len(new_support[key]) == len(old_support[key])
            for trial, original in zip(new_support[key], old_support[key]):
                assert np.isclose(trial["level_cm"], original["level_cm"])
                assert trial["status"] == original["status"]
                for name, decision in original["policies"].items():
                    assert trial["policies"][name]["eligible"] == decision["eligible"]
                    assert trial["policies"][name]["reasons"] == decision["reasons"]
    base = new["fits"]["reference"]
    v = base["volume_km3"]
    cap = new["unobserved_proximal_cap"]
    passing = [(name,fit["volume_km3"]) for name,fit in new["fits"].items() if fit["passed"]]
    return {"eruption":record["name"],"slug":slug,"lower95":v["lower95"],"upper95":v["upper95"],
        "previous_lower95":old["fits"]["baseline"]["volume"]["lower95"], "previous_upper95":old["fits"]["baseline"]["volume"]["upper95"],
        "footprint_median_fraction":base["parts"]["footprint"]["fraction"]["median"],
        "geographic_score":val["equal_region_log_score"],
        "previous_score":v0["metrics"]["equal_region_log_score"],
        "exact_coverage":val["metrics"]["coverage95_observations"],
        "previous_coverage":v0["metrics"]["coverage95_observations"],
        "quadrant_score":new["validation"]["quadrants"]["equal_region_log_score"],
        "quadrant_scored_rows":new["validation"]["quadrants"]["scored_rows"],
        "quadrant_total_rows":len(val["row_ids"]),
        "reference_passed":base["passed"],"nearest_accepted_km":cap["nearest_accepted_chord_km"],
        "cap_geographic_score":cap["validation"]["geographic"]["equal_region_log_score"],
        "cap_passed":cap["passed"],
        "passing_policy_lower_endpoint":min((i["lower95"] for _,i in passing),default=None),
        "passing_policy_upper_endpoint":max((i["upper95"] for _,i in passing),default=None),
        "policies_passed":sum(v["passed"] for v in new["fits"].values()),"policies":len(new["fits"]),
        "previous_geographic_checks_passed":all(f["fit"]["diagnostics"]["passed"] for f in v0["fits"]),
        "geographic_checks_passed":val["all_estimable_passed"],
        "quadrant_checks_passed":new["validation"]["quadrants"]["all_estimable_passed"],
        "extra_fan_passed":new["fits"]["possible_missing_fan"]["passed"],
        "input_sha256":new["input"]["input_sha256"]}


def stress(runs, output):
    records, evidence = [], {}
    old = pd.read_csv("comparisons/spherical-mixture/stress-comparison.csv").query("assumption == 'baseline'")
    for scenario in (*SCENARIOS, "invisible_sector"):
        for seed in (10, 11, 70, 71):
            data, truth = trial(scenario,seed)
            folder = runs/"synthetic"/f"{scenario}-{seed}"/"fits/reference"
            fit = read(folder/"integral.json")
            key = f"{scenario}-{seed}"
            evidence[key] = {"truth":truth,"fit":fit,"design":read(folder/"design.json")}
            v = fit["volume_km3"]
            row = {"scenario":scenario,"seed":seed,"n":len(data.xy),"passed":fit["passed"],
                   "truth_km3":truth["total_km3"],"lower95":v["lower95"],"upper95":v["upper95"],
                   "covers_truth":v["lower95"] <= truth["total_km3"] <= v["upper95"]}
            match = old[(old.scenario == scenario) & (old.seed == seed)]
            if len(match):
                o = match.iloc[0]
                assert np.isclose(o.truth_km3, truth["total_km3"])
                row.update(previous_covers_truth=bool(o.covers_truth),previous_passed=bool(o.sampling_passed))
            if "NE_km3" in truth:
                row.update(truth,hidden_truth_covered=v["lower95"] <= truth["same_data_hidden_total_km3"] <= v["upper95"])
                row.update({"NE_"+k:a for k,a in fit["compass"]["NE"]["volume"].items() if k in ("lower95","upper95")})
            with np.load(folder/fit["attempts"][-1]["file"]) as draws, np.load(folder/"mass.npz") as mass:
                _,_,_,extra = cap_volumes(data,Geometry(data,0),dict(draws))
                cap_total = interval(mass["mass"].sum(axis=(-1,-2))+extra)
            row.update({"source_cap_"+k:v for k,v in cap_total.items() if k in ("lower95","upper95")})
            evidence[key]["source_cap_volume_km3"] = cap_total
            records.append(row)
    frame = pd.DataFrame(records)
    frame.to_csv(output/"known-volumes.csv",index=False)
    raw = json.dumps(evidence,separators=(",",":"),allow_nan=False).encode()
    (output/"known-volume-fits.json.gz").write_bytes(gzip.compress(raw,mtime=0))
    return frame




def case_extremes(record, report, by_octant=False):
    """Outermost exact site by direction, predicted under its 90-degree holdout."""
    rows = []
    data, _ = audited_data(record["path"])
    axis = report["designs"]["reference"]["geometry"]["authalic_axis_scale"]
    radius, angle = polar(data.xy, axis)
    validation = report["validation"]["quadrants"]
    by_id = {int(r):i for i,r in enumerate(validation["row_ids"])}
    octants = np.floor((angle+np.pi/8)/(np.pi/4)).astype(int)%8
    by_fold = {f["fold"]:f for f in validation["fits"]}
    rows_fold = np.array([validation["folds"][by_id[int(i)]] for i in data.row_ids])
    groups = octants if by_octant else rows_fold
    for group in np.unique(groups):
        idx = np.flatnonzero(~data.censored & (groups == group))
        if not len(idx):
            continue
        i = idx[np.lexsort((data.row_ids[idx],-radius[idx]))[0]]
        j = by_id[int(data.row_ids[i])]
        fold = by_fold[int(rows_fold[i])]
        row = {"slug":record["slug"],"octant":COMPASS[octants[i]],"quadrant":fold["fold"],"training_status":fold["status"],
               "sampling_passed":fold.get("passed"),"row_id":int(data.row_ids[i]),
               "source_chord_km":float(radius[i]),"bearing_deg":float(np.rad2deg(angle[i])),
               "observed_cm":float(np.exp(data.log_value[i]))}
        for label in ("median","lower","upper"):
            value = validation["predictions"][label][j]
            row["predicted_"+label+"_cm"] = None if value is None else float(np.exp(value))
        rows.append(row)
    return rows


def directional_extremes(cases, by_octant=False):
    rows = []
    for record in datasets():
        report = read(cases/record["slug"]/"report.json")
        rows.extend(case_extremes(record, report, by_octant))
    return pd.DataFrame(rows)



def compass_maps(cases, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
    from src.empirical_isopach.models.directional.inference import field

    fig, axes = plt.subplots(2,3,figsize=(12.4,8.0),subplot_kw={"projection":"polar"},layout="constrained")
    theta = np.linspace(0,2*np.pi,161)
    fractions = np.linspace(0,1,61)
    angle, rings = np.meshgrid(theta,fractions)
    norm = Normalize(-1.,2.5)
    for ax, record in zip(axes.flat,datasets(),strict=True):
        slug = record["slug"]
        data, meta = audited_data(record["path"])
        g = Geometry(data,meta["vent_lonlat"][1])
        site_r, site_theta = polar(data.xy,g.axis)
        radius = site_r.max()
        xy = np.c_[(radius*rings*np.sin(angle)*g.axis).ravel(),(radius*rings*np.cos(angle)/g.axis).ravel()]
        with np.load(cases/slug/"posterior.npz") as archive:
            n = archive["source"].shape[1]
            idx = np.linspace(0,n-1,32).round().astype(int)
            fields = {k:v[:,idx].reshape((-1,*v.shape[2:])) for k,v in archive.items()}
        logcm = np.concatenate([field(fields,g,xy[i:i+512]).mean(0)/np.log(10.)
                                for i in range(0,len(xy),512)]).reshape(rings.shape)
        ax.set_theta_zero_location("N")
        ax.set_theta_direction(-1)
        image = ax.contourf(angle,rings,logcm,levels=np.linspace(-1,2.5,29),norm=norm,cmap="viridis",extend="both")
        levels = [level for level in (0.,1.,2.) if logcm.min()<level<logcm.max()]
        if levels:
            ax.contour(angle,rings,logcm,levels=levels,colors="white",linestyles="dashed",alpha=.85,linewidths=.7)
        for mask,marker in ((~data.censored,"o"),(data.lower_censored,"^"),(data.censored&~data.lower_censored,"v")):
            ax.scatter(site_theta[mask],site_r[mask]/radius,s=14,marker=marker,c="white",edgecolors="#242829",linewidths=.55,zorder=4,clip_on=False)
        ax.scatter(0,0,marker="*",s=75,c="#e65c35",edgecolors="white",linewidths=.6,zorder=5)
        ax.set_xticks(np.deg2rad([0,90,180,270]),labels=["N","E","S","W"],fontsize=8)
        ax.set_yticks([.5],labels=[f"{radius/2:,.0f} km"],color="white",fontsize=7)
        ax.set_rlabel_position(55)
        ax.set_ylim(0,1)
        ax.grid(color="white",linewidth=.55,alpha=.26)
        ax.set_title(f"{record['name']}  ·  outer radius {radius:,.0f} km",fontsize=10.5,pad=7)
    bar = fig.colorbar(image,ax=list(axes.flat),location="bottom",shrink=.65,aspect=52,pad=.04)
    ticks = np.arange(-1,2.51,.5)
    bar.set_ticks(ticks,labels=[f"{10**n:.2g}" for n in ticks])
    bar.set_label("Conditional bulk thickness (cm, logarithmic scale)")
    fig.suptitle("Conditional isopachs in all directions from each vent",fontsize=15,fontweight="bold")
    fig.savefig(output/"compass-maps.png",dpi=155)
    plt.close(fig)


def aso_sector_plot(aso, previous, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter

    fig, ax = plt.subplots(figsize=(8.8,4.9),layout="constrained")
    y = np.arange(len(COMPASS))[::-1]
    families = (("Directional",-.13,"#086970","o",[aso["compass"][k]["volume"] for k in COMPASS]),
                ("Spherical mixture",.13,"#9d5231","s",[previous["volume_km3"][k] for k in COMPASS]))
    for label, shift, color, marker, vals in families:
        lo, hi = np.array([[v[k] for v in vals] for k in ("lower95","upper95")])
        ax.hlines(y+shift,lo,hi,color=color,linewidth=2.2,label=label)
    ax.set_xscale("log")
    ax.set_xlabel("Sector bulk fallout (km³, logarithmic scale)")
    ax.set_yticks(y,[f"{k}  ·  {aso['compass'][k]['exact']} exact" for k in COMPASS])
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x,pos:f"{x:g}"))
    ax.grid(axis="x",alpha=.24)
    ax.spines[["top","right"]].set_visible(False)
    ax.legend(frameon=False,loc="upper right")
    ax.set_title("Aso-4 conditional 95% volume intervals by direction", loc="left",fontweight="bold",pad=10)
    fig.savefig(path,dpi=165)
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for a in ("runs","regional","cases","output","reference","replication","spherical-support"):
        parser.add_argument("--"+a,type=Path,required=True)
    args = parser.parse_args(argv)
    output = args.output
    output.mkdir(parents=True,exist_ok=True)
    rows = [matched_rows(r,args.runs,args.regional,args.cases) for r in datasets()]
    replicated = {}
    for row in rows:
        slug = row["slug"]
        folder = args.replication/slug
        rep = read(folder/"integral.json")
        design = read(folder/"design.json")
        ref = read(args.runs/slug/"fits/reference/design.json")
        assert design["geometry"] == ref["geometry"]
        for key in ("source_sha256","arrays_sha256","policy","runtime"):
            assert design["signature"][key] == ref["signature"][key]
        assert design["signature"]["seed"] != ref["signature"]["seed"]
        assert design["replication"]["input_sha256"] == row["input_sha256"]
        assert rep["signature"]["design_sha256"] == digest(folder/"design.json")
        assert rep["mass_sha256"] == digest(folder/"mass.npz")
        assert rep["signature"]["posterior_sha256"] == digest(folder/"attempt-1.npz")
        row.update(replication_passed=rep["passed"], **{"replication_"+k:v for k,v in rep["volume_km3"].items() if k in ("lower95","upper95")})
        replicated[slug] = {"design":design,"integral":rep}
    (output/"independent-replications.json.gz").write_bytes(gzip.compress(
        json.dumps(replicated,allow_nan=False,separators=(",",":")).encode(),mtime=0))
    (output/"spherical-support").mkdir(exist_ok=True)
    for row in rows:
        slug = row["slug"]
        source, archived = args.spherical_support, output/"spherical-support"
        with gzip.open(source/(slug+".json.gz"),"rt") as f:
            screen = json.load(f)
        assert screen["input_sha256"] == row["input_sha256"]
        assert digest(source/(slug+".npz")) == screen["surface_samples_sha256"]
        row["spherical_qualifying_support_policies"] = sum(p["selected_band"] is not None for p in screen["policies"].values())
        directional = read(args.cases/slug/"complete-contour-audit.json")
        row["directional_qualifying_support_policies"] = sum(p["selected_band"] is not None for p in directional["policies"].values())
        for ext in (".json.gz",".npz"):
            shutil.copyfile(source/(slug+ext),archived/(slug+ext))
    frame = pd.DataFrame(rows)
    frame.to_csv(output/"comparison.csv",index=False)
    extremes = directional_extremes(args.cases)
    octants = directional_extremes(args.cases, by_octant=True)
    octants.to_csv(output/"octant-extremes.csv",index=False)
    extremes.to_csv(output/"quadrant-extremes.csv",index=False)
    previous = read(args.reference)
    write_json(output/"spherical-mixture-sectors.json",previous)
    examples = stress(args.runs,output)
    compass_maps(args.cases,output)
    regional_trials = read(args.regional/"synthetic/synthetic.json")["runs"]
    ref_trials = read(Path("comparisons/regional/synthetic/synthetic.json"))["runs"]
    assert len(regional_trials) == len(ref_trials) == 40
    for a,b in zip(regional_trials,ref_trials):
        assert (a["scenario"],a["seed"]) == (b["scenario"],b["seed"])
        for method in ("stacked","spline"):
            left, right = a["methods"][method], b["methods"][method]
            assert left["status"] == right["status"]
            if left["status"] == "complete":
                assert left["conditional_volume_interval_covers_truth"] == right["conditional_volume_interval_covers_truth"]
    verification = {"matched_inputs_folds_buffers_training_and_test_ids":True,"regional_methods_checked":["stacked","spline"],
        "regional_volume_and_geographic_score_relative_tolerance":1e-6,
        "regional_synthetic_trials":len(regional_trials),"rebuilt_regional_contour_decisions_match":True,
        "regional_synthetic_attempt_status_and_interval_decisions_match":True,
        "cases":{r["slug"]:digest(args.cases/r["slug"]/"report.json") for r in rows},
        "old_spherical_posterior_inputs_checked":all(previous[r["slug"]]["input_sha256"] == r["input_sha256"] for r in rows)}
    assert verification["old_spherical_posterior_inputs_checked"]
    write_json(output/"verification.json",verification)
    aso=read(args.cases/"aso-4/report.json")["fits"]["reference"]
    aso_sector_plot(aso,previous["aso-4"],output/"aso-4-sectors.png")
    write_study_page(frame,examples,output)


def write_study_page(cases,trials,output):
    lines = ["# Single-structure directional sensitivity", "",
             "This six-eruption study tests the older single-structure continuation against a spherical fit on the same observations. The B-Tm comparison uses the 84-record input. The current case pages give the median-thickness maps and their integrated volumes. [Current directional fits](../README.md) · [B-Tm evidence revision](../../../eruptions/b-tm/README.md)", "",
             "## Geographic validation and volume uncertainty", "",
             "| Eruption | Directional 95% interval (km³) | Spherical 95% interval (km³) | Geographic score, directional / spherical | Exact-site coverage, directional / spherical | Passing directional policies |",
             "|---|---:|---:|---:|---:|---:|"]
    for row in cases.itertuples():
        lines.append(f"| {row.eruption} | {row.lower95:,.1f}–{row.upper95:,.1f} | "
                     f"{row.previous_lower95:,.1f}–{row.previous_upper95:,.1f} | {row.geographic_score:.3f} / {row.previous_score:.3f} | "
                     f"{row.exact_coverage:.1%} / {row.previous_coverage:.1%} | {row.policies_passed}/{row.policies} |")
    fresh = trials[trials.seed.isin([70,71]) & (trials.scenario!="invisible_sector")]
    passed = fresh[fresh.passed]
    lines += ["", "The CSV includes the independent replication intervals, the envelope across passing policies, and geographic scores with a proximal cap. [Paired cases](comparison.csv) · [Independent fit diagnostics](independent-replications.json.gz) · [Matched-fold verification](verification.json)", "",
              "## Unsampled sectors at Aso-4", "",
              "The southeast and south each have two exact readings. The west has none. With the southeast quadrant withheld, a 15 cm offshore observation at 709 km fell above the directional 95% predictive interval of 0.0003–1.95 cm. An unsampled sector can contain a real lobe. [Outermost site by octant](octant-extremes.csv) · [Outermost site by quadrant](quadrant-extremes.csv)", "",
              "![Conditional sector volume intervals for both Aso-4 fits](aso-4-sectors.png)", "",
              "![Fitted thickness and measurements across six eruptions](compass-maps.png)", "",
              "## Known-deposit tests", "",
              f"Of {len(fresh)} fresh trials, {len(passed)} passed the numerical checks. Their conditional volume intervals contained the true volume in {int(passed.covers_truth.sum())}/{len(passed)}. The full trial table includes failures, deliberately missing directions, and the known volumes of the simulated deposits. [All trials](known-volumes.csv) · [Fit records](known-volume-fits.json.gz)", ""]
    lines += ["## Reproduce", "",
              "Run the single-structure and geographic benchmarks with a fresh results directory. The comparison command pairs geographic folds, rebuilds the two interval tables and checks each input identity.", "",
              "~~~sh",
              "python -m benchmarks.directional --help",
              "python -m benchmarks.directional_replication --help",
              "python -m benchmarks.directional_comparison --help",
              "python -m benchmarks.verify_directional",
              "~~~", "", "[Eruption workflow](../../../docs/NEW_ERUPTION.md) · [Benchmark scripts](../../../benchmarks/README.md)", ""]
    (output/"README.md").write_text("\n".join(lines))


if __name__=="__main__":
    main()
