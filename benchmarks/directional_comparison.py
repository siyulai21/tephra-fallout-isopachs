"""Publish paired six-eruption comparisons and known-volume stress tests."""
import argparse
import gzip
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from benchmarks.directional import SCENARIOS, trial
from src.empirical_isopach.core.catalog import datasets
from src.empirical_isopach.models.directional.estimate import interval
from src.empirical_isopach.models.directional.geometry import COMPASS, Geometry, polar
from src.empirical_isopach.models.directional.source_cap import cap_volumes
from src.empirical_isopach.models.mass_mixture.study import audited_data
from src.empirical_isopach.models.directional.study import digest, write_json


def read(path):
    return json.loads(path.read_text())


def matched_rows(record, run, regional, output):
    slug = record["slug"]
    home = Path("eruptions")/slug/"results"
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
        assert np.isclose(a["expected_km3"], b["expected_km3"],rtol=1e-6)
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
    low = min(passing,key=lambda x:x[1]["median"]) if passing else (None,{})
    high = max(passing,key=lambda x:x[1]["median"]) if passing else (None,{})
    return {"eruption":record["name"],"slug":slug,**v,"previous_median":old["fits"]["baseline"]["volume"]["median"],
        "previous_lower95":old["fits"]["baseline"]["volume"]["lower95"], "previous_upper95":old["fits"]["baseline"]["volume"]["upper95"],
        "footprint_median_fraction":base["parts"]["footprint"]["fraction"]["median"],
        "footprint_median_km3":base["parts"]["footprint"]["volume_km3"]["median"],
        "geographic_score":val["equal_region_log_score"],
        "previous_score":v0["metrics"]["equal_region_log_score"],
        "exact_coverage":val["metrics"]["coverage95_observations"],
        "previous_coverage":v0["metrics"]["coverage95_observations"],
        "quadrant_score":new["validation"]["quadrants"]["equal_region_log_score"],
        "quadrant_scored_rows":new["validation"]["quadrants"]["scored_rows"],
        "quadrant_total_rows":len(val["row_ids"]),
        "reference_passed":base["passed"],"nearest_accepted_km":cap["nearest_accepted_chord_km"],
        "cap_extra_median":cap["extra_km3"]["median"],"cap_total_median":cap["volume_km3"]["median"],
        "cap_geographic_score":cap["validation"]["geographic"]["equal_region_log_score"],
        "cap_passed":cap["passed"],"lowest_passing_policy":low[0],"highest_passing_policy":high[0],
        "lowest_passing_median":low[1].get("median"),"highest_passing_median":high[1].get("median"),
        "passing_policy_lower_endpoint":min((i["lower95"] for _,i in passing),default=None),
        "passing_policy_upper_endpoint":max((i["upper95"] for _,i in passing),default=None),
        "policies_passed":sum(v["passed"] for v in new["fits"].values()),"policies":len(new["fits"]),
        "previous_geographic_checks_passed":all(f["fit"]["diagnostics"]["passed"] for f in v0["fits"]),
        "geographic_checks_passed":val["all_estimable_passed"],
        "quadrant_checks_passed":new["validation"]["quadrants"]["all_estimable_passed"],
        "extra_fan_median":new["fits"]["possible_missing_fan"]["volume_km3"]["median"],
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
                   "truth_km3":truth["total_km3"],**v,
                   "covers_truth":v["lower95"] <= truth["total_km3"] <= v["upper95"],
                   "absolute_log_volume_error":float(abs(np.log(v["median"]/truth["total_km3"])))}
            match = old[(old.scenario == scenario) & (old.seed == seed)]
            if len(match):
                o = match.iloc[0]
                assert np.isclose(o.truth_km3, truth["total_km3"])
                row.update(previous_median=o["median"],previous_covers_truth=bool(o.covers_truth),
                           previous_passed=bool(o.sampling_passed),
                           previous_absolute_log_volume_error=abs(np.log(o["median"]/truth["total_km3"])))
            if "NE_km3" in truth:
                row.update(truth,hidden_truth_covered=v["lower95"] <= truth["same_data_hidden_total_km3"] <= v["upper95"])
                row.update({"NE_"+k:a for k,a in fit["compass"]["NE"]["volume"].items()})
            with np.load(folder/fit["attempts"][-1]["file"]) as draws, np.load(folder/"mass.npz") as mass:
                _,_,_,extra = cap_volumes(data,Geometry(data,0),dict(draws))
                cap_total = interval(mass["mass"].sum(axis=(-1,-2))+extra)
            row.update({"source_cap_"+k:v for k,v in cap_total.items()})
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
        lo, med, hi = np.array([[v[k] for v in vals] for k in ("lower95","median","upper95")])
        ax.errorbar(med, y+shift, xerr=(med-lo,hi-med), fmt=marker, color=color,
                    elinewidth=1.6, capsize=2, markersize=5, label=label)
    ax.set_xscale("log")
    ax.set_xlabel("Sector bulk fallout (km³, logarithmic scale)")
    ax.set_yticks(y,[f"{k}  ·  {aso['compass'][k]['exact']} exact" for k in COMPASS])
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x,pos:f"{x:g}"))
    ax.grid(axis="x",alpha=.24)
    ax.spines[["top","right"]].set_visible(False)
    ax.legend(frameon=False,loc="upper right")
    ax.set_title("Aso-4 fallout by direction", loc="left",fontweight="bold",pad=10)
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
        row.update(replication_passed=rep["passed"], **{"replication_"+k:v for k,v in rep["volume_km3"].items()})
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
    aro = read(args.cases/"aso-4"/"report.json")
    aso = aro["fits"]["reference"]; missing = aro["fits"]["possible_missing_fan"]
    matched = examples.dropna(subset=["previous_median"])
    successful = matched[matched.passed & matched.previous_passed.astype(bool)]
    fresh = examples[examples.seed.isin([70,71]) & (examples.scenario!="invisible_sector")]
    draw = np.load(args.cases/"aso-4"/"directional-mass.npz")["mass"]
    area = draw.sum(axis=-2)
    group = interval(area[..., [3,4,6]].sum(-1)/area.sum(-1))
    wins = int((frame.geographic_score < frame.previous_score).sum())
    reliable = frame[frame.geographic_checks_passed & frame.previous_geographic_checks_passed]
    reliable_wins = int((reliable.geographic_score < reliable.previous_score).sum())
    lines = ["# Single-structure directional sensitivity study", "",
        "This study retains the original single-structure results and their assumption tests. [Principal uncertain-direction fits and paired checks](../README.md)", "",
        "The directional model interpolates log thickness across locally supported triangles. A positive source-centred thinning law continues across gaps. Its reach can increase towards measured bearings, with less angular spread from isolated sites. The same policies, input acceptance rules and geographic tests apply to all six eruptions. Each case report shows the data-bracketed arcs, the extrapolated field, and its volume by compass direction. [Equations and assumptions](../../../docs/methods/directional-continuation.md).", "",
        "| Eruption | Directional median km³ [conditional 95%] | Spherical-mixture median km³ [95%] | Inside local triangles | Geographic score, directional / spherical | 95% exact-site coverage, directional / spherical |",
        "|---|---:|---:|---:|---:|---:|"]
    for r in rows:
        lines.append(f"| [{r['eruption']}](../../../eruptions/{r['slug']}/results/directional/single-structure/README.md) | {r['median']:,.1f} [{r['lower95']:,.1f}–{r['upper95']:,.1f}] | {r['previous_median']:,.1f} [{r['previous_lower95']:,.1f}–{r['previous_upper95']:,.1f}] | {r['footprint_median_fraction']:.1%} | {r['geographic_score']:.3f} / {r['previous_score']:.3f} | {r['exact_coverage']:.1%} / {r['previous_coverage']:.1%} |")
    lines += ["",f"Lower scores mean better prediction of held-out thickness or a documented bound. The directional score is lower in **{wins}/6** eruptions overall and **{reliable_wins}/{len(reliable)}** where both models passed sampling checks on every geographic fold. Each case report lists numerical warnings and quadrant-deletion results. These are development comparisons using four matched outer folds within each eruption. The true global volume of each real eruption remains unknown. Both global estimates are published as conditional alternatives.", "",
        "The shared complete-contour screen was also run on both global families, using the same bracketing, support, resolution and four-deletion criteria as the regional peers. "
        f"The directional model has a qualifying band in {int((frame.directional_qualifying_support_policies>0).sum())}/6 cases and the spherical mixture in {int((frame.spherical_qualifying_support_policies>0).sum())}/6. "
        "Each case contains its directional decisions. The spherical packet includes all rejected levels, numerical fold flags and checksum-verified field subsets for replay. [Spherical support evidence](spherical-support/) · [Regional support evidence](../../support/README.md).", "",
        "![Vent-centred conditional maps for all six eruptions](compass-maps.png)", "",
        "Each circle shows all compass directions to the most distant accepted observation in that eruption. White symbols locate the observations and bounds. Dashed contours mark 1, 10 and 100 cm where present. The star marks the vent. Colours in empty sectors show the reference continuation. The circle edge is only a display limit. The bulk-volume integral continues across the whole Earth.", "",
        "## Sensitivity to unmeasured fallout", "",
        "The reference interval conditions on one policy. The next table compares the same 13 predeclared policies in every eruption. The endpoint range runs from the lowest 2.5th percentile to the highest 97.5th percentile among numerically passing fits. It has no joint probability attached. The individual case reports retain every fit, including failures.", "",
        "| Eruption | Passing policies | Median range km³ | Endpoint range km³ | Extra missing-fan median km³ |", "|---|---:|---:|---:|---:|"]
    for r in rows:
        def span(low,high):
            return f"{r[low]:,.0f}–{r[high]:,.0f}" if r[low] is not None else "unavailable"
        warning = "" if r["extra_fan_passed"] else " (warning)"
        lines.append(f"| {r['eruption']} | {r['policies_passed']}/{r['policies']} | {span('lowest_passing_median','highest_passing_median')} | {span('passing_policy_lower_endpoint','passing_policy_upper_endpoint')} | {r['extra_fan_median']:,.0f}{warning} |")
    lines += ["", "### A source cap invisible to the observations", "",
        "Local triangles or their transition can cross the vent even when all measurements are some distance away. This additional sensitivity places a compact proximal deposit inside the nearest accepted radius, with amplitude equal to the positive difference between the fitted background and local vent thickness. Each accepted site and bound receives exactly zero addition. Its analytic volume is added to each coherent posterior field. Geographic scores use training-only cap radii and the same held-out rows.", "",
        "| Eruption | Nearest site km | Median added km³ | Cap-inclusive median km³ | Geographic score with cap |", "|---|---:|---:|---:|---:|"]
    for r in rows:
        warn = "" if r["cap_passed"] else " (warning)"
        lines.append(f"| {r['eruption']} | {r['nearest_accepted_km']:.1f} | {r['cap_extra_median']:.2f} | {r['cap_total_median']:,.1f}{warn} | {r['cap_geographic_score']:.3f} |")
    lines += ["", "## Independent numerical replication", "",
        "A residual divergence in the Tambora reference prompted one additional run for **every** eruption. The separate run used the same statistical model, a fresh fixed seed, 1,800 warmup iterations and 4,000 draws in each of four chains. It was executed once per eruption with the original numerical thresholds. The standard run and this check are reported separately. A similar median across runs can coexist with a sampling warning.", "",
        "| Eruption | Standard reference checks | Independent median km³ [95%] | Independent checks |", "|---|---|---:|---|"]
    for r in rows:
        lines.append(f"| {r['eruption']} | {'pass' if r['reference_passed'] else 'warning'} | {r['replication_median']:,.1f} [{r['replication_lower95']:,.1f}–{r['replication_upper95']:,.1f}] | {'pass' if r['replication_passed'] else 'warning'} |")
    lines += ["", "[Independent designs, posterior checksums, convergence and sector results](independent-replications.json.gz).", "",
        "## Aso-4 and its unsampled directions", "",
        "The measured distribution is highly asymmetric. SE and S each have two exact rows, W has none, and NE has 60. A 15 cm offshore layer about 709 km to the southeast provides positive evidence in that direction. It cannot determine thickness all around the vent at that radius. [Source-level decisions](../../../data/audits/Aso-4/README.md).", "",
        "| Aso-4 sector | Directional median km³ [95%] | Spherical-mixture median km³ [95%] | Exact rows |","|---|---:|---:|---:|"]
    for key in COMPASS:
        c = aso["compass"][key]; o = previous["aso-4"]["volume_km3"][key]; n = c["volume"]
        lines.append(f"| {key} | {n['median']:,.1f} [{n['lower95']:,.1f}–{n['upper95']:,.1f}] | {o['median']:,.1f} [{o['lower95']:,.1f}–{o['upper95']:,.1f}] | {c['exact']} |")
    aso_sector_plot(aso, previous["aso-4"], output/"aso-4-sectors.png")
    se = octants.query("slug == 'aso-4' and octant == 'SE'").iloc[0]
    due_east = extremes.query("slug == 'aso-4' and quadrant == 1").iloc[0]
    old_aso = previous["aso-4"]
    p = aso["parts"]["footprint"]
    v = missing["volume_km3"]
    lines += ["",f"Together SE, S and W carry a median **{group['median']:.1%}** of directional volume [{group['lower95']:.1%}–{group['upper95']:.1%}], calculated by summing the same three sectors within every draw. Only **{p['volume_km3']['median']:.1f} km³** lies inside the entire local footprint. The remaining mass depends on the assumed continuation, including in the northeast between scattered measurements. Allowing an extra fan in the largest distal gap gives **{v['median']:,.0f} [{v['lower95']:,.0f}–{v['upper95']:,.0f}] km³**. Preferential sampling and marine redistribution introduce additional uncertainty.", "",
        f"The northeast median changes from {old_aso['volume_km3']['NE']['median']:,.0f} to {aso['compass']['NE']['volume']['median']:,.0f} km³ and the south from {old_aso['volume_km3']['S']['median']:,.0f} to {aso['compass']['S']['volume']['median']:,.1f} km³. The spherical-mixture sector partition uses {old_aso['draws']} fields from its verified posterior and {old_aso['sobol_points_per_component']:,} integration points per component. Halving the integration points changes the Aso-4 sector fractions by up to {old_aso['largest_sector_fraction_change_on_doubling']:.4f} in absolute fraction.", "",
        f"The directional rule also risks **underestimating a real off-axis fan**. When all observations from bearings 90–180° were withheld with a buffer, the most distant exact site in the southeast octant (row {int(se.row_id)}, {se.source_chord_km:.0f} km, bearing {se.bearing_deg:.1f}°) had **{se.observed_cm:g} cm**. The held-out median was **{se.predicted_median_cm:.2g} cm** and the 95% observation interval was **{se.predicted_lower_cm:.2g}–{se.predicted_upper_cm:.2f} cm**. This site is included in the full-data estimate. In the same fold the more easterly site at {due_east.bearing_deg:.1f}° had {due_east.observed_cm:g} cm and a predictive interval of {due_east.predicted_lower_cm:.2f}–{due_east.predicted_upper_cm:.1f} cm. The same position-only selections are published across all eruptions. [Outermost site by octant](octant-extremes.csv) · [Outermost site by quadrant](quadrant-extremes.csv).", "",
        "![Sector volume medians and conditional 95% intervals for both Aso-4 reconstructions](aso-4-sectors.png)", "",
        "![Aso-4 evidence and extrapolation maps](../../../eruptions/aso-4/results/directional/single-structure/isopachs.png)", "",
        "## Known-volume and missing-direction tests", "",
        "The matched trials reuse the exact synthetic deposits and noise seeds 10 and 11 from the spherical-mixture study. Seeds 70 and 71 generate new observations after the final model choice. Tests deliberately include secondary maxima, a missing sector, an invisible proximal cap and a diffuse far tail. The trial table retains numerical failures. The displayed comparison of median errors uses only pairs passing numerical checks in both models.", "",
        f"Of {len(matched)} matched trials, {len(successful)} passed for both families. Their median absolute log volume errors are **{successful.absolute_log_volume_error.median():.3f}** for the directional model and **{successful.previous_absolute_log_volume_error.median():.3f}** for the spherical mixture. In the {len(fresh)} fresh trials across the original six scenarios, {int(fresh.passed.sum())} passed numerical checks. Their conditional intervals contained the true volume in **{int((fresh.covers_truth & fresh.passed).sum())}/{int(fresh.passed.sum())}** passing trials. These small, designed experiments measure specific failure modes.", "",
        "| Scenario, fresh seeds | True km³ | Median estimates km³ | Numerically usable | Truth in interval among usable |","|---|---:|---:|---:|---:|"]
    for scenario,g in fresh.groupby("scenario",sort=False):
        lines.append(f"| {scenario.replace('_',' ')} | {g.truth_km3.iloc[0]:.1f} | "+", ".join(f"{x:.1f}" for x in g["median"])+f" | {g.passed.sum()}/{len(g)} | {(g.covers_truth & g.passed).sum()}/{g.passed.sum()} |")
    twin = examples[examples.scenario=="invisible_sector"]
    t = twin.iloc[0]
    lines += ["",f"The additional paired-truth experiment has *identical observations*: a smooth deposit of **{t.truth_km3:.1f} km³**, or the same deposit with a compact lobe wholly inside the unsampled northeast, giving **{t.same_data_hidden_total_km3:.1f} km³**. The true NE sector changes from {t.NE_km3:.1f} to {t.same_data_hidden_NE_km3:.1f} km³. Each data realization yields exactly one posterior for either truth. Across {int(twin.passed.sum())} passing realizations, {int((twin.hidden_truth_covered.astype(bool) & twin.passed).sum())} intervals include the larger hidden deposit. A weakly identified lobe remains a separate structural uncertainty even when plotted isopachs look convincing.", "",
        "[Complete paired results](known-volumes.csv) · [All stress-test fits, policies and retry diagnostics](known-volume-fits.json.gz) · [Development models and failures](development/README.md)", "",
        "## Reproduce", "", "~~~sh",
        "python -m benchmarks.directional --output results/new-study --reports results/new-cases",
        "for scenario in "+" ".join((*SCENARIOS,"invisible_sector"))+"; do",
        "  python -m benchmarks.directional --synthetic \"$scenario\" --output results/new-study",
        "done",
        "python -m benchmarks.regional --output results/new-regional",
        "python -m benchmarks.directional_reference --output results/new-mixture-sectors.json",
        "python -m benchmarks.directional_replication --runs results/new-study --output results/new-replication",
        "python -m benchmarks.directional_spherical_support --published --output results/new-spherical-support",
        "python -m benchmarks.directional_comparison --spherical-support results/new-spherical-support --runs results/new-study --cases results/new-cases --replication results/new-replication --regional results/new-regional --reference results/new-mixture-sectors.json --output results/new-comparison",
        "~~~","", "The normal case command produces the complete plots, checks and report. [Case workflow](../../../docs/NEW_ERUPTION.md) · [Comparison CSV](comparison.csv) · [Paired-input and regional regression checks](verification.json) · [Spherical-mixture sector estimates and quadrature refinement](spherical-mixture-sectors.json).",""]
    (output/"README.md").write_text("\n".join(lines))


if __name__=="__main__":
    main()
