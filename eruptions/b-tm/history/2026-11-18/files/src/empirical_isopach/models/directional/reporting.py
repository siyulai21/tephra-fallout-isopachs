"""Isopach fragments, sector budgets and reproducible whole-deposit reports."""
from dataclasses import asdict
from pathlib import Path
from tempfile import TemporaryDirectory
import json
import shutil

import numpy as np
import pandas as pd

from ...core.catalog import dataset_record
from ..mass_mixture.study import audited_data
from .estimate import estimate
from .geometry import BASE, COMPASS, POLICIES, Geometry
from .inference import flat
from .integration import PARTS
from .study import digest, run_fits, write_json


def contour_fragments(data, g, draws):
    """Every triangle and level, including failures. Never join across gaps."""
    samples = flat(draws)["nodes"]
    values = samples.mean(0)
    sites = {tuple(x): data.log_value[np.all(data.xy == x, axis=1) & ~data.censored] for x in g.vertices}
    exact = data.log_value[~data.censored]
    levels = np.unique(np.linspace(*np.quantile(exact, [.05, .95]), 25))
    rows = []
    for t in np.flatnonzero(g.inside):
        ids = g.node_ids[t]
        xy = g.vertices[ids]
        z = values[ids]
        low, high = samples[:, ids].min(1), samples[:, ids].max(1)
        observations = [sites[tuple(x)] for x in xy]
        for level in levels:
            bracket = any(v.max() < level for v in observations) and any(v.min() > level for v in observations)
            prob = float(np.mean((low < level) & (level < high)))
            endpoints = []
            for i, j in ((0,1), (1,2), (2,0)):
                if min(z[i], z[j]) <= level < max(z[i], z[j]):
                    endpoints.append(xy[i]+(xy[j]-xy[i])*(level-z[i])/(z[j]-z[i]))
            segment = np.unique(endpoints, axis=0) if endpoints else []
            row = {"triangle": int(t), "level_cm": float(np.exp(level)), "node_1": ids[0],
                   "node_2": ids[1], "node_3": ids[2], "measured_bracket": bracket,
                   "posterior_crossing_frequency": prob, "has_segment": len(segment) == 2}
            row["displayed"] = bool(row["has_segment"] and bracket and prob >= .9)
            if row["has_segment"]:
                (row["x1_km"], row["y1_km"]), (row["x2_km"], row["y2_km"]) = segment
            rows.append(row)
    columns = ["triangle","level_cm","node_1","node_2","node_3","measured_bracket",
               "posterior_crossing_frequency","has_segment","displayed","x1_km","y1_km","x2_km","y2_km"]
    return pd.DataFrame(rows, columns=columns)


def figures(output, title, data, meta, g, draws, arcs, result):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    from ...visualization.global_maps import global_maps
    map_result = global_maps(output, title, data, meta, draws, result, g=g, arcs=arcs)
    fig, ax = plt.subplots(figsize=(8.5, 4.2), layout="constrained")
    for j,s in enumerate(COMPASS):
        item = result["compass"][s]
        v = item["volume"]
        ax.plot([j,j],[v["lower95"],v["upper95"]],c="#235164",lw=2)
        ax.scatter(j,v["median"],c="#235164",s=32,zorder=3)
        ax.scatter(j,item["parts"]["footprint"]["median"],s=28,c="#e08548",marker="D",zorder=3)
    ax.set(xticks=range(8),xticklabels=[f"{s}\n{result['compass'][s]['exact']} rows" for s in COMPASS],
           ylabel="Bulk fallout (km³)",title=title+"  |  Volume by bearing from the vent")
    ax.set_ylim(bottom=0)
    ax.grid(axis="y",alpha=.22)
    ax.legend([Line2D([0],[0],c="#235164",marker="o"),Line2D([0],[0],c="#e08548",marker="D",ls="")],
              ["Sector median and conditional 95% interval","Median inside the local footprint"],fontsize=9)
    fig.savefig(output/"directions.png",dpi=145,bbox_inches="tight"); plt.close(fig)
    return map_result


def _range(v):
    return f"{v['median']:,.1f} [{v['lower95']:,.1f}–{v['upper95']:,.1f}]"


def render(path, work, output, name=None):
    data, meta = audited_data(path)
    latitude = meta["vent_lonlat"][1]
    record = dataset_record(path)
    name = name or (record["name"] if record else Path(path).stem)
    work, output = Path(work), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    estimates, totals, records = {}, {}, []
    for p in POLICIES:
        folder = work/"fits"/p.name
        item = estimate(data, latitude, p, folder)
        estimates[p.name] = item
        with np.load(folder/"mass.npz") as f:
            totals[p.name] = f["mass"].sum(axis=(-1,-2))
        records.append({"policy":p.name, **item["volume_km3"], "passed":item["passed"]})
    pd.DataFrame(records).to_csv(output/"estimates.csv", index=False)
    np.savez_compressed(output/"volume-draws.npz", **totals)
    folder = work/"fits"/BASE.name
    base = estimates[BASE.name]
    shutil.copyfile(folder/base["attempts"][-1]["file"], output/"posterior.npz")
    shutil.copyfile(folder/"mass.npz", output/"directional-mass.npz")
    g = Geometry(data, latitude)
    with np.load(output/"posterior.npz") as f:
        draws = dict(f)
    arcs = contour_fragments(data, g, draws)
    arcs.to_csv(output/"isopach-fragments.csv", index=False)
    map_result = figures(output, name, data, meta, g, draws, arcs, base)
    validation = {k: json.loads((work/f"{k}.json").read_text()) for k in ("geographic","quadrants")}
    from .contours import shared_audit
    screen = shared_audit(data, latitude, work, validation["geographic"])
    write_json(output/"complete-contour-audit.json", screen)
    pred = pd.DataFrame({"row_id": data.row_ids, "log_observation_or_limit": data.log_value,
                         "upper": data.censored & ~data.lower_censored, "lower": data.lower_censored})
    for kind, v in validation.items():
        pred[kind+"_fold"] = v["folds"]
        for key, values in v["predictions"].items():
            pred[kind+"_"+key] = values
    pred.to_csv(output/"heldout.csv", index=False)
    designs = {p.name: json.loads((work/"fits"/p.name/"design.json").read_text()) for p in POLICIES}
    report = {"dataset":name, "protocol":"supported-isopachs-directional-thinning-v3",
              "input":{k:v for k,v in meta.items() if k!="transform_inv"},
              "policies":[asdict(p) for p in POLICIES], "fits":estimates, "designs":designs,
              "validation":validation, "map":map_result,
              "complete_bands":{k:v["selected_band"] for k,v in screen["policies"].items()}, "contours":{"displayed_segments":int(arcs.displayed.sum()),
                  "triangle_level_pairs":len(arcs), "display_rule":"Exact sites bracket the level at distinct vertices and at least 90% of draws cross it in this triangle. Display only."}}
    write_json(output/"report.json", report)
    v = validation["geographic"]; w = validation["quadrants"]
    largest = g.gap
    policies = sorted(estimates.values(), key=lambda a:a["volume_km3"]["median"])
    n_pass = sum(r["passed"] for r in estimates.values())
    def fold_checks(validation):
        fits = validation["fits"]
        estimated = [f for f in fits if f["status"] == "estimated"]
        failures = [str(f["fold"]) for f in estimated if not f["passed"]]
        omitted = [f"{f['fold']} ({f['status']})" for f in fits if f["status"] != "estimated"]
        result = f"{sum(f['passed'] for f in estimated)}/{len(estimated)} fitted folds passed sampling checks"
        if failures:
            result += ", with a numerical warning in fold " + ", ".join(failures)
        if omitted:
            result += ". Unscored folds were " + ", ".join(omitted)
        return result + "."

    parts_text = {"footprint":"Retained measurement triangles", "transition":"Transition around triangle edges",
        "proximal_background":"Remaining area closer than the nearest exact site",
        "gap_background":"Remaining area between the nearest and farthest exact sites",
        "distal_background":"Remaining area beyond the farthest exact site"}
    reference_status = ("The reference passed the sampling and integration checks." if base["passed"] else
                        "**Numerical warning.** The reference failed at least one sampling or integration check.")
    geographic_score = v["equal_region_log_score"]
    score_text = f"{geographic_score:.3f}" if geographic_score is not None else "unavailable"
    coverage = v["metrics"].get("coverage95_observations")
    coverage_text = f"{coverage:.1%}" if coverage is not None else "unavailable"
    lines = [f"# {name} directional fallout estimate", "",
        f"**{_range(base['volume_km3'])} km³ of bulk fallout** under the common two-scale continuation with an inferred distal thinning law. The brackets give the conditional 95% credible interval. A median **{base['parts']['footprint']['fraction']['median']:.1%}** of the volume lies inside retained measurement triangles. The remainder depends on extrapolation. {reference_status}", "",
        "![Supported isopach fragments and conditional continuation](isopachs.png)", "",
        f"The first map shades the interpolation footprint and shows {int(arcs.displayed.sum())} measured-bracketed line segments. The second map shows the geometric mean of posterior thickness with labelled contours. The panels deliberately use different extents: every observation is visible locally, while the continuation view expands around the 0.1 cm contour. Circles mark exact observations, triangles mark bounds, and the star marks the source. Volume comes from integration of each complete field to the antipode.", "",
        "![Local thickness and latent uncertainty](local.png)", "",
        "![Whole-Earth conditional continuation](world.png)", "",
        "Local detail can cut contours at its window; longitude edges join in the whole-Earth view. Colours saturate below 0.1 cm and above the displayed upper limit. Fine coastlines supply context, not fitting evidence. [Display bounds, finite-grid edge check and posterior subset](map-display.json).", "",
        f"The displayed grid reaches {map_result['grid_maximum_cm']:,.1f} cm at a source distance of {map_result['grid_maximum_chord_distance_km']:.1f} km ({map_result['grid_maximum_region']}). At the supplied vent it gives {map_result['source_geometric_mean_cm']:,.1f} cm. These are summaries of the fitted map.", "",
        "## Where the volume comes from", "", "| Region | Median bulk km³ [95% interval] | Median fraction |","|---|---:|---:|"]
    for k in PARTS:
        vpart = base["parts"][k]
        lines.append(f"| {parts_text[k]} | {_range(vpart['volume_km3'])} | {vpart['fraction']['median']:.1%} |")
    lines += ["","The five regions exhaust the Earth in each draw. Their separately reported medians can sum to a different value from the median total.","",
        "![Volume by compass sector](directions.png)","", "| Bearing | Exact rows | Bounds | Exact rows in distal half | Footprint km² | Median sector km³ [95% interval] |",
        "|---|---:|---:|---:|---:|---:|"]
    for s, val in base["compass"].items():
        lines.append(f"| {s} | {val['exact']} | {val['bounds']} | {val['distal_half_exact']} | {val['footprint_km2']:,.0f} | {_range(val['volume'])} |")
    lines += ["",f"Octants span 45° centred on each compass bearing. Distal half refers to radius at or beyond the median distinct exact-site radius. The largest gap between distal bearings spans **{largest['width_deg']:.1f}°** around **{largest['midpoint_deg']:.1f}°**. Measurements at the vent have no bearing. The explicit extra-fan sensitivity gives **{_range(estimates['possible_missing_fan']['volume_km3'])} km³**.", "",
        "## Assumptions and validation", "", "| Common policy | Median bulk km³ [95% interval] | Numerical checks |", "|---|---:|---|"]
    lines.extend(f"| {p.name.replace('_',' ')} | {_range(estimates[p.name]['volume_km3'])} | {'pass' if estimates[p.name]['passed'] else 'review'} |" for p in POLICIES)
    lines += ["",f"The {len(estimates)} median estimates span {policies[0]['volume_km3']['median']:,.1f}–{policies[-1]['volume_km3']['median']:,.1f} km³ across named assumptions. {n_pass}/{len(estimates)} fits passed the parameter, volume and integration checks. The intervals condition on the fitted field family, sampling locations and deposit identification. Unknown fans, preservation bias and marine redistribution can move the true total outside them.", "",
        f"The matched buffered geographic score is **{score_text}** (lower is better). Its 95% predictive intervals cover **{coverage_text}** of exact held-out measurements. The quadrant-deletion test scored {w['scored_rows']}/{w['total_rows']} rows with an equal-quadrant score of "+(f"{w['equal_region_log_score']:.3f}." if w["equal_region_log_score"] is not None else "unavailable.")+" Both tests rebuild the geometry from training rows.", "",
        "Geographic deletion used four clusters. "+fold_checks(v)+" Quadrant deletion used four compass quadrants. "+fold_checks(w), "",
        f"The shared whole-contour audit found {sum(v['selected_band'] is not None for v in screen['policies'].values())} policies with a qualifying band out of {len(screen['policies'])}. Its record retains rejected contours, holes and geographic-deletion checks. [Complete contour audit](complete-contour-audit.json)", "",
        "[Numerical report and fold identities](report.json) · [Every contour decision](isopach-fragments.csv) · [Held-out predictions](heldout.csv) · [Assumption estimates](estimates.csv) · [Joint parameter and node draws](posterior.npz) · [Sector and support mass by chain and draw](directional-mass.npz) · [All sensitivity volume draws](volume-draws.npz)", ""]
    (output/"README.md").write_text("\n".join(lines))
    write_json(output/"artifacts.json", {p:digest(output/p) for p in ("README.md","report.json","complete-contour-audit.json","isopachs.png","local.png","world.png","map-display.json","directions.png","estimates.csv","posterior.npz","directional-mass.npz","volume-draws.npz","heldout.csv","isopach-fragments.csv")})
    from .source_cap import publish as publish_source_cap
    return publish_source_cap(data, g, draws, output, report, work, latitude)


def directional_dataset(path, output, name=None, resume=False):
    output = Path(output)
    if output.exists() and any(output.iterdir()) and not resume:
        raise ValueError("Use a fresh directional output directory or request resume.")
    data, meta = audited_data(path)
    # Check that buffered outer folds are feasible before starting inference.
    from ...validation import automatic_buffer
    automatic_buffer(data)
    run_fits(data, meta["vent_lonlat"][1], output/"work")
    # Keep a previous ensemble page readable while the reference and cap are regenerated.
    with TemporaryDirectory(dir=output, prefix=".reference-") as temporary:
        reference = Path(temporary)
        render(path, output/"work", reference, name)
        (reference/"README.md").replace(reference/"single-structure.md")
        manifest_path = reference/"artifacts.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["single-structure.md"] = manifest.pop("README.md")
        write_json(manifest_path, manifest)
        for filename in manifest:
            (reference/filename).replace(output/filename)
        manifest_path.replace(output/"artifacts.json")
    from .ensemble.report import render as ensemble_report, reuse_reference_fits
    from .ensemble.members import run_members
    work = output/"work"/"ensemble"
    reuse_reference_fits(output/"work", work)
    run_members(data, meta["vent_lonlat"][1], work)
    return ensemble_report(path, work, output/"ensemble", name, family_page=True)
