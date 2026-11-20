"""Compare the B-Tm reconstructions on identical footprints and diagnose regional tail sensitivity."""
import argparse
import json
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.special import logsumexp

from benchmarks.btm_cryptotephra import AUDIT, NEW, OLD, digest
from benchmarks.publication import BTM_THICKNESS_ONLY
from benchmarks.refresh_figures import regional_models
from src.empirical_isopach.core.audit import audited_data
from src.empirical_isopach.core.distributions import predictive_distribution
from src.empirical_isopach.region import integrate_region
from src.empirical_isopach.visualization.maps import frame_bounds, observations, observation_legend, setup_map


def record_benchmark(benchmark, output):
    """Check the other ten published case/method rows and all synthetic trial results."""
    previous = BTM_THICKNESS_ONLY/"results/source-audit/matched-regional.csv"
    reference = Path("comparisons/regional/synthetic/comparison.csv")
    matched = benchmark/"comparison.csv"
    simulated = benchmark/"synthetic/comparison.csv"
    current, earlier = (pd.read_csv(p).set_index(["slug","method"]).sort_index() for p in (matched,previous))
    pd.testing.assert_index_equal(current.index,earlier.index)
    other = current.index.get_level_values("slug") != "b-tm"
    pd.testing.assert_frame_equal(current[other],earlier[other],check_exact=True)
    assert digest(simulated) == digest(reference)
    trials = pd.read_csv(simulated)
    record = {"principal_input_sha256":digest(NEW),"real_cases":current.index.get_level_values("slug").nunique(),
              "case_method_rows":len(current),"same_non_btm_rows":int(other.sum()),"previous_comparison":str(previous),
              "matched_regional_sha256":digest(matched),"synthetic_results":str(reference),
              "synthetic_sha256":digest(simulated),"synthetic_method_rows":len(trials),
              "synthetic_method_rows_completed":int(trials.completed.sum()),
              "synthetic_method_rows_with_insufficient_training":int(trials.completed.eq(0).sum())}
    output.mkdir(parents=True,exist_ok=True)
    shutil.copyfile(matched,output/"matched-regional.csv")
    (output/"regional-benchmark-check.json").write_text(json.dumps(record,indent=2)+"\n")


def refined_draws(spline, hull):
    """Paired triangle rules, same 20,000 correlated draws, independent reproducible seed."""
    coeff = spline.draw_coefficients(20000, np.random.default_rng(136919))
    checks = []
    for order in (12, 24, 36):
        xy, area = hull.quadrature(order)
        design = spline.basis.design(xy)
        w = np.log(area*1e-5)[:,None]
        v = np.concatenate([np.exp(logsumexp(w + design@chunk, axis=0))
                            for chunk in np.array_split(coeff, 100, axis=1)])
        def q(values):
            return np.quantile(values, [.025, .5, .975]).tolist()
        checks.append({"order":order,"draws":len(v),"conditional_quantiles_km3":q(v),
                       "half_sample_quantiles":[q(v[::2]), q(v[1::2])]})
    return checks


def tail_audit(out, model, region, old, data, meta):
    xy, area = region.quadrature(48)
    moments, logs = [], []
    for points in np.array_split(xy, max(1, len(xy)//1024)):
        d = predictive_distribution(model, points)
        moments.append(d.moments())
        logs.extend(d.log_expected_thickness())
    _, var = np.concatenate(moments, axis=1)
    logs = np.array(logs)
    mass = area*np.exp(logs)*1e-5
    order = np.argsort(-logs)
    shares = np.cumsum(area[order])/area.sum()
    top1 = float(np.interp(.01, np.r_[0,shares], np.r_[0,np.cumsum(mass[order])/mass.sum()]))
    peak = np.argmax(logs)
    lon,lat = meta["transform_inv"].transform(*xy[peak])
    fig, ax = plt.subplots(figsize=(7.7,5.0),layout="constrained")
    colors = ax.scatter(*xy.T,c=np.sqrt(var),s=8,edgecolors="none",cmap="magma",rasterized=True)
    setup_map(ax,frame_bounds(data.xy),meta["vent_lonlat"])
    v = np.vstack((old.vertices, old.vertices[0]))
    ax.plot(*v.T,color="#21a8b3",lw=1.4,ls="--",label="Original hull")
    ax.legend(loc="lower left",fontsize=8)
    observations(ax,data,meta=meta)
    observation_legend(fig,data,meta=meta)
    fig.colorbar(colors,ax=ax,label="Standard deviation of log thickness")
    ax.set_title("B-Tm cubic spline uncertainty inside the expanded hull", fontsize=12)
    fig.savefig(out/"spline-uncertainty.png",dpi=160)
    plt.close(fig)
    return {"quadrature_order":48,"share_expected_volume_outside_original_hull":float(mass[~old.contains(xy)].sum()/mass.sum()),
            "share_expected_volume_from_top_one_percent_area":top1,"peak_expected_thickness_lonlat":[lon,lat],
            "peak_node_latent_log_sd":float(np.sqrt(var[peak])),"peak_node_median_cm":float(np.exp(logs[peak]-.5*var[peak])),
            "joint_draw_check":refined_draws(model,region)}


def run(previous, primary, weaker, analog, output):
    output.mkdir(parents=True,exist_ok=True)
    summary, probes = [], []
    sites = json.loads((AUDIT/"cryptotephra-probes.json").read_text())["sites"]
    from src.empirical_isopach.models.directional.archive_readings import positions
    old = None
    previous_spline = json.loads((previous/"report.json").read_text())["methods"]["spline"]["model"]
    for label, path, folder in (("source_only",OLD,previous),("principal",NEW,primary),
        ("one_tenth_response",AUDIT/"cryptotephra-one_tenth_response.csv",weaker),
        ("recognition_analogs",AUDIT/"cryptotephra-recognition_analogs.csv",analog)):
        data, meta = audited_data(path)
        report = json.loads((folder/"report.json").read_text())
        assert report["input"]["input_sha256"] == meta["input_sha256"], label
        models, hull, _ = regional_models(data,report)
        if old is None:
            old = hull
        xy = positions(sites,meta["vent_lonlat"])
        for family,model in models.items():
            item = report["methods"][family]
            common = integrate_region(model,old)
            summary.append({"case":label,"method":family,"input_sha256":meta["input_sha256"],
                            "fitted_hull_km2":hull.area, "expected_km3":item["regional_volume"]["expected_km3"],
                            "median_map_km3":item["regional_volume"]["median_surface_km3"],
                            **{"volume_q"+k:v for k,v in item["joint_volume"]["conditional_quantiles_km3"].items()},
                            "original_hull_km2":old.area, "original_hull_expected_km3":common["expected_km3"],
                            "original_hull_median_map_km3":common["median_surface_km3"]})
            for site,point,inside in zip(sites,xy,hull.contains(xy),strict=True):
                row = {"case":label,"method":family,"id":site["id"],"inside_regional_hull":bool(inside)}
                if inside:
                    d = predictive_distribution(model,point[None,:])
                    row.update(zip(("latent_lower95_cm","latent_median_cm","latent_upper95_cm"),
                                   [float(np.exp(d.quantile(p)[0])) for p in (.025,.5,.975)],strict=True))
                probes.append(row)
            if label=="principal" and family=="spline":
                audit = tail_audit(output,model,hull,old,data,meta)
                from src.empirical_isopach.models.cubic_spline.model import SplineCandidate, TrendRegularizedSpline
                fixed = TrendRegularizedSpline(SplineCandidate(**previous_spline["candidate"]),
                    quadrature_order=previous_spline["quadrature_order"]).fit(data,bounds=report["fit_bounds_km"])
                counterfactual = integrate_region(fixed,hull)
                audit["previous_settings_on_principal_data"] = {"candidate":previous_spline["candidate"],
                    "expected_km3":counterfactual["expected_km3"],"median_surface_km3":counterfactual["median_surface_km3"],
                    "interpretation":"All 71 observations and the expanded hull, keeping the pre-revision tuning settings fixed."}
                (output/"spline-tail.json").write_text(json.dumps(audit,indent=2,allow_nan=False)+"\n")
    pd.DataFrame(summary).to_csv(output/"regional-sensitivities.csv",index=False)
    pd.DataFrame(probes).to_csv(output/"regional-archive-predictions.csv",index=False)


if __name__ == "__main__":
    p=argparse.ArgumentParser(description=__doc__)
    for name in ("previous","primary","weaker","analog","output"):
        p.add_argument("--"+name,type=Path,required=True)
    run(**vars(p.parse_args()))
