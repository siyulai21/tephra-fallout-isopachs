"""Assemble matched four-family tables and before/after directional maps from saved B-Tm fits."""
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

from benchmarks.btm_cryptotephra import AUDIT, NEW, OLD, SUIGETSU_YIELD, unrounded_limits
from benchmarks.publication import BTM_THICKNESS_ONLY
from src.empirical_isopach.core.audit import audited_data
from src.empirical_isopach.models.directional.archive_readings import positions
from src.empirical_isopach.models.directional.azimuth.primary import read_field
from src.empirical_isopach.models.directional.ensemble.maps import MapView
from src.empirical_isopach.models.mass_mixture.archive_check import archive_checks
from src.empirical_isopach.visualization.maps import frame_bounds, grid_axes, setup_map


def read(path):
    return json.loads(path.read_text())


def saved_globals(home, path, inventory):
    data,meta=audited_data(path)
    az=read(home/"directional/azimuth/report.json")
    sphere=read(home/"spherical-mixture/report.json")
    assert az["input"]["input_sha256"] == sphere["input"]["input_sha256"] == meta["input_sha256"]
    marine_cm = 9529 * 2 / SUIGETSU_YIELD
    cutoffs=sorted({.1,1.,10.,marine_cm,*(float(s["principal_effective_upper_cm"]) for s in inventory["sites"]
                                  if "principal_effective_upper_cm" in s)})
    with np.load(home/"spherical-mixture/posterior.npz") as p:
        probes,_=archive_checks(dict(p),data,meta["vent_lonlat"],inventory,cutoffs)
    point = next(s for s in inventory["sites"] if s["id"] == "old-085")
    bounded = [site for site in inventory["sites"] if "principal_effective_upper_cm" in site]
    xy = positions([point,*bounded],meta["vent_lonlat"])
    directional = read_field(home/"directional/azimuth",data,az)
    local = directional.distribution(xy,observation=True)
    thresholds = [marine_cm,*(site["principal_effective_upper_cm"] for site in bounded)]
    cdf = local.cdf(np.log(thresholds))
    from scipy.special import ndtr
    dir_p = float(local.weights @ ndtr((local.means[:,0]-np.log(marine_cm))/np.sqrt(local.variances[:,0])))
    at_limits = {site["id"]:float(prob) for site,prob in zip(bounded,cdf[1:],strict=True)}
    marine = next(s for s in probes["quantitative"]["sites"] if s["id"] == point["id"])
    sphere_p = float(np.clip(1-marine["p_local_below_scale"][str(marine_cm)],0,1))
    return {"directional":{"volume":az["volume"],"fits":az["members"],"marine_probability":dir_p,
                           "probability_below_principal_limit":at_limits,
                           "sites":az["archive"]["scenarios"]["quantitative"]["sites"]},
            "spherical":{"volume":sphere["fits"]["baseline"]["volume"],
                         "fits":sphere["fits"]["baseline"],"sites":probes["quantitative"]["sites"],"marine_probability":sphere_p}}


def overlay(home, old, out, inventory):
    sites=[s for s in inventory["sites"] if "principal_effective_upper_cm" in s]
    fig,axes=plt.subplots(1,2,figsize=(11.2,5.7),layout="constrained")
    from src.empirical_isopach.models.directional.geometry import COMPASS, RADIUS
    models=[]
    extent=[]
    for folder,path in ((old,OLD),(home,NEW)):
        az=folder/"directional/azimuth"
        data,meta=audited_data(path)
        r=read(az/"report.json")
        field = read_field(az,data,r)
        models.append(MapView(field,meta["vent_lonlat"][1]))
        axis = field.members[0].geometry.axis
        for row in r["source_connected_reach"]:
            if row["method"] == "revised" and row["level_cm"] == 1:
                radians = COMPASS.index(row["sector"]) * np.pi/4
                chord = 2*RADIUS*np.sin(row["first_crossing_great_circle_km"]/(2*RADIUS))
                extent.append(chord*np.array([axis*np.sin(radians),np.cos(radians)/axis]))
    xy=positions(sites,meta["vent_lonlat"])
    wide=frame_bounds(np.vstack((data.xy,extent)))
    near=frame_bounds(np.vstack((np.zeros(2),xy[:2])),margin=.15)
    middle=(near[:2]+near[2:])/2
    half=(near[2:]-near[:2])/2
    half[1]=max(half[1],.74*half[0])
    near=np.r_[middle-half,middle+half]
    marine=next(site for site in inventory["sites"] if site["id"]=="old-085")
    marine_xy, = positions([marine],meta["vent_lonlat"])
    for panel,(ax,bounds) in enumerate(zip(axes,(wide,near),strict=True)):
        x,y=grid_axes(bounds,longest=181)
        gx,gy=np.meshgrid(x,y)
        pts=np.column_stack((gx.ravel(),gy.ravel()))
        setup_map(ax,bounds,meta["vent_lonlat"])
        for model,ink,style in zip(models,("#a55d36","#076a86"),("dashed","solid"),strict=True):
            grid=model.central(pts).reshape(gx.shape)
            levels=[v for v in (-1.,0.,1.) if np.nanmin(grid)<v<np.nanmax(grid)]
            if levels:
                cs=ax.contour(x,y,grid,levels=levels,colors=ink,linestyles=style,linewidths=1.25)
                ax.clabel(cs,fmt={v:f"{10**v:g}" for v in levels},fontsize=8)
        measured=data.xy[~data.censored]
        ax.scatter(*measured.T,s=8,facecolors="white",edgecolors="#656969",linewidths=.55,zorder=4)
        for station,(xx,yy) in zip(sites,xy,strict=True):
            if bounds[0]<=xx<=bounds[2] and bounds[1]<=yy<=bounds[3]:
                ax.scatter(xx,yy,marker="D",color="#e6ab32",edgecolors="#463308",s=35,zorder=6)
                if panel or station["id"]=="old-084":
                    ax.annotate(station["name"].replace("Lake ","").replace(" peat",""),(xx,yy),
                                xytext=(3,6),textcoords="offset points",fontsize=8,zorder=7)
        ax.scatter(0,0,marker="*",s=110,c="#b13430",edgecolors="white",lw=.8,zorder=7)
        if panel:
            ax.annotate("Paektu",(0,0),xytext=(-5,-13),ha="right",textcoords="offset points",fontsize=8)
        else:
            ax.scatter(*marine_xy,marker="^",s=43,facecolor="white",edgecolor="#793e86",lw=1.3,zorder=7)
            ax.annotate("Ulleung Basin",marine_xy,xytext=(4,-12),textcoords="offset points",fontsize=8)
        ax.set_title(("Survey and source-connected 1 cm extent","Western sites near Paektu")[panel],loc="left")
    fig.suptitle("B-Tm directional median isopachs before and after the core evidence",fontweight="bold")
    fig.legend(handles=[Line2D([],[],color="#a55d36",ls="--",label="68 directly quantitative sites"),
                        Line2D([],[],color="#076a86",label="71 sites including interpreted bounds"),
                        Line2D([],[],ls="",marker="D",color="#e6ab32",label="Interpreted upper bound"),
                        Line2D([],[],ls="",marker="^",color="#793e86",markerfacecolor="white",label="Independent marine occurrence")],
               loc="outside lower center",ncols=2,frameon=False)
    fig.savefig(out/"directional-before-after.png",dpi=175)
    plt.close(fig)


def volume_plot(volumes, out):
    cases = ["source_only","principal","one_tenth_response","recognition_analogs"]
    labels = ["68 direct sites","Site-derived limits","Tenfold weaker limits","Visible-bed analogues"]
    families = [("stacked","GP ensemble · regional hull"),("spline","Cubic spline · regional hull"),
                ("directional","Uncertain fan direction · Earth"),("spherical","Spherical mixture · Earth")]
    fig, axes = plt.subplots(2,2,figsize=(10.6,6.3),sharex=True,layout="constrained")
    table = volumes.set_index(["case","method"])
    for ax,(method,title) in zip(axes.flat,families,strict=True):
        for y,case in enumerate(cases):
            r = table.loc[(case,method)]
            color = "#08768b" if case == "principal" else "#6d7377"
            ax.plot([r.lower95,r.upper95],[y,y],color=color,lw=2,solid_capstyle="round")
            ax.scatter(r.volume_median_km3,y,s=42,facecolor=color,edgecolor="white",lw=.6,zorder=3)
            if method == "directional":
                ax.scatter(r.median_map_km3,y+.15,marker="D",s=32,color="#be7229",zorder=3)
        ax.set(xscale="log",yticks=range(len(labels)),yticklabels=labels,ylim=(3.6,-.55),title=title)
        ax.grid(axis="x",alpha=.23)
        ax.spines[["top","right"]].set_visible(False)
    fig.suptitle("B-Tm bulk volumes across four interpretations of the cores",fontweight="bold")
    for ax in axes[-1]:
        ax.set_xlabel("Bulk volume (km³, logarithmic scale)")
        ax.set_xticks([20,50,100,300,1000,3000],labels=["20","50","100","300","1,000","3,000"])
        ax.tick_params(axis="x",which="minor",length=0)
    fig.legend(handles=[Line2D([],[],color="#687178",marker="o",label="Complete deposits · median and conditional 95% interval"),
                        Line2D([],[],ls="",marker="D",color="#be7229",label="Directional median-map integral")],
               loc="outside lower center",ncols=2,frameon=False)
    fig.savefig(out/"volume-sensitivity.png",dpi=170)
    plt.close(fig)


def assemble(home,old,out):
    inventory=read(AUDIT/"cryptotephra-probes.json")
    cases={label:saved_globals(folder,path,inventory) for label,folder,path in (
           ("source_only",old,OLD),("principal",home,NEW))}
    for label in ("one_tenth_response","recognition_analogs"):
        cases[label]={family:read(out/f"{label}-{family}.json") for family in ("directional","spherical")}
    marine = [{"case":case,"method":family,"site":"Ulleung Basin 13PT-P4_026",
               "single_slice_equivalent_cm":9529 * 2 / SUIGETSU_YIELD,
               "p_local_at_least_equivalent":cases[case][family]["marine_probability"]}
              for case in ("source_only","principal") for family in ("directional","spherical")]
    pd.DataFrame(marine).to_csv(out/"marine-occurrence-check.csv",index=False)
    regional=pd.read_csv(out/"regional-sensitivities.csv")
    sources=pd.read_csv(out/"regional-archive-predictions.csv")
    stations={s["id"]:s for s in inventory["sites"]}
    fitrows=[]
    for label,methods in cases.items():
        for family in ("stacked","spline"):
            r=regional[(regional.case==label)&(regional.method==family)].iloc[0]
            fitrows.append(dict(case=label,method=family,scope="observation hull",median_map_km3=r.median_map_km3,
                                volume_median_km3=r["volume_q0.5"],lower95=r["volume_q0.025"],upper95=r["volume_q0.975"],
                                analytic_mean_km3=r.expected_km3))
        for family,r in methods.items():
            v=r["volume"] if family=="spherical" else r["volume"]["coherent_volume_km3"]
            fitrows.append(dict(case=label,method=family,scope="whole Earth",volume_median_km3=v["median"],
                                lower95=v["lower95"],upper95=v["upper95"],median_map_km3=r["volume"].get("median_map_km3")))
    volumes=pd.DataFrame(fitrows)
    volumes.to_csv(out/"four-method-sensitivities.csv",index=False)
    volume_plot(volumes,out)
    rows=sources.to_dict("records")
    for label,methods in cases.items():
        for sid,s in methods["directional"]["sites"].items():
            if sid not in stations:
                continue
            lo,median,hi=s["latent_cm_p025_p50_p975"]
            rows.append(dict(case=label,method="directional",id=sid,latent_lower95_cm=lo,
                             latent_median_cm=median,latent_upper95_cm=hi))
        for s in methods["spherical"]["sites"]:
            v=s["latent_cm"]
            rows.append(dict(case=label,method="spherical",id=s["id"],latent_lower95_cm=v["lower95"],
                             latent_median_cm=v["median"],latent_upper95_cm=v["upper95"]))
    sites=pd.DataFrame(rows)
    sites["site"]=sites.id.map({k:s["name"] for k,s in stations.items()})
    sites.to_csv(out/"four-method-site-predictions.csv",index=False)
    response = []
    for sid, upper in unrounded_limits().items():
        selected = sites[(sites.id == sid) & (sites.method == "directional")]
        fields = [(row.case, row.latent_median_cm) for row in selected.itertuples()]
        for label, thickness in [("isopach",v) for v in (.1,1.,10.,100.)] + fields:
            # For a lake, relative response can exceed unity. For peat, retention is a fraction.
            maximum = upper/thickness
            response.append({"id":sid,"site":stations[sid]["name"],"comparison":label,
                             "local_thickness_cm":thickness,
                             "response_quantity":"peat mass retention fraction" if sid=="quanyang"
                                                 else "lake count response relative to Suigetsu",
                             "maximum_compatible_response":min(1.,maximum) if sid=="quanyang" else maximum})
    pd.DataFrame(response).to_csv(out/"response-required-by-contours.csv",index=False)
    data,meta=audited_data(NEW)
    limits={s["row_id"]:s for s in meta["interpreted_upper_limits"]}
    from benchmarks.refresh_figures import regional_models
    from src.empirical_isopach.core.distributions import predictive_distribution
    models,_,_ = regional_models(data,read(home/"report.json"))
    bounded = list(limits.values())
    bound_xy = positions([stations[s["id"]] for s in bounded],meta["vent_lonlat"])
    levels = [s["limit_cm"] for s in bounded]
    known = [data.log_sd[np.flatnonzero(data.row_ids == s["row_id"]).item()] for s in bounded]
    fitted = {}
    for family,model in models.items():
        cdf = predictive_distribution(model,bound_xy,observation=True,log_sd=known).cdf(np.log(levels))
        fitted[family] = {s["id"]:float(p) for s,p in zip(bounded,cdf,strict=True)}
    fitted["directional"] = cases["principal"]["directional"]["probability_below_principal_limit"]
    spherical_sites = {s["id"]:s for s in cases["principal"]["spherical"]["sites"]}
    fitted["spherical"] = {s["id"]:spherical_sites[s["id"]]["p_local_below_scale"][str(s["limit_cm"])]
                           for s in bounded}
    held=[]
    regional_cv=pd.read_csv(home/"heldout_predictions.csv").set_index("row_id")
    sphere_cv=pd.read_csv(home/"spherical-mixture/heldout_predictions.csv").set_index("row_id")
    dir_cv=pd.read_csv(home/"directional/azimuth/heldout.csv")
    dir_cv=dir_cv[(dir_cv.design=="geographic")&(dir_cv.method=="revised")].set_index("row_id")
    score_tables = (("stacked",regional_cv,"stacked_log_score"),("spline",regional_cv,"spline_log_score"),
                    ("spherical",sphere_cv,"log_score"),("directional",dir_cv,"integrated_log_score"))
    input_table = pd.read_csv(NEW)
    input_table["row_id"] = np.arange(2,2+len(input_table))
    reference_id, = input_table.loc[input_table.EvidenceID.eq("old-082"),"row_id"]
    audit_scores = []
    for family, table, score in score_tables:
        for label, members in (("all",data.row_ids),("exact_only",data.row_ids[~data.censored]),
                               ("direct_excluding_suigetsu",set(data.row_ids)-set(limits)-{reference_id})):
            selected = table.loc[table.index.isin(members),["fold",score]].dropna()
            by_fold = selected.groupby("fold")[score].mean()
            audit_scores.append({"method":family,"evidence_subset":label,"planned_rows":len(members),
                                 "scored_rows":len(selected),"scored_folds":len(by_fold),
                                 "equal_fold_negative_log_score":-by_fold.mean()})
    pd.DataFrame(audit_scores).to_csv(out/"heldout-by-evidence.csv",index=False)
    for row_id,site in limits.items():
        for family,table,score in score_tables:
            r=table.loc[row_id] if row_id in table.index else None
            median = None
            if r is not None:
                if family == "directional":
                    median = r["predicted_site_median_cm"]
                else:
                    column = family+"_median" if family in ("stacked","spline") else "median"
                    median = np.exp(r[column])
            held.append(dict(site=site["name"],row_id=row_id,limit_cm=site["limit_cm"],method=family,
                             fitted_repeat_probability_of_bound=fitted[family][site["id"]],
                             heldout_local_median_cm=median, fold=None if r is None else int(r["fold"]),
                             heldout_probability_of_bound=None if r is None else np.exp(r[score])))
    held=pd.DataFrame(held)
    held.to_csv(out/"heldout-archive-probabilities.csv",index=False)
    reaches=[]
    for label,folder in (("source_only",old),("principal",home)):
        az=read(folder/"directional/azimuth/report.json")
        for r in az["source_connected_reach"]:
            if r["method"]=="revised":
                reaches.append({"case":label,**r})
    pd.DataFrame(reaches).to_csv(out/"directional-reach.csv",index=False)
    overlay(home,old,out,inventory)
    from benchmarks.btm_cryptotephra_narrative import write_study
    write_study(home,out,volumes,sites,held)
    return volumes,sites,held


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--home",type=Path,default=Path("eruptions/b-tm/results"))
    p.add_argument("--previous",type=Path,default=BTM_THICKNESS_ONLY/"results")
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--matched-benchmark",type=Path,help="Fresh six-case regional benchmark with synthetic trials.")
    a=p.parse_args()
    if a.matched_benchmark:
        from benchmarks.btm_cryptotephra_regional import record_benchmark
        record_benchmark(a.matched_benchmark,a.output)
    assemble(a.home,a.previous,a.output)
