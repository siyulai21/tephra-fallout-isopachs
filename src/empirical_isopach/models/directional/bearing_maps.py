"""Map agreement, disagreement and measured support without inventing closure points."""
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm,ListedColormap
from matplotlib.collections import LineCollection,PolyCollection
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np

from ...core.catalog import dataset_record
from ...visualization.global_maps import footprint
from ...visualization.maps import frame_bounds,grid_axes,observations,setup_map
from ..mass_mixture.study import audited_data
from .archive_readings import ELIGIBLE,load_inventory,positions
from .bearing_report import load_fields
from .ensemble.fields import Ensemble,load_packet
from .ensemble.maps import MapView,fragments
from .geometry import COMPASS
from .study import digest,write_json


def read(path):
    return json.loads(Path(path).read_text())


def median_grid(model,latitude,bounds):
    x,y = grid_axes(bounds,longest=181,minimum=65)
    xx,yy = np.meshgrid(x,y)
    xy = np.column_stack((xx.ravel(),yy.ravel()))
    z = MapView(model,latitude).central(xy).reshape(xx.shape)
    return x,y,z


def archive_symbols(ax,sites,xy,bounds):
    for s,point in zip(sites,xy,strict=True):
        x,y = point
        if not (bounds[0] <= x <= bounds[2] and bounds[1] <= y <= bounds[3]):
            continue
        if s["reading"] in ELIGIBLE and s["confidence"]=="secure":
            ax.scatter(x,y,s=41,marker="D",facecolor="#e96e38",edgecolor="#412e25",linewidth=.65,zorder=9)
            ax.annotate(s["name"].removeprefix("Lake ").removesuffix(" peat"),(x,y),xytext=(4,6),
                        textcoords="offset points",fontsize=7,zorder=10,
                        bbox=dict(facecolor="white",alpha=.82,edgecolor="none",pad=1.1))
        else:
            ax.scatter(x,y,s=24,marker="x",color="#6e655d",linewidth=1.2,zorder=9)


def views(output,reports,data,meta):
    models = {k:load_fields(output/k,data,meta["vent_lonlat"][1]) for k,r in reports.items() if "members" in r}
    control_dir = output.parent/"ensemble"
    if (control_dir/"artifacts.json").exists():
        models["six_fan_structures"] = load_packet(control_dir,data,meta["vent_lonlat"][1],read(control_dir/"report.json"))
    return models


def draw(output,reports,inventory,data,meta,title):
    models = views(output,reports,data,meta)
    primary = models["quantitative"]
    sites = inventory["sites"]
    xy = positions(sites,meta["vent_lonlat"])
    full_bounds = frame_bounds(np.vstack((data.xy,xy)),margin=.08)
    check_ids = {sid for s in inventory["scenarios"] for sid in s["effective_limit_cm"]}
    probes = np.array([point for s,point in zip(sites,xy,strict=True) if s["id"] in check_ids]).reshape(-1,2)
    if len(probes):
        radius = max(np.linalg.norm(probes,axis=1))
        nearby = data.xy[np.linalg.norm(data.xy,axis=1) <= radius*1.15]
        focus = frame_bounds(np.vstack((probes,nearby)),margin=.17)
        # A cross-section bounded only by the two archives can hide the unsurveyed south.
        # Give the display geographic context while leaving the volume integrals untouched.
        mid=(focus[:2]+focus[2:])/2
        span=focus[2:]-focus[:2]
        span=np.maximum(span,span[::-1]*.74)
        focus=np.r_[mid-span/2,mid+span/2]
    else:
        focus = full_bounds.copy()
    geom = primary.members[0].geometry
    results = {}
    for key,m in models.items():
        x,y,z = median_grid(m,meta["vent_lonlat"][1],focus)
        results[key]=z
    fields = np.stack(list(results.values()))
    low,high = np.nanmin(fields,axis=0),np.nanmax(fields,axis=0)
    levels = (.1,1.,10.)
    fig,axes=plt.subplots(1,3,figsize=(13.2,4.6),layout="constrained",squeeze=False)
    shades=ListedColormap(["#f2f2ec","#ddac79","#548784"])
    for ax,level in zip(axes.flat,levels,strict=True):
        t=np.log10(level)
        status=(high>=t).astype(int)+(low>=t).astype(int)
        ax.pcolormesh(x,y,np.ma.masked_where(~np.isfinite(high),status),shading="auto",cmap=shades,
                      norm=BoundaryNorm([-.5,.5,1.5,2.5],3),rasterized=True)
        setup_map(ax,focus,meta["vent_lonlat"])
        footprint(ax,geom.mesh,color="#24484d",lw=.9)
        observations(ax,data,size=15)
        archive_symbols(ax,sites,xy,focus)
        ax.set_title(f"{level:g} cm median extent",fontsize=11)
    fig.suptitle(title+" · where the median isopachs agree",fontsize=15)
    legend=[Patch(color="#548784",label="All mapped scenarios reach level"),
            Patch(color="#ddac79",label="Scenarios disagree"),Patch(color="#f2f2ec",label="No scenario median reaches level"),
            Line2D([],[],color="#24484d",ls="--",label="Measurement footprint")]
    if sites:
        legend.append(Line2D([],[],marker="D",ls="",color="#e96e38",label="Inspected archive"))
    legend.extend([Line2D([],[],marker="*",color="#c34e32",ls="",label="Vent"),
                   Line2D([],[],marker="o",markerfacecolor="white",color="#324746",ls="",label="Measured thickness")])
    fig.legend(handles=legend,loc="outside lower center",ncols=4,frameon=False,fontsize=8)
    fig.savefig(output/"isopach-agreement.png",dpi=175,bbox_inches="tight")
    plt.close(fig)
    display=[]
    if "six_fan_structures" in models: display.append(("six_fan_structures","Missing-fan sensitivity"))
    display.append(("quantitative","Measured fan directions"))
    reference=inventory.get("validation_scenario")
    if reference in models:display.append((reference,"Archive transfer sensitivity"))
    panel_count=len(display)
    fig,axes=plt.subplots(1,panel_count,figsize=(4.45*panel_count,4.65),layout="constrained",squeeze=False)
    ticks=np.array([.01,.1,1,10,100,1000])
    for ax,(key,label) in zip(axes.flat,display,strict=True):
        z=results[key]
        image=ax.pcolormesh(x,y,z,cmap="viridis",vmin=-2,vmax=3,shading="auto",rasterized=True)
        for level in (.1,1.,10.,100.):
            lv=np.log10(level)
            if np.nanmin(z)<lv<np.nanmax(z):
                cs=ax.contour(x,y,z,levels=[lv],colors="#f1f4ec",linestyles="--",linewidths=.9)
                ax.clabel(cs,fmt={lv:f"{level:g}"},fontsize=7,inline=True)
        setup_map(ax,focus,meta["vent_lonlat"])
        footprint(ax,geom.mesh,color="#a5f2d0")
        observations(ax,data,size=15)
        archive_symbols(ax,sites,xy,focus)
        ax.set_title(label,fontsize=11)
    bar=fig.colorbar(image,ax=list(axes.flat),location="bottom",shrink=.79,pad=.07,aspect=50)
    bar.set_ticks(np.log10(ticks),labels=[f"{v:g}" for v in ticks])
    bar.set_label("Conditional median primary fallout thickness (cm)")
    fig.suptitle(title+" · source-region continuation",fontsize=15)
    fig.savefig(output/"conditional-isopachs.png",dpi=175,bbox_inches="tight")
    plt.close(fig)
    deposited={}
    for key,label in display:
        ex,ey,z=median_grid(models[key],meta["vent_lonlat"][1],full_bounds)
        deposited[key]=z
    fig,axes=plt.subplots(1,len(display),figsize=(4.5*len(display),4.7),layout="constrained",squeeze=False)
    for ax,(key,label) in zip(axes.flat,display,strict=True):
        z=deposited[key]
        im=ax.pcolormesh(ex,ey,z,cmap="viridis",vmin=-2,vmax=3,shading="auto",rasterized=True)
        valid=z[np.isfinite(z)]
        contour_levels=np.array([-1.,0.,1.,2.])
        contour_levels=contour_levels[(contour_levels>valid.min()) & (contour_levels<valid.max())]
        if len(contour_levels):
            cs=ax.contour(ex,ey,z,levels=contour_levels,colors="white",linestyles="--",linewidths=.85)
            ax.clabel(cs,fmt=lambda v:f"{10**v:g}",fontsize=7)
        setup_map(ax,full_bounds,meta["vent_lonlat"])
        footprint(ax,geom.mesh,color="#a5f2d0")
        observations(ax,data,size=11)
        inspected=[(s,pt) for s,pt in zip(sites,xy,strict=True) if s["reading"] in ELIGIBLE and s["confidence"]=="secure"]
        if inspected:
            points=np.array([pt for _,pt in inspected])
            ax.scatter(*points.T,s=22,marker="D",facecolor="#ed7740",edgecolor="#412e25",linewidth=.6,zorder=9)
        ax.set_title(label,fontsize=11)
    fig.suptitle(title+" · median isopachs across the surveyed deposit",fontsize=14)
    bar=fig.colorbar(im,ax=list(axes.flat),location="bottom",shrink=.79,pad=.07,aspect=50)
    bar.set_ticks(np.log10(ticks),labels=[f"{t:g}" for t in ticks])
    bar.set_label("Conditional median fallout thickness (cm)")
    fig.savefig(output/"median-deposit.png",dpi=175,bbox_inches="tight")
    plt.close(fig)
    supported=[key for key in ("quantitative",reference) if key is not None and key in deposited]
    fig,axes=plt.subplots(1,len(supported),figsize=(5.3*len(supported),4.9),layout="constrained",squeeze=False)
    for ax,key in zip(axes.flat,supported,strict=True):
        decisions,vertices,segments,values=fragments(data,models[key],ex,ey,deposited[key])
        decisions.to_csv(output/key/"fragment-decisions.csv",index=False)
        vertices.to_csv(output/key/"supported-fragments.csv",index=False)
        ax.add_collection(PolyCollection(geom.mesh.triangles,facecolors="#d3e9e5",edgecolors="none"))
        colors=plt.Normalize(-2,3)
        lines=LineCollection(segments,cmap="viridis",norm=colors,linewidths=1.5,zorder=4)
        lines.set_array(np.asarray(values));ax.add_collection(lines,autolim=False)
        setup_map(ax,full_bounds,meta["vent_lonlat"]);observations(ax,data,size=13)
        ax.set_title("Measured fan directions" if key=="quantitative" else "Archive transfer sensitivity",fontsize=11)
    bar=fig.colorbar(lines,ax=list(axes.flat),location="bottom",shrink=.8,pad=.06,aspect=45)
    bar.set_ticks(np.log10(ticks),labels=[f"{t:g}" for t in ticks])
    bar.set_label("Median isopach level (cm)")
    fig.suptitle(title+" · measured-bracketed isopach segments",fontsize=14)
    fig.savefig(output/"measured-fragments.png",dpi=175,bbox_inches="tight");plt.close(fig)
    arrays={"local_x_km":x,"local_y_km":y,"full_x_km":ex,"full_y_km":ey,
            **{"local_"+k:v for k,v in results.items()},**{"full_"+k:v for k,v in deposited.items()}}
    np.savez_compressed(output/"grids-log10-cm.npz",**arrays)
    return {"local_bounds_km":focus.tolist(),"full_bounds_km":full_bounds.tolist(),"models":list(results),
            "displayed_levels_cm":levels,"archive_coordinates_xy_km":xy.tolist(),
            "quantity":"pointwise median of each structural or archive scenario; unresolved area if medians disagree"}


def render_bearing_reports(path,output,inventory=None,spherical_posterior=None):
    output=Path(output)
    data,meta=audited_data(path)
    data=data.canonical()
    inv,_=load_inventory(path,inventory)
    rec=dataset_record(path)
    title=rec["name"] if rec else Path(path).stem
    reports={f.parent.name:read(f) for f in sorted(output.glob("*/result.json"))}
    geo=draw(output,reports,inv,data,meta,title)
    write_json(output/"map-design.json",geo)
    rows=[]
    for key,r in reports.items():
        trained=(r["scenario"] or {}).get("effective_limit_cm",{})
        for site in r["sites"]:
            obs=next(s for s in inv["sites"] if s["id"]==site["id"])
            vals=site["probability_local_below"]
            for scale,prob in vals.items():
                usable=obs["reading"] in ELIGIBLE and obs["confidence"]=="secure"
                rows.append({"scenario":key,"site":obs["name"],"reading":obs["reading"],"eligible":usable,
                             "used_in_fit":site["id"] in trained or obs.get("in_principal_as") == "interpreted_upper_limit","effective_limit_scenario_cm":float(scale),
                             "median_field_cm":site["latent_cm"]["median"],"local_median_cm":site["new_site_cm"]["median"],
                             "five_percent_conflict_transition_cm":site["new_site_cm"]["lower90"],
                             "p_local_below_scale":prob,"conditional_conflict_at_5pct":usable and prob<.05})
    if rows:
        with (output/"archive-checks.csv").open("w",newline="") as stream:
            writer=csv.DictWriter(stream,fieldnames=list(rows[0]),lineterminator="\n");writer.writeheader();writer.writerows(rows)
    labels={"quantitative":"Measured fan directions"}
    labels.update({s["id"]:s["title"] for s in inv["scenarios"]})
    base=reports["quantitative"];central=base["median_surface"]["volume_km3"]
    lines=[f"# {title} directional continuation","",
           f"The median of the ensemble using measured fan directions integrates to **{central:,.1f} km³ of bulk fallout**. "
           "Angular reach varies across extrapolation assumptions. Changing reach alone preserves the field inside measured triangles. Conditioning on archive evidence can update the fitted thicknesses. "
           "A hypothetical extra fan in the largest sampling gap remains a separate structural sensitivity.","",
           "![Agreement and disagreement about the median isopachs](isopach-agreement.png)","",
           "Each panel compares the same thickness level across the mapped scenarios. Ochre marks locations where changing the assumptions changes which side of an isopach the site occupies. The dashed outline encloses the measured interpolation footprint. The diamond marks an inspected archive where one is available. A scenario median below a level still permits ash at that site.","",
           "![Conditional median isopachs on a common scale](conditional-isopachs.png)","","![Median isopachs across the complete survey region](median-deposit.png)","","The colour scale is shared across panels. Values below 0.01 cm share the lowest colour. Diamonds mark inspected archives. The full-deposit integrals extend beyond the displayed survey area through the antipode.","",
           "![Median isopach segments bracketed by measured sites](measured-fragments.png)","","Each segment lies inside an accepted interpolation triangle with measurements on both sides of the level and at least 90% conditional probability that the level crosses its nodes. Gaps and disconnected patches remain. Candidate narrow rays in the full-field panels can extend beyond the station that fixes their bearing. Only the segments here passed the local bracketing test.","",
           "| Continuation | Integral of median map (km³) | Median complete deposit (conditional 95%, km³) | Numerical checks |",
           "|---|---:|---:|---|"]
    sector_rows = ["| Continuation | " + " | ".join(COMPASS) + " |", "|---|" + "---:|"*len(COMPASS)]
    for key in ["quantitative",*(s["id"] for s in inv["scenarios"])]:
        r=reports[key]
        if "median_surface" not in r:continue
        m,j=r["median_surface"],r["coherent_fields"]
        q=j["volume_km3"]
        sector_volumes=np.asarray(m["parts_sectors_km3"]).sum(axis=0)
        sector_rows.append("| " + labels[key] + " | " + " | ".join(f"{v:,.3g}" for v in sector_volumes) + " |")
        flags=[]
        if not m["passed"]:flags.append("median integral")
        bad=sum(not b["passed"] for b in j["members"].values())
        if bad:flags.append(f"{bad} member diagnostics")
        lines.append(f"| {labels[key]} | {m['volume_km3']:,.2f} | {q['median']:,.1f} ({q['lower95']:,.1f}–{q['upper95']:,.1f}) | {', '.join(flags) or 'Passed'} |")
    lines+=["","### Bulk volume of the median map by direction","",
            "Each entry is km³ in a 45° sector centred on the named bearing from the vent.","",*sector_rows,"",
            "The sector volumes sum through the antipode and include extrapolated proximal, unsampled and distal fallout. The conditional intervals integrate individual deposits. The sector draws are retained for each fitted deposit. Agreement among medians cannot establish an observation-supported closed band.",""]
    if inv["scenarios"]:
        lines+=["## Field readings and transfer assumptions",""]
        lines.extend(f"- **{s['title']}.** {s['assumption']}" for s in inv["scenarios"])
        lines+=["","[Site probabilities and held-out archive checks](archive-checks.csv). The threshold at the fifth predictive percentile is the assumed effective visibility scale below which the reading would have less than 5% predictive probability. Each row identifies archives used while fitting. An uninspected or disputed horizon contributes no evidence against a contour.",""]
    for key,r in reports.items():
        p=output/key/"validation.json"
        if not p.exists():continue
        val=read(p);audit=read(output/key/"complete-contour-audit.json")
        scores=[]
        for kind,v in val.items():
            s=v["equal_region_negative_log_score"]
            scores.append(f"{kind} {s:.3f}" if s is not None else f"{kind} unscored")
        bad=[str(f["fold"]) for f in val["geographic"]["folds"] if f["status"]=="estimated" and not f["passed"]]
        bands=sum(v["selected_band"] is not None for v in audit["policies"].values())
        lines.append(f"**{labels[key]}** scored "+", ".join(scores)+f". Geographic sampling warnings occurred in {', '.join(bad) if bad else 'zero folds'}. The shared screen accepted {bands} complete bands across its policies. [Geographic and directional folds]({key}/validation.json) · [Contour decisions]({key}/complete-contour-audit.json)")
        lines.append("")
    from .archive_reliability import publish as reliability
    for key in reliability(output,reports):
        lines+=["## Partial preservation of the bed signal","",
                "The two archives can independently lose their ability to constrain original thickness even while preserving identifiable shards. For this sensitivity the effective visibility scales are held at the values in "+labels[key]+". The four independently fitted posteriors condition on neither archive, either one alone, or both. Each is weighted by its predictive probability of the observed readings and the assumed chances of retaining a recognizable bed. The plot varies both loss probabilities together. The table also varies them separately. These probabilities are sensitivity parameters.","",
                f"![Archive preservation sensitivity](archive-reliability-{key}.png)","",
                "[All pairs of preservation assumptions and repeat-site probabilities](archive-reliability.csv)",""]
    peer = output/"spherical-archive-checks.json"
    if peer.exists():
        sphere=read(peer)
        if spherical_posterior:
            if digest(spherical_posterior)!=sphere["posterior_sha256"]:
                raise ValueError("Spherical archive map and visibility checks use different draw files")
            from ..mass_mixture.archive_map import render
            with np.load(spherical_posterior) as raw:
                render(dict(raw),output/"spherical-archive-weights.npz",sphere,data,meta,inv,output,geo["local_bounds_km"])
        lines+=["## Independent spherical mixture","",
                ("These predictions use the spherical posterior fitted with the same interpreted upper limits. Each saved deposit contributes its original weight." if meta.get("interpreted_upper_limits") and not inv["scenarios"] else "The same readings update the weights of complete mass-normalized spherical deposits. The likelihood includes the spherical model’s local scatter and contamination component. The whole-deposit volumes below come from the updated draws. Each retains its own physical normalization. The effective sample size and the consistency of the independent chain estimates check whether the update is numerically usable."),"",
                "| Archive transfer | Complete-deposit median (conditional 95%, km³) | Weight ESS | Numerical check |","|---|---:|---:|---|"]
        for key,v in sphere["scenarios"].items():
            if (v["scenario"] or {}).get("role")=="cross_prediction":continue
            q=v["full_field_volume_km3"]
            evidence_label = "Principal observation bounds and thicknesses" if key == "quantitative" else labels[key]
            lines.append(f"| {evidence_label} | {q['median']:,.1f} ({q['lower95']:,.1f}–{q['upper95']:,.1f}) | {v['raw_weight_ess']:,.0f} | {'Passed' if v['importance_checks_passed'] else 'Requires independent refit'} |")
        if (output/"spherical-archive-isopachs.png").exists():
            lines+=["","![Independent spherical median isopachs before and after archive conditioning](spherical-archive-isopachs.png)"]
        lines+=["","[Spherical archive probabilities, chain checks and hull membership](spherical-archive-checks.json) · [Posterior weights by chain](spherical-archive-weights.npz)",""]
    lines+=["## Reproducible results",""]
    for key in labels:
        desc=f"[{labels[key]}]({key}/result.json)"
        if "members" in reports[key]:
            desc+=f" · [Median field packet]({key}/map-fields.npz) · [Complete-field sector draws]({key}/sector-draws.npz)"
        lines.append("- "+desc)
    lines+=["","[Map design](map-design.json) · [Numerical map grids](grids-log10-cm.npz)",""]
    (output/"README.md").write_text("\n".join(lines))
    write_json(output/"artifacts.json",{p.relative_to(output).as_posix():digest(p) for p in output.rglob("*")
                                         if p.is_file() and p!=output/"artifacts.json"})
