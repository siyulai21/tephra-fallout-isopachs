"""Evidence-styled directional isopachs and the chance of exceeding 10 cm."""
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

from ....visualization.maps import grid_axes, observation_legend, observations, setup_map
from ..archive_readings import positions
from ..ensemble.fields import inverse_cdf
from ..ensemble.maps import fragments
from ..geometry import RADIUS, polar

LEVELS = np.array([1.,10.,100.])
COLOURS = ["#167893","#9b390e","#603488"]


def supported_lines(ax, data, model, x, y, z, levels, colors):
    """Paint exact-site bracketed, high-probability crossings over the conditional dashes."""
    table,vertices,segments,heights = fragments(data,model,x,y,z,np.asarray(levels)*np.log(10.))
    if segments:
        indices = np.abs(np.asarray(heights)[:,None]-levels).argmin(1)
        ax.add_collection(LineCollection(segments,colors=np.asarray(colors)[indices],linewidths=2.3,zorder=6),
                          autolim=False)
    return table,vertices


def mapped_archives(inventory):
    mapped = next((s for s in inventory["scenarios"] if s["id"]==inventory.get("validation_scenario")),{})
    return [s for s in inventory["sites"] if s.get("in_principal_as") == "interpreted_upper_limit"
            or s["id"] in mapped.get("effective_limit_cm",{})]


def mark_archives(ax, sites, vent, bounds, *, labels=True):
    centre = (np.asarray(bounds[:2])+bounds[2:])/2
    half = (np.asarray(bounds[2:])-bounds[:2])/2
    for index,(site,point) in enumerate(zip(sites,positions(sites,vent),strict=True)):
        if not np.all(abs(point-centre)<.96*half):
            continue
        ax.scatter(*point,marker="D",s=31,color="#f5be45",edgecolors="#332807",lw=.8,zorder=8)
        if labels:
            offset = (-5,6) if -.50*half[0]<point[0]-centre[0]<0 else (5,6)
            if index==len(sites)-1 and len(sites)>1:
                offset=(4,-12)
            ax.annotate(site["name"].replace("Lake ","").replace(" peat",""),point,xytext=offset,
                        textcoords="offset points",ha="right" if offset[0]<0 else "left",fontsize=7,zorder=9,
                        bbox={"facecolor":"white","edgecolor":"none","alpha":.86,"pad":1})


def render(output,data,meta,r,model,inventory,archive_weights):
    output = Path(output)
    reach = [v["first_crossing_great_circle_km"] for v in r["source_connected_reach"]
             if v["method"]=="control" and v["level_cm"]==10 and v["first_crossing_great_circle_km"]>0]
    radius = np.median(reach) if reach else np.median(np.linalg.norm(data.xy[~data.censored],axis=1))
    span = max(1.,min(1.35*radius,RADIUS*.95))
    bounds = np.array([-span,-span,span,span])
    x,y = grid_axes(bounds,longest=149)
    xx,yy = np.meshgrid(x,y)
    xy = np.column_stack((xx.ravel(),yy.ravel()))
    fields = []
    for i in range(0,len(xy),model.median_grid_batch):
        points=xy[i:i+model.median_grid_batch]
        old,new,bary = model.components(points)
        central = inverse_cdf(new.T,model.weights)[0]
        h,_ = model.support.evaluate(points,bary)
        values = [inverse_cdf(old.T,model.base.weights)[0]/np.log(10.),central/np.log(10.),
                  model.weights@(new>=np.log(10.)),h]
        if archive_weights is not None:
            values.append(archive_weights@(new>=np.log(10.)))
        fields.append(np.stack(values))
    z = np.concatenate(fields,axis=1).reshape(-1,*xx.shape)
    saved = dict(zip(("control_log10_cm","revised_log10_cm","probability_ge_10cm","exposure","archive_probability_ge_10cm"),
                     z))
    np.savez_compressed(output/"map-grid.npz",x_km=x,y_km=y,**saved)
    vent = meta["vent_lonlat"]
    archive_sites = mapped_archives(inventory)
    def geography(ax):
        setup_map(ax,bounds,vent)
        observations(ax,data,size=16)
        mark_archives(ax,archive_sites,vent,bounds)
    fig,axes = plt.subplots(1,2,figsize=(11.4,5.8),layout="constrained",sharex=True,sharey=True)
    tables,vertices_all=[],[]
    for ax,field,m,label,key in zip(axes,z[:2],(model.base,model),("Fixed fan directions","Uncertain fan direction"),("control","revised"),strict=True):
        ax.set_facecolor("#f9faf7")
        levels = np.log10(LEVELS)
        ax.contour(x,y,field,levels=levels,colors=COLOURS,linewidths=1.15,linestyles="dashed",zorder=4)
        tab,vert = supported_lines(ax,data,m,x,y,field,levels,COLOURS)
        tables.append(tab.assign(method=key))
        vertices_all.append(vert.assign(method=key))
        geography(ax)
        ax.set_title(label,loc="left",fontweight="bold",fontsize=12)
    fig.suptitle(f"{r['dataset']} · median isopachs near the source",fontsize=15)
    handles = [Line2D([],[],color=c,lw=2,label=f"{v:g} cm") for v,c in zip(LEVELS,COLOURS,strict=True)]
    handles += [Line2D([],[],color="#4a5657",lw=2.4,label="Measured-bracketed fragment"),
                Line2D([],[],color="#4a5657",ls="--",label="Model continuation")]
    if archive_sites:
        handles += [Line2D([],[],color="#97741f",ls="",marker="D",label="Inspected archive")]
    observation_legend(fig,data,handles)
    fig.savefig(output/"median-isopachs.png",dpi=158,bbox_inches="tight")
    plt.close(fig)
    pd.concat(tables,ignore_index=True).to_csv(output/"fragment-decisions.csv",index=False)
    pd.concat(vertices_all,ignore_index=True).to_csv(output/"bracketed-fragments.csv",index=False)
    panels = [(z[2],"Thickness observations")]
    if archive_weights is not None:
        panels.append((z[4],"With assumed archive visibility"))
    else:
        panels.append((z[3],"Exposure to unsampled bearings"))
    fig,axes = plt.subplots(1,2,figsize=(11.3,5.7),layout="constrained",sharex=True,sharey=True)
    for i,(ax,(values,label)) in enumerate(zip(axes,panels,strict=True)):
        prob = i==0 or archive_weights is not None
        im = ax.pcolormesh(x,y,values,cmap="Blues" if prob else "Oranges",vmin=0,vmax=1,shading="auto")
        if prob:
            ax.contour(x,y,values,levels=[.5,.9],colors=["#c54e15","#f4ac29"],linestyles=["--","-"],linewidths=[1.2,1.5])
        geography(ax)
        ax.set_title(label,loc="left",fontweight="bold",fontsize=12)
        fig.colorbar(im,ax=ax,location="bottom",shrink=.78,aspect=40,pad=.06,
                     label="Probability thickness reaches 10 cm" if prob else "Directional exposure (0 to 1)")
    fig.suptitle(f"{r['dataset']} · where does fallout reach 10 cm?",fontsize=15)
    observation_legend(fig,data,[Line2D([],[],color="#c54e15",ls="--",label="50% probability"),
                                 Line2D([],[],color="#e19a19",label="90% probability")])
    fig.savefig(output/"exceedance.png",dpi=158,bbox_inches="tight")
    plt.close(fig)
