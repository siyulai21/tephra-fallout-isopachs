"""Draw the matched, unsmoothed geographic comparisons and nine-case deposit plate."""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
import numpy as np

from .nested_footprints import cases
from src.empirical_isopach.core.audit import audited_data
from src.empirical_isopach.visualization.deposit_field import PolarField
from src.empirical_isopach.visualization.deposit_maps import draw_land, frames, regular_axes

COLORS = {1.:"#176ca3",64.:"#a94823"}
LEVELS = .5*2**np.arange(13)


def surface(path):
    with np.load(path) as archive:
        return PolarField(archive["radii"],archive["z"],float(archive["axis"]))


def draw(ax, field, data, meta, bounds, fine=True):
    draw_land(ax,bounds,meta["vent_lonlat"],300 if fine else 165)
    x,y=regular_axes(bounds,430 if fine else 255)
    z=field.on_grid(x,y)
    levels=LEVELS[(LEVELS>10**np.nanmin(z)) & (LEVELS<10**np.nanmax(z))]
    if len(levels):
        ax.contour(x,y,z,levels=np.log10(levels),colors=[COLORS.get(v,"#50605b") for v in levels],
                   linewidths=[1.4 if v in COLORS else .65 for v in levels],zorder=3)
    iv=np.isfinite(data.interval_log_upper)
    for mask,symbol in ((~data.censored,"o"),(data.lower_censored & ~iv,"^"),
                        (data.censored & ~data.lower_censored & ~iv,"v"),(iv,"s")):
        pts=data.xy[mask]
        ax.scatter(pts[:,0],pts[:,1],marker=symbol,s=9 if fine else 5,c="#2e3540",alpha=.52,linewidths=0,zorder=4)
    ax.plot(0,0,"*",ms=10 if fine else 8,mfc="#fff6bc",mec="#5e301b",mew=.9,zorder=8)
    ax.set(aspect="equal",xlim=bounds[::2],ylim=bounds[1::2])
    ax.tick_params(labelsize=7,length=2)
    for axis in (ax.xaxis,ax.yaxis):
        axis.set_major_locator(MaxNLocator(4))
    for spine in ax.spines.values():
        spine.set_color("#b5c2bd")


def legend(fig):
    handles=[Line2D([],[],c=c,lw=2,label=f"{v:g} cm") for v,c in COLORS.items()]
    handles.append(Line2D([],[],c="#50605b",lw=.9,label="Other doubling contours from 0.5 cm"))
    for marker,name in (("o","Measurement"),("^","Lower limit"),("v","Upper limit"),("s","Interval"),("*","Vent")):
        handles.append(Line2D([],[],marker=marker,ls="",ms=6,color="#424940" if marker!="*" else "#a06c37",label=name))
    fig.legend(handles=handles,loc="outside lower center",ncols=4,fontsize=9,frameon=False)


def focus(slug, scratch, trial, output):
    rec=cases()[slug]
    data,meta=audited_data(rec["path"])
    folder=scratch/"morphology"/slug
    fields=[surface(folder/f"surface-{i}.npz") for i in (0,1)]
    rejected = (trial/"morphology"/slug/"surface-2.npz" if trial else
                Path("archive/2026-11-23-nested-geography/fixed-secondary/surfaces")/(slug+".npz"))
    fields.extend([surface(rejected),surface(folder/"surface-2.npz")])
    full=np.array([frames(f,data)[0] for f in fields])
    box=np.r_[full[:,:2].min(axis=0),full[:,2:].max(axis=0)]
    local=frames(fields[-1],data)[1]
    # The comparative close-up includes the source and the adjacent thick-contour trough.
    near=np.minimum(local[2:],(100.,100.))
    boxes=[box,np.r_[-near,near]]
    fig,axes=plt.subplots(2,4,figsize=(15.2,8.8),layout="constrained")
    for j,(field,title) in enumerate(zip(fields,("Station corridors","Original whole area","Fixed secondary reach · trial","Shared core and inferred reach"),strict=True)):
        for i,b in enumerate(boxes):
            ax=axes[i,j]
            draw(ax,field,data,meta,b)
            if not i: ax.set_title(title,fontsize=12)
            ax.set_xlabel("Easting (km)",fontsize=8)
            if j==0: ax.set_ylabel(("Entire 0.5 cm footprint" if i==0 else "Near vent")+"\nNorthing (km)",fontsize=10)
    fig.suptitle(f"{rec['name']} · four geographic models on the same {len(data.xy)} localities",fontsize=16)
    legend(fig)
    fig.savefig(output/(slug+".png"),dpi=150)
    plt.close(fig)


def gallery(scratch, output):
    fig,axes=plt.subplots(3,3,figsize=(12.7,13.5),layout="constrained")
    for ax,(slug,rec) in zip(axes.flat,[(s,r) for s,r in cases().items() if s==r["slug"]],strict=True):
        data,meta=audited_data(rec["path"])
        field=surface(scratch/"morphology"/slug/"surface-2.npz")
        draw(ax,field,data,meta,frames(field,data)[0],False)
        ax.set_title(f"{rec['name']}  ·  {len(data.xy)} sites",fontsize=10)
    fig.suptitle("Principal nested median isopachs · full deposits in vent-centred kilometres",fontsize=15)
    legend(fig)
    fig.savefig(output/"all-eruptions.png",dpi=135)
    plt.close(fig)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scratch",type=Path,required=True)
    parser.add_argument("--trial",type=Path,help="Optional directory with a freshly replayed fixed-secondary trial")
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--case",action="append")
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    for slug in args.case or ("mazama","pinatubo-1991-layer-c"):
        focus(slug,args.scratch,args.trial,args.output)
    if not args.case:
        gallery(args.scratch,args.output)
