"""Compare the same 10 cm isopach before and after directional continuation."""
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from .publication import SPHERICAL_REFERENCE, read_hashes, preceding_spherical_bytes
from src.empirical_isopach.core.audit import audited_data
from src.empirical_isopach.core.catalog import datasets
from src.empirical_isopach.models.mass_mixture.bearings.maps import BearingView, medians
from src.empirical_isopach.models.mass_mixture.bearings.study import read, digest
from src.empirical_isopach.visualization.maps import coastlines, frame_bounds, global_frame, grid_axes, observations
from src.empirical_isopach.visualization.global_maps import PosteriorView
from src.empirical_isopach.visualization.style import STYLE


def grid(view, bounds, size):
    x,y = grid_axes(bounds, longest=size, minimum=size//2)
    xx,yy = np.meshgrid(x,y)
    xy = np.column_stack((xx.ravel(),yy.ravel()))
    valid = view.domain.rho(xy)<view.domain.rho_max
    fields = np.full((2,len(xy)), np.nan)
    fields[:,valid] = medians(view,xy[valid])
    return x,y,xy,fields.reshape(2,*xx.shape)


def draw(output, cases):
    previous, current = "#a44926", "#075d72"
    sources = {}
    records=sorted(datasets(),key=lambda r:r["slug"])
    archived=read_hashes(SPHERICAL_REFERENCE/"previous-files.sha256")
    rows=(len(records)+1)//2
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(rows,2,figsize=(10.2,3.6*rows),squeeze=False,layout="constrained")
        for ax in axes.flat[len(records):]:
            ax.set_visible(False)
        fig.suptitle("How the spherical 10 cm isopach changes", fontsize=18)
        for ax,rec in zip(axes.flat,records):
            slug=rec["slug"]
            folder=cases/slug/"results/spherical-mixture"
            report=read(folder/"report.json")
            data,meta=audited_data(rec["path"])
            with np.load(folder/"posterior.npz") as raw:
                samples=dict(raw)
            lat=meta["vent_lonlat"][1]
            view=BearingView(samples,data,lat,report["bearing"]["map_draw_check"]["draws_per_chain"])
            name=folder/"map-display.json"
            key=f"eruptions/{slug}/results/spherical-mixture/map-display.json"
            if key in archived:
                earlier_bounds=json.loads(preceding_spherical_bytes(key))["global_bounds_km"]
            else:
                earlier_bounds,_=global_frame(data.xy,lat,PosteriorView(samples,lat).central,minimum_cm=10.)
            boxes=np.array([earlier_bounds,read(name)["global_bounds_km"]])
            full=np.r_[boxes[:,:2].min(0),boxes[:,2:].max(0)]
            x,y,xy,coarse=grid(view,full,91)
            positive=xy[coarse[0].ravel()>=np.log(10.)]
            bounds=frame_bounds(positive,margin=.17)+np.array([-1,-1,1,1])*np.tile([x[1]-x[0],y[1]-y[0]],2) if len(positive) else full
            edge_open=False
            for step in range(4):
                x,y,_,field=grid(view,bounds,127)
                edge=np.r_[field[0,0,:],field[0,-1,:],field[0,:,0],field[0,:,-1]]
                finite=edge[np.isfinite(edge)]
                edge_max=float(np.max(finite)) if len(finite) else None
                edge_open=edge_max is not None and edge_max>=np.log(10.)
                if not edge_open or step==3:
                    break
                width=bounds[2:]-bounds[:2]
                bounds+=.18*np.r_[-width,width]
            assert not np.any((field[1]>=np.log(10.))&(field[0]<np.log(10.)))
            # Nested fills give the difference between the two interpolated contour boundaries.
            for z,color in zip(field,("#e1bba6","#bde2e5"),strict=True):
                if np.nanmax(z)>=np.log(10.):
                    ax.contourf(x,y,z,levels=[np.log(10.),np.nextafter(np.nanmax(z),np.inf)],colors=[color],zorder=1)
            coastlines(ax,meta["vent_lonlat"],bounds)
            for z,color,style in zip(field,(previous,current),("dotted","dashed"),strict=True):
                if np.nanmin(z)<np.log(10.)<np.nanmax(z):
                    ax.contour(x,y,z,levels=[np.log(10.)],colors=[color],linestyles=style,linewidths=1.65,zorder=5)
            observations(ax,data,size=13,meta=meta)
            title=rec["name"]
            if edge_open:
                title+=" · contour reaches frame"
            elif np.nanmax(field[0])<np.log(10.):
                title+=" · 10 cm unresolved"
            ax.set(xlim=bounds[[0,2]],ylim=bounds[[1,3]],aspect="equal",title=title)
            ax.tick_params(labelsize=8)
            if ax.get_subplotspec().is_last_row():
                ax.set_xlabel("Easting from source (km)",fontsize=9)
            sources[slug]={"report_sha256":digest(folder/"report.json"),"parent_posterior_sha256":digest(folder/"posterior.npz"),
                           "bounds_km":bounds.tolist(),"maximum_edge_cm":float(np.exp(edge_max)) if edge_max is not None else None}
        handles=[Patch(facecolor="#e1bba6",edgecolor=previous,linestyle=":",label="Earlier estimate only"),
                 Patch(facecolor="#bde2e5",edgecolor=current,linestyle="--",label="Principal estimate ≥10 cm"),
                 Line2D([],[],ls="",marker="o",ms=5,mec="#183843",mfc="white",label="Exact thickness"),
                 Line2D([],[],ls="",marker="^",ms=6,mec="#183843",mfc="white",label="Lower bound"),
                 Line2D([],[],ls="",marker="v",ms=6,mec="#183843",mfc="white",label="Upper bound"),
                 Line2D([],[],ls="",marker="D",ms=6,mec="#332807",mfc="#f5be45",label="Interpreted bound"),
                 Line2D([],[],ls="",marker="*",ms=10,color="#c34e32",label="Source")]
        fig.legend(handles=handles,loc="outside lower center",ncols=4,frameon=False,fontsize=9)
        fig.supylabel("Northing from source (km)",fontsize=10)
        fig.savefig(output/"paired-isopachs.png",dpi=170,metadata={"Description":json.dumps(sources)})
        plt.close(fig)
