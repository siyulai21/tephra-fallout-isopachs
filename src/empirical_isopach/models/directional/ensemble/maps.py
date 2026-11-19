"""Draw median isopachs, measured-bracketed fragments and pointwise uncertainty."""
from pathlib import Path

import contourpy
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection, PolyCollection
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
import shapely

from src.probabilistic_isopach.figures import STYLE
from src.probabilistic_isopach.global_volume import EarthDomain
from ....visualization.global_maps import field_panel, footprint, grid, save, thickness_bar, world_map
from ....visualization.maps import frame_bounds, global_frame, observation_legend, observations, setup_map, panel_layout
from ..geometry import polar
from .fields import inverse_cdf


class MapView:
    statistic = "Pointwise ensemble median thickness"

    def __init__(self, model, latitude):
        self.model = model
        self.domain = EarthDomain.wgs84(latitude)

    def values(self, xy, probs):
        xy = np.asarray(xy, float)
        out = np.full((len(probs),len(xy)),np.nan)
        valid = np.flatnonzero(self.domain.rho(xy) < self.domain.rho_max)
        for start in range(0,len(valid),256):
            ids = valid[start:start+256]
            d = self.model.distribution(xy[ids])
            out[:,ids] = inverse_cdf(d.values.T, d.weights, probs)/np.log(10.)
        return out

    def evaluate(self, xy):
        return self.values(xy,(.025,.5,.975))

    def central(self, xy):
        return self.values(xy,(.5,))[0]


def fragments(data, model, x, y, log10_median, levels_log_cm=None):
    """Clip level lines of the actual median grid to individually qualified triangles."""
    g = model.members[0].geometry
    samples = np.concatenate([m.samples["nodes"] for m in model.members])
    by_site = {tuple(v):data.log_value[(data.xy==v).all(1)&~data.censored] for v in g.vertices}
    low, high = np.quantile(data.log_value[~data.censored],[.05,.95])
    gen = contourpy.contour_generator(x=x,y=y,z=np.ma.masked_invalid(log10_median))
    records, points, segments, levels = [], [], [], []
    levels_to_check = np.linspace(low,high,25) if levels_log_cm is None else levels_log_cm
    for j, level in enumerate(np.unique(levels_to_check)):
        cm = float(np.exp(level))
        lines = [shapely.LineString(s) for s in gen.lines(level/np.log(10.)) if len(s)>1]
        contours = shapely.MultiLineString(lines)
        for t in np.flatnonzero(g.inside):
            ids = g.node_ids[t]
            obs = [by_site[tuple(v)] for v in g.vertices[ids]]
            bracket = any(v.max()<level for v in obs) and any(v.min()>level for v in obs)
            z = samples[:,ids]
            prob = float(model.weights@((z.min(1)<level)&(z.max(1)>level)))
            clipped = contours.intersection(g.polygons[t])
            parts = [q for q in shapely.get_parts(clipped) if q.geom_type=="LineString" and q.length>0.]
            displayed = bool(bracket and prob>=.9 and parts)
            records.append({"triangle":int(t),"level_cm":cm,"measured_bracket":bool(bracket),
                            "crossing_probability":prob,"grid_segments":len(parts),"displayed":displayed})
            if displayed:
                for seg, line in enumerate(parts):
                    xy = np.asarray(line.coords)
                    segments.append(xy)
                    levels.append(level/np.log(10.))
                    points.extend((int(t),j,cm,seg,k,*v) for k,v in enumerate(xy))
    table = pd.DataFrame(records,columns=["triangle","level_cm","measured_bracket","crossing_probability","grid_segments","displayed"])
    vertices = pd.DataFrame(points,columns=["triangle","level_id","level_cm","segment","vertex","x_km","y_km"])
    return table, vertices, segments, levels


def render_maps(output, name, data, meta, model):
    output = Path(output)
    view = MapView(model,meta["vent_lonlat"][1])
    geom = model.members[0].geometry
    bounds, extent = global_frame(data.xy,meta["vent_lonlat"][1],view.central)
    local = frame_bounds(data.xy)
    lx,ly,lq = grid(view,local)
    x,y,q = grid(view,bounds)
    norm = Normalize(-1.,max(1.,float(np.quantile(data.log_value[~data.censored]/np.log(10.),.98)),
                             float(np.nanquantile(lq[1],.995))))
    table, vertices, segments, levels = fragments(data,model,lx,ly,lq[1])
    table.to_csv(output/"fragment-decisions.csv",index=False)
    vertices.to_csv(output/"supported-fragments.csv",index=False)
    np.savez_compressed(output/"median-grids.npz",local_x_km=lx,local_y_km=ly,local_log10_cm=lq,
                        expanded_x_km=x,expanded_y_km=y,expanded_log10_cm=q,probabilities=[.025,.5,.975])
    with plt.rc_context(STYLE):
        ratios = [(b[2]-b[0])/(b[3]-b[1]) for b in (local,bounds)]
        if min(ratios)>1.6:
            shape,size,kw = (2,1),(12.,10.6/ratios[0]+10.6/ratios[1]+2.4),{}
        else:
            shape,size,kw = (1,2),(13.2,max(5.8,11.7/sum(ratios)+2.2)),{"width_ratios":ratios}
        fig, axes = plt.subplots(*shape,figsize=size,layout="constrained",squeeze=False,gridspec_kw=kw)
        a,b = axes.flat
        a.add_collection(PolyCollection(geom.mesh.triangles,facecolors="#D3E9E5",edgecolors="none",zorder=1))
        if segments:
            lines = LineCollection(segments,cmap="viridis",norm=norm,linewidths=2,zorder=4)
            lines.set_array(np.asarray(levels))
            a.add_collection(lines,autolim=False)
        setup_map(a,local,meta["vent_lonlat"])
        observations(a,data)
        a.set_title("Measured-bracketed median isopach fragments",fontsize=11)
        image = field_panel(b,x,y,q[1],bounds,data,meta,norm,geom.mesh)
        b.set_title("Conditional median isopachs across the deposit (cm)",fontsize=11)
        fig.suptitle(f"{name} · directional ensemble",fontsize=15)
        observation_legend(fig,data,[Line2D([],[],color="#278B86",ls="--",label="Measurement footprint")])
        thickness_bar(fig,image,list(axes.flat),norm)
        save(fig,output,"isopachs.png")
        shape,size = panel_layout(local)
        fig,axes = plt.subplots(*shape,figsize=size,layout="constrained",squeeze=False)
        a,b = axes.flat
        image = field_panel(a,lx,ly,lq[1],local,data,meta,norm,geom.mesh)
        a.set_title("Median thickness and measurement footprint",fontsize=11)
        u = b.pcolormesh(lx,ly,lq[2]-lq[0],cmap="magma",shading="auto",vmin=0,zorder=1)
        setup_map(b,local,meta["vent_lonlat"])
        observations(b,data)
        footprint(b,geom.mesh,color="#9BE3DB")
        b.set_title("Pointwise 95% latent thickness uncertainty",fontsize=11)
        fig.colorbar(u,ax=b,location="bottom",shrink=.8,pad=.06,aspect=44,label="Orders of magnitude between upper and lower thickness")
        thickness_bar(fig,image,[a],norm)
        fig.suptitle(f"{name} · local detail",fontsize=14)
        observation_legend(fig,data)
        save(fig,output,"local.png")
    world = world_map(output,name,data,meta,view,norm)
    maxima = []
    for xx, yy, zz in ((x,y,q[1]),(lx,ly,lq[1])):
        iy,ix = np.unravel_index(np.nanargmax(zz),zz.shape)
        maxima.append((float(zz[iy,ix]),[float(xx[ix]),float(yy[iy])]))
    peak,point = max(maxima,key=lambda item:item[0])
    return {"statistic":view.statistic,"pointwise_probabilities":[.025,.5,.975],
            "fields_per_structure":model.sizes,"local_bounds_km":local.tolist(),"global_bounds_km":bounds.tolist(),
            "extent_audit":extent,"world":world,"vent_median_cm":float(10**view.central([[0.,0.]])[0]),
            "max_on_display_grids_cm":float(10**peak),"max_xy_km":point,
            "max_source_distance_chord_km":float(polar(np.array([point]),geom.axis)[0][0]),
            "displayed_triangle_levels":int(table.displayed.sum()),"triangle_levels":len(table)}
