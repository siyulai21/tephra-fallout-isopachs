"""Principal global deposit plate. Distances and areas use the source-centred WGS84 equal-area chart."""
from functools import lru_cache
import gzip
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import colors, patheffects
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter, MaxNLocator
import numpy as np
import shapely

from src.Laea import make_laea_transform
from .deposit_field import FIRST_CM
from .contour_limits import local_half_width, outer_contour, near_frame, near_levels, visible_levels
from .maps import frame_bounds
from .shorelines import detail, local_mask

LAND = Path(__file__).parents[1]/"assets/ne_10m_land.geojson.gz"
INK = "#203d47"
WATER = "#d8eaf0"
GROUND = "#ebe3ce"
UNDER = "#9cabb1"
# Fixed lower-edge colours, shared across all eruptions and families.
PALETTE = ("#137889", "#168f8c", "#40a488", "#77b477", "#b2b85f", "#d4ad46",
           "#e39541", "#dd7443", "#c8584a", "#ad4356", "#8b365c", "#70365f",
           "#563256", "#452945", "#38395c", "#2c4970", "#20577d", "#0c6681", "#08748b")
FMT = FuncFormatter(lambda v, _: f"{v:,.0f}")


@lru_cache(maxsize=1)
def geography():
    land, lakes = shapely.get_parts(shapely.from_geojson(gzip.decompress(LAND.read_bytes())))
    shapely.prepare(land)
    shapely.prepare(lakes)
    return land, lakes


def geography_mask(x, y, vent):
    xx, yy = np.meshgrid(x,y)
    _, inv, _ = make_laea_transform(*vent)
    lon, lat = inv.transform(xx,yy)
    ok = np.isfinite(lon) & np.isfinite(lat)
    land, lakes = geography()
    mask = np.full(xx.shape,-1,dtype=np.int8)
    mask[ok] = (shapely.contains_xy(land,lon[ok],lat[ok]) & ~shapely.contains_xy(lakes,lon[ok],lat[ok]))
    return mask


def frames(field, data, outer_cm=FIRST_CM):
    outer = field.outer_crossings(outer_cm)
    ok = np.isfinite(outer)
    reach = outer[ok]
    angle = field.angles[ok]
    rim = bool(np.any(reach >= field.radii[-1]*(1.-1e-9)))
    if rim:
        a, b = field.radii[-1]*field.axis, field.radii[-1]/field.axis
        bounds = np.array([-a,-b,a,b])*1.025
    elif len(reach) and np.max(reach) > 0:
        xy = np.column_stack((reach*np.sin(angle)*field.axis,reach*np.cos(angle)/field.axis))
        bounds = frame_bounds(xy,.09)
        # Include extrema lying between angular samples, including detached lobes.
        pad = reach.max()*2*np.pi/len(outer)*max(field.axis,1/field.axis)
        bounds += pad*np.array([-1.,-1.,1.,1.])
    else:
        bounds = frame_bounds(data.xy)
    # Prevent long, thin lobes from forcing unreadably flat panels.
    centre = (bounds[:2]+bounds[2:])/2
    span = bounds[2:]-bounds[:2]
    span = np.maximum(span,[span[1]*.91,span[0]/2.12])
    bounds = np.r_[centre-span/2,centre+span/2]
    local = local_half_width(data)
    return bounds, np.array([-local,-local,local,local]), rim


def regular_axes(bounds, longest):
    delta = bounds[2:]-bounds[:2]
    return tuple(np.linspace(a,b,max(61,round(longest*d/max(delta))))
                 for a,b,d in zip(bounds[:2],bounds[2:],delta,strict=True))


def colours(count):
    # The same colour always denotes the same thickness bracket, also below 0.5 cm.
    extra = [colors.to_hex(plt.get_cmap("cividis")(min(.9,.22+.07*i))) for i in range(max(0,count-len(PALETTE)))]
    return list(PALETTE[:count])+extra


def plot_observations(ax, data, meta, levels, palette, bounds):
    ranks = np.searchsorted(np.log(levels),data.log_value,side="right")
    fill = np.array([UNDER,*palette])[ranks]
    interpreted = np.isin(data.row_ids,[row["row_id"] for row in meta.get("interpreted_upper_limits",[])])
    data = data.reported_bounds()
    upper = data.censored & ~data.lower_censored
    interval = np.isfinite(getattr(data,"interval_log_upper",np.full(len(ranks),np.nan)))
    kinds = [(~data.censored & ~interval,"o","Measurement"), (data.lower_censored & ~interval,"^","Lower limit"),
             (upper & ~interpreted & ~interval,"v","Upper limit"), (interpreted,"D","Interpreted upper limit"),
             (interval,"s","Interval (lower edge)")]
    handles = []
    visible = ((data.xy>=bounds[:2]) & (data.xy<=bounds[2:])).all(axis=1)
    for mask,mark,label in kinds:
        if not mask.any():
            continue
        select = mask & visible
        ax.scatter(*data.xy[select].T,s=23 if mark=="o" else 38,c=fill[select],marker=mark,
                   edgecolors=INK,linewidths=.65,zorder=7)
        handles.append(Line2D([],[],ls="",marker=mark,mfc="white",mec=INK,ms=6,label=label))
    ax.scatter(0,0,marker="*",s=125,c="white",edgecolors="#222b30",linewidths=1.1,zorder=8)
    handles.append(Line2D([],[],ls="",marker="*",mfc="white",mec=INK,ms=10,label="Vent"))
    return handles, int(visible.sum())


def support_lines(ax,family,data,model,x,y,z,levels):
    from ..models.directional.multiscale.posterior import DirectionalField
    if family=="directional" and not isinstance(model,DirectionalField):
        from ..models.directional.azimuth.maps import supported_lines
        supported_lines(ax,data,model,x,y,z,levels,[INK]*len(levels))
    elif family=="nested-isopachs" or isinstance(model,DirectionalField):
        import contourpy
        from ..support import SupportMesh
        from ..models.nested.reach_report import bracketed_regions, line_parts
        regions,_ = bracketed_regions(data,SupportMesh(data),model,levels)
        lines = contourpy.contour_generator(x=x,y=y,z=np.ma.masked_invalid(z))
        pieces = [piece for level in levels for line in lines.lines(level) if len(line)>1
                  for piece in line_parts(shapely.LineString(line).intersection(regions[level]))]
        ax.add_collection(LineCollection(pieces,colors=INK,linewidths=1.65,zorder=5),autolim=False)


def draw_land(ax,bounds,vent,resolution,*,shoreline=None):
    bx,by = regular_axes(bounds,resolution)
    mask = local_mask(bx,by,vent,shoreline["tier"]) if shoreline else geography_mask(bx,by,vent)
    ax.pcolormesh(bx,by,mask,cmap=colors.ListedColormap(["white",WATER,GROUND]),
                  vmin=-1,vmax=1,shading="nearest",rasterized=True,zorder=0)
    if np.any(mask==1) and np.any(mask==0):
        ax.contour(bx,by,np.ma.masked_less(mask,0),levels=[.5],colors="#829798",linewidths=.40,zorder=2)


def panel(ax, field, data, meta, family, model, bounds, levels, palette, large, outer_cm=FIRST_CM, inner_cm=np.inf):
    x,y = regular_axes(bounds,470 if large else 301)
    z = field.on_grid(x,y)
    shoreline = None if large else detail(bounds)
    draw_land(ax,bounds,meta["vent_lonlat"],shoreline["raster_long_side"] if shoreline else 890,shoreline=shoreline)
    show = (levels >= outer_cm) & (levels <= inner_cm)
    selected, band_colours = levels[show],np.asarray(palette)[show]
    if not large and inner_cm<=FIRST_CM:
        selected = near_levels(field,inner_cm)
        band_colours = [UNDER if v<FIRST_CM else palette[0] for v in selected]
    tiers = np.log10(selected)
    finite = z[np.isfinite(z)]
    visible = tiers[(tiers>finite.min()) & (tiers<finite.max())] if len(finite) else []
    if len(tiers) and len(finite) and finite.max()>=tiers[0]:
        ax.contourf(x,y,z,levels=np.r_[tiers,max(tiers[-1]+.31,finite.max()+.01)],
                    colors=band_colours,alpha=.30,antialiased=True,zorder=1)
    if len(visible):
        lines = ax.contour(x,y,z,levels=visible,colors=INK,linewidths=[1.05 if v==tiers[0] else .57 for v in visible],
                           linestyles="dashed",zorder=3)
        support_lines(ax,family,data,model,x,y,z,visible)
        # Use at most four widely spaced labels; the shared colour key identifies intermediate bands.
        chosen = visible[::max(1,int(np.ceil(len(visible)/4)))]
        labels = ax.clabel(lines,levels=list(chosen),fmt=lambda v:f"{10**v:g}",fontsize=8 if large else 7,
                           inline=True,inline_spacing=4)
        for label in labels:
            label.set_path_effects([patheffects.withStroke(linewidth=2,foreground="#fcfcf5")])
    handles, count = plot_observations(ax,data,meta,levels,palette,bounds)
    ax.set(xlim=(bounds[0],bounds[2]),ylim=(bounds[1],bounds[3]),aspect="equal")
    for axis in (ax.xaxis,ax.yaxis):
        axis.set_major_locator(MaxNLocator(5 if large else 4))
        axis.set_major_formatter(FMT)
    ax.tick_params(colors=INK,labelsize=8,length=3)
    ax.grid(alpha=.18,linewidth=.45,color="#62787a")
    for spine in ax.spines.values():
        spine.set_color("#97aaad")
    ax.set_xlabel("Easting from vent (km)",fontsize=8)
    ax.set_ylabel("Northing (km)",fontsize=8)
    return handles,count


def plate(path, name, family, field, data, meta, model, levels, rows, receipt, near_half=None):
    from .deposit_field import display_name
    selection = receipt["outer_contour"]
    outer_cm = selection["level_cm"]
    inner_cm = receipt["inner_contour"]["level_cm"]
    near_half = near_frame([field],data) if near_half is None else near_half
    bounds,_,rim=frames(field,data,outer_cm)
    near = np.array([-near_half,-near_half,near_half,near_half])
    height=np.clip(7.1*(bounds[3]-bounds[1])/(bounds[2]-bounds[0]),3.6,7.1)
    palette=colours(len(levels))
    fig=plt.figure(figsize=(8.25,height+5.65),layout="constrained")
    grid=fig.add_gridspec(4,2,height_ratios=[height,.55,3.12,.44],width_ratios=[1.03,1.])
    volume=receipt["median_map_volume_km3"]
    volume_label=f"{volume:,.0f}" if volume>=1000 else f"{volume:,.3g}"
    edition = f"   |   historical {len(data.xy)} sites" if receipt.get("historical_input") else ""
    fig.suptitle(f"{name}  ·  {display_name(family, receipt['sources_sha256'])}\nMedian thickness   |   {volume_label} km³ over Earth{edition}",
                 fontsize=15,color=INK,fontweight="bold")
    ax=fig.add_subplot(grid[0,:])
    handles,count=panel(ax,field,data,meta,family,model,bounds,levels,palette,True,outer_cm,inner_cm)
    covered = selection["inside"]
    floor_note = "  ·  1 cm floor" if not selection["target_met"] else ""
    full_scale = f"{outer_cm:g}–{inner_cm:g} cm" if inner_cm>outer_cm else f"{outer_cm:g} cm" if inner_cm==outer_cm else "no site-enclosing line at 1 cm"
    ax.set_title(f"Full deposit  ·  {full_scale}  ·  {covered}/{len(data.xy)} at or above {outer_cm:g} cm "
                 f"({covered/len(data.xy):.0%}){floor_note}",fontsize=10,loc="left")
    local=fig.add_subplot(grid[2,0])
    _,near_count=panel(local,field,data,meta,family,model,near,levels,palette,False,min(FIRST_CM,inner_cm),inner_cm)
    local_levels = visible_levels(field,near_half,near_levels(field,inner_cm))
    if not local_levels:
        local.text(.03,.03,"No site-enclosing contour crosses this view",transform=local.transAxes,fontsize=8,
                   bbox=dict(facecolor="white",alpha=.85,edgecolor="none"),zorder=10)
    local.set_title(f"Near vent  ·  ±{near[2]:g} km  ·  {near_count} sites in view",fontsize=10,loc="left")
    key=fig.add_subplot(grid[1,:]);key.set_axis_off()
    n=len(palette)+1
    for i,(colour,label) in enumerate(zip([UNDER,*palette],["<0.5",*[f"{v:g}" for v in levels]],strict=True)):
        key.add_patch(plt.Rectangle((i/n,.50),1/n,.22,transform=key.transAxes,facecolor=colour,ec="white",lw=.8))
        key.text((i+.5)/n,.43,label,ha="center",va="top",fontsize=9,color=INK)
    key.text(0,.89,"Thickness bracket starts at (cm)",fontsize=9,color=INK)
    curve=fig.add_subplot(grid[2,1])
    active=[r for r in rows if r["area_km2"]>0 and r["thickness_cm"]<=inner_cm]
    cm=np.array([r["thickness_cm"] for r in active])
    if len(cm):
        distance=[r["sqrt_area_km"] for r in active]
        curve.plot(distance,cm,color=INK,lw=1.25)
        curve.scatter(distance,cm,c=palette[:len(cm)],s=35,edgecolors=INK,linewidths=.7,zorder=3)
        curve.set_yscale("log",base=2)
        ticks=cm[::max(1,int(np.ceil(len(cm)/7)))]
        curve.set_yticks(ticks,labels=[f"{v:g}" for v in ticks])
        curve.set_ylim(cm.min()/1.45,cm.max()*1.45)
    else:
        curve.text(.5,.5,"No site-enclosing level from 0.5 cm",ha="center",transform=curve.transAxes)
    curve.set_xlim(left=0)
    curve.xaxis.set_major_formatter(FMT)
    curve.xaxis.set_major_locator(MaxNLocator(4))
    curve.set_xlabel("Square root of cumulative area (km)",fontsize=9)
    curve.set_ylabel("Thickness (cm)",fontsize=9)
    curve.set_title("Thickness versus √area",fontsize=11,loc="left")
    curve.tick_params(labelsize=9)
    curve.spines[["top","right"]].set_visible(False)
    curve.grid(color="#b1bfbd",alpha=.5,linewidth=.5)
    leg=fig.add_subplot(grid[3,:]);leg.set_axis_off()
    leg.legend(handles=handles,loc="upper center",ncol=len(handles),frameon=False,fontsize=8,handletextpad=.3,columnspacing=1.1)
    note="Dashed isopachs are conditional."
    if family!="spherical-mixture":note+=" Solid fragments have local measurement brackets."
    leg.text(.5,-.03,note,
             ha="center",va="top",fontsize=8.5,color=INK)
    receipt.update(full_bounds_km=bounds.tolist(),near_bounds_km=near.tolist(),full_sites=count,near_sites=near_count,
                   near_half_width_km=near[2], near_visible_levels_cm=local_levels, near_shoreline=detail(near),
                   full_levels_cm=levels[(levels>=outer_cm)&(levels<=inner_cm)].tolist(),
                   reached_antipodal_rim=rim,levels_cm=levels.tolist())
    fig.savefig(path,dpi=143,metadata={"Description":json.dumps(receipt,allow_nan=False)})
    plt.close(fig)
