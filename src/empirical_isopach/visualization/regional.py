"""Complete-hull maps for regional peers, with common scales and frames."""
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Polygon
from scipy.interpolate import RegularGridInterpolator

from .contour_limits import inner_contour

from ..core.distributions import predictive_distribution
from src.empirical_isopach.visualization.style import STYLE
from .maps import (ASSET, INK, AMBER, frame_bounds, grid_axes, panel_layout,
                   level_labels)


def regional_grid(fit_bounds, region, models):
    """Evaluate the published regional display grid in the original fitting domain."""
    # Plot frame and spline fitting domain are distinct. Never enlarge or refit
    # a spline merely to include a source outside the observation hull.
    x, y = grid_axes(fit_bounds, longest=361, minimum=121)
    xx, yy = np.meshgrid(x, y)
    xy = np.column_stack((xx.ravel(), yy.ravel()))
    inside = region.contains(xy)
    fields, table = {}, {"x_km":xy[:, 0], "y_km":xy[:, 1], "inside_observation_hull":inside}
    for method, model in models.items():
        parts = []
        for start in range(0, len(xy), 2048):
            d = predictive_distribution(model, xy[start:start+2048])
            parts.append(np.array([d.quantile(.5), d.log_expected_thickness(), d.quantile(.025), d.quantile(.975)]))
        values = np.concatenate(parts, axis=1)
        fields[method] = values[0].reshape(xx.shape)/np.log(10.)
        for key, row in zip(("median", "conditional_mean", "conditional_lower95", "conditional_upper95"), values):
            table[f"{method}_{key}_cm"] = np.exp(row)
    return x, y, inside, fields, table


def regional_maps(output, name, data, metadata, region, models, report):
    from ..reporting import report_label
    output = Path(output)
    fit_bounds = np.array(report["fit_bounds_km"])
    bounds = frame_bounds(np.vstack((region.vertices, data.xy)))
    x, y, inside, fields, table = regional_grid(fit_bounds, region, models)
    pd.DataFrame(table).to_csv(output/"predictions.csv", index=False)
    values = np.concatenate([v.ravel()[inside] for v in fields.values()])
    from .deposit_field import doubling_levels
    from .deposit_maps import LAND, UNDER, FMT, colours, draw_land, plot_observations
    limits = {}
    for method, field in fields.items():
        interpolant = RegularGridInterpolator((y,x),field)
        site_values = interpolant(data.xy[:,::-1])
        selection = {"mapped_log10_cm":site_values.tolist(),"row_ids":data.row_ids.tolist()}
        limits[method] = {**inner_contour(selection),"mapped_site_log10_cm":site_values.tolist()}
    maximum = max(np.exp(data.log_value.max()),max(item["level_cm"] for item in limits.values()))
    levels = doubling_levels(maximum)
    palette = colours(len(levels))
    cut = np.r_[min(-5.,values.min()-.1), np.log10(levels), max(values.max()+1,np.log10(levels[-1])+1)]
    cmap = ListedColormap([UNDER,*palette])
    norm = BoundaryNorm(cut,cmap.N,clip=True)
    methods = tuple(report["methods"])
    layout, size = panel_layout(bounds,len(methods))
    probes = {}
    with plt.rc_context(STYLE):
        fig, axs = plt.subplots(*layout,figsize=size,layout="constrained",squeeze=False)
        for ax, method in zip(axs.flat,methods,strict=True):
            ax.set(xlim=(bounds[0],bounds[2]),ylim=(bounds[1],bounds[3]),aspect="equal",
                   xlabel="Easting from vent (km)",ylabel="Northing (km)")
            ax.xaxis.set_major_formatter(FMT)
            ax.yaxis.set_major_formatter(FMT)
            draw_land(ax,bounds,metadata["vent_lonlat"],720)
            ceiling = limits[method]["level_cm"]
            shaded = np.minimum(fields[method],np.log10(ceiling)+1e-12)
            image = ax.pcolormesh(x,y,shaded,shading="auto",cmap=cmap,norm=norm,alpha=.43,rasterized=True,zorder=1)
            image.set_clip_path(Polygon(region.vertices,closed=True,transform=ax.transData))
            z = np.ma.array(fields[method],mask=(~inside).reshape(len(y),len(x)))
            line_levels = np.unique(np.r_[levels[levels<=ceiling],ceiling])
            selected = np.log10(line_levels)
            selected = selected[(selected>z.min()) & (selected<z.max())]
            limits[method]["drawn_levels_cm"] = np.exp2(np.rint(selected/np.log10(2.))).tolist()
            if len(selected):
                contours = ax.contour(x,y,z,levels=selected,colors=INK,linewidths=.66,zorder=3)
                labels = ax.clabel(contours,levels=list(selected[::max(1,int(np.ceil(len(selected)/5)))]),
                                   fmt=lambda v:f"{10**v:g}",fontsize=8)
                for lab in labels:
                    lab.set_path_effects([pe.withStroke(linewidth=2,foreground="white")])
            ax.add_patch(Polygon(region.vertices,closed=True,fill=False,ec=AMBER,ls="--",lw=1.1,zorder=5))
            handles,_ = plot_observations(ax,data,metadata,levels,palette,bounds)
            volume = report["methods"][method]["regional_volume"]["median_surface_km3"]
            enclosed = limits[method]["inside"]
            site_label = "site" if enclosed == 1 else "sites"
            ax.set_title(f"{report_label(report,method)} · {volume:,.3g} km³ inside hull\n"
                         f"Highest band ≥{ceiling:,.6g} cm · {enclosed} {site_label}",fontsize=10.5)
            k = np.flatnonzero(inside)[np.linspace(0,inside.sum()-1,9).astype(int)]
            iy,ix = np.unravel_index(k,z.shape)
            probes[method] = np.column_stack((x[ix],y[iy],fields[method][iy,ix])).tolist()
        legend = plt.cm.ScalarMappable(norm=norm,cmap=cmap)
        bar = fig.colorbar(legend,ax=list(axs.flat),location="bottom",shrink=.85,pad=.075,aspect=46,
                           label="Thickness bracket starts at (cm)")
        bar.set_ticks((cut[1:]+cut[:-1])/2,labels=["<0.5",*[f"{v:g}" for v in levels]])
        handles.append(Line2D([],[],color=AMBER,ls="--",label="Observation hull"))
        fig.legend(handles=handles,loc="outside lower center",ncol=min(6,len(handles)),frameon=False,fontsize=8)
        fig.suptitle(f"{name} · regional median thickness",fontsize=15)
        fig.savefig(output/"isopachs.png",dpi=155)
        plt.close(fig)
    contained = bool(np.all(region.vertices >= bounds[:2]) and np.all(region.vertices <= bounds[2:]))
    if not contained:
        raise AssertionError("The map must contain every hull vertex")
    display = {"protocol":"regional-median-deposit-v3", "input_sha256":metadata["input_sha256"],
        "display_bounds_km":bounds.tolist(), "unchanged_fit_bounds_km":fit_bounds.tolist(),
        "entire_observation_hull_in_frame":contained, "hull_raster_clip":"Exact polygon, not grid-centre mask",
        "all_accepted_locations_in_frame":bool(np.all(data.xy >= bounds[:2]) and np.all(data.xy <= bounds[2:])),
        "grid_shape":[len(y), len(x)], "coastline_scale":"1:10 million", "coastline_sha256":sha256(ASSET.read_bytes()).hexdigest(),
        "land_sha256":sha256(LAND.read_bytes()).hexdigest(), "check_points_x_y_km_log10_cm":probes,
        "levels_cm":levels.tolist(), "inner_contours":limits, "accepted_row_ids":data.row_ids.tolist(), "central_statistic":"Posterior median thickness", "volume_calculation_changed":False}
    (output/"map-display.json").write_text(json.dumps(display, indent=2)+"\n")
    return display
