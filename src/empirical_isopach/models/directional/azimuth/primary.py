"""Replay the quantitative uncertain-azimuth field for the principal directional maps."""
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
import numpy as np

from src.probabilistic_isopach.figures import STYLE
from ....visualization.global_maps import save, thickness_bar, world_map
from ....visualization.maps import (frame_bounds, global_frame, grid_axes, level_labels,
                                  observation_legend, observations, setup_map, THICKNESS_LEVELS)
from ..ensemble.fields import Ensemble, Member
from ..ensemble.maps import MapView
from ..ensemble.members import MEMBERS
from ..geometry import Geometry
from ..study import digest, write_json
from .field import AzimuthField
from .maps import mapped_archives, mark_archives, supported_lines


def read_field(folder, data, report):
    """Rebuild the exact saved joint fields, including the concentration, phase and weights."""
    folder = Path(folder)
    path = folder/"fields.npz"
    hashes = json.loads((folder/"artifacts.json").read_text())
    if digest(path) != hashes["fields.npz"]:
        raise ValueError("Uncertain-azimuth field checksum mismatch")
    if sorted(data.row_ids.tolist()) != sorted(report["data_rows"]):
        raise ValueError("Field packet and observations have different row identities")
    policies = report["weights"]
    members = []
    with np.load(path) as a:
        for policy in MEMBERS:
            prefix = policy.name+"_"
            draws = {key[len(prefix):]:a[key] for key in a
                     if key.startswith(prefix) and key != prefix+"indices"}
            geom = Geometry(data,report["input"]["vent_lonlat"][1],policy)
            members.append(Member(geom,draws,a[prefix+"indices"],[]))
        model = AzimuthField(Ensemble(members,policies["structure"]),data,policies["concentration"])
        if (not np.array_equal(model.k,a["concentration_index"])
                or not np.allclose(model.phase,a["azimuth_rad"],rtol=0,atol=1e-13)
                or not np.allclose(model.weights,a["field_weights"],rtol=1e-12,atol=0)):
            raise ValueError("Restored directional mixture differs from the published field states")
        # Downstream maps use the stored quantitative weights, even when a packet
        # also contains importance weights for a qualitative archive sensitivity.
        model.k, model.phase, model.weights = a["concentration_index"], a["azimuth_rad"], a["field_weights"]
    return model


def median_grid(view, bounds):
    x,y = grid_axes(bounds,longest=241)
    xx,yy = np.meshgrid(x,y)
    median = view.central(np.column_stack((xx.ravel(),yy.ravel()))).reshape(xx.shape)
    return x,y,median


def draw_primary(folder,data,meta,report,model,inventory,source):
    with np.load(folder/"median-grids.npz") as a:
        x,y,wide,local = [a[k] for k in ("expanded_x_km","expanded_y_km","expanded_log10_cm","local_log10_cm")]
    with np.load(source/"map-grid.npz") as a:
        sx,sy,source_median = a["x_km"],a["y_km"],a["revised_log10_cm"]
    bounds = [x[0],y[0],x[-1],y[-1]]
    near = [sx[0],sy[0],sx[-1],sy[-1]]
    norm = Normalize(-1.,max(1.,np.ceil(max(np.nanmax(wide),np.nanmax(local),np.nanmax(source_median)))))
    sites = mapped_archives(inventory)
    with plt.rc_context(STYLE):
        fig,axes = plt.subplots(1,2,figsize=(12,6),layout="constrained")
        for ax,gx,gy,z,frame,title in (
            (axes[0],x,y,wide,bounds,"Deposit extent"),(axes[1],sx,sy,source_median,near,"Near the volcano")
        ):
            image = ax.pcolormesh(gx,gy,z,shading="auto",cmap="viridis",norm=norm,rasterized=True)
            levels = np.log10(THICKNESS_LEVELS)
            levels = levels[(levels>=norm.vmin)&(levels>np.nanmin(z))&(levels<np.nanmax(z))]
            if len(levels):
                outlines = ax.contour(gx,gy,z,levels=levels,colors="white",linewidths=.85,linestyles="dashed")
                supported_lines(ax,data,model,gx,gy,z,levels,["white"]*len(levels))
                major = levels[np.isclose(levels,levels.round())]
                if len(major):
                    ax.clabel(outlines,levels=major,fmt=level_labels,fontsize=8)
            setup_map(ax,frame,meta["vent_lonlat"])
            observations(ax,data,size=16)
            mark_archives(ax,sites,meta["vent_lonlat"],frame,labels=ax is axes[1])
            ax.set_title(title,loc="left",fontweight="bold")
        fig.suptitle(f"{report['dataset']} · uncertain fan direction\nMedian isopachs and whole-Earth map volume of "
                     f"{report['volume']['median_map_km3']:,.1f} km³",fontsize=14)
        thickness_bar(fig,image,list(axes),norm)
        handles = [Line2D([],[],color="#324851",lw=2,label="Measured bracket"),
                   Line2D([],[],color="#324851",ls="--",label="Model continuation")]
        if sites:
            handles.append(Line2D([],[],ls="",marker="D",color="#b88917",label="Interpreted archive limit" if meta.get("interpreted_upper_limits") else "Inspected archive"))
        observation_legend(fig,data,handles)
        save(fig,folder,"isopachs.png")
    return norm,near


def render_primary(folder,data,meta,report,model,inventory,*,source=None):
    """Use the same field as the whole-Earth integral at every plotted location."""
    folder = Path(folder)
    source = Path(source) if source is not None else folder
    view = MapView(model,meta["vent_lonlat"][1])
    view.statistic = "Uncertain fan direction · pointwise median"
    bounds,extent = global_frame(data.xy,meta["vent_lonlat"][1],view.central)
    x,y,wide = median_grid(view,bounds)
    local_bounds = frame_bounds(data.xy)
    lx,ly,local = median_grid(view,local_bounds)
    np.savez_compressed(folder/"median-grids.npz",local_x_km=lx,local_y_km=ly,local_log10_cm=local,
                        expanded_x_km=x,expanded_y_km=y,expanded_log10_cm=wide)
    norm,near = draw_primary(folder,data,meta,report,model,inventory,source)
    world = world_map(folder,report["dataset"],data,meta,view,norm,show_contours=False)
    write_json(folder/"map-display.json",{
        "protocol":report["protocol"],"surface":"quantitative pointwise median",
        "input_sha256":meta["input_sha256"],"fields_sha256":digest(source/"fields.npz"),
        "median_map_km3":report["volume"]["median_map_km3"],"colour_limits_log10_cm":[norm.vmin,norm.vmax],
        "local_bounds_km":local_bounds.tolist(),"expanded_bounds_km":bounds.tolist(),"extent_audit":extent,
        "source_bounds_km":near,"world":world})


def numerical_notes(report):
    volume = report["volume"]
    checks = volume["numerical"]
    notes = []
    failed = sum(not m["diagnostics"]["passed"] for m in report["members"].values())
    if failed:
        notes.append(f"{failed} of six full-data structures carry sampling warnings.")
    if not checks["spatial_passed"]:
        notes.append("The whole-Earth spatial integration check flagged instability.")
    if not checks["split_map_passed"]:
        levels = [f"{a['level_cm']:g} cm ({a['symmetric_difference_fraction']:.0%} area disagreement)"
                  for a in checks["split_areas"] if a["effective_union_nodes"]>=64
                  and a["symmetric_difference_fraction"]>=.1]
        notes.append("Split posterior draws flagged "+(", ".join(levels) if levels else "the median map volume")+".")
    if not checks["joint_quantile_split_passed"]:
        notes.append("The complete-volume quantiles changed by more than 10% across draw splits.")
    return notes


def family_page(folder):
    """Publish the signed standalone report at the directional family entry point."""
    folder = Path(folder)
    assets = folder / "azimuth"
    manifest = assets / "artifacts.json"
    hashes = json.loads(manifest.read_text())
    standalone = assets / "README.md"
    page = folder / "README.md"
    heading = "\n## Related directional studies\n"
    promote = "README.md" in hashes
    if promote:
        if digest(standalone) != hashes.pop("README.md"):
            raise ValueError("The standalone directional report differs from its saved checksum")
        content = standalone.read_text()
        for name in (*hashes, "artifacts.json"):
            content = content.replace(f"]({name})", f"](azimuth/{name})")
    else:
        if digest(page) != hashes["../README.md"]:
            raise ValueError("The directional family report differs from its saved checksum")
        content = page.read_text().partition(heading)[0]
    links = [(name, title) for name, title in (
        ("ensemble/README.md", "Earlier fixed-direction ensemble and its separate calibration"),
        ("single-structure/README.md" if (folder/"single-structure/README.md").is_file() else "single-structure.md",
         "Single-structure and proximal-cap sensitivities"),
        ("bearing/README.md", "Measured-fan continuation and inspected-site evidence"),
    ) if (folder / name).is_file()]
    if links:
        content = content.rstrip() + "\n" + heading + "\n" + "\n".join(
            f"- [{title}]({name})" for name, title in links)
    temporary = page.with_suffix(".md.tmp")
    temporary.write_text(content.rstrip() + "\n")
    temporary.replace(page)
    hashes["../README.md"] = digest(page)
    write_json(manifest, hashes)
    if promote:
        standalone.unlink(missing_ok=True)
