"""Show the offshore archives against the principal Pinatubo posterior median field."""
from argparse import ArgumentParser
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from PIL import Image

from .global_case_validation import METHODS
from .pinatubo_nested_review import OUT
from .pinatubo_offshore import INPUT
from src.empirical_isopach.core.audit import audited_data
from src.empirical_isopach.models.directional.archive_readings import load_inventory, positions
from src.Laea import make_laea_transform
from src.empirical_isopach.visualization.deposit import read_json
from src.empirical_isopach.visualization.deposit_field import PolarField, digest
from src.empirical_isopach.visualization.deposit_maps import draw_land
from src.empirical_isopach.visualization.maps import coast_rings, frame_bounds, grid_axes
from src.empirical_isopach.visualization.shorelines import detail


def comparison_figure(fields, xy, sites, bounds, vent, target, receipt):
    scheme = {
        "directional": ("Directional ensemble", "#784a9c", "solid"),
        "spherical-mixture": ("Spherical mixture", "#17668b", "dashed"),
        "nested-isopachs": ("Nested footprints", "#aa4620", "dashdot"),
    }
    symbols = {
        "inspected_no_bed": ("No visible bed", "o", "none"),
        "occurrence_only": ("Ash observed", "o", "#4c6369"),
        "unclear_thickness": ("Disturbed or unclear", "x", "#657174"),
        "unresolved_horizon": ("Unresolved horizon", "+", "#657174"),
        "dispersed_no_continuous_bed": ("Dispersed ash", "D", "#a98f62"),
    }
    fig, axes = plt.subplots(1, 2, figsize=(10.3, 6.2), layout="constrained")
    fig.suptitle("Pinatubo Layer C · the 1 cm isopach offshore", fontsize=17)
    readings = np.array([s["reading"] for s in sites])
    for ax, box, title in zip(axes, bounds, ("All 151 marine survey records", "Extent of the three 1 cm footprints"), strict=True):
        shore = detail(box)
        draw_land(ax, box, vent, shore["raster_long_side"], shoreline=shore)
        x, y = grid_axes(box, longest=430, minimum=175)
        xx, yy = np.meshgrid(x, y)
        for family, field in fields.items():
            _, color, style = scheme[family]
            grid = field.at_xy(xx, yy)
            ax.contour(x, y, grid, levels=[0.0], colors=[color], linestyles=style, linewidths=1.9, zorder=4)
        for kind, (_, marker, fill) in symbols.items():
            points = xy[readings == kind]
            if marker in ("x", "+"):
                ax.scatter(*points.T, marker=marker, c=fill, s=24, linewidths=.85, zorder=5)
            else:
                ax.scatter(*points.T, marker=marker, facecolors=fill, edgecolors="#384c50",
                           s=20 if marker=="o" else 29, linewidths=.7, zorder=5)
        ax.scatter(0, 0, marker="*", facecolor="white", edgecolor="#1b252a", s=110, linewidth=.9, zorder=6)
        ax.set(xlim=(box[0],box[2]), ylim=(box[1],box[3]), aspect="equal",
               xlabel="Easting from vent (km)", title=title)
        ax.grid(alpha=.16)
    axes[0].set_ylabel("Northing from vent (km)")
    model_handles = [Line2D([], [], color=c, ls=ls, lw=2.2, label=label) for label,c,ls in scheme.values()]
    site_handles = [Line2D([], [], color="#384c50", marker=m, markerfacecolor=fill, ls="", ms=5, label=label)
                    for label,m,fill in symbols.values()]
    site_handles.append(Line2D([], [], color="#1b252a", marker="*", markerfacecolor="white", ls="", ms=9, label="Vent"))
    fig.legend(handles=[*model_handles,*site_handles], loc="outside lower center", ncols=3, frameon=False, fontsize=9)
    fig.savefig(target, dpi=180, metadata={"Description": json.dumps(receipt)})
    plt.close(fig)


def station_comparison(home, target, *, verify=False):
    """Read the displayed median maps at the exact positions in the offshore audit."""
    _, meta = audited_data(INPUT)
    inventory, _ = load_inventory(INPUT)
    sites = inventory["sites"]
    xy = positions(sites, meta["vent_lonlat"])
    table = pd.DataFrame({
        "site_id": [s["id"] for s in sites],
        "reading": [s["reading"] for s in sites],
        "latitude": [s["latitude"] for s in sites],
        "longitude": [s["longitude"] for s in sites],
        "quantitative_role_in_audit": [s.get("in_principal_as", "") for s in sites],
    })
    fields, grids, outlines = {}, {}, []
    for family in METHODS:
        folder = home / family
        grid = folder / "median-grid.npz"
        receipt = read_json(folder / "deposit.json")
        assert receipt["input_sha256"] == meta["input_sha256"] and receipt["median_grid_sha256"] == digest(grid)
        with np.load(grid, allow_pickle=False) as packet:
            field = PolarField(packet["radii_km"], packet["log10_cm"], float(packet["axis"]))
            values = 10.0 ** field.at_xy(xy[:, 0], xy[:, 1])
        fields[family] = field
        grids[family] = receipt["median_grid_sha256"]
        radius = field.outer_crossings(1.0)
        arc = np.column_stack((radius * field.axis * np.sin(field.angles), radius / field.axis * np.cos(field.angles)))
        outlines.extend(arc[np.isfinite(radius)])
        assert np.isfinite(values).all() and (values >= 0).all()
        table[family + "_mapped_median_cm"] = values
    assert table.site_id.is_unique and len(table) == 151
    bounds = [frame_bounds(np.vstack((xy, outlines)), margin=.07), frame_bounds(np.asarray(outlines), margin=.10)]
    from src.empirical_isopach.core.catalog import data_directory, dataset_record
    audit = data_directory() / dataset_record(INPUT)["archive_evidence"]
    figure = target.with_name("marine-comparison.png")
    receipt = {"input_sha256": meta["input_sha256"], "marine_inventory_sha256": digest(audit), "contour_cm": 1.,
               "median_grid_sha256": grids, "bounds_km": [b.tolist() for b in bounds],
               "sites": len(table), "at_or_above_1_cm": {f: int((table[f + "_mapped_median_cm"] >= 1).sum()) for f in fields}}
    if verify:
        with Image.open(figure) as image:
            observed = json.loads(image.info["Description"])
        np.testing.assert_allclose(observed.pop("bounds_km"), receipt.pop("bounds_km"), rtol=1e-8, atol=1e-6)
        assert observed == receipt
        saved = pd.read_csv(target, keep_default_na=False, float_precision="round_trip")
        pd.testing.assert_frame_equal(table, saved, check_dtype=False, rtol=1e-6, atol=1e-12)
    else:
        table.to_csv(target, index=False)
        comparison_figure(fields, xy, sites, bounds, meta["vent_lonlat"], figure, receipt)


def plot(folder):
    readings = pd.read_csv(folder / "marine-predictions.csv")
    accepted = pd.read_csv(INPUT)
    vent = accepted.iloc[0][["Vent Lon", "Vent Lat"]].to_numpy(dtype=float)
    project, _, _ = make_laea_transform(*vent)
    with np.load(folder / "median-grid.npz", allow_pickle=False) as packet:
        field = PolarField(packet["radii_km"], packet["log10_cm"], float(packet["axis"]))
    categories = [
        ("inspected_no_bed", "No visible bed", "o", "none", "#28627e", 25),
        ("occurrence_only", "Ash observed", "o", "#c77b34", "#744b29", 22),
        ("unclear_thickness", "Disturbed deposit", "x", "#67696b", None, 23),
        ("unresolved_horizon", "Unresolved symbol", "+", "#626065", None, 29),
        ("dispersed_no_continuous_bed", "Mixed ash on X-ray", "D", "#842955", "#842955", 42),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(11.1, 6.1), gridspec_kw={"width_ratios":[0.83, 1.17]})
    fig.subplots_adjust(top=.86, bottom=.22, wspace=.18, left=.055, right=.98)
    handles = []
    for i, (ax, bounds, title) in enumerate(zip(axes, [(108.8,121.8,5.4,22.2),(112.5,121.2,12,17.8)],
                                               ["All survey positions", "Western fallout and its margins"])):
        west,east,south,north = bounds
        ax.set(xlim=(west,east), ylim=(south,north), aspect=1/np.cos(np.radians(15)),
               xlabel="Longitude (°E)", title=title)
        ax.grid(alpha=.18)
        if i==0:
            ax.set_ylabel("Latitude (°N)")
        xx,yy = np.meshgrid(np.linspace(west,east,450),np.linspace(south,north,450))
        x,y = project.transform(xx,yy)
        z = field.at_xy(x,y)
        lines = ax.contour(xx,yy,z,levels=np.log10([.1,.5,1,2,4]),colors="#1d7571",
                           linewidths=.9,linestyles="dashed")
        ax.clabel(lines, fmt=lambda v:f"{10**v:g} cm", fontsize=7)
        rings = [v for v in coast_rings() if v[:,0].min() <= east and v[:,0].max() >= west
                 and v[:,1].min() <= north and v[:,1].max() >= south]
        ax.add_collection(LineCollection(rings, colors="#87938b", linewidths=.52, zorder=1))
        for category,label,mark,fill,edge,size in categories:
            selected=readings[readings.reading.eq(category)]
            style=dict(c=fill) if edge is None else dict(facecolors=fill,edgecolors=edge)
            artist=ax.scatter(selected.lon,selected.lat,marker=mark,s=size,linewidths=.95,zorder=3,label=label,**style)
            if i==0:
                handles.append(artist)
        star=ax.scatter(*vent,marker="*",s=85,color="#28353a",zorder=5,label="Pinatubo")
        if i==0:
            handles.append(star)
        else:
            mixed=readings.loc[readings.site_id.eq("W09-47")].iloc[0]
            ax.annotate("SO132-47",(mixed.lon,mixed.lat),xytext=(-7,12),textcoords="offset points",
                        ha="right",color="#672744",fontsize=8)
    fig.suptitle("Pinatubo 1991  |  Offshore core evidence",fontsize=17,fontweight="bold")
    fig.legend(handles=handles,ncol=3,frameon=False,loc="lower center",bbox_to_anchor=(.5,.025),fontsize=9)
    fig.savefig(folder/"marine-survey.png",dpi=160,facecolor="white")
    plt.close(fig)


if __name__=="__main__":
    p=ArgumentParser(description=__doc__)
    p.add_argument("--folder",type=Path,default=OUT)
    p.add_argument("--station-table", type=Path, help="Compare all three principal maps at the audited marine stations")
    p.add_argument("--verify", action="store_true", help="Replay the saved station table")
    args = p.parse_args()
    if args.station_table:
        station_comparison(args.folder.parent, args.station_table, verify=args.verify)
    elif args.verify:
        p.error("--verify requires --station-table")
    else:
        plot(args.folder)
