"""Show the offshore archives against the principal Pinatubo posterior median field."""
from argparse import ArgumentParser
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.collections import LineCollection

from .pinatubo_nested_review import OUT
from .pinatubo_offshore import INPUT
from src.Laea import make_laea_transform
from src.empirical_isopach.visualization.deposit_field import PolarField
from src.empirical_isopach.visualization.maps import coast_rings


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
    plot(p.parse_args().folder)
