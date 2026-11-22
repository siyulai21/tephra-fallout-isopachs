"""Plot accepted Pinatubo observations by location, bearing, distance and measurement type."""
import json
from hashlib import sha256

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from .pinatubo import AUDIT, COMPASS, INPUT
from src.Laea import make_laea_transform
from src.empirical_isopach.visualization.maps import frame_bounds, setup_map
from src.empirical_isopach.visualization.style import STYLE

GROUPS = {"Terrestrial surveys":"#176f77", "Marine cores":"#9d5425", "Airfield limits":"#794c94"}
KINDS = {"measurement":("o","Measured"), "lower_limit":("^","Lower bound"), "upper_limit":("v","Upper bound")}


def figure():
    records = pd.read_csv(AUDIT/"selection.csv").dropna(subset=["accepted_csv_row"]).copy()
    raw = pd.read_csv(INPUT)
    assert records.id.tolist() == raw.EvidenceID.tolist()
    vent = raw.loc[0, ["Vent Lon", "Vent Lat"]].to_numpy(float)
    proj, _, _ = make_laea_transform(*vent)
    records["x"], records["y"] = proj.transform(records.longitude, records.latitude)
    records["group"] = np.select([records.source.isin(["P2","K3","P7"]), records.source.eq("AIR")],
                                 ["Terrestrial surveys","Airfield limits"], default="Marine cores")
    idx = records.sector.map({v:i for i,v in enumerate(COMPASS)})
    records["sector_offset"] = idx + ((records.bearing_deg-45*idx+180)%360-180)/55
    land = records[records.group.eq("Terrestrial surveys")]
    panels = [(frame_bounds(records[["x","y"]].to_numpy(),margin=.07), "All accepted sites"),
              (frame_bounds(land[["x","y"]].to_numpy(),margin=.10), "Luzon field surveys")]
    with plt.rc_context(STYLE):
        fig = plt.figure(figsize=(10.8,8.6),layout="constrained")
        grid = fig.add_gridspec(2,2,height_ratios=(1,.85))
        axs = [fig.add_subplot(grid[0,0]),fig.add_subplot(grid[0,1])]
        bearing = fig.add_subplot(grid[1,:])
        for ax,(bounds,title) in zip(axs,panels,strict=True):
            setup_map(ax,bounds,vent)
            ax.set(title=title,xlabel="Easting from vent (km)",ylabel="Northing (km)")
            ax.scatter([0],[0],marker="*",s=115,c="white",edgecolors="#293e42",zorder=7)
        for group, color in GROUPS.items():
            for kind,(marker,_) in KINDS.items():
                sub=records[records.group.eq(group)&records.kind.eq(kind)]
                for ax in axs:
                    ax.scatter(sub.x,sub.y,marker=marker,s=25,facecolor=color,edgecolor="white",linewidth=.45,zorder=5)
                bearing.scatter(sub.km,sub.sector_offset,marker=marker,s=30,c=color,edgecolor="white",linewidth=.4,zorder=4)
        for ax,ident,offset,label in [
            (axs[0],"S99-SO-114-23",(-28,-24),"SO-114-23 · 1 cm"),
            (axs[0],"H01-16",(-75,-18),"SO-114-13 · 1–2 mm"),
            (axs[1],"P7-096",(-57,-18),"P7-096 · 33 cm"),
            (axs[1],"P7-099",(12,-21),"P7-099 · 32 cm")]:
            row=records.loc[records.id.eq(ident)].iloc[0]
            ax.annotate(label,(row.x,row.y),xytext=offset,textcoords="offset points",fontsize=8,zorder=8,
                        bbox=dict(facecolor="white",edgecolor="none",alpha=.88,pad=1.5),
                        arrowprops=dict(arrowstyle="-",color="#424f50",lw=.75))
        bearing.set_xscale("log")
        ticks=[10,20,50,100,200,500,900]
        bearing.set_xticks(ticks,labels=[str(t) for t in ticks])
        bearing.set(xlim=(7,1020),ylim=(7.55,-.55),yticks=range(8),yticklabels=COMPASS,
                    xlabel="Distance from Pinatubo (km, logarithmic scale)",ylabel="Bearing from vent",
                    title="Distance and direction of each accepted observation")
        bearing.grid(axis="x",alpha=.25)
        handles=[Line2D([],[],marker="o",ls="",c=c,label=f"{k} ({sum(records.group.eq(k))})") for k,c in GROUPS.items()]
        handles += [Line2D([],[],marker=m,ls="",color="#4d5d60",label=s) for m,s in KINDS.values()]
        handles += [Line2D([],[],marker="*",ls="",markerfacecolor="white",markeredgecolor="#293e42",label="Vent")]
        fig.legend(handles=handles,loc="outside lower center",ncols=4,frameon=False)
        fig.suptitle("Pinatubo 1991 · measured Layer C thicknesses and published bounds",fontsize=14,weight="bold")
        fig.savefig(AUDIT/"coverage.png",dpi=175,metadata={"Description":json.dumps({
            "input_sha256":sha256(INPUT.read_bytes()).hexdigest(),"observations":len(records),
            "group_counts":{k:int(records.group.eq(k).sum()) for k in GROUPS}})})
        plt.close(fig)


if __name__=="__main__":
    figure()
