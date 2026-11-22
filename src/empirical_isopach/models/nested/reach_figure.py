"""Full-compass comparison of nested superlevel sets, including disconnected regions."""
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np

from src.probabilistic_isopach.global_volume import EarthDomain
from .geometry import DIAMETER, positions


def compass_map(out, name, data, model, previous):
    # In this authalic chart, chord radius is 2R sin(arc / 2R).
    # Logarithmic plotting distances display both survey and global scales.
    scale = 25.
    maximum = np.pi * DIAMETER / 2.
    y = np.linspace(0., np.log1p(maximum/scale), 165)
    angle = np.linspace(0., 2*np.pi, 193)
    theta, radius_axis = np.meshgrid(angle, y)
    chord = DIAMETER * np.sin(scale*np.expm1(radius_axis)/DIAMETER)
    axis = EarthDomain.wgs84(model.latitude).axis_scale
    xy = np.column_stack(((axis*chord*np.sin(theta)).ravel(), (chord*np.cos(theta)/axis).ravel()))
    part, old = model.reduced(512), previous.reduced(512)
    levels = (1., 10.)
    chances = np.empty((2,len(levels),len(xy)))
    for ids in np.array_split(np.arange(len(xy)),128):
        for j, field in enumerate((old, part)):
            value = field.field(xy[ids])
            for k, cm in enumerate(levels):
                chances[j,k,ids] = (value >= np.log(cm)).mean(axis=0)
    r_obs,t_obs = positions(data.xy, model.latitude)
    y_obs = np.log1p(DIAMETER*np.arcsin(np.clip(r_obs/DIAMETER,0.,1.))/scale)
    fig, axs = plt.subplots(1,2,figsize=(11.9,6.),subplot_kw={"projection":"polar"},layout="constrained")
    for j,(ax,level) in enumerate(zip(axs,levels,strict=True)):
        ax.set_theta_zero_location("N")
        ax.set_theta_direction(-1)
        ax.set_xticks(np.arange(8)*np.pi/4, ("N","NE","E","SE","S","SW","W","NW"))
        ticks = [50,250,1000,5000,20000]
        ax.set_yticks(np.log1p(np.asarray(ticks)/scale), [f"{k:,}" for k in ticks],fontsize=7)
        ax.set_rlabel_position(30.)
        ax.set_ylim(0., y[-1])
        ax.grid(alpha=.33, linewidth=.65)
        p = chances[1,j].reshape(theta.shape)
        # Separate the conditional uncertainty belt from the median contour.
        if p.max() > .1:
            ax.contourf(theta,radius_axis,p,levels=(.1,.9,1.01),colors=("#dcece9","#f7faf9"),zorder=0)
        for i, color, style, lw in ((0,"#a04a2c","dashed",1.9),(1,"#005a67","solid",2.2)):
            prob = chances[i,j].reshape(theta.shape)
            if prob.min() < .5 < prob.max():
                ax.contour(theta,radius_axis,prob,levels=[.5],colors=color,linewidths=lw,linestyles=style,zorder=3)
        for above, face in ((False,"white"),(True,"#273034")):
            take = ~data.censored & ((data.log_value>=np.log(level)) == above)
            ax.scatter(t_obs[take],y_obs[take],s=10,facecolors=face,edgecolors="#273034",linewidths=.45,zorder=4)
        select = data.censored
        ax.scatter(t_obs[select],y_obs[select],s=17,marker="D",color="#db9d35",edgecolors="#584723",linewidths=.4,zorder=4)
        ax.set_title(f"{level:g} cm isopach",fontsize=12,pad=13)
    handles = [Line2D([],[],color="#a04a2c",ls="--",label="Whole-area median"),
               Line2D([],[],color="#005a67",lw=2,label="Corridor median"),
               Patch(facecolor="#dcece9",label="10–90% exceedance probability"),
               Line2D([],[],marker="o",mfc="#273034",color="#273034",ls="",ms=4,label="Measured ≥ level"),
               Line2D([],[],marker="o",mfc="white",color="#273034",ls="",ms=4,label="Measured < level"),
               Line2D([],[],marker="D",color="#db9d35",ls="",ms=5,label="Bounded reading")]
    fig.legend(handles=handles,loc="outside lower center",ncols=3,fontsize=9,frameon=False)
    fig.suptitle(name + " · Full compass from the vent\nGreat-circle distance in km on a logarithmic radius",fontsize=14)
    fig.savefig(out/"compass.png",dpi=155)
    plt.close(fig)
