"""Source-region figures with the vent included exactly in the numerical grid."""
from pathlib import Path
import numpy as np
from .models.mass_mixture.inference import flatten
from .models.mass_mixture.geometry import log_field, to_sphere


def source_figure(output, name, data, meta, samples, previous=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from src.probabilistic_isopach.figures import STYLE, _coastlines
    distance = np.linalg.norm(data.xy, axis=1)
    extent = min(400., max(50., .25*distance.max(), 3*distance.min()))
    axis = np.linspace(-extent, extent, 161)
    xx, yy = np.meshgrid(axis, axis)
    unit = to_sphere(np.column_stack((xx.ravel(), yy.ravel())), meta["vent_lonlat"][1])
    panels = [("Current reconstruction", samples)] if previous is None else [("Previous displaced model", previous), ("Source-constrained reconstruction", samples)]
    medians = []
    for _, posterior in panels:
        flat = flatten(posterior)
        index = np.unique(np.linspace(0, len(flat["log_volume"])-1, min(1024, len(flat["log_volume"]))).astype(int))
        flat = {k: v[index] for k, v in flat.items()}
        median = np.concatenate([np.median(log_field(flat, unit[i:i+512]), axis=0) for i in range(0,len(unit),512)])
        medians.append(median.reshape(xx.shape)/np.log(10))
    hi = max(np.nanmax(a) for a in medians)
    lo = max(hi-3., min(np.nanmin(a) for a in medians))
    levels = np.arange(np.ceil(lo*2)/2, np.floor(hi*2)/2+.01, .5)
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(1,len(panels),figsize=(7*len(panels),6.7),layout="constrained",squeeze=False)
        for ax,(title,_),field in zip(axes[0],panels,medians):
            plot = ax.pcolormesh(axis,axis,field,shading="auto",cmap="viridis",vmin=lo,vmax=hi)
            lines = ax.contour(axis,axis,field,levels=levels,colors="white",linewidths=.65)
            ax.clabel(lines,fmt=lambda v:f"{10**v:.2g}",fontsize=8)
            _coastlines(ax,meta["vent_lonlat"])
            ax.scatter(*data.xy.T,s=22,c="white",edgecolors="#253B43",linewidths=.7,zorder=4)
            ax.scatter(0,0,s=170,marker="*",c="#E47445",edgecolors="white",zorder=6,label="Source volcano")
            maximum = np.unravel_index(np.argmax(field),field.shape)
            ax.scatter(axis[maximum[1]],axis[maximum[0]],s=80,marker="+",c="#EE5666",zorder=7,label="Grid maximum")
            ax.set(xlim=(-extent,extent),ylim=(-extent,extent),aspect="equal",title=title,
                   xlabel="Easting from source (km)",ylabel="Northing (km)")
            ax.legend(loc="lower left",fontsize=8)
        fig.colorbar(plot,ax=axes[0].tolist(),shrink=.7,label="log₁₀ median thickness (cm); contours in cm")
        fig.suptitle(f"{name} · source-region check\nPoints are accepted observations; map extent does not set volume.",fontsize=14)
        fig.savefig(Path(output)/"source.png",dpi=170)
        plt.close(fig)
