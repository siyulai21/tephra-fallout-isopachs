"""Before-and-after contour maps on matched coordinates and the same posterior subsets."""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import MaxNLocator

from .extent_challenges import paired
from .footprint_study import controls
from src.empirical_isopach.core.audit import audited_data
from src.empirical_isopach.core.catalog import datasets
from src.empirical_isopach.models.extent_integrals import parent_fields
from src.empirical_isopach.models.extent_weights import calibration
from src.empirical_isopach.visualization.deposit_field import PolarField
from src.empirical_isopach.visualization.contour_limits import inner_contour, outer_contour
from src.empirical_isopach.visualization.deposit_maps import colours, draw_land, frames, plot_observations, regular_axes
from src.probabilistic_isopach.global_volume import EarthDomain

CASES=(("pinatubo-1991-layer-c","spherical-mixture","Pinatubo Layer C · spherical mixture"),
       ("ilopango-tbj","spherical-mixture","Ilopango Unit G · spherical mixture"),
       ("aso-4","spherical-mixture","Aso-4 · spherical mixture"),
       ("mazama","nested-isopachs","Mazama · nested corridors"),
       ("tambora","directional","Tambora · directional ensemble"))


def surfaces(slug,family):
    rec=next(r for r in datasets() if r["slug"]==slug)
    data,meta=audited_data(rec["path"],allow_intervals=True)
    latitude=meta["vent_lonlat"][1]
    dom=EarthDomain.wgs84(latitude)
    folder = controls(slug,rec)[0] if family=="nested-isopachs" else Path("eruptions")/slug/"results"/family
    fields=parent_fields(folder,family,data,latitude,count=192 if family=="spherical-mixture" else 1024)
    cal=calibration(family,meta["input_sha256"],exclude_slug=slug)
    radius=np.r_[0.,np.geomspace(.1,dom.rho_max*(1.-1e-8),176)]
    theta=np.arange(240)*2*np.pi/240.
    xy=np.stack((dom.axis_scale*np.outer(radius,np.sin(theta)),np.outer(radius,np.cos(theta))/dom.axis_scale),axis=-1).reshape(-1,2)
    z=np.concatenate([paired(fields,xy[k:k+128],cal) for k in range(0,len(xy),128)],axis=1)/np.log(10.)
    return data,meta,[PolarField(radius,a.reshape(len(radius),-1),dom.axis_scale) for a in z]


def figure(out):
    out.mkdir(parents=True,exist_ok=True)
    fig,axes=plt.subplots(5,2,figsize=(9,16),layout="constrained")
    ink="#244c59"
    rows=[]
    for pair,(slug,family,title) in zip(axes,CASES,strict=True):
        data,meta,fields=surfaces(slug,family)
        cover = [outer_contour(field,data) for field in fields]
        minimum = min(r["level_cm"] for r in cover)
        limits=np.array([frames(field,data,minimum)[0] for field in fields])
        lo,hi=np.min(limits[:,:2],axis=0),np.max(limits[:,2:],axis=0)
        bounds=np.r_[lo,hi]
        x,y=regular_axes(bounds,340)
        for ax,field,selection,heading in zip(pair,fields,cover,("Earlier","Revised"),strict=True):
            largest = inner_contour(selection)["level_cm"]
            levels = 2.**np.arange(np.log2(selection["level_cm"]),np.log2(largest)+1)
            brackets = 2.**np.arange(-1,np.log2(max(largest,.5))+1)
            draw_land(ax,bounds,meta["vent_lonlat"],320)
            z=field.on_grid(x,y)
            use=levels[(levels>10.**np.nanmin(z)) & (levels<10.**np.nanmax(z))]
            if len(use):
                ax.contourf(x,y,z,levels=np.log10(np.r_[use,use[-1]*2048.]),cmap="YlGnBu",alpha=.60)
                lines=ax.contour(x,y,z,levels=np.log10(use),colors=ink,linewidths=.7)
                labelled=[np.log10(a) for a in (1.,4.,16.,64.) if a in use]
                ax.clabel(lines,levels=labelled,fmt=lambda v:f"{10**v:g}",fontsize=6)
            plot_observations(ax,data,meta,brackets,colours(len(brackets)),bounds)
            ax.set(xlim=(lo[0],hi[0]),ylim=(lo[1],hi[1]),aspect="equal")
            ax.set_title(title+"\n"+heading,fontsize=10,loc="left")
            ax.xaxis.set_major_locator(MaxNLocator(5)); ax.yaxis.set_major_locator(MaxNLocator(4))
            ax.tick_params(labelsize=7)
            for level in (.5,1.,4.,16.):
                cross=field.outer_crossings(level)
                if np.isfinite(cross).all() and (cross>0).all():
                    harmonics=abs(np.fft.rfft(np.log(cross)-np.mean(np.log(cross))))**2
                    fraction=harmonics[5:].sum()/max(harmonics[1:].sum(),1e-30)
                    rows.append(dict(slug=slug,input_sha256=meta["input_sha256"],family=family,model=heading,level_cm=level,
                                     high_harmonic_fraction=fraction))
    fig.suptitle("How the inferred fallout footprint changes",fontsize=15)
    fig.supxlabel("Easting from the vent (km)   ·   contours in centimetres",fontsize=9)
    fig.supylabel("Northing from the vent (km)",fontsize=9)
    fig.savefig(out/"affected-contours.png",dpi=165)
    plt.close(fig)
    pd.DataFrame(rows).to_csv(out/"angular-variation.csv",index=False)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output",type=Path,required=True)
    figure(p.parse_args().output)
