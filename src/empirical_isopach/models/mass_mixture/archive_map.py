"""Independent spherical median maps under stated archive-visibility assumptions."""
import numpy as np
import matplotlib.pyplot as plt
from src.probabilistic_isopach.global_volume import EarthDomain

from ...visualization.maps import grid_axes,observations,setup_map
from ..directional.archive_readings import positions
from ..directional.bearing_maps import archive_symbols
from ..directional.ensemble.fields import inverse_cdf
from ..directional.study import write_json
from .geometry import log_field,to_sphere
from .inference import flatten


def selected_fields(samples,weights,count):
    p = flatten(samples)
    rows,mass = [],[]
    for c,chain in enumerate(weights):
        chain_mass=chain.sum()
        cdf=np.cumsum(chain)/chain_mass
        ix=np.searchsorted(cdf,(np.arange(count)+.5)/count)
        unique,n=np.unique(ix,return_counts=True)
        rows.extend(c*len(chain)+unique)
        mass.extend(n*chain_mass/count)
    return {k:v[rows] for k,v in p.items()},np.array(mass)


def median(p,w,xy,lat):
    domain = EarthDomain.wgs84(lat)
    valid = np.flatnonzero(domain.rho(xy) < domain.rho_max)
    values = np.full(len(xy),np.nan)
    for start in range(0,len(valid),128):
        rows = valid[start:start+128]
        z = log_field(p,to_sphere(xy[rows],lat))
        values[rows] = inverse_cdf(z.T,w)[0]/np.log(10)
    return values


def render(samples,weight_file,report,data,meta,inventory,output,bounds):
    reference=inventory.get("validation_scenario")
    if not reference or not report["scenarios"][reference]["importance_checks_passed"]:
        return
    vent=meta["vent_lonlat"]
    sites=inventory["sites"]
    site_xy=positions(sites,vent)
    x,y=grid_axes(np.array(bounds),longest=157,minimum=65)
    xx,yy=np.meshgrid(x,y)
    xy=np.column_stack((xx.ravel(),yy.ravel()))
    grids,diagnostics={},{}
    with np.load(weight_file) as archive:
        for key in ("quantitative",reference):
            target=np.log10([s["latent_cm"]["median"] for s in report["scenarios"][key]["sites"]])
            for count in (64,128,256,512):
                p,w=selected_fields(samples,archive[key],count)
                at=median(p,w,site_xy,vent[1])
                delta=float(max(abs(at-target)))
                if delta<.08:
                    break
            diagnostics[key]={"target_draws_per_chain":count,"maximum_site_median_error_log10":delta,
                              "site_check_passed":delta<.08}
            grids[key]=median(p,w,xy,vent[1]).reshape(xx.shape)
    fig,axes=plt.subplots(1,2,figsize=(9.2,4.7),layout="constrained")
    for ax,(key,title) in zip(axes,(("quantitative","Thickness evidence"),(reference,"Conditional on archive transfer")),strict=True):
        z=grids[key]
        image=ax.pcolormesh(x,y,z,cmap="viridis",vmin=-2,vmax=3,shading="auto",rasterized=True)
        for level in (.1,1.,10.,100.):
            l=np.log10(level)
            if np.nanmin(z)<l<np.nanmax(z):
                cs=ax.contour(x,y,z,levels=[l],colors="white",linestyles="--",linewidths=.9)
                ax.clabel(cs,fmt={l:f"{level:g}"},fontsize=7)
        setup_map(ax,bounds,vent)
        observations(ax,data,size=15)
        archive_symbols(ax,sites,site_xy,bounds)
        ax.set_title(title,fontsize=11)
    fig.suptitle("Spherical mixture · source-region median isopachs",fontsize=15)
    bar=fig.colorbar(image,ax=axes,location="bottom",shrink=.85,pad=.06,aspect=45)
    bar.set_ticks(np.arange(-2,4),labels=["0.01","0.1","1","10","100","1,000"])
    bar.set_label("Conditional median thickness (cm)")
    fig.savefig(output/"spherical-archive-isopachs.png",dpi=175,bbox_inches="tight")
    plt.close(fig)
    np.savez_compressed(output/"spherical-archive-grids.npz",x=x,y=y,**grids)
    write_json(output/"spherical-map-checks.json",diagnostics)
