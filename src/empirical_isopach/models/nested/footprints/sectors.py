"""Quantify the volume assigned to each compass sector and the presence of thickness evidence."""
import numpy as np
import pandas as pd

from ..geometry import positions
from ..reach_integrals import SECTORS, nodes


def sector(theta):
    return np.floor((theta+np.pi/8)/(np.pi/4)).astype(int)%8


def budgets(model, data, map_draws):
    subset=model.reduced(512)
    central=model.reduced(map_draws)
    trials, areas = [], []
    for seed in (81203,75641):
        xy,w,_,theta=nodes(14,seed,model.latitude)
        labels=sector(theta)
        sectors=np.eye(8)[labels]
        sums=np.zeros((len(subset.z)+1,8))
        counts=np.zeros(8)
        for start in range(0,len(xy),128):
            sl=slice(start,start+128)
            logs=subset.field(xy[sl])
            values=np.exp(logs)
            if map_draws==len(subset.z):
                med=np.exp(np.median(logs,axis=0))
            else:
                med=np.exp(np.median(central.field(xy[sl]),axis=0))
            mass=1e-5*w[sl,None]*sectors[sl]/len(w)
            sums[:-1]+=values@mass
            sums[-1]+=med@mass
            counts+=(med>=1.)@(w[sl,None]*sectors[sl]/len(w))
        trials.append(sums)
        areas.append(counts)
    mass=np.mean(trials,axis=0)
    qr=np.quantile(mass[:-1],[.025,.5,.975],axis=0)
    err=np.abs(mass[:-1].sum(axis=1)/subset.volume()-1)
    scrambles=np.abs(trials[0][-1]-trials[1][-1])/mass[-1]
    radius,bearing=positions(data.xy,model.latitude)
    site_sector=sector(bearing)
    two=np.isfinite(data.interval_log_upper)
    rows=[]
    for j,compass in enumerate(SECTORS):
        group=(site_sector==j) & (radius>1e-8)
        rows.append(dict(sector=compass,exact=sum(group & ~data.censored),lower=sum(group & data.lower_censored & ~two),
                         upper=sum(group & data.censored & ~data.lower_censored & ~two),interval=sum(group & two),
                         farthest_site_km=float(radius[group].max()) if np.any(group) else None,
                         median_map_km3=mass[-1,j],median_map_fraction=mass[-1,j]/mass[-1].sum(),
                         median_map_area_1cm_km2=np.mean(areas,axis=0)[j],draw_lower95_km3=qr[0,j],
                         draw_median_km3=qr[1,j],draw_upper95_km3=qr[2,j],map_relative_scramble_difference=scrambles[j]))
    check=dict(scrambles=[81203,75641],points_per_scramble=2**14,complete_draws=len(subset.z),
               central_draws=len(central.z),norm="equal-area source chart",sectors="45 degrees centred on compass label",
               analytic_total_relative_error_p95=float(np.quantile(err,.95)),median_map_km3=float(mass[-1].sum()),
               unassigned_vent_sites=int(sum(radius<=1e-8)),
               median_map_max_sector_scramble_difference=float(scrambles.max()))
    return pd.DataFrame(rows),check
