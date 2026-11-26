"""Matched site diagnostics and unsmoothed three-scale plates for every eruption."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd

from .publication import result_home
from .footprint_morphology import outlines
from src.empirical_isopach.core.audit import audited_data
from src.empirical_isopach.core.catalog import datasets
from src.empirical_isopach.models.nested.footprints.posterior import FootprintField, predictions
from src.empirical_isopach.models.nested.footprints.continuous import FlowField
from src.empirical_isopach.models.nested.report import MapView
from src.empirical_isopach.models.nested.study import digest, write_json
from src.empirical_isopach.visualization.contour_limits import local_half_width
from src.empirical_isopach.visualization.deposit_field import sample_field, PolarField
from src.empirical_isopach.visualization.deposit_maps import draw_land, frames, regular_axes, plot_observations, colours


def contour_geometry(field, cm, previous=None):
    """Unsmoothed radial curvature, orientation and spacing; topology separately.

    These descriptors are diagnostics, never objectives for fitting. A real
    lobe can have negative curvature. Coarsened replay exposes grid sensitivity.
    """
    from scipy.ndimage import label
    mask = field.log10_cm >= np.log10(cm)
    components, n = label(mask)
    parent = np.arange(n+1)
    def root(i):
        while parent[i] != i:
            i = parent[i]
        return i
    for a,b in zip(components[:,0],components[:,-1],strict=True):
        if a and b:
            parent[root(a)] = root(b)
    result = dict(connected_components=len({root(i) for i in range(1,n+1)}),
                  radial_reentries=int(np.sum(np.diff(mask.astype(int),axis=0)>0)))
    if not mask[0].all() or mask[-1].any() or result['radial_reentries']:
        return result
    r = field.outer_crossings(cm)
    theta = field.angles
    xy = np.column_stack((r*np.sin(theta),r*np.cos(theta)))
    centred = xy-xy.mean(axis=0)
    eigen, vectors = np.linalg.eigh(centred.T@centred/len(r))
    axis = vectors[:,-1]
    step = 2*np.pi/len(r)
    d = (np.roll(r,-1)-np.roll(r,1))/(2*step)
    dd = (np.roll(r,-1)-2*r+np.roll(r,1))/step**2
    curvature = (r*r+2*d*d-r*dd)/(r*r+d*d)**1.5
    result.update(boundary_eccentricity=float(np.sqrt(1.-eigen[0]/eigen[1])),
                  principal_axis_degrees=float(np.rad2deg(np.arctan2(axis[0],axis[1]))%180),
                  concave_perimeter_fraction=float(np.mean(curvature<0)),
                  curvature_radius_scaled_p95=float(np.quantile(abs(curvature)*np.mean(r),.95)),
                  minimum_reach_km=float(r.min()),maximum_reach_km=float(r.max()))
    if previous is not None:
        outer = field.outer_crossings(previous)
        gap = outer-r
        result.update(minimum_radial_gap_km=float(np.nanmin(gap)),
                      relative_gap_spread=float(np.nanstd(gap)/np.nanmean(gap)))
    return result


def read_field(path):
    with np.load(path) as a:
        return PolarField(a['radii_km'], a['log10_cm'], float(a['axis']))


def assess(job):
    record, root, output, source, baseline, draws = job
    slug = record['slug']
    data, meta = audited_data(record['path'])
    old = baseline/slug if baseline else result_home(record)/'nested-isopachs'
    out = output/slug
    out.mkdir(parents=True,exist_ok=True)
    fields, models, labels = [], [], []
    sources=[old/'footprints/full-selected.npz']
    if (out/'previous.npz').is_file():
        fields.append(read_field(out/'previous.npz'))
    else:
        field = read_field(old/'median-grid.npz')
        fields.append(field)
        np.savez_compressed(out/'previous.npz',radii_km=field.radii,log10_cm=field.log10_cm,axis=field.axis)
    models.append(FootprintField.load(old/'footprints/full-selected.npz'))
    labels.append('Previous footprints')
    if root is not None:
        path = root/slug/('footprints/full-selected.npz' if source=='release' else 'full.npz')
        with np.load(path,allow_pickle=False) as packet:
            cls = FlowField if 'warp' in packet else FootprintField
        model = cls.load(path)
        sources.append(path)
        field = (read_field(root/slug/'median-grid.npz') if source=='release' else
                 sample_field(MapView(model.reduced(draws)),data,meta['vent_lonlat'][1],257,384))
        np.savez_compressed(out/'growth.npz',radii_km=field.radii,log10_cm=field.log10_cm,axis=field.axis)
        fields.append(field); models.append(model); labels.append('Continuous area growth' if cls is FlowField else 'Positive area growth')
    box = np.array([frames(f,data,1.)[0] for f in fields])
    full = np.r_[box[:,:2].min(axis=0),box[:,2:].max(axis=0)]
    near = local_half_width(data)
    intermediate = max(3*near, np.quantile(np.linalg.norm(data.xy,axis=1),.55))
    boxes = [np.array([-near,-near,near,near]),
             np.array([-intermediate,-intermediate,intermediate,intermediate]), full]
    fig, axes = plt.subplots(len(fields),3,figsize=(15,4.8*len(fields)),squeeze=False,layout='constrained')
    levels = .5*2.**np.arange(15)
    for row,(field,label) in enumerate(zip(fields,labels,strict=True)):
        for col,(bound,scale) in enumerate(zip(boxes,('Proximal','Intermediate','Full deposit'),strict=True)):
            ax = axes[row,col]
            draw_land(ax,bound,meta['vent_lonlat'],300)
            x,y=regular_axes(bound,560)
            z=field.on_grid(x,y)
            lev=levels[(levels>10.**np.nanmin(z)) & (levels<10.**np.nanmax(z))]
            if col==2:
                lev=lev[lev>=1.]
            if len(lev):
                cs=ax.contour(x,y,z,levels=np.log10(lev),colors=['#176c83' if v<16 else '#a54b29' for v in lev],linewidths=.8)
                ax.clabel(cs,fmt={np.log10(v):f'{v:g}' for v in lev},fontsize=7,inline=True)
            plot_observations(ax,data,meta,levels,colours(len(levels)),bound)
            if col==2:
                for half,color in ((near,'#742f8c'),(intermediate,'#777777')):
                    ax.add_patch(Rectangle((-half,-half),2*half,2*half,fill=False,ec=color,lw=1.1,ls=':'))
            ax.set(aspect='equal',xlim=bound[::2],ylim=bound[1::2],title=f'{label} · {scale}',xlabel='Easting (km)',ylabel='Northing (km)')
            ax.tick_params(labelsize=8)
    fig.suptitle(f"{record['name']} · identical observations and viewports · contours in cm",fontsize=15)
    fig.savefig(out/'scales.png',dpi=140)
    plt.close(fig)
    write_json(out/'review.json',dict(input_sha256=meta['input_sha256'],
               source_posteriors_sha256={label:digest(path) for label,path in zip(labels,sources,strict=True)},
               field_mode='published median grid' if source=='release' else f'development subset of {draws} draws',
               matched_bounds_km=[b.tolist() for b in boxes],levels_cm=levels.tolist(),full_minimum_cm=1.,
               source_grid_sha256={label:digest(out/name) for label,name in zip(labels,('previous.npz','growth.npz'))},
               figure_sha256=digest(out/'scales.png')))
    rows, metrics, morph = [], [], []
    for field,model,label in zip(fields,models,labels,strict=True):
        log=model.field(data.xy)
        median=np.median(log,axis=0)
        pred=predictions(model,data)
        err=median-data.log_value
        exact=~data.censored
        upper=data.censored & ~data.lower_censored
        interval=data.interval
        event=np.array([p['log_score'] for p in pred])
        metrics.append(dict(case=slug,method=label,input_sha256=meta['input_sha256'],rows=len(data.xy),
                            measured_log_rmse=float(np.sqrt(np.mean(err[exact]**2))),
                            measured_median_abs_log_error=float(np.median(abs(err[exact]))),
                            full_negative_log_score=float(-event.mean()),
                            bounded_negative_log_probability=float(-event[~exact].mean()) if (~exact).any() else None,
                            upper_median_violations=int(np.sum(upper & (err>0))),
                            interval_median_violations=int(np.sum(interval & ((err<0)|(median>data.interval_log_upper))))))
        rows.extend(dict(case=slug,method=label,row_id=int(i),observed_log_cm=float(y),median_log_cm=float(m),
                         kind='interval' if iv else 'upper' if up else 'point',log_score=p['log_score'])
                    for i,y,m,iv,up,p in zip(data.row_ids,data.log_value,median,interval,upper,pred,strict=True))
        for stride in (1,2):
            g = PolarField(field.radii[::stride],field.log10_cm[::stride,::stride],field.axis)
            for k,cm in enumerate(levels):
                detail=outlines(g,cm)
                morph.append(dict(case=slug,method=label,level_cm=cm,bearings=len(g.angles),area_km2=g.area(cm),
                                  **(detail or {}),**contour_geometry(g,cm,levels[k-1] if k else None)))
    raw=pd.read_csv(record['path'])
    raw.index+=2
    source_ids=[str(raw.loc[i,'EvidenceID']) if 'EvidenceID' in raw else f'row-{i}' for i in data.row_ids]
    radius=np.linalg.norm(data.xy,axis=1)
    bearing=np.rad2deg(np.arctan2(data.xy[:,0],data.xy[:,1]))%360
    evidence=pd.DataFrame(dict(row_id=data.row_ids,source_id=source_ids,easting_km=data.xy[:,0],northing_km=data.xy[:,1],
                               distance_km=radius,bearing_degrees=bearing,
                               sector=np.asarray(['N','NE','E','SE','S','SW','W','NW'])[np.floor(((bearing+22.5)%360)/45).astype(int)],
                               scale=np.where(radius<=near,'proximal',np.where(radius<=intermediate,'intermediate','distal'))))
    site_frame=pd.DataFrame(rows).merge(evidence,on='row_id',validate='many_to_one')
    site_frame['signed_log_error']=site_frame.median_log_cm-site_frame.observed_log_cm
    site_frame.to_csv(out/'sites.csv.gz',index=False,compression={'method':'gzip','mtime':0})
    exact=site_frame.loc[site_frame.kind=='point']
    groups=exact.groupby(['method','sector','scale']).signed_log_error
    groups.agg(['count','mean','median',lambda x:np.sqrt(np.mean(x*x))]).rename(columns={'<lambda_0>':'log_rmse'}).to_csv(out/'sector-errors.csv')
    pd.DataFrame(metrics).to_csv(out/'metrics.csv',index=False)
    pd.DataFrame(morph).to_csv(out/'morphology.csv',index=False)
    print(slug,'review plate ready',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--fits',type=Path)
    p.add_argument('--source',choices=('fits','release'),default='fits')
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--case',action='append')
    p.add_argument('--baseline',type=Path)
    p.add_argument('--draws',type=int,default=1024)
    p.add_argument('--workers',type=int,default=2)
    a=p.parse_args()
    records=[r for r in datasets() if not a.case or r['slug'] in a.case]
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        list(pool.map(assess,[(r,a.fits,a.output,a.source,a.baseline,a.draws) for r in records]))
