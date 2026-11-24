"""Numbered source-map registrations and legible close-ups of measured and omitted sites."""
import csv
import json

import matplotlib.pyplot as plt
import numpy as np

from benchmarks.mazama import AUDIT
from benchmarks.mazama_lake_sections import MARKERS, PANELS, marcott_position


def scatter(ax, rows, chosen, labels=True):
    for sid, name, lat, lon in rows:
        use = sid in chosen
        ax.scatter(lon,lat,marker='^' if use else 'o',s=20,color='#a6482f' if use else '#829497',zorder=2)
        if labels:
            ax.annotate(name,(lon,lat),xytext=(3,3),textcoords='offset points',fontsize=7)
    ax.ticklabel_format(useOffset=False,axis='both')
    ax.set_xlabel('Longitude (°)')
    ax.set_ylabel('Latitude (°)')
    ax.set_aspect(1/np.cos(np.deg2rad(np.mean([r[2] for r in rows]))))
    ax.grid(alpha=.2)


def run():
    with (AUDIT/'model-selection.csv').open() as f:
        chosen={r['record_id'] for r in csv.DictReader(f)}
    pos=[dict(site=n,panel=p,x=x,y=y,latitude=marcott_position(n)[0],longitude=marcott_position(n)[1]) for n,(p,x,y) in sorted(MARKERS.items())]
    (AUDIT/'marcott-registration.json').write_text(json.dumps(dict(source='Marcott et al. 2009, Figure 2, PDF page 3, original landscape map printed within portrait page. Marcott 2005 logs and archive search preceded digitization; no numeric field GPS table for these sites was recovered.',image_size_px=[1088,1450],pixel_origin='upper left',panels=PANELS,
        controls_order=['x_west','x_east','longitude_west','longitude_east','y_north','y_south','latitude_north','latitude_south'],
        uncertainty='Symbols and graticule were read at this resolution. A five-pixel displacement corresponds to roughly 40–90 m depending on panel and direction. This describes marker reading only. Source map accuracy and within-exposure position introduce additional uncertainty.',
        positions=pos),indent=2)+'\n')
    fig,axs=plt.subplots(2,2,figsize=(11,8),constrained_layout=True)
    rows=[(f"MAR2009-S{p['site']:02}",str(p['site']),p['latitude'],p['longitude']) for p in pos]
    for ax,(name,title) in zip(axs.flat,[(None,'All field sections'),('middle','Middle Sister'),('south','South Sister'),('broken','Broken Top northern flank')],strict=True):
        group=[r for r,p in zip(rows,pos,strict=True) if name is None or p['panel']==name]
        scatter(ax,group,chosen,labels=name is not None)
        if len(group)==1:
            lat,lon=group[0][2:4]
            ax.set_xlim(lon-.01,lon+.01);ax.set_ylim(lat-.008,lat+.008)
        ax.set_title(title)
    fig.suptitle('Marcott field sections • triangles enter the principal input')
    fig.savefig(AUDIT/'marcott-map.png',dpi=155);plt.close(fig)
    source=list(csv.DictReader((AUDIT/'young1990-section-positions.csv').open()))
    rows=[(f"Y1990-A{int(r['site']):03}",r['site'],float(r['latitude']),float(r['longitude'])) for r in source]
    fig,axs=plt.subplots(2,2,figsize=(14,10),constrained_layout=True)
    scatter(axs.flat[0],rows,chosen,False);axs.flat[0].set_title('All 138 sections')
    for ax,page in zip(list(axs.flat)[1:],[321,323,324],strict=True):
        group=[r for r,t in zip(rows,source,strict=True) if int(t['map_pdf_page'])==page]
        scatter(ax,group,chosen);ax.set_title(f'Appendix 1d • PDF page {page}')
    fig.suptitle('Young field sections • numbers are Appendix 1 site IDs')
    fig.savefig(AUDIT/'young1990-maps.png',dpi=170);plt.close(fig)
    # Individual source groups retain a readable row for every number even where printed symbols overlap.
    for page in (321,323,324):
        group=[r for r,t in zip(rows,source,strict=True) if int(t['map_pdf_page'])==page]
        ordered=sorted(group,key=lambda r:(r[3],r[2]))
        fig,axs=plt.subplots(2,2,figsize=(13,9),constrained_layout=True)
        for ax,idx in zip(axs.flat,np.array_split(np.arange(len(ordered)),4),strict=True):
            members=[ordered[i] for i in idx]
            scatter(ax,members,chosen)
            ax.set_title('Appendix sites '+', '.join(r[1] for r in members),fontsize=8,wrap=True)
        fig.savefig(AUDIT/f'young1990-detail-{page}.png',dpi=155);plt.close(fig)


if __name__=='__main__':
    run()
