"""Map the Mazama evidence inventory without treating occurrences as thickness bounds."""
import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
import numpy as np

from benchmarks.mazama import AUDIT, INPUT, sha
from src.empirical_isopach.visualization.maps import coast_rings, ASSET


def render(output):
    ledger = list(csv.DictReader((AUDIT/'revised-locality-ledger.csv').open()))
    accepted = list(csv.DictReader(INPUT.open()))
    plotted = [r for r in ledger if r['latitude'] and r['accepted_for_model']!='True']
    controls = [r for r in plotted if r['quantity_kind'] in ('not_a_Mazama_measurement','target_not_established','other_eruption_measured_bed','cross_reference_without_target_confirmation')
                or 'not_confirmed_Mazama' in r['deposit_class']]
    plotted = [r for r in plotted if r not in controls]
    tentative = [r for r in plotted if r['quantity_kind'] in {'tentative_signal','tentative_lens_with_measured_thickness','mixed_unidentified_pod','ambiguous_phase_and_unmeasured_bed'} or r['fallout_model_status']=='tentative_occurrence_only']
    known = [r for r in plotted if r['quantity_kind']=='cryptotephra' and r not in tentative]
    other = [r for r in plotted if r not in known and r not in tentative]
    fig,axes = plt.subplots(1,2,figsize=(13.5,6.5),gridspec_kw={'width_ratios':[1,1.45]})
    fig.subplots_adjust(left=.055,right=.99,bottom=.20,top=.86,wspace=.17)
    for ax,extent,title in zip(axes,[(-127,-105,39,57),(-155,10,25,85)],['Western sites and fitted observations','Distant occurrences and evidence gaps'],strict=True):
        ax.add_collection(LineCollection(coast_rings(),colors='#9baeb2',linewidths=.4,zorder=1),autolim=False)
        for records,marker,color,size in [(other,'o','#b6c3c6',12),(known,'D','#6252a0',30),(tentative,'x','#bb703d',45)]:
            ax.scatter([float(r['longitude']) for r in records],[float(r['latitude']) for r in records],s=size,marker=marker,c=color,zorder=3)
        ax.scatter([float(r['longitude']) for r in controls],[float(r['latitude']) for r in controls],s=14,marker='s',facecolors='none',edgecolors='#abb3b5',linewidths=.6,zorder=2)
        for kind,marker,color,size in [('measurement','o','#117c80',31),('lower_limit','^','#c14c32',55),('upper_limit','v','#4f62a5',60)]:
            a=[r for r in accepted if r['Kind']==kind]
            ax.scatter([float(r['Lon']) for r in a],[float(r['Lat']) for r in a],s=size,marker=marker,c=color,edgecolors='white',linewidths=.5,zorder=5)
        ax.scatter(-122.1,42.95,c='#182f38',marker='*',s=120,zorder=6)
        ax.set(xlim=extent[:2],ylim=extent[2:],xlabel='Longitude (degrees)',ylabel='Latitude (degrees)',title=title)
        ax.grid(alpha=.13)
        ax.set_aspect(1/np.cos(np.deg2rad(50)))
    left,right=axes
    a=next(r for r in accepted if r['EvidenceID']=='B2020-257')
    left.annotate('Green Lake corrected\nto Whistler study lake',xy=(float(a['Lon']),float(a['Lat'])),xytext=(-120.7,53.8),arrowprops={'arrowstyle':'->','color':'#52686b'},fontsize=8)
    left.plot([-121.231433,float(a['Lon'])],[51.403267,float(a['Lat'])],ls=':',color='#aa6850',lw=1.1,zorder=4)
    left.scatter(-121.231433,51.403267,marker='+',s=50,c='#aa6850',zorder=5)
    left.annotate('Wrong compiled lake',(-121.231433,51.403267),xytext=(-119.5,51.6),fontsize=7.5,arrowprops={'arrowstyle':'-','color':'#aa6850'})
    right.annotate('Greenland ice records',(-42.3,75.1),xytext=(-72,81),fontsize=8,arrowprops={'arrowstyle':'-','color':'#52686b'})
    right.annotate('Ireland: tentative\nreassignment',(-9.505,53.05833),xytext=(-38,38),fontsize=8,arrowprops={'arrowstyle':'-','color':'#52686b'})
    handles=[Line2D([],[],ls='',marker=m,color=c,label=t) for m,c,t in [
        ('o','#117c80',f"{sum(r['Kind']=='measurement' for r in accepted)} fitted point values"),('^','#c14c32',f"{sum(r['Kind']=='lower_limit' for r in accepted)} conditional lower bounds"),
        ('v','#4f62a5',f"{sum(r['Kind']=='upper_limit' for r in accepted)} conditional upper bound"),
        ('o','#b6c3c6','Other records, omitted from fitting'),('D','#6252a0','Reported cryptotephra'),('x','#bb703d','Tentative signals'),('s','#abb3b5','Coverage / other-event controls')]]
    fig.legend(handles=handles,loc='lower center',bbox_to_anchor=(.5,.008),ncol=4,fontsize=8.5,frameon=False)
    fig.suptitle(f'Mazama: {len(ledger)} evidence records, {len(accepted)} fitting observations',fontsize=15,y=.98)
    fig.text(.5,.12,f"{sum(not r['latitude'] for r in ledger)} records lack decimal coordinates. Locations can be approximate; occurrence does not supply a numerical fallout limit.",ha='center',fontsize=9,color='#445b61')
    output.parent.mkdir(parents=True,exist_ok=True)
    metadata=dict(ledger_sha256=sha(AUDIT/'revised-locality-ledger.csv'),input_sha256=sha(INPUT),coastline_sha256=sha(ASSET),source='Source coordinates and qualified site proxies. Young sections, Stillwater, the Oregon fens and other digitized localities carry separate georeferencing evidence.',unlocated_records=sum(not r['latitude'] for r in ledger))
    fig.savefig(output,dpi=160,metadata={'Description':json.dumps(metadata,sort_keys=True)})
    plt.close(fig)
    output.with_suffix('.json').write_text(json.dumps(metadata,indent=2)+'\n')


def yellowstone(output):
    """Core-position close-up: ten ash cores, eight locations, seven principal rows."""
    ledger = list(csv.DictReader((AUDIT/'revised-locality-ledger.csv').open()))
    rows = [r for r in ledger if r['record_id'].startswith('M2023-YL') and r['observed_point_cm']]
    assert len(rows) == 10
    groups = {}
    for r in rows:
        groups.setdefault((float(r['longitude']),float(r['latitude'])),[]).append(r)
    assert len(groups) == 8
    fig, ax = plt.subplots(figsize=(9.4,7.7))
    fig.subplots_adjust(left=.12,right=.96,bottom=.23,top=.86)
    positions = {
        'M2023-YL16-2C':(-85,23), 'M2023-YL16-3A':(-130,-13),
        'M2023-YL16-4C':(12,21), 'M2023-YL16-5A':(15,10),
        'M2023-YL16-6A':(15,-29), 'M2023-YL92-1A':(12,8),
        'M2023-YL92-2A':(12,-10), 'M2023-YL92-3A':(-34,-31)}
    for (lon,lat),group in groups.items():
        chosen=next((r for r in group if r['accepted_for_model']=='True'),group[0])
        rid=chosen['record_id']; excluded=rid=='M2023-YL16-6A'
        ax.scatter(lon,lat,s=85,marker='x' if excluded else 'o',c='#bc583e' if excluded else '#127d80',zorder=4)
        label=' / '.join(r['record_id'].removeprefix('M2023-') for r in group)
        label+='\n'+('4 mm (older report 5 mm)' if rid=='M2023-YL16-2C' else '5 mm')
        if excluded:label+='; slump qualification'
        elif len(group)>1:label+='\npaired cores counted once'
        ax.annotate(label,(lon,lat),xytext=positions[rid],textcoords='offset points',fontsize=9,
                    arrowprops={'arrowstyle':'-','color':'#98a9ac','lw':.7})
    ax.set(xlim=(-110.59,-110.20),ylim=(44.355,44.575),xlabel='Longitude (°W shown negative)',ylabel='Latitude (°N)')
    ax.set_aspect(1/np.cos(np.deg2rad(44.46)));ax.grid(alpha=.2)
    ax.set_title('Yellowstone Lake — reported Mazama ash cores',fontsize=15,pad=18)
    fig.text(.5,.94,'10 cores • 8 locations • 7 principal observations',ha='center',fontsize=12,color='#425b60')
    fig.text(.5,.045,'Positions from Morgan et al.2023, supplementary TableS1. Labels preserve each core identity.\nCoordinate plot; no lake shoreline inferred. Core pairs share one reported position.\nYL16-6A is a prediction location only. Seven distinct stations enter the fit.',ha='center',fontsize=9,color='#425b60')
    metadata=dict(ledger_sha256=sha(AUDIT/'revised-locality-ledger.csv'),input_sha256=sha(INPUT),
                  source='Morgan et al.2023 doi:10.1130/B36190.1, TableS1; main-text thicknesses',
                  cores=[r['record_id'] for r in rows],locations=8,principal_locations=7)
    fig.savefig(output,dpi=170,metadata={'Description':json.dumps(metadata,sort_keys=True)})
    plt.close(fig);output.with_suffix('.json').write_text(json.dumps(metadata,indent=2)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();render(a.output)
    yellowstone(a.output.with_name('yellowstone-cores.png'))
