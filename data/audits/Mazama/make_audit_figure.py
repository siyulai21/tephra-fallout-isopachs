"""Optional coordinate diagnostic; requires matplotlib. Not a fitted isopach map."""
import csv
import math
import os
import tempfile
from pathlib import Path
os.environ.setdefault('MPLCONFIGDIR', tempfile.mkdtemp(prefix='mazama-mpl-'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
H=Path(__file__).resolve().parent
r=[r for r in csv.DictReader((H/'supplied-row-crosswalk.csv').open()) if r['record_id']]
east=[(float(x['source_longitude'])-float(x['project_longitude']))*111.195*math.cos(math.radians(float(x['source_latitude']))) for x in r]
north=[(float(x['source_latitude'])-float(x['project_latitude']))*111.195 for x in r]
fig,ax=plt.subplots(1,2,figsize=(11,4.6),layout='constrained')
fig.suptitle('Mazama input: candidate source-coordinate discrepancies',fontsize=15)
ax[0].scatter(east,north,c='#23648a',s=34,alpha=.8)
ax[0].axhline(0,color='#999',lw=.8);ax[0].axvline(0,color='#999',lw=.8)
for x,y,rw in zip(east,north,r):
 if rw['record_id'] in ['B2020-291','B2020-264']:
  ax[0].annotate(rw['locality'].replace('Banff National Park, ',''),(x,y),xytext=(5,-14),textcoords='offset points',fontsize=8)
ax[0].set(xlabel='Eastward displacement (km)',ylabel='Northward displacement (km)',title='Published S1 position minus supplied position',xlim=(-.25,4),ylim=(-.8,4.7))
ax[0].grid(alpha=.2)
d=[float(x['coordinate_displacement_km']) for x in r]
ax[1].hist(d,bins=[0,.5,1,1.5,2,2.5,3,3.5,4,4.5,5],color='#23648a',edgecolor='white')
ax[1].set(xlabel='Great-circle displacement (km)',ylabel='Number of candidate matches',title='45 non-pseudo project rows',xlim=(0,5))
ax[1].text(.98,.94,'Median 1.216 km\nRange 0.836–4.486 km',ha='right',va='top',transform=ax[1].transAxes,fontsize=10)
fig.supxlabel('Diagnostic of source lineage; neither a datum correction nor a measure of true site-location error.',fontsize=9)
fig.savefig(H/'coordinate-discrepancies.png',dpi=180)
plt.close(fig)
