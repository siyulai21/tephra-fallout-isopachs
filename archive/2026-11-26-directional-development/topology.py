import argparse
from pathlib import Path
import pandas as pd,numpy as np,shapely
from benchmarks.directional_scale_maps import read_grid
from benchmarks.directional_morphology import level_metrics
from src.empirical_isopach.core.catalog import datasets
from src.empirical_isopach.core.audit import audited_data
from src.empirical_isopach.contour_support import level_polygons
from src.empirical_isopach.visualization.deposit_maps import regular_axes
from src.empirical_isopach.visualization.deposit_field import doubling_levels
p=argparse.ArgumentParser();p.add_argument('variant');a=p.parse_args();root=Path('results/directional-geography-v6');rows=[];evidence=[]
for rec in datasets():
 slug=rec['slug'];data,_=audited_data(rec['path']);points=shapely.points(data.xy)
 path=root/'baseline'/slug/'median-grid.npz' if a.variant=='v5' else root/'experiments'/a.variant/slug/'full/grid.npz'
 if not path.exists():continue
 f=read_grid(path)
 for cm in doubling_levels(10**f.log10_cm.max()):
  metrics=level_metrics(f,cm,385)
  if metrics['status']!='closed':continue
  r=f.outer_crossings(cm);mask=np.isfinite(r)
  x=r[mask]*np.sin(f.angles[mask])*f.axis;y=r[mask]*np.cos(f.angles[mask])/f.axis;pad=np.nanmax(r)*.06
  bounds=np.array([min(0,min(x))-pad,min(0,min(y))-pad,max(0,max(x))+pad,max(0,max(y))+pad])
  x,y=regular_axes(bounds,385);_,polys=level_polygons((x,y,f.on_grid(x,y)*np.log(10)),[cm]);poly=polys[cm]
  positive=(~data.censored|data.interval)&(data.log_value>=np.log(cm))
  negative=(~data.lower_censored)&(data.log_value<np.log(cm))|data.interval&(data.interval_log_upper<np.log(cm))
  unsupported=[0,0];areas=[0.,0.]
  for part in shapely.get_parts(poly):
   if not part.covers(shapely.Point(0,0)):
    ids=data.row_ids[shapely.covers(part,points)&positive].tolist()
    unsupported[0]+=not bool(ids);areas[0]+=part.area if not ids else 0
    evidence.append(dict(slug=slug,level_cm=cm,kind='patch',area_km2=part.area,x=part.centroid.x,y=part.centroid.y,evidence_row_ids=' '.join(map(str,ids))))
   for ring in part.interiors:
    hole=shapely.Polygon(ring);ids=data.row_ids[shapely.covers(hole,points)&negative].tolist()
    unsupported[1]+=not bool(ids);areas[1]+=hole.area if not ids else 0
    evidence.append(dict(slug=slug,level_cm=cm,kind='hole',area_km2=hole.area,x=hole.centroid.x,y=hole.centroid.y,evidence_row_ids=' '.join(map(str,ids))))
  rows.append(dict(slug=slug,level_cm=cm,**metrics,empty_patches=unsupported[0],empty_holes=unsupported[1],empty_patch_km2=areas[0],empty_hole_km2=areas[1]))
pd.DataFrame(rows).to_csv(root/(a.variant+'-geometry.csv'),index=False)
pd.DataFrame(evidence).to_csv(root/(a.variant+'-features.csv'),index=False)
print(pd.DataFrame(rows).groupby('slug')[['components','holes','empty_patches','empty_holes','length_weighted_absolute_turns']].mean().to_string())
