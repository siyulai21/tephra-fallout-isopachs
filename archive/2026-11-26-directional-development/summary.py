import json
from pathlib import Path
import pandas as pd,numpy as np
from src.empirical_isopach.core.catalog import datasets
root=Path('results/directional-geography-v6');rows=[]
for rec in datasets():
 slug=rec['slug'];old=root/'baseline'/slug/'multiscale'
 f=pd.read_csv(old/'measured-sites.csv.gz');g=pd.read_csv(old/'heldout.csv.gz')
 rows.append(dict(variant='v5',slug=slug,site=f.median_absolute_log_error.mean(),geographic=-g[g.design=='geographic'].groupby('fold').principal_log_score.mean().mean(),folds=4,passed=True))
 for d in (root/'experiments').iterdir():
  case=d/slug;full=case/'full/scores.csv'
  if not full.exists():continue
  f=pd.read_csv(full);gs=[pd.read_csv(p) for p in case.glob('geographic-*/scores.csv')]
  rows.append(dict(variant=d.name,slug=slug,site=f.median_absolute_log_error.mean(),geographic=np.mean([-g.principal_log_score.mean() for g in gs]) if gs else np.nan,folds=len(gs),passed=all(g.fit_passed.all() for g in [f,*gs])))
r=pd.DataFrame(rows);r.to_csv(root/'all-attempts.csv',index=False)
x=r.groupby('variant').agg(cases=('slug','count'),site=('site','mean'),geo=('geographic','mean'),folds=('folds','sum'),passed=('passed','all'))
print(x.sort_values('geo').to_string())
