"""Refresh only current principal comparison readers after installing reviewed v6 packets."""
from pathlib import Path
import json,shutil,subprocess
from benchmarks.publication import result_home,published_methods
from benchmarks.global_case_validation import field_validation
from benchmarks.minimum_fit_results import assess
from src.empirical_isopach.core.catalog import datasets
from src.empirical_isopach.visualization.principal import volume_rows
from src.empirical_isopach.visualization.deposit_field import digest
root=Path('results/directional-geography-v6')
for r in datasets():
 if not r.get('reported_file'):continue
 h=result_home(r);families=[k for k in ('directional','spherical-mixture','nested-isopachs') if k in published_methods(r)]
 field_validation(h,r['path'],r['slug'],verify=False,families=families)
 field_validation(h,r['path'],r['slug'],families=families)
 assess(r,write=True)
 path=h/'release.json'
 if path.exists():
  receipt=json.loads(path.read_text());receipt['methods']=volume_rows(h,input_sha256=digest(r['path']))
  receipt['directional_median_map_km3']=next(v['median_map_volume_km3'] for v in receipt['methods'] if v['method']=='directional')
  receipt['files']={name:digest(Path(name)) for name in receipt['files']}
  path.write_text(json.dumps(receipt,indent=2,allow_nan=False)+'\n')
 print(r['slug'],'source categories, exact-site maps and geographic predictions refreshed',flush=True)
commands=[('minimum_fit_summary',['--output',str(root/'catalogue-minima')]),('regional',['--summarize-only','--published','--output',str(root/'catalogue-regional')]),('global_posterior',['--summarize-only','--published','--output',str(root/'catalogue-spherical')])]
for module,args in commands:
 subprocess.run(['.venv/bin/python','-m','benchmarks.'+module,*args],check=True)
for source,target,names in [('catalogue-regional','regional',['comparison.csv','comparison.png','comparison.svg']),('catalogue-spherical','spherical-mixture',['comparison.csv','comparison.png','comparison.svg','report.json']),('catalogue-minima','reported-minima',['README.md','all-distances.csv','site-ratios.png'])]:
 for name in names:shutil.copy2(root/source/name,Path('comparisons')/target/name)
