from pathlib import Path
import shutil,sys,types,importlib
root=Path('results/directional-geography-v6');source=Path('src/empirical_isopach/models/directional/multiscale');dest=root/'broad-amplitude-engine';dest.mkdir(exist_ok=True)
for p in source.glob('*.py'):shutil.copyfile(p,dest/p.name)
p=dest/'geometry.py';s=p.read_text();s=s.replace('return np.column_stack((columns, amplitude * local_part)), distance','n = len(local.centers) // 2\n    broad = np.any(local.averaging[:, :n] != 0, axis=1)\n    gains = np.where(broad, amplitude, LOCAL_SD)\n    return np.column_stack((columns, gains * local_part)), distance');p.write_text(s)
p=dest/'fit.py';s=p.read_text().replace('SPREAD, AMPLITUDES, LocalBasis','SPREAD, AMPLITUDES, LOCAL_SD, LocalBasis').replace('(amplitude * rim2) ** 2','(0.8 * amplitude**2 + 0.2 * LOCAL_SD**2) * rim2**2');p.write_text(s)
p=dest/'posterior.py';s=p.read_text().replace('c.amplitude\n                    * rim[:, None] ** 2\n                    * self.local[c.length].features(xy, self.domain.axis_scale)','np.where(np.any(self.local[c.length].averaging[:, :len(self.local[c.length].centers)//2] != 0, axis=1), c.amplitude, LOCAL_SD)[None, :] * rim[:, None] ** 2 * self.local[c.length].features(xy, self.domain.axis_scale)').replace('(c.amplitude * rim2) ** 2','(0.8 * c.amplitude**2 + 0.2 * LOCAL_SD**2) * rim2**2');p.write_text(s)
name='src.empirical_isopach.models.directional._broad_probe';package=types.ModuleType(name);package.__path__=[str(dest.resolve())];sys.modules[name]=package
fit=importlib.import_module(name+'.fit').fit
import benchmarks.directional_challenges as ch
ch.fit=fit
p=root/'broad-amplitude-probe/smooth-90';p.mkdir(parents=True,exist_ok=True);ch.known('smooth',90,p);print((p/'volume-intervals.csv').read_text(),flush=True)
from src.empirical_isopach.core.catalog import datasets
from src.empirical_isopach.core.audit import audited_data
from src.empirical_isopach.models.directional.multiscale.validation import predictions,write_json
import pandas as pd
for rec in datasets():
 if rec['slug'] not in ('minoan','sakurajima-satsuma'):continue
 data,meta=audited_data(rec['path']);p=root/'broad-amplitude-probe'/rec['slug'];p.mkdir(exist_ok=True)
 model,receipt=ch.checked_fit(data,meta['vent_lonlat'][1],p)
 pred=pd.DataFrame(predictions(model,data,'full','',receipt['passed']));pred.to_csv(p/'scores.csv',index=False)
 model.save(p/'field.npz')
 print(rec['slug'],pred.median_absolute_log_error.mean(),receipt['passed'],flush=True)
