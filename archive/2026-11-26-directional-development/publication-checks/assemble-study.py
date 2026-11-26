from pathlib import Path
import json,tarfile,shutil,hashlib
import pandas as pd
base=Path('results/directional-geography-v6'); source=base/'principal-study'; target=Path('comparisons/directional/geographic-regularity')
for f in source.iterdir():
 if f.is_file(): shutil.copy2(f,target/f.name)
shutil.copytree(source/'fold-marginals',target/'fold-marginals',dirs_exist_ok=True)
(target/'regional').mkdir(exist_ok=True)
for f in (base/'regional').glob('*'):
 if f.is_file() and f.suffix in ('.csv','.json','.gz'): shutil.copy2(f,target/'regional'/f.name)
with tarfile.open(target/'qualification-attempts.tar.gz','w:gz') as tar:
 for root,label in [(base/'principal-fits','partitions'),(base/'principal-challenges','current-challenges'),(base/'v5-matched-challenges','previous-challenges'),(source/'resolution-attempts','coherent-fraction')]:
  for f in sorted(root.rglob('*')):
   if f.is_file() and (f.suffix in ('.json','.csv','.gz') or f.name=='volume-draws.npz'):
    info=tar.gettarinfo(str(f),arcname=str(Path(label)/f.relative_to(root)));info.uid=info.gid=0;info.uname=info.gname='';info.mtime=0
    with f.open('rb') as stream:tar.addfile(info,stream)
frame=pd.read_csv(base/'principal-matched-scores.csv')
lines=['| Eruption | Site error v5 → v6 | Geographic score v5 → v6 | Quadrant score v5 → v6 |','| --- | ---: | ---: | ---: |']
for r in frame.itertuples():lines.append(f'| {r.slug} | {r.old_site:.3f} → {r.new_site:.3f} | {r.old_geographic:.3f} → {r.new_geographic:.3f} | {r.old_quadrants:.3f} → {r.new_quadrants:.3f} |')
p=target/'README.md';s=p.read_text();start=s.index('<!-- matched-results -->');end=s.index('\n\nAll 78',start);s=s[:start]+'<!-- matched-results -->\n\n'+'\n'.join(lines)+s[end:];p.write_text(s)
print('Collected',len(list(target.rglob('*'))),'files/directories; archive MB',round((target/'qualification-attempts.tar.gz').stat().st_size/1e6,2))
