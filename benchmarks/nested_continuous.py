"""Fit the continuous-direction candidate on every registered geographic partition."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from .nested_footprints import cases
from src.empirical_isopach.core.audit import audited_data
from src.empirical_isopach.models.directional.ensemble.members import splits
from src.empirical_isopach.models.nested.footprints.continuous import write_fit
from src.empirical_isopach.models.nested.footprints.release import fit_part
from src.empirical_isopach.models.nested.study import write_json


def job(args):
    slug,rec,index,output,design=args
    data,meta=audited_data(rec['path'])
    assert not (data.lower_censored & ~data.interval).any(), 'Use the registered minimum-fitting edition'
    key,train,test='full',data,None
    if index is not None:
        key=('fold-' if design=='geographic' else 'quadrant-')+str(index)
        _,train,test=next(s for s in splits(data,meta['vent_lonlat'][1],design) if s[0]==index)
        if train is None or len(train.xy)<3 or not (~train.censored).any():
            folder=output/slug;folder.mkdir(parents=True,exist_ok=True)
            write_json(folder/(key+'-outcome.json'),dict(status='insufficient_training_or_empty_sector',
                       input_sha256=meta['input_sha256'],train=[] if train is None else train.row_ids.tolist(),test=test.row_ids.tolist()))
            return
    receipt=(fit_part if design=='quadrants' else write_fit)(output/slug,key,train,test,meta)
    if design=='quadrants':
        write_json(output/slug/(key+'-outcome.json'),dict(status='estimated',**receipt))
    print(slug,key,receipt['numerics']['passed'],receipt['draws'],flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',required=True,type=Path)
    p.add_argument('--case',action='append')
    p.add_argument('--workers',type=int,default=2)
    p.add_argument('--design',choices=('geographic','quadrants'),default='geographic')
    p.add_argument('--full-only',action='store_true')
    p.add_argument('--skip-full',action='store_true')
    a=p.parse_args()
    folds=[None] if a.full_only else [None,0,1,2,3] if a.design=='geographic' else [0,1,2,3]
    if a.skip_full:
        folds=[i for i in folds if i is not None]
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        list(pool.map(job,[(s,r,i,a.output,a.design) for i in folds for s,r in cases().items() if not a.case or s in a.case]))
