"""Complete one production partition, including uniform numerical fallbacks."""
import argparse
from pathlib import Path

from src.empirical_isopach.core.audit import audited_data
from src.empirical_isopach.core.catalog import datasets
from src.empirical_isopach.models.directional.ensemble.members import splits
from src.empirical_isopach.models.nested.footprints.release import fit_part


def run(slug,key,root):
    rec=next(r for r in datasets() if r['slug']==slug)
    train,meta=audited_data(rec['path'])
    test=None
    if key!='full':
        _,train,test=next(p for p in splits(train,meta['vent_lonlat'][1],'geographic') if p[0]==int(key[-1]))
    result=fit_part(root/slug,key,train,test,meta)
    print(slug,key,'production passed',result['numerics']['passed'],flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('case')
    p.add_argument('part',choices=('full','fold-0','fold-1','fold-2','fold-3'))
    p.add_argument('--fits',type=Path,required=True)
    a=p.parse_args()
    run(a.case,a.part,a.fits)
