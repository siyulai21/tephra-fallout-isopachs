"""Build checked principal packets from the matched continuous-growth fits."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from .nested_footprints import cases
from src.empirical_isopach.models.nested.footprints.release import run


def job(args):
    slug,record,fits,output=args
    missing=[key for key in ('full','fold-0','fold-1','fold-2','fold-3')
             if not (fits/slug/(key+'.json')).is_file()]
    if missing:
        raise ValueError(f'{slug}: missing completed fit receipts: {missing}')
    report=run(record['path'],output/slug,record['name'],resume=True,upstream=fits/slug)
    print(slug,'release',report['draw_checks_passed'],report['validation'],flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--fits',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--case',action='append')
    p.add_argument('--workers',type=int,default=2)
    a=p.parse_args()
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        list(pool.map(job,[(s,r,a.fits,a.output) for s,r in cases().items() if not a.case or s in a.case]))
