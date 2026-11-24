"""Rebuild each principal nested deposit, independent integrals, support, and case plate."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from .nested_footprints import cases
from src.empirical_isopach.models.nested.footprints.release import run


def generate(job):
    slug,record,inputs,out = job
    run(record["path"],out/slug,record["name"],resume=True,upstream=inputs/slug)
    print(slug,flush=True)


if __name__ == "__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--fits",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--workers",type=int,default=2)
    p.add_argument("--case",action="append")
    a=p.parse_args()
    jobs=[(s,r,a.fits,a.output) for s,r in cases().items() if not a.case or s in a.case]
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        list(pool.map(generate,jobs))
