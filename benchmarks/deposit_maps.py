"""Replay the verified global fields for each registered principal eruption input."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from .principal_volumes import principal_home
from src.empirical_isopach.core.catalog import datasets
from src.empirical_isopach.core.audit import audited_data
from src.empirical_isopach.visualization.deposit import FAMILIES, render_deposit


def render(job):
    rec, family, output = job
    data,meta = audited_data(rec["path"],allow_intervals=True)
    target = output/rec["slug"]/family
    receipt = render_deposit(principal_home(rec)/family,family,data,meta,rec["name"],output=target)
    print(rec["slug"],family,receipt["median_map_volume_km3"],flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--case",choices=[r["slug"] for r in datasets()])
    parser.add_argument("--workers",type=int,default=1)
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("Choose an empty output directory")
    jobs = [(r,f,args.output) for r in datasets() if args.case in (None,r["slug"])
            for f in FAMILIES if (principal_home(r)/f/"deposit.json").is_file()]
    with ProcessPoolExecutor(args.workers) as pool:
        list(pool.map(render,jobs))


if __name__=="__main__":
    main()
