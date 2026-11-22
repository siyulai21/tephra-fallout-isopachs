"""Replay the three global deposits from the eight registered cases and the separate Pinatubo comparison."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from .nested_reach import case_home
from .nested_reach_challenge import records
from src.empirical_isopach.core.audit import audited_data
from src.empirical_isopach.visualization.deposit import FAMILIES, render_deposit


def render(job):
    rec, family, output = job
    data,meta = audited_data(rec["path"],allow_intervals=True)
    target = output/rec["slug"]/family
    receipt = render_deposit(case_home(rec["slug"])/family,family,data,meta,rec["name"],output=target)
    print(rec["slug"],family,receipt["median_map_volume_km3"],flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--case",choices=[r["slug"] for r in records()])
    parser.add_argument("--workers",type=int,default=1)
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("Choose an empty output directory")
    jobs = [(r,f,args.output) for r in records() if args.case in (None,r["slug"]) for f in FAMILIES]
    with ProcessPoolExecutor(args.workers) as pool:
        list(pool.map(render,jobs))


if __name__=="__main__":
    main()
