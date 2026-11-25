"""Rebuild smooth directional posteriors on all accepted sites and matched buffered holdouts."""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from src.empirical_isopach.core.audit import audited_data
from src.empirical_isopach.core.catalog import datasets
from src.empirical_isopach.models.directional.multiscale.validation import run_partition


def run(job):
    record, work, kind, fold, power = job
    data, meta = audited_data(record["path"])
    run_partition(data, meta["vent_lonlat"][1], work / record["slug"], kind, fold, resume=True, power=power)
    return record["slug"], kind, fold


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--work", type=Path, required=True)
    p.add_argument("--jobs", type=int, default=3)
    p.add_argument("--power", type=int, default=8)
    p.add_argument("--slugs", nargs="*")
    p.add_argument("--designs", nargs="+", default=["full", "geographic", "quadrants"])
    a = p.parse_args()
    jobs = [
        (r, a.work, k, f, a.power)
        for k in a.designs
        for f in ([None] if k == "full" else range(4))
        for r in datasets()
        if not a.slugs or r["slug"] in a.slugs
    ]
    with ProcessPoolExecutor(max_workers=a.jobs) as pool:
        for future in as_completed([pool.submit(run, j) for j in jobs]):
            print(*future.result(), flush=True)
