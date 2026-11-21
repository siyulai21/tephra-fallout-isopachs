"""Reproduce the complete nested-isopach case reports in a fresh results directory."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from src.empirical_isopach.core.catalog import datasets
from src.empirical_isopach.models.nested.report import nested_dataset


def report(job):
    path,folder,name,resume,budget,base = job
    result = nested_dataset(path,folder,name,resume=resume,budget=budget,base_cache=base)
    print(name,result["conditional_volume_km3"],flush=True)


if __name__=="__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--workers",type=int,default=1)
    p.add_argument("--resume",action="store_true")
    p.add_argument("--initial-profile",action="store_true",help="Reproduce the archived prior on distal slopes")
    p.add_argument("--base-cache",type=Path,help="Reuse verified intrinsic profile packets by eruption")
    p.add_argument("--extra",type=Path,help="Also evaluate a separate, unregistered CSV snapshot")
    p.add_argument("--extra-slug",default="unregistered")
    p.add_argument("--extra-name",default="Unregistered input")
    a = p.parse_args()
    jobs = [(r["path"],a.output/r["slug"],r["name"],a.resume,not a.initial_profile,a.base_cache/r["slug"] if a.base_cache else None) for r in datasets()]
    if a.extra:
        if any(job[1].name==a.extra_slug for job in jobs):
            p.error("Use the registered case for a slug already in the catalogue")
        jobs.append((a.extra,a.output/a.extra_slug,a.extra_name,a.resume,not a.initial_profile,a.base_cache/a.extra_slug if a.base_cache else None))
    with ProcessPoolExecutor(a.workers) as executor:
        list(executor.map(report,jobs))
