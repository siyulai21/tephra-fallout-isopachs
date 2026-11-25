"""Run the paired nested geographic study using verified, published baseline packets."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from .nested_reach_challenge import records
from src.empirical_isopach.models.nested.reach_report import reach_dataset


def case_home(slug):
    """Locate the baseline used by the matched reach study, including its development case."""
    if slug == "pinatubo-1991":
        return Path("comparisons/nested-isopachs/pinatubo-preview")
    from .publication import matched_edition, result_home
    from src.empirical_isopach.core.catalog import datasets
    record = next(r for r in datasets() if r["slug"] == slug)
    return result_home(matched_edition(record))


def run(job):
    rec, out, parents, resume = job
    slug = rec["slug"]
    parent = parents / slug if parents else case_home(slug) / "nested-whole-area"
    result = reach_dataset(rec["path"], out / slug, rec["name"], parent=parent, resume=resume)
    print(slug, result["median_map_volume_km3"], result["numerics"], flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--parents", type=Path, help="Override the directory of published baseline packets")
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--resume", action="store_true")
    a = p.parse_args()
    jobs = [(rec, a.output, a.parents, a.resume) for rec in records(published=a.parents is None)]
    with ProcessPoolExecutor(a.workers) as pool:
        list(pool.map(run,jobs))


if __name__ == "__main__":
    main()
