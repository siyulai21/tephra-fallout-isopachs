"""Publish the reviewed current directional family from independently trained fold packets."""

import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from src.empirical_isopach.core.catalog import datasets
from src.empirical_isopach.models.directional.multiscale.release import assemble


def run(args):
    record, root, output = args
    slug = record["slug"]
    folder = output / slug
    assemble(record["path"], folder, record["name"], work=root / slug)
    print(slug, flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--work", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--jobs", type=int, default=3)
    p.add_argument("--slugs", nargs="*")
    a = p.parse_args()
    jobs = [
        (r, a.work, a.output)
        for r in sorted(datasets(), key=lambda r: r["slug"])
        if not a.slugs or r["slug"] in a.slugs
    ]
    with ProcessPoolExecutor(a.jobs) as pool:
        list(pool.map(run, jobs))
