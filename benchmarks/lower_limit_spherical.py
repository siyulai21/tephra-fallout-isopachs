"""Fit independent spherical priors and geographic deletions on the registered minimum-fit inputs."""
import argparse
from dataclasses import replace
from pathlib import Path

from src.empirical_isopach.core.audit import audited_data
from src.empirical_isopach.core.catalog import datasets
from src.empirical_isopach.core.validation import spatial_folds, training_mask
from src.empirical_isopach.models.mass_mixture.inference import Sampling
from src.empirical_isopach.models.mass_mixture.source_protocol import protocol
from src.empirical_isopach.models.mass_mixture.study import fit_checked
from src.empirical_isopach.validation import automatic_buffer


def partition(record, key, folder):
    data, meta = audited_data(record["path"])
    latitude = meta["vent_lonlat"][1]
    baseline, alternatives = protocol(record.get("global_geometry", "displaced"))
    settings = Sampling()
    if key.startswith("fold-"):
        fold = int(key.removeprefix("fold-"))
        if fold not in range(4):
            raise ValueError("Use a registered geographic fold from 0 to 3.")
        data = data.canonical()
        train, _ = training_mask(data, spatial_folds(data.xy, 4), fold, automatic_buffer(data)[0])
        data = data.subset(train)
        prior = baseline
        settings = replace(settings, seed=settings.seed + fold + 1)
        folder = folder / "validation" / key
    else:
        prior = {p.name: p for p in (baseline, *alternatives)}[key]
        folder = folder / key
    return fit_checked(data, latitude, prior, folder, settings, progress=print)[1]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("case", choices=[r["slug"] for r in datasets() if r.get("reported_file")])
    parser.add_argument("part", help="A fixed prior name, fold-0 through fold-3, or assemble")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    rec = next(r for r in datasets() if r["slug"] == args.case)
    if args.part == "assemble":
        from src.empirical_isopach.global_reporting import global_dataset
        from src.empirical_isopach.models.continuation_review import spherical_review
        global_dataset(rec["path"], args.out, rec["name"], resume=True)
        spherical_review(rec["path"], args.out, args.out / "sensitivity-study")
    else:
        partition(rec, args.part, args.out / "posteriors")
