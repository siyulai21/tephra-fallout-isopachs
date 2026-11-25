"""Archive the conditional mixture at every deleted site so its likelihood can be replayed."""

import argparse
import json
from pathlib import Path

import numpy as np

from src.empirical_isopach.core.audit import audited_data
from src.empirical_isopach.core.catalog import datasets
from src.empirical_isopach.models.directional.ensemble.members import splits
from src.empirical_isopach.models.directional.multiscale.posterior import DirectionalField
from src.empirical_isopach.models.directional.multiscale.release import digest


def archive_predictions(work, output):
    output.mkdir(parents=True, exist_ok=True)
    for record in datasets():
        slug = record["slug"]
        data, meta = audited_data(record["path"])
        arrays = dict(input_sha256=meta["input_sha256"])
        for design in ("geographic", "quadrants"):
            for fold, _, test in splits(data, meta["vent_lonlat"][1], design):
                key = f"{design}-{fold}"
                folder = work / slug / key
                receipt = json.loads((folder / "fit.json").read_text())
                if receipt["status"] != "estimated":
                    continue
                field = folder / "field.npz"
                assert (
                    digest(field) == receipt["packet_sha256"]
                    and test.row_ids.tolist() == receipt["test_row_ids"]
                )
                d = DirectionalField.load(field).distribution(test.xy, observation=True, log_sd=test.log_sd)
                for name, value in dict(
                    mean=d.means,
                    variance=d.variances,
                    weight=d.weights,
                    row_id=test.row_ids,
                    posterior_sha256=digest(field),
                ).items():
                    arrays[key + "_" + name] = value
        np.savez_compressed(output / (slug + ".npz"), **arrays)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    archive_predictions(args.work, args.output)
