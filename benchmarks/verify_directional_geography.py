"""Replay the matched v6 geography study and each promoted directional posterior."""

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .publication import released_datasets, result_home
from .verify_directional_multiscale import verify_case
from src.empirical_isopach.core.catalog import datasets
from src.empirical_isopach.models.directional.multiscale import engine_sha256
from src.empirical_isopach.models.directional.multiscale.geometry import (
    LENGTHS,
    NOISE,
    RANGES,
    SPREAD,
    TAILS,
    SCHEMA,
)
from src.empirical_isopach.visualization.deposit_field import digest

STUDY = Path("comparisons/directional/geographic-regularity")
BASELINE = Path("archive/2026-11-25-directional-v5/current-cases")


def verify_publication_identity(path, expected):
    """Verify study sources at their signed revisions and verify reviewed replacements."""
    current = digest(path)
    if current == expected:
        return True
    name = path.as_posix()
    for manifest in sorted(Path("archive").glob("*/directional-study-sources.json")):
        revisions = json.loads(manifest.read_text())
        record = revisions.get(name)
        if record and record["current_sha256"] == current:
            assert record["original_sha256"] == expected == digest(Path(record["archived_source"]))
            return True
    allowed = {
        'benchmarks/principal_volumes.py', 'data/datasets.json',
        'src/empirical_isopach/visualization/deposit_field.py',
        'src/empirical_isopach/visualization/summary.py',
        'benchmarks/verify_directional_geography.py',
        'eruptions/minoan/results/minimum-fit-2026-11-25/directional/deposit.json',
        'eruptions/minoan/results/minimum-fit-2026-11-25/directional/deposit.png',
    }
    assert name in allowed, name
    revisions = json.loads(Path('archive/2026-11-25-nested-growth/upstream-directional-publication/revisions.json').read_text())
    record = revisions['files'][name]
    assert record['original_sha256'] == expected == digest(Path(record['archived_source']))
    assert record['current_sha256'] == current
    return True


def one(job):
    record, pair, home, study, identifiers, count = job
    paths = {
        "input_sha256": record["path"],
        "fit_report_sha256": home / "multiscale/report.json",
        "posterior_sha256": home / "multiscale/posterior.npz",
        "figure_sha256": home / "deposit.png",
        "map_receipt_sha256": home / "deposit.json",
        "median_grid_sha256": home / "median-grid.npz",
        "probability_figure_sha256": home / "exceedance.png",
        "probability_receipt_sha256": home / "exceedance.json",
        "fold_marginals_sha256": study / "fold-marginals" / (record["slug"] + ".npz"),
        "paired_figure_sha256": study / (record["slug"] + ".png"),
        "paired_figure_receipt_sha256": study / (record["slug"] + ".json"),
    }
    assert all(verify_publication_identity(path, identifiers[key]) for key, path in paths.items())
    report = json.loads((home / "multiscale/report.json").read_text())
    assert report["fit"]["engine_sha256"] == engine_sha256()
    assert report["fit"]["passed"] and report["integration_passed"]
    assert all(
        f["status"] == "empty" or (f["passed"] and f["engine_sha256"] == engine_sha256())
        for f in report["folds"]
    )
    assert json.loads((home / "deposit.json").read_text())["contour_evaluation"] == (
        "direct posterior median at Cartesian display nodes"
    )
    plate = json.loads((study / (record["slug"] + ".json")).read_text())
    assert plate["input_sha256"] == identifiers["input_sha256"]
    assert plate["posterior_sha256"] == digest(home / "multiscale/posterior.npz")
    assert plate["previous_posterior_sha256"] == digest(BASELINE / record["slug"] / "multiscale/posterior.npz")
    assert plate["figure_sha256"] == digest(study / (record["slug"] + ".png"))
    assert len(plate["windows_km"]) == 3 and plate["cartesian_long_side"] == 425
    verify_case(record, pair, home=home, study=study, previous=BASELINE, protocol=SCHEMA, state_count=count)
    return record["slug"]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--study", type=Path, default=STUDY)
    p.add_argument("--release", type=Path, help="A staged release; defaults to registered case homes")
    p.add_argument("--jobs", type=int, default=2)
    a = p.parse_args()
    provenance = json.loads((a.study / "provenance.json").read_text())
    assert provenance["protocol"] == SCHEMA
    assert provenance["numerical_engine_sha256"] == engine_sha256()
    assert provenance["baseline_provenance_sha256"] == digest(BASELINE / "provenance.json")
    for name, expected in provenance["source_files_sha256"].items():
        verify_publication_identity(Path(name), expected)
    count = (
        len(RANGES)
        * 2
        * len(LENGTHS)
        * len(NOISE)
        * sum(t != 0 or s != 0 for t in TAILS for s in SPREAD)
    )
    assert provenance["states_per_partition"] == count
    cases = pd.read_csv(a.study / "paired-results.csv").set_index("slug")
    editions = datasets() if a.release else released_datasets("directional")
    assert set(cases.index) == {r["slug"] for r in editions}
    jobs = [
        (
            r,
            cases.loc[r["slug"]],
            a.release / r["slug"] if a.release else result_home(r) / "directional",
            a.study,
            provenance["cases"][r["slug"]],
            count,
        )
        for r in editions
    ]
    with ProcessPoolExecutor(a.jobs) as pool:
        for slug in pool.map(one, jobs):
            print(slug, "field, evidence, geography, probabilities and integrals replayed", flush=True)
    pairs = pd.read_csv(a.study / "paired-known-deposits.csv.gz")
    assert (pairs.input_sha256_current == pairs.input_sha256_previous).all()
    np.testing.assert_allclose(pairs.truth_volume_km3_current, pairs.truth_volume_km3_previous)
    for prefix in ("", "previous-"):
        intervals = pd.read_csv(a.study / (prefix + "known-volume-intervals.csv"))
        history = json.loads((a.study / (prefix + "known-volume-refinements.json")).read_text())
        assert np.array_equal(
            intervals.covers_truth,
            intervals.truth_volume_km3.between(intervals.lower95_km3, intervals.upper95_km3),
        )
        for row in intervals.itertuples():
            last = history[f"{row.design}-{row.seed}"][-1]
            assert row.integration_passed == last["passed"]
            np.testing.assert_allclose(
                [row.lower95_km3, row.upper95_km3], np.array(last["conditional_quantiles_km3"])[[0, 2]]
            )
    sensitivity = pd.read_csv(a.study / "resolution-sensitivity.csv")
    assert len(sensitivity) == 3 * len(cases) and set(sensitivity.shared_fraction) == {0.0, 0.5, 1.0}
    assert set(sensitivity.input_sha256) == set(cases.input_sha256)
    geometry = pd.read_csv(a.study / "geometry.csv.gz")
    assert set(geometry.slug) == set(cases.index)
    assert set(geometry.cartesian_long_side) == {193, 385}
    assert set(geometry.method) == {"previous", "multiscale"}
    features = pd.read_csv(a.study / "secondary-features.csv.gz")
    assert set(features.method) == {"previous", "multiscale"}
    assert set(features.cartesian_long_side) == {193, 385}
    surveys = pd.read_csv(a.study / "matched-surveys.csv.gz")
    assert surveys.fit_passed_current.all() and surveys.fit_passed_previous.all()
    np.testing.assert_allclose(surveys.observed_log_cm_current, surveys.observed_log_cm_previous)
    current = json.loads((a.study / "survey-fits.json").read_text())
    previous = json.loads((a.study / "previous-survey-fits.json").read_text())
    assert len(current) == len(previous) == 8
    for name, receipt in current.items():
        assert not set(receipt["training_ids"]) & set(receipt["withheld_ids"])
        for key in ("training_ids", "withheld_ids"):
            assert receipt[key] == previous[name][key]
    for name, expected in provenance["study_files_sha256"].items():
        assert digest(a.study / name) == expected, name
    print("Verified the current directional geography study and its matched historical controls.")


if __name__ == "__main__":
    main()
