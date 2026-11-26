"""Collect simulations, declared-out survey tests and development ablations without changing their fits."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .directional_evaluation import save
from src.empirical_isopach.core.catalog import datasets


def collect(challenges, output, iterations=None, baseline=None):
    output.mkdir(parents=True, exist_ok=True)
    known = pd.concat([pd.read_csv(f) for f in sorted(challenges.glob("*/recovery.csv"))])
    if baseline:
        old = pd.concat(pd.read_csv(f) for f in sorted(baseline.glob("*/recovery.csv")))
    else:
        old = pd.read_csv("comparisons/global-extent/known-deposits.csv.gz")
        old = old[(old.method == "directional") & (old.model == "principal")].copy()
    keys = ["design", "seed", "scramble", "level_cm", "truth"]
    paired = known.merge(old, on=keys, suffixes=("_current", "_previous"), validate="one_to_one")
    assert len(paired) == len(old)
    np.testing.assert_allclose(paired.truth_volume_km3_current, paired.truth_volume_km3_previous)
    if baseline:
        assert (paired.input_sha256_current == paired.input_sha256_previous).all()
        save(old, output / "previous-known-deposits.csv.gz")
        old_intervals = pd.concat(pd.read_csv(f) for f in sorted(baseline.glob("*/volume-intervals.csv")))
        save(old_intervals, output / "previous-known-volume-intervals.csv")
        old_refinements = {f.parent.name: json.loads(f.read_text())
                           for f in sorted(baseline.glob("*/volume-refinements.json"))}
        (output / "previous-known-volume-refinements.json").write_text(
            json.dumps(old_refinements, indent=2, allow_nan=False) + "\n")
    save(known, output / "known-deposits.csv.gz")
    intervals = pd.concat([pd.read_csv(f) for f in sorted(challenges.glob("*/volume-intervals.csv"))])
    save(intervals, output / "known-volume-intervals.csv")
    refinements = {
        f.parent.name: json.loads(f.read_text()) for f in sorted(challenges.glob("*/volume-refinements.json"))
    }
    (output / "known-volume-refinements.json").write_text(
        json.dumps(refinements, indent=2, allow_nan=False) + "\n"
    )
    save(paired, output / "paired-known-deposits.csv.gz")
    radii = []
    for source in sorted(challenges.glob("*/radii.csv")):
        frame = pd.read_csv(source)
        frame.insert(0, "bearing_deg", np.arange(len(frame)))
        radii.append(frame.assign(design=source.parent.name))
    save(pd.concat(radii), output / "rotated-survey-radii.csv.gz")
    campaigns = pd.concat([pd.read_csv(f) for f in sorted(challenges.glob("*/survey.csv"))])
    save(campaigns, output / "survey-predictions.csv.gz")
    receipts = {
        f.parent.name.rsplit("-", 1)[0]: json.loads(f.read_text())
        for f in sorted(challenges.glob("*/survey-fit.json"))
    }
    (output / "survey-fits.json").write_text(json.dumps(receipts, indent=2, allow_nan=False) + "\n")
    previous_surveys = sorted(baseline.glob("*/survey.csv")) if baseline else []
    if previous_surveys:
        earlier = pd.concat(pd.read_csv(f) for f in previous_surveys)
        matched = campaigns.merge(
            earlier, on=["survey", "row_id", "input_sha256", "evidence"],
            suffixes=("_current", "_previous"), validate="one_to_one",
        )
        assert len(matched) == len(campaigns) == len(earlier)
        np.testing.assert_allclose(matched.observed_log_cm_current, matched.observed_log_cm_previous)
        previous_receipts = {
            f.parent.name.rsplit("-", 1)[0]: json.loads(f.read_text())
            for f in sorted(baseline.glob("*/survey-fit.json"))
        }
        for name, current in receipts.items():
            for key in ("training_ids", "withheld_ids"):
                assert current[key] == previous_receipts[name][key]
        save(matched, output / "matched-surveys.csv.gz")
        save(earlier, output / "previous-survey-predictions.csv.gz")
        (output / "previous-survey-fits.json").write_text(
            json.dumps(previous_receipts, indent=2, allow_nan=False) + "\n")
    if iterations:
        records = []
        for stage in ("candidate", "refined", "final", "neighborhood", "pooled"):
            for record in datasets():
                for receipt in sorted((iterations / stage / record["slug"]).glob("*/fit.json")):
                    fit = json.loads(receipt.read_text())
                    scores = receipt.with_name("scores.csv.gz")
                    if not scores.exists() or fit["status"] != "estimated":
                        continue
                    frame = pd.read_csv(scores)
                    records.append(
                        dict(
                            stage=stage,
                            slug=record["slug"],
                            protocol=fit["protocol"],
                            design=fit["design"],
                            fold=fit["fold"],
                            train_rows=len(fit["train_row_ids"]),
                            test_rows=len(frame),
                            training_identity=fit["training_identity"],
                            posterior_sha256=fit["packet_sha256"],
                            passed=fit["passed"],
                            negative_log_score=-frame.principal_log_score.mean(),
                            absolute_log_error_exact=frame.median_absolute_log_error.mean(),
                        )
                    )
        save(pd.DataFrame(records), output / "development-ablations.csv.gz")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--challenges", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--iterations", type=Path)
    parser.add_argument("--baseline", type=Path, help="Matched synthetic observations fitted with the archived engine")
    args = parser.parse_args()
    collect(args.challenges, args.output, args.iterations, args.baseline)
