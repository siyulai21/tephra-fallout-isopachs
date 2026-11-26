"""Compare matched site probabilities and geographic deletions with the preceding release."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from .directional_scale_maps import preceding_control
from .publication import result_home
from src.empirical_isopach.core.audit import audited_data
from src.empirical_isopach.core.catalog import datasets
from src.empirical_isopach.models.directional.multiscale.validation import predictions
from src.empirical_isopach.visualization.deposit import read_json
from src.empirical_isopach.visualization.deposit_field import open_field


def save(frame, path):
    frame.to_csv(
        path,
        index=False,
        lineterminator="\n",
        compression={"method": "gzip", "mtime": 0} if path.suffix == ".gz" else None,
    )


def evaluation(release, output, baseline=None):
    output.mkdir(parents=True, exist_ok=True)
    cases = []
    events = []
    pairings = []
    measured = []
    for rec in datasets():
        slug = rec["slug"]
        data, meta = audited_data(rec["path"])
        folder = result_home(rec) / "directional"
        previous = baseline / slug if baseline else preceding_control(rec, release)
        prior = read_json((previous / "multiscale" if baseline else folder / "extent") / "report.json")
        old = read_json(previous / "deposit.json")
        new = read_json(release / slug / "deposit.json")
        assert prior["input_sha256"] == old["input_sha256"] == new["input_sha256"] == meta["input_sha256"]
        if baseline:
            pred = pd.read_csv(previous / "multiscale/measured-sites.csv.gz").set_index("row_id")
        else:
            view, model, _ = open_field(folder, "directional", data, meta, review=folder / "extent/report.json")
            pred = pd.DataFrame(predictions(model, data, "full", "", old["posterior_checks_passed"])).set_index("row_id")
            np.testing.assert_allclose(view.central(data.xy)*np.log(10.0), pred.median_log_cm, atol=2e-6)
        current = release / slug / "multiscale"
        now = read_json(current / "report.json")
        trained = pd.read_csv(current / "measured-sites.csv.gz").set_index("row_id").loc[pred.index]
        historic = pd.read_csv((previous / "multiscale" if baseline else folder / "extent") / "heldout.csv.gz")
        newtest = pd.read_csv(current / "heldout.csv.gz")
        pair = historic.merge(
            newtest, on=["design", "fold", "row_id"], suffixes=("_previous", "_new"), validate="one_to_one"
        )
        pair["slug"] = slug
        if "evidence_previous" in pair:
            assert (pair.evidence_previous == pair.evidence_new).all()
            pair["evidence"] = pair.evidence_new
        pairings.append(
            pair[
                [
                    "slug",
                    "design",
                    "fold",
                    "row_id",
                    "evidence",
                    "fit_passed_previous",
                    "fit_passed_new",
                    "principal_log_score_previous",
                    "principal_log_score_new",
                    "principal_cdf_previous",
                    "principal_cdf_new",
                ]
            ]
        )
        measured.append(
            trained.reset_index()
            .merge(pred.reset_index(), on="row_id", suffixes=("_new", "_previous"))
            .assign(slug=slug)
        )
        row = dict(
            slug=slug,
            input_sha256=meta["input_sha256"],
            accepted_rows=len(data.xy),
            previous_km3=old["median_map_volume_km3"],
            median_map_km3=new["median_map_volume_km3"],
            q25_km3=new["thickness_band"]["volumes_km3"][0],
            q75_km3=new["thickness_band"]["volumes_km3"][-1],
            conditional95_low_km3=now["conditional_interval95_km3"][0],
            conditional95_high_km3=now["conditional_interval95_km3"][1],
            previous_posterior_passed=old["posterior_checks_passed"],
            posterior_passed=new["posterior_checks_passed"],
            geographical_integration_passed=new["integration_passed"] and new["thickness_band"]["passed"],
        )
        for k, frame in (("previous", pred), ("new", trained)):
            exact = frame.evidence == "exact"
            row[k + "_site_mae_log_cm"] = float(
                frame.loc[exact, "median_absolute_log_error"].astype(float).mean()
            )
            for kind, g in frame.groupby("evidence"):
                events.append(
                    dict(
                        slug=slug,
                        method=k,
                        design="fitted",
                        evidence=kind,
                        rows=len(g),
                        negative_log_score=-g.principal_log_score.mean(),
                    )
                )
        for design in ("geographic", "quadrants"):
            matched = pair.loc[pair.design == design]
            row[design + "_matched_rows"] = len(matched)
            for label in ("previous", "new"):
                score = "principal_log_score_" + label
                cdf = "principal_cdf_" + label
                row[label + "_" + design + "_score"] = -matched.groupby("fold")[score].mean().mean()
                row[label + "_" + design + "_coverage95"] = (
                    matched.loc[matched.evidence == "exact", cdf].between(0.025, 0.975).mean()
                )
                for kind, g in matched.groupby("evidence"):
                    events.append(
                        dict(
                            slug=slug,
                            method=label,
                            design=design,
                            evidence=kind,
                            rows=len(g),
                            negative_log_score=-g[score].mean(),
                        )
                    )
            row["previous_" + design + "_passed_folds"] = (
                prior["validation"][design]["passed_folds"]
                if "passed_folds" in prior["validation"][design]
                else sum(
                    f.get("passed", f.get("fit_passed", False))
                    for f in prior["folds"]
                    if f["design"] == design
                )
            )
            row["new_" + design + "_passed_folds"] = now["validation"][design]["passed_folds"]
            row["new_" + design + "_scored_rows"] = now["validation"][design]["scored_rows"]
        cases.append(row)
    save(pd.DataFrame(cases), output / "paired-results.csv")
    save(pd.DataFrame(events), output / "events.csv")
    save(pd.concat(pairings), output / "matched-holdouts.csv.gz")
    save(pd.concat(measured), output / "matched-sites.csv.gz")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--release", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--baseline", type=Path, help="Archived matched multiscale release; otherwise compare the extent control")
    a = p.parse_args()
    evaluation(a.release, a.output, a.baseline)
