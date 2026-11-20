"""Publish geographic validation and posterior break parameters for B-Tm thinning laws."""
import argparse
import json
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd

from benchmarks.btm_thinning import audit_input
from benchmarks.btm_thinning_fit import VARIANTS, warp
from benchmarks.btm_thinning_mass import load_full
from src.empirical_isopach.models.directional.study import digest, write_json

LABELS = {"smooth": "Published smooth reference", "one_break": "One shared break",
          "two_breaks": "Two shared breaks", "exponential": "Exponential kernels",
          "broken_exponential": "Exponential kernels, one break", "wide_prior": "One break, wider slope prior"}
from benchmarks.publication import BTM_THICKNESS_ONLY

HOME = BTM_THICKNESS_ONLY / "results"
PUBLISHED = HOME/"source-audit/thinning"
FILES = ("README.md", "fits.json", "heldout.csv", "summary.csv", "break-draws.npz")


def read(path):
    return json.loads(path.read_text())


def summarize(output, runs, held):
    scores = held.groupby(["variant", "fold"]).negative_log_score.mean()
    rows = []
    for variant, run in runs.items():
        folds = run["folds"]
        rows.append({"variant": variant, "full_sampling_pass": run["full_fit"]["passed"],
                     "equal_fold_nll": scores.loc[variant].mean() if folds else None,
                     "flagged_folds": [f["fold"] for f in folds if not f["passed"]]})
    pd.DataFrame(rows).to_csv(output/"summary.csv", index=False)
    lines = ["# Can a change in radial thinning improve geographic predictions?", "",
             "Six radial thinning laws use the same 68 accepted B-Tm observations and the same reference fan geometry. "
             "Five have predictions for all four withheld geographic regions. The wider slope prior is a full-data sensitivity. "
             "[Accepted observations](../../../../../data/eruptions/B-Tm_reviewed.csv) · [Whole-deposit volumes](volumes/README.md)", "",
             "| Thinning law | Full-data sampling | Mean held-out score | Geographic folds with warnings |",
             "|---|---|---:|---|"]
    for row in rows:
        score = row["equal_fold_nll"]
        score_text = "—" if score is None else f"{score:.3f}"
        flags = ", ".join(map(str, row["flagged_folds"])) or "None"
        lines.append(f"| {LABELS[row['variant']]} | {'Passed' if row['full_sampling_pass'] else 'Warning'} | "
                     f"{score_text} | {flags if score is not None else 'Full fit only'} |")
    change = scores.loc["one_break"] - scores.loc["smooth"]
    shifts = ", ".join(f"{value:+.3f}" for value in change)
    q = runs["one_break"]["full_fit"]["parameter_quantiles"]
    lo, med, hi = q["break_km"][0]
    slower, middle, faster = np.exp(q["log_slope_change"][0])
    lines += ["", "Each region receives equal weight. Lower negative log predictive scores indicate better predictions "
              f"at withheld measurements and bounds. Adding one break changes the four scores by {shifts}. "
              "[Individual held-out predictions](heldout.csv) · [Model scores](summary.csv)", "",
              f"The one-break fit places the change at a median radius of {med:.1f} km "
              f"(conditional 95% interval {lo:,.1f}–{hi:,.0f} km). The radial slope multiplier is {middle:.2f} "
              f"({slower:.2f}–{faster:.2f}). The interval allows both steeper and gentler thinning.", "",
              "The smooth reference combines short-range and bearing-dependent components. The break models change the "
              "radial slope continuously at one or two distances shared across bearings. Each training fold sets its own "
              "prior range from its nearest and farthest positive site radii. The classical segmented-exponential approach "
              "uses the square root of enclosed isopach area. These fits use distance from the vent. "
              "[Directional equations](../../../../../docs/methods/directional-continuation.md) · "
              "[Fierstein and Nathenson (1992)](https://pubs.usgs.gov/publication/70017275)", "",
              "[Fold membership, attempts and parameter quantiles](fits.json) · [Break-parameter draws](break-draws.npz) · "
              "[Checksums and software versions](design.json)", "", "## Reproduce", "", "\x60\x60\x60sh",
              "for v in smooth one_break two_breaks exponential broken_exponential wide_prior; do",
              '  python -m benchmarks.btm_thinning --work results/btm-break-check --variant "$v"',
              "done", "python -m benchmarks.btm_thinning_report --work results/btm-break-check --output results/btm-break-report",
              "\x60\x60\x60", ""]
    (output/"README.md").write_text("\n".join(lines))


def publish(work, output):
    output.mkdir(parents=True, exist_ok=True)
    runs, parameters, held = {}, {}, []
    fit_code_hashes = set()
    for variant, (n_breaks, _, _) in VARIANTS.items():
        run = read(work/variant/"result.json")
        if run["principal_input_sha256"] != digest(audit_input()):
            raise ValueError(f"{variant} used a different observation table")
        # Older fit caches also contain optional predictions. Only training records
        # and the geographic holdouts belong to this comparison.
        record = {key: run[key] for key in (
            "variant", "n_breaks", "exponential_kernels", "slope_ratio_log_sd", "principal_input_sha256", "folds")}
        record["full_fit"] = {key: run["full_fit"][key] for key in (
            "fold", "training_row_ids", "prior_break_domain_km", "attempts", "passed", "parameter_quantiles")}
        runs[variant] = record
        if run["folds"]:
            held.append(pd.read_csv(work/variant/"heldout.csv"))
        if n_breaks:
            fit_code_hashes.add(read(work/variant/"full/design.json")["experiment_sha256"])
            draws, _, _ = load_full(work, variant)
            for key in ("break_km", "log_slope_change"):
                parameters[f"{variant}__{key}"] = draws[key]
    if len(fit_code_hashes) != 1:
        raise ValueError("Break fits came from different versions of the thinning model")
    held = pd.concat(held, ignore_index=True)
    held.to_csv(output/"heldout.csv", index=False)
    write_json(output/"fits.json", runs)
    np.savez_compressed(output/"break-draws.npz", **parameters)
    summarize(output, runs, held)
    write_json(output/"design.json", {
        "input_sha256": digest(audit_input()),
        "versions": {k: version(k) for k in ("numpy", "numpyro", "jax", "jaxlib", "scipy", "arviz", "shapely")},
        "fit_code_sha256": fit_code_hashes.pop(),
        "report_fit_module_sha256": digest(Path(__file__).with_name("btm_thinning_fit.py")),
        "all_files": {name: digest(output/name) for name in FILES},
    })


def verify(folder=PUBLISHED, reference=HOME/"directional/single-structure/report.json"):
    design = read(folder/"design.json")
    assert design["input_sha256"] == digest(audit_input())
    assert design["report_fit_module_sha256"] == digest(Path(__file__).with_name("btm_thinning_fit.py"))
    assert set(design["all_files"]) == set(FILES)
    for name, sha in design["all_files"].items():
        assert digest(folder/name) == sha, name
    held = pd.read_csv(folder/"heldout.csv")
    summary = pd.read_csv(folder/"summary.csv").set_index("variant")
    assert summary.index.is_unique
    runs = read(folder/"fits.json")
    ref = read(reference)["validation"]["geographic"]
    expected = pd.DataFrame({"row_id": ref["row_ids"], "fold": ref["folds"],
                             "log_score": ref["predictions"]["log_score"]}).set_index("row_id")
    smooth = held[held.variant == "smooth"].set_index("row_id").loc[expected.index]
    np.testing.assert_allclose(smooth.negative_log_score, -expected.log_score, rtol=1e-8, atol=1e-10)
    np.testing.assert_array_equal(smooth.fold, expected.fold)
    assert set(runs) == set(summary.index) == set(VARIANTS)
    with np.load(folder/"break-draws.npz") as saved:
        keys = set()
        for variant, run in runs.items():
            full = run["full_fit"]
            assert run["principal_input_sha256"] == design["input_sha256"]
            assert set(full["training_row_ids"]) == set(expected.index)
            assert summary.at[variant, "full_sampling_pass"] == full["passed"]
            scores = held[held.variant == variant]
            if variant == "wide_prior":
                assert not run["folds"] and scores.empty and pd.isna(summary.at[variant, "equal_fold_nll"])
            else:
                assert set(scores.row_id) == set(expected.index) and scores.row_id.is_unique
                np.testing.assert_allclose(scores.groupby("fold").negative_log_score.mean().mean(),
                                           summary.at[variant, "equal_fold_nll"])
                for fold, original in zip(run["folds"], ref["fits"], strict=True):
                    for field in ("training_row_ids", "withheld_row_ids"):
                        assert fold[field] == original[field]
            count = run["n_breaks"]
            last = full["attempts"][-1]
            assert set(full["parameter_quantiles"]) == ({"break_km", "log_slope_change"} if count else set())
            for field, quantiles in full["parameter_quantiles"].items():
                key = f"{variant}__{field}"
                values = saved[key]
                assert values.shape == (last["chains"], last["draws_per_chain"], count)
                np.testing.assert_allclose(np.quantile(values, [.025, .5, .975], axis=(0, 1)).T, quantiles)
                keys.add(key)
        assert set(saved.files) == keys
    radii = np.array([0., 9.999999, 10., 10.000001, 20., 25.])
    np.testing.assert_allclose(warp(radii, np.array([10., 20.]), np.zeros(2)), radii)
    np.testing.assert_allclose(warp(radii, np.array([10., 20.]), np.log([2., .25])),
                               [0., 9.999999, 10., 10.000002, 30., 32.5])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference", type=Path, default=HOME/"directional/single-structure/report.json",
                        help="Published smooth reference on the same observations")
    args = parser.parse_args()
    publish(args.work, args.output)
    verify(args.output, args.reference)
