"""Replay directional fields, observed events, deleted surveys and coherent-volume intervals."""

import gzip
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .directional_scale_maps import ARCHIVE
from .publication import result_home
from src.empirical_isopach.core.audit import audited_data
from src.empirical_isopach.core.catalog import datasets
from src.empirical_isopach.models.directional.ensemble.members import splits
from src.empirical_isopach.models.directional.multiscale.fit import KEEP_MASS
from src.empirical_isopach.models.directional.multiscale.geometry import LEGACY_SCHEMA as SCHEMA
from src.empirical_isopach.models.directional.multiscale.posterior import DirectionalField, Marginals
from src.empirical_isopach.models.directional.multiscale.release import archive_probes, digest
from src.empirical_isopach.visualization.deposit import read_json
from src.empirical_isopach.visualization.deposit_integrals import within_cells
from .directional_scale_maps import read_grid

STUDY = Path("comparisons/directional/multiscale")


def matched_events(distribution, data, rows):
    assert rows.row_id.tolist() == data.row_ids.tolist()
    np.testing.assert_allclose(distribution.score(data), rows.principal_log_score, rtol=2e-7, atol=1e-8)
    np.testing.assert_allclose(distribution.cdf(data.log_value), rows.principal_cdf, atol=2e-7)
    bounded = data.censored
    np.testing.assert_allclose(
        np.exp(rows.principal_log_score[bounded]), rows.event_probability[bounded], atol=1e-12
    )
    assert np.array_equal(data.interval, rows.evidence == "interval")
    if np.any(data.interval):
        cdf = distribution.cdf(np.where(data.interval, data.interval_log_upper, data.log_value))
        np.testing.assert_allclose(cdf[data.interval], rows.upper_endpoint_cdf[data.interval], atol=2e-7)


def verify_case(record, pair, *, home=None, study=STUDY, previous=ARCHIVE,
                protocol=SCHEMA, state_count=576):
    home = Path(home) if home is not None else result_home(record) / "directional"
    folder = home / "multiscale"
    data, meta = audited_data(record["path"])
    data = data.canonical()
    slug = record["slug"]
    report = read_json(folder / "report.json")
    assert report["protocol"] == protocol
    assert report["input_sha256"] == pair.input_sha256 == meta["input_sha256"] == digest(record["path"])
    assert len(data.xy) == report["accepted_rows"] == pair.accepted_rows
    for filename, checksum in report["artifacts_sha256"].items():
        assert digest(folder / filename) == checksum
    with gzip.open(folder / "fit-diagnostics.json.gz", "rt") as stream:
        histories = json.load(stream)
    field = DirectionalField.load(folder / "posterior.npz")
    probability = read_json(home / "exceedance.json")
    dep = read_json(home / "deposit.json")
    assert probability["input_sha256"] == meta["input_sha256"]
    assert probability["median_grid_sha256"] == digest(home / "median-grid.npz")
    assert probability["posterior_sha256"] == digest(folder / "posterior.npz")
    assert probability["thresholds_cm"] == [1.0, max(2.0, dep["inner_contour"]["level_cm"] / 2)]
    assert probability.pop("figure_sha256") == digest(home / "exceedance.png")
    from PIL import Image

    with Image.open(home / "exceedance.png") as image:
        assert json.loads(image.info["Description"]) == probability
    from src.empirical_isopach.visualization.deposit_maps import regular_axes

    for box, limits in zip(
        probability["windows_km"], probability["probability_bounds_by_window"], strict=True
    ):
        x, y = regular_axes(np.array(box), 177)
        points = np.column_stack([v.ravel() for v in np.meshgrid(x, y)])
        points = points[field.domain.rho(points) <= field.domain.rho_max]
        # Verify extremal probabilities on the complete displayed grid; arbitrary geographic slices replay identically.
        bounds = np.tile([1.0, 0.0], (2, 1))
        for start in range(0, len(points), 512):
            dist = field.distribution(points[start : start + 512])
            for i, level in enumerate(probability["thresholds_cm"]):
                value = 1.0 - dist.cdf(np.log(level))
                bounds[i] = [min(bounds[i, 0], value.min()), max(bounds[i, 1], value.max())]
        np.testing.assert_allclose(bounds, limits, rtol=1e-5, atol=2e-7)
    full = pd.read_csv(folder / "measured-sites.csv.gz")
    matched_events(field.distribution(data.xy, observation=True, log_sd=data.log_sd), data, full)
    np.testing.assert_allclose(
        field.central(data.xy) * np.log(10.0), full.median_log_cm, rtol=2e-7, atol=2e-7
    )
    np.testing.assert_allclose(full.median_absolute_log_error.mean(), pair.new_site_mae_log_cm)
    archive = read_json(folder / "archive-probes.json")
    predicted = archive_probes(record["path"], None, data, meta, field)
    assert predicted["inventory_sha256"] == archive["inventory_sha256"]
    assert [r["id"] for r in archive["sites"]] == [r["id"] for r in predicted["sites"]]
    for row, other in zip(archive["sites"], predicted["sites"], strict=True):
        assert row["coincident_row_ids"] == other["coincident_row_ids"]
        np.testing.assert_allclose(row["latent_cm"], other["latent_cm"], rtol=5e-6)
    for key, v in predicted["scenarios"].items():
        np.testing.assert_allclose(
            v["probability_below"], archive["scenarios"][key]["probability_below"], atol=2e-7
        )
    for receipt in histories.values():
        if receipt["status"] == "empty":
            continue
        states = receipt["states"]
        assert len(states) == state_count
        mass = np.array([s["probability"] for s in states])
        np.testing.assert_allclose(mass.sum(), 1)
        retained = sum(s["probability"] for s in states if s["retained"])
        np.testing.assert_allclose(receipt["retained_mass"], retained)
        assert retained >= KEEP_MASS - 1e-10
        assert receipt["shared_fraction"] == 0.5
        passed = (
            receipt["max_pooled_cdf_error"] < 0.035
            and receipt["max_independent_cdf_difference"] < 0.06
            and receipt["weighted_scramble_difference"] < 0.10
            and receipt["weighted_relative_ess"] > 0.14
            and receipt["proposal_residual"] < 1e-4
        )
        if protocol != SCHEMA:
            passed = passed and all(s.get("minimum_corrected_covariance_ratio", 1.) > 0
                                    for s in states if s["retained"])
        assert receipt["passed"] == passed
        assert receipt["attempts"][-1]["passed"] == passed
    held = pd.read_csv(folder / "heldout.csv.gz")
    with np.load(study / "fold-marginals" / (slug + ".npz"), allow_pickle=False) as packet:
        assert str(packet["input_sha256"]) == meta["input_sha256"]
        for design in ("geographic", "quadrants"):
            rows = held[held.design == design]
            assert not rows.row_id.duplicated().any() and set(rows.row_id) == set(data.row_ids)
            for fold, train, test in splits(data, meta["vent_lonlat"][1], design):
                key = f"{design}-{fold}"
                f = histories[key]
                assert test.row_ids.tolist() == f["test_row_ids"]
                if train is None or not len(test.xy):
                    assert f["status"] == "empty" and key + "_weight" not in packet
                    continue
                assert train.row_ids.tolist() == f["train_row_ids"] and not set(test.row_ids) & set(
                    train.row_ids
                )
                assert str(packet[key + "_posterior_sha256"]) == f["packet_sha256"]
                np.testing.assert_array_equal(packet[key + "_row_id"], test.row_ids)
                marginal = Marginals(*(packet[key + "_" + k] for k in ("weight", "mean", "variance")))
                block = rows[rows.fold == fold]
                matched_events(marginal, test, block)
                assert (block.fit_passed == f["passed"]).all()
            value = -rows.groupby("fold").principal_log_score.mean().mean()
            np.testing.assert_allclose(value, report["validation"][design]["negative_log_score"])
            np.testing.assert_allclose(value, pair["new_" + design + "_score"])
    with np.load(folder / "volumes.npz", allow_pickle=False) as draws:
        mass = draws["volume"]
        np.testing.assert_allclose(draws["sector"].sum(1), mass, rtol=3e-12)
        fraction = draws["allocation"] / mass[:, None]
        assert (fraction >= 0).all() and (fraction <= 1 + 2e-12).all()
        np.testing.assert_allclose(np.quantile(mass, [0.025, 0.975]), report["conditional_interval95_km3"])
        np.testing.assert_allclose(
            np.quantile(fraction, [0.025, 0.5, 0.975], axis=0).T, report["allocation_fraction_quantiles"]
        )
    last = report["independent_draw_integration"][-1]
    assert (
        report["integration_passed"]
        == last["passed"]
        == (
            last["spatial_relative_difference"] < 0.04
            and last["population_relative_difference"] < 0.10
            and last["refinement_relative_difference"] < 0.08
        )
    )
    with gzip.open(folder / "support.json.gz", "rt") as stream:
        support = json.load(stream)
    assert support["input_sha256"] == meta["input_sha256"]
    assert set(support["policies"]) == {"strict", "standard", "permissive"}
    assert sorted(report["support_policies_with_band"]) == sorted(
        k for k, v in support["policies"].items() if v["selected_band"] is not None
    )
    for row in support["levels"] + support["refined_levels"]:
        assert all(v["eligible"] == (not v["reasons"]) for v in row["policies"].values())
    for where, value in ((home, pair.median_map_km3), (previous / slug, pair.previous_km3)):
        dep = read_json(where / "deposit.json")
        assert (
            dep["input_sha256"] == meta["input_sha256"]
            and digest(where / "median-grid.npz") == dep["median_grid_sha256"]
        )
        calculated, _ = within_cells(read_grid(where / "median-grid.npz"), dep["levels_cm"])
        np.testing.assert_allclose(calculated, value, rtol=2e-7)


def main():
    from src.empirical_isopach.models.directional.multiscale import engine_sha256

    from .publication import reported_edition
    records = [reported_edition(record) for record in datasets()]
    provenance = read_json(STUDY / "provenance.json")
    original_reader = Path("archive/2026-11-25-lower-limit-policy/observations.py")
    assert digest(original_reader) == "37dc55a1ca9af46998c21dbfe7d7b4f2f23304a9ec37b39ac7ab6956eb1b1424"
    assert provenance["protocol"] == SCHEMA
    original_engine = Path("archive/2026-11-25-directional-v5")
    assert provenance["numerical_engine_sha256"] == engine_sha256(
        observation_source=original_reader, legacy_source=original_engine)
    cases = pd.read_csv(STUDY / "paired-results.csv").set_index("slug")
    geometry = pd.read_csv(STUDY / "geometry.csv.gz")
    assert set(cases.index) == {r["slug"] for r in records} == set(geometry.slug)
    for record in records:
        slug = record["slug"]
        home = result_home(record) / "directional"
        identifiers = provenance["cases"][slug]
        if digest(home / "multiscale/report.json") != identifiers["fit_report_sha256"]:
            home = original_engine / "current-cases" / slug
        for key, path in (
            ("input_sha256", record["path"]),
            ("fit_report_sha256", home / "multiscale/report.json"),
            ("figure_sha256", home / "deposit.png"),
            ("map_receipt_sha256", home / "deposit.json"),
            ("fold_marginals_sha256", STUDY / "fold-marginals" / (slug + ".npz")),
        ):
            assert digest(path) == identifiers[key]
        verify_case(record, cases.loc[slug], home=home)
    assert set(geometry.cartesian_long_side) == {193, 385} and set(geometry.method) == {
        "previous",
        "multiscale",
    }
    sens = pd.read_csv(STUDY / "resolution-sensitivity.csv")
    assert len(sens) == 3 * len(cases) and set(sens.shared_fraction) == {0.0, 0.5, 1.0}
    assert set(sens.input_sha256) == set(cases.input_sha256)
    sims = pd.read_csv(STUDY / "known-deposits.csv.gz")
    hidden = sims[sims.design == "invisible_sector"]
    assert (hidden.groupby(["seed", "scramble", "level_cm"]).median_map_km3.nunique() == 1).all()
    assert set(hidden.truth) == {"specified", "hidden"}
    totals = pd.read_csv(STUDY / "known-volume-intervals.csv")
    assert set(totals.input_sha256) == set(sims.input_sha256)
    assert np.array_equal(
        totals.covers_truth, totals.truth_volume_km3.between(totals.lower95_km3, totals.upper95_km3)
    )
    histories = read_json(STUDY / "known-volume-refinements.json")
    for row in totals.itertuples():
        last = histories[f"{row.design}-{row.seed}"][-1]
        assert row.integration_passed == last["passed"]
        np.testing.assert_allclose(
            [row.lower95_km3, row.upper95_km3], np.array(last["conditional_quantiles_km3"])[[0, 2]]
        )
    print(
        f"Verified {len(records)} directional fields, fitted and deleted site distributions, map volumes and support."
    )


if __name__ == "__main__":
    main()
