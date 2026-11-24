"""Replay matched all-phase maps and independently refit the two source interpretations."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .ilopango_field_review import REV, read, csv_bytes, digest
from .ilopango_total_airfall import FIELD, INPUT, RESULT, workbook
from .verify_deposit_maps import verify_global
from src.empirical_isopach.core.audit import audited_data
from src.empirical_isopach.models.nested.footprints.posterior import FootprintField, predictions
from src.empirical_isopach.models.nested.footprints.release import fit_part

SCENARIOS = ("without-osa", "without-historical-T2")
SITES = ("D19-CR", "D19-L08", "D19-L10", "D19-L12", "HS83-44", "SM1-P116")


def scenario_rows(scenario):
    rows = read(INPUT)
    if scenario == "without-osa":
        return [r for r in rows if r["EvidenceID"] != "D19-CR"]
    assert scenario == "without-historical-T2"
    stops = {r["id"]: r for r in read(FIELD / "hart-steen-mcintyre-stops.csv")}
    previous = {r["id"]: r for r in read(REV / "localities.csv")}
    sections = {r["id"]: r for r in read(REV / "workbook-sections.csv")}
    accepted = []
    for row in rows:
        code = row["EvidenceID"]
        if code in ("SM1-P45", "SM1-P89"):
            # Keep the independent contemporary workbook minimum at a grouped station.
            stop = stops[{"SM1-P45": "HS83-16", "SM1-P89": "HS83-48"}[code]]
            _, alt, _ = workbook(sections[code], previous[code])
            row["Limit_cm"] = str(max(float(alt), float(stop["sum_cm"]) - float(stop["T2_used_cm"])))
        elif code in stops and float(stops[code]["T2_used_cm"] or 0) > 0:
            remaining = float(stops[code]["sum_cm"]) - float(stops[code]["T2_used_cm"])
            if code == "HS83-44":
                remaining = 20.0  # Smaller documented reading of the upper ash at Laguna Seca.
            if remaining <= 0:
                continue
            row.update(Kind="lower_limit", Limit_cm=f"{remaining:g}", Lower_cm="", Upper_cm="")
        accepted.append(row)
    return accepted


def fit_scenarios(folder):
    for name in SCENARIOS:
        out = folder / "source-sensitivity" / name
        out.mkdir(parents=True, exist_ok=True)
        source = out / "input.csv"
        source.write_bytes(csv_bytes(scenario_rows(name)))
        data, meta = audited_data(source)
        receipt = fit_part(out, "full", data, None, meta)
        selected = (
            out / "full.npz"
            if json.loads((out / "full.json").read_text())["numerics"]["passed"]
            else out / "refinement/full.npz"
        )
        assert digest(selected) == receipt["posterior_sha256"]
        print(name, len(data.xy), receipt["numerics"]["passed"], flush=True)


def tabulate(folder, verify):
    data, meta = audited_data(INPUT)
    frame = read(INPUT)
    by_id = {r["EvidenceID"]: i for i, r in enumerate(frame, 2)}
    selected = [by_id[k] for k in SITES]
    positions = np.array([data.xy[np.flatnonzero(data.row_ids == i)[0]] for i in selected])
    rows = []
    for name in ("principal", *SCENARIOS):
        if name == "principal":
            packet = folder / "nested-isopachs/footprints"
            fit = json.loads((packet / "report.json").read_text())["fits"]["full"]
            path = packet / "full-selected.npz"
            count = len(frame)
        else:
            packet = folder / "source-sensitivity" / name
            source = packet / "input.csv"
            assert source.read_bytes() == csv_bytes(scenario_rows(name))
            rec = json.loads((packet / "full.json").read_text())
            assert rec["identity"]["input_sha256"] == digest(source)
            assert digest(packet / "full.npz") == rec["posterior_sha256"]
            if not rec["numerics"]["passed"]:
                packet = packet / "refinement"
                rec = json.loads((packet / "full.json").read_text())
            assert rec["identity"]["input_sha256"] == digest(source)
            fit = dict(posterior_sha256=rec["posterior_sha256"], passed=rec["numerics"]["passed"])
            path, count = packet / "full.npz", len(read(source))
        assert digest(path) == fit["posterior_sha256"]
        model = FootprintField.load(path)
        vol = np.quantile(model.volume(), [0.025, 0.5, 0.975])
        latent = np.exp(np.quantile(model.field(positions), [0.025, 0.5, 0.975], axis=0))
        for i, key in enumerate(SITES):
            rows.append(
                dict(
                    fit=name,
                    sites=count,
                    passed=fit["passed"],
                    site=key,
                    p025_km3=vol[0],
                    draw_median_km3=vol[1],
                    p975_km3=vol[2],
                    p025_cm=latent[0, i],
                    median_cm=latent[1, i],
                    p975_cm=latent[2, i],
                )
            )
    csv = folder / "source-sensitivity/comparison.csv"
    if verify:
        pd.testing.assert_frame_equal(pd.DataFrame(rows), pd.read_csv(csv), check_dtype=False, rtol=2e-7)
    else:
        pd.DataFrame(rows).to_csv(csv, index=False)


def verify_maps(folder):
    from src.empirical_isopach.models.directional.ensemble.members import splits

    data, meta = audited_data(INPUT)
    frame = read(INPUT)
    for family in ("directional", "spherical-mixture", "nested-isopachs"):
        verify_global(folder / family, family, data, meta)
    packet = folder / "nested-isopachs/footprints"
    report = json.loads((packet / "report.json").read_text())
    assert report["accepted_rows"] == len(frame)
    assert digest(packet / "support.json.gz") == report["support_sha256"]
    assert digest(packet / "heldout.csv.gz") == report["heldout_sha256"]
    held = pd.read_csv(packet / "heldout.csv.gz").set_index("row_id")
    assert held.index.is_unique and set(held.index) == set(data.row_ids)
    for i, tr, te in splits(data, meta["vent_lonlat"][1], "geographic"):
        key = f"fold-{i}"
        f = report["fits"][key]
        assert f["train"] == tr.row_ids.tolist() and f["test"] == te.row_ids.tolist()
        path = packet / (key + "-selected.npz")
        assert digest(path) == f["posterior_sha256"]
        for pred in predictions(FootprintField.load(path), te):
            np.testing.assert_allclose(held.loc[pred["row_id"], "log_score"], pred["log_score"], rtol=2e-7)
    for i, row in enumerate(frame, 2):
        if row["Kind"] == "interval":
            assert float(row["Lower_cm"]) < float(row["Upper_cm"])
            assert 0 < np.exp(held.loc[i, "log_score"]) <= 1


def field_validation(folder, verify):
    """Match each observation, buffered fold and predictive event across the three maps."""
    import gzip
    from src.empirical_isopach.models.directional.ensemble.members import splits
    from src.empirical_isopach.models.extent_release import draw_check, predictive_summary
    from src.empirical_isopach.models.extent_weights import ASSET, calibration

    data, meta = audited_data(INPUT)
    raw = read(INPUT)
    partitions = {
        i: (tr.row_ids.tolist(), te.row_ids.tolist())
        for i, tr, te in splits(data, meta["vent_lonlat"][1], "geographic")
    }
    results = []
    for family in ("directional", "spherical-mixture", "nested-isopachs"):
        root = folder / family
        nested = family == "nested-isopachs"
        packet = root / ("footprints" if nested else "extent")
        screen = json.loads(gzip.decompress((packet / "support.json.gz").read_bytes()))
        assert screen["status"] != "completed" or set(screen["policies"]) == {
            "strict",
            "standard",
            "permissive",
        }
        receipt = json.loads((packet / "report.json").read_text())
        assert receipt["input_sha256"] == digest(INPUT)
        assert receipt["heldout_sha256"] == digest(packet / "heldout.csv.gz")
        held = pd.read_csv(packet / "heldout.csv.gz")
        if nested:
            score_col = "log_score"
        else:
            cal = receipt["calibration"]
            actual = calibration(family, meta["input_sha256"], exclude_slug="ilopango-tbj")
            assert cal["donors"] == actual["donors"] and "ilopango-tbj" not in cal["donors"]
            np.testing.assert_allclose(cal["weights"], actual["weights"])
            assert digest(ASSET) == receipt["calibration_sha256"]
            assert all(digest(root / k) == v for k, v in receipt["parent_sha256"].items())
            assert digest(packet / "volumes.npz") == receipt["volume_draws_sha256"]
            with np.load(packet / "volumes.npz") as stored:
                q, _ = draw_check(stored["volume_km3"], stored["weights"], cal)
                if not receipt["exact_parent_identity"]:
                    np.testing.assert_allclose(q[[0, 2]], receipt["conditional_interval95_km3"], rtol=1e-5)
            rebuilt, metrics = predictive_summary(held, receipt["folds"], cal)
            np.testing.assert_allclose(held.principal_log_score, rebuilt.principal_log_score)
            assert (
                abs(
                    metrics["geographic"]["negative_log_score"]
                    - receipt["validation"]["geographic"]["negative_log_score"]
                )
                < 1e-8
            )
            for f in receipt["folds"]:
                if f["design"] == "geographic":
                    assert f["status"] == "estimated"
                    assert (f["train"], f["test"]) == partitions[f["fold"]]
            held = held.loc[held.design == "geographic"].copy()
            score_col = "principal_log_score"
            if family == "spherical-mixture":
                priors = json.loads((packet / "prior-sensitivity.json").read_text())
                assert len(priors) == 8 and set(priors) == set(
                    json.loads((root / "report.json").read_text())["fits"]
                )
        assert held.row_id.is_unique and set(held.row_id) == set(data.row_ids)
        for h in held.to_dict("records"):
            source = raw[int(h["row_id"]) - 2]
            fold = int(h["fold"])
            assert h["row_id"] in partitions[fold][1]
            kind = source["Kind"]
            assert bool(h["exact"]) == (kind == "measurement")
            score = float(h[score_col])
            event = np.exp(score) if kind != "measurement" else None
            if event is not None:
                assert 0 <= event <= 1 + 1e-10
            if kind == "interval" and not nested:
                cdf_diff = sum(
                    w * (h[s + "_upper_endpoint_cdf"] - h[s + "_cdf"])
                    for s, w in zip(cal["states"], cal["weights"])
                )
                np.testing.assert_allclose(cdf_diff, event, rtol=2e-6, atol=2e-10)
            results.append(
                dict(
                    method=family,
                    fold=fold,
                    source_id=source["EvidenceID"],
                    site=source["Site"],
                    kind=kind,
                    thickness_cm=source["Thickness"],
                    limit_cm=source["Limit_cm"],
                    lower_cm=source["Lower_cm"],
                    upper_cm=source["Upper_cm"],
                    fold_passed=h["fit_passed"],
                    log_predictive_score=score,
                    probability_of_reported_bound_or_range=event,
                )
            )
    table = pd.DataFrame(results).sort_values(["method", "fold", "source_id"]).reset_index(drop=True)
    filename = folder / "geographic-site-validation.csv"
    if verify:
        numeric = ["thickness_cm", "limit_cm", "lower_cm", "upper_cm"]
        for col in numeric:
            table[col] = pd.to_numeric(table[col], errors="raise")
        pd.testing.assert_frame_equal(table, pd.read_csv(filename), check_dtype=False, rtol=2e-7, atol=1e-10)
    else:
        table.to_csv(filename, index=False)


def verify_sampling(folder):
    """Validate every raw attempt, including repeated use and unsuccessful earlier fits."""
    index = read(folder / "sampling/attempts.csv")
    receipts = json.loads((folder / "sampling/diagnostics.json").read_text())
    lookup = {(r["family"], r["run"], r["attempt"]): r for r in receipts}
    assert len(lookup) == len(index)
    seen = {}
    for row in index:
        detail = lookup[(row["family"], row["run"], int(row["attempt"]))]["receipt"]
        path = folder / row["posterior_path"]
        assert path.stat().st_size == int(row["bytes"])
        if path not in seen:
            seen[path] = digest(path)
        assert seen[path] == row["sha256"] == detail["posterior_sha256"]
        assert str(detail["diagnostics"]["passed"]) == row["raw_sampler_passed"]
    assert {r["family"] for r in index} == {"directional", "spherical-mixture"}


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--folder", type=Path, default=RESULT)
    p.add_argument("--fit-sources", action="store_true")
    p.add_argument("--write-comparison", action="store_true")
    args = p.parse_args()
    if args.fit_sources:
        fit_scenarios(args.folder)
    elif args.write_comparison:
        tabulate(args.folder, False)
        field_validation(args.folder, False)
    else:
        from .ilopango_total_map import draw

        verify_maps(args.folder)
        verify_sampling(args.folder)
        draw(args.folder, verify=True)
        field_validation(args.folder, True)
        tabulate(args.folder, True)
