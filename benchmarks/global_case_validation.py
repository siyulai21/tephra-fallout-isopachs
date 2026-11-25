"""Compare published global fits against one audited input and its buffered holdouts."""
import argparse
import csv
import json
import shutil
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

from .publication import published_methods, reported_edition, result_home
from src.empirical_isopach.core.audit import audited_data
from src.empirical_isopach.core.catalog import datasets
from src.empirical_isopach.visualization.deposit_field import digest


METHODS = ("directional", "spherical-mixture", "nested-isopachs")


def read_rows(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def field_validation(folder, input_path, slug, *, verify=True):
    """Match each observation, buffered fold and predictive event across the three maps."""
    import gzip
    from src.empirical_isopach.models.directional.ensemble.members import splits
    from src.empirical_isopach.models.extent_release import draw_check, predictive_summary
    from src.empirical_isopach.models.extent_weights import ASSET, calibration

    data, meta = audited_data(input_path)
    raw = read_rows(input_path)
    partitions = {
        i: (tr.row_ids.tolist(), te.row_ids.tolist())
        for i, tr, te in splits(data, meta["vent_lonlat"][1], "geographic")
    }
    regional_path = folder / "report.json"
    if regional_path.is_file():
        regional = json.loads(regional_path.read_text())
        assert regional["input"]["input_sha256"] == meta["input_sha256"]
        for method in regional["validation"]["methods"].values():
            recorded = {int(f["fold"]): (f["training_row_ids"], f["withheld_row_ids"])
                        for f in method["selections"]}
            assert recorded == partitions
    results = []
    for family in METHODS:
        root = folder / family
        nested = family == "nested-isopachs"
        from src.empirical_isopach.visualization.deposit_field import review_path
        packet = review_path(root, family).parent
        multiscale = packet.name == "multiscale"
        screen = json.loads(gzip.decompress((packet / "support.json.gz").read_bytes()))
        assert (not multiscale and screen["status"] != "completed") or set(screen["policies"]) == {
            "strict",
            "standard",
            "permissive",
        }
        receipt = json.loads((packet / "report.json").read_text())
        assert receipt["input_sha256"] == digest(input_path)
        heldout_hash = receipt["artifacts_sha256"]["heldout.csv.gz"] if multiscale else receipt["heldout_sha256"]
        assert heldout_hash == digest(packet / "heldout.csv.gz")
        held = pd.read_csv(packet / "heldout.csv.gz")
        if nested:
            for fold, rows in partitions.items():
                fitted = receipt["fits"][f"fold-{fold}"]
                assert (fitted["train"], fitted["test"]) == rows
            score_col = "log_score"
        elif multiscale:
            for f in receipt["folds"]:
                if f["design"] == "geographic":
                    assert (f["train_row_ids"], f["test_row_ids"]) == partitions[f["fold"]]
            held = held.loc[held.design == "geographic"].copy()
            held["exact"] = held.evidence == "exact"
            score_col = "principal_log_score"
            np.testing.assert_allclose(-held.groupby("fold")[score_col].mean().mean(),
                                       receipt["validation"]["geographic"]["negative_log_score"])
        else:
            cal = receipt["calibration"]
            actual = calibration(family, meta["input_sha256"], exclude_slug=slug)
            assert cal["donors"] == actual["donors"] and slug not in cal["donors"]
            # Use the shared extent benchmark tolerance for the fitted donor weights.
            np.testing.assert_allclose(cal["weights"], actual["weights"], atol=1e-5)
            assert digest(ASSET) == receipt["calibration_sha256"]
            assert all(digest(root / k) == v for k, v in receipt["parent_sha256"].items())
            assert digest(packet / "volumes.npz") == receipt["volume_draws_sha256"]
            with np.load(packet / "volumes.npz") as stored:
                q, _ = draw_check(stored["volume_km3"], stored["weights"], cal)
                if not receipt["exact_parent_identity"]:
                    np.testing.assert_allclose(q[[0, 2]], receipt["conditional_interval95_km3"], rtol=1e-5)
            rebuilt, metrics = predictive_summary(held, receipt["folds"], cal)
            # A probability indistinguishable from one has a log score near zero.
            columns = ["principal_log_score", "principal_cdf"]
            np.testing.assert_allclose(held[columns], rebuilt[columns],
                                       rtol=1e-11, atol=8 * np.finfo(float).eps)
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
            kind = (source.get("Kind") or "measurement").strip().lower()
            assert bool(h["exact"]) == (kind == "measurement")
            score = float(h[score_col])
            event = np.exp(score) if kind != "measurement" else None
            if event is not None:
                assert 0 <= event <= 1 + 1e-10
            if kind == "interval" and multiscale:
                np.testing.assert_allclose(h["upper_endpoint_cdf"] - h["principal_cdf"], event,
                                           rtol=2e-6, atol=2e-10)
            if kind == "interval" and not (nested or multiscale):
                cdf_diff = sum(
                    w * (h[s + "_upper_endpoint_cdf"] - h[s + "_cdf"])
                    for s, w in zip(cal["states"], cal["weights"])
                )
                np.testing.assert_allclose(cdf_diff, event, rtol=2e-6, atol=2e-10)
            results.append(
                dict(
                    method=family,
                    fold=fold,
                    source_id=source.get("EvidenceID") or f"row-{int(h['row_id'])}",
                    site=source.get("Site") or f"{source['Lat']}, {source['Lon']}",
                    kind=kind,
                    thickness_cm=source["Thickness"],
                    limit_cm=source.get("Limit_cm", ""),
                    lower_cm=source.get("Lower_cm", ""),
                    upper_cm=source.get("Upper_cm", ""),
                    fold_passed=h["fit_passed"],
                    log_predictive_score=score,
                    probability_of_reported_bound_or_range=event,
                )
            )
    table = pd.DataFrame(results).sort_values(["method", "fold", "source_id"]).reset_index(drop=True)
    filename = folder / "geographic-site-validation.csv"
    if verify:
        numeric = ["thickness_cm", "limit_cm", "lower_cm", "upper_cm", "probability_of_reported_bound_or_range"]
        for col in numeric:
            table[col] = pd.to_numeric(table[col], errors="raise")
        pd.testing.assert_frame_equal(table, pd.read_csv(filename), check_dtype=False, rtol=2e-7, atol=1e-10)
    else:
        table.to_csv(filename, index=False)


def collect_sampling(production, folder):
    """Retain each attempted posterior once and keep its design and diagnostics."""
    with TemporaryDirectory(dir=folder) as workspace:
        staged = Path(workspace) / "sampling"
        staged.mkdir()
        _write_sampling(production, folder, staged)
        target = folder / "sampling"
        if target.exists():
            verify_sampling(folder)
            def identities(path):
                return {(r["family"], r["run"], r["attempt"]): r["sha256"]
                        for r in read_rows(path / "attempts.csv")}
            if identities(staged) != identities(target):
                raise ValueError("The existing sampling release belongs to different attempts")
        else:
            staged.rename(target)


def _write_sampling(production, folder, target, families=("directional", "spherical-mixture")):
    work_folders = {"directional": "work", "spherical-mixture": "posteriors"}
    sources = {family: work_folders[family] for family in families}
    canonical = {}
    for family in sources:
        for path in sorted((folder / family).rglob("*.npz")):
            canonical.setdefault(digest(path), path.relative_to(folder).as_posix())
    rows, details = [], []
    for family, subdir in sources.items():
        work = production / family / subdir
        attempts = sorted(work.rglob("attempt-*.json"))
        if not attempts:
            raise ValueError(f"No sampler attempts in {work}")
        for file in attempts:
            receipt = json.loads(file.read_text())
            packet = file.with_suffix(".npz")
            checksum = digest(packet)
            if checksum != receipt["posterior_sha256"]:
                raise ValueError(f"Posterior differs from its sampling receipt at {packet}")
            run = file.parent.relative_to(work).as_posix()
            number = int(file.stem.split("-")[-1])
            if checksum not in canonical:
                dest = target / family / run / packet.name
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(packet, dest)
                canonical[checksum] = (Path("sampling") / family / run / packet.name).as_posix()
            identity = dict(family=family, run=run, attempt=number)
            rows.append(dict(**identity, raw_sampler_passed=receipt["diagnostics"]["passed"],
                             sha256=checksum, bytes=packet.stat().st_size, posterior_path=canonical[checksum]))
            design = (json.loads(file.with_name("design.json").read_text()) if family == "directional"
                      else receipt["signature"])
            details.append(dict(**identity, design=design, receipt=receipt))
    with (target / "attempts.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    (target / "diagnostics.json").write_text(json.dumps(details, indent=2) + "\n")


def verify_sampling(folder):
    """Validate every raw attempt, including repeated use and unsuccessful earlier fits."""
    index = read_rows(folder / "sampling/attempts.csv")
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
    directional = folder / "directional"
    last = {r["run"]: r["sha256"] for r in sorted(index, key=lambda r: int(r["attempt"]))
            if r["family"] == "directional"}
    members = json.loads((directional / "azimuth/report.json").read_text())["members"]
    for name, fit in members.items():
        assert last[f"ensemble/fits/{name}"] == fit["posterior_sha256"]
    folds = json.loads((directional / "extent/report.json").read_text())["folds"]
    for fold in folds:
        if fold["status"] == "estimated":
            for name, checksum in fold["source_sha256"].items():
                run = f"ensemble/{fold['design']}/fold-{fold['fold']}/{name}"
                assert last[run] == checksum
    sphere = json.loads((folder / "spherical-mixture/report.json").read_text())
    full_and_folds = [*sphere["fits"].values(), *(v["fit"] for v in sphere["validation"]["fits"])]
    released = {row["sha256"] for row in index if row["family"] == "spherical-mixture"}
    assert all(fit["posterior_sha256"] in released for fit in full_and_folds)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    cases = {r["slug"]: r for r in datasets()}
    parser.add_argument("--case", required=True, choices=sorted(cases))
    parser.add_argument("--reported-edition", action="store_true", help="Replay the previous lower-censored source edition")
    parser.add_argument("--folder", type=Path, help="Override the published results directory")
    parser.add_argument("--write-site-table", action="store_true")
    parser.add_argument("--collect-attempts", type=Path, metavar="PRODUCTION",
                        help="Collect verified directional/work and spherical-mixture/posteriors into the report")
    args = parser.parse_args()
    case = reported_edition(cases[args.case]) if args.reported_edition else cases[args.case]
    if not set(METHODS).issubset(published_methods(case)):
        parser.error("All three global families must be published on the same input")
    folder = args.folder or result_home(case)
    if args.collect_attempts:
        collect_sampling(args.collect_attempts, folder)
    field_validation(folder, case["path"], case["slug"], verify=not args.write_site_table)
    if not args.write_site_table:
        verify_sampling(folder)
