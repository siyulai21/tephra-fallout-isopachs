"""Assemble reproducible directional maps, evidence checks and deletion predictions."""

import gzip
from hashlib import sha256
import json
from pathlib import Path
import shutil

import numpy as np

from ....core.audit import audited_data
from ....core.geometry import survey_bounds
from ....contour_support import audit_contours
from ..archive_readings import load_inventory, positions
from .geometry import SCHEMA
from .integrals import integrate
from .posterior import DirectionalField
from .validation import read_rows, run_partition, summaries, write_json, write_rows


def digest(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def packed(path, record):
    path.write_bytes(
        gzip.compress((json.dumps(record, separators=(",", ":"), allow_nan=False) + "\n").encode(), mtime=0)
    )


def archive_probes(path, archives, data, meta, model):
    inventory, source = load_inventory(path, archives)
    result = dict(inventory_sha256=digest(source) if source else None, sites=[], scenarios={})
    sites = inventory["sites"]
    if not sites:
        return result
    xy = positions(sites, meta["vent_lonlat"])
    latent = model.distribution(xy)
    quantiles = np.exp([latent.quantile(p) for p in (0.025, 0.5, 0.975)]).T
    for row, p, q in zip(sites, xy, quantiles, strict=True):
        result["sites"].append(
            dict(
                id=row["id"],
                site=row["name"],
                reading=row["reading"],
                latent_cm=q.tolist(),
                coincident_row_ids=data.row_ids[np.linalg.norm(data.xy - p, axis=1) < 1e-6].tolist(),
            )
        )
    for scenario in inventory["scenarios"]:
        ix = [
            next(i for i, s in enumerate(sites) if s["id"] == key) for key in scenario["effective_limit_cm"]
        ]
        probabilities = model.distribution(xy[ix], observation=True).cdf(
            np.log(list(scenario["effective_limit_cm"].values()))
        )
        result["scenarios"][scenario["id"]] = dict(
            assumption=scenario["assumption"],
            probability_below=probabilities.tolist(),
            effective_limit_cm=scenario["effective_limit_cm"],
        )
    return result


def assemble(path, folder, name, *, work=None, archives=None, render=True):
    data, meta = audited_data(path)
    latitude = meta["vent_lonlat"][1]
    dest = Path(folder) / "multiscale"
    dest.mkdir(parents=True, exist_ok=True)
    work = Path(work) if work else Path(folder) / "work" / "multiscale"
    scores, folds, checks = [], [], []
    histories = {}
    for kind in ("full", "geographic", "quadrants"):
        for fold in [None] if kind == "full" else range(4):
            run_partition(data, latitude, work, kind, fold, resume=True)
            key = "full" if fold is None else f"{kind}-{fold}"
            home = work / key
            receipt = json.loads((home / "fit.json").read_text())
            simple = {k: v for k, v in receipt.items() if k != "states"}
            histories[key] = receipt
            if kind != "full":
                folds.append(simple)
            if receipt["status"] != "estimated":
                continue
            if digest(home / "field.npz") != receipt["packet_sha256"]:
                raise ValueError(f"Changed directional packet for {key}")
            rows = read_rows(home / "scores.csv.gz")
            if kind == "full":
                shutil.copyfile(home / "field.npz", dest / "posterior.npz")
                write_rows(dest / "measured-sites.csv.gz", rows)
                full = DirectionalField.load(dest / "posterior.npz")
                fit = simple
                measured = summaries(rows, [dict(fold="", passed=fit["passed"], status="estimated")])
            else:
                scores.extend(rows)
                if kind == "geographic" and receipt["passed"]:
                    checks.append(DirectionalField.load(home / "field.npz"))
    write_rows(dest / "heldout.csv.gz", scores)
    packed(dest / "fit-diagnostics.json.gz", histories)
    audit, *_ = audit_contours(full, data, survey_bounds(data.xy), checks)
    audit.update(
        usable_geographic_deletions=len(checks),
        input_sha256=meta["input_sha256"],
        median_surface="Pointwise marginal median of the fitted directional ensemble",
    )
    packed(dest / "support.json.gz", audit)
    write_json(dest / "archive-probes.json", archive_probes(path, archives, data, meta, full))
    volume = integrate(full, data, dest / "volumes.npz")
    validation = {
        k: summaries([r for r in scores if r["design"] == k], [f for f in folds if f["design"] == k])
        for k in ("geographic", "quadrants")
    }
    report = dict(
        protocol=SCHEMA,
        dataset=name,
        input_sha256=meta["input_sha256"],
        accepted_rows=len(data.xy),
        fit=fit,
        measured=measured,
        folds=folds,
        validation=validation,
        support_policies_with_band=[
            k for k, v in audit["policies"].items() if v["selected_band"] is not None
        ],
        draw_checks_passed=bool(fit["passed"] and volume["integration_passed"]),
        artifacts_sha256={
            p.name: digest(p) for p in sorted(dest.iterdir()) if p.is_file() and p.name != "report.json"
        },
        **volume,
    )
    write_json(dest / "report.json", report)
    if render:
        from ....visualization.deposit import render_deposit

        render_deposit(folder, "directional", data, meta, name)
    return report
