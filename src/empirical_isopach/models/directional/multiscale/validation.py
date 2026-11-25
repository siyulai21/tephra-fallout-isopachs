"""Refit geography and every model weight inside each deletion and score every observed event."""

import csv
import gzip
from hashlib import sha256
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import os

import numpy as np

from ..ensemble.members import splits
from . import engine_sha256
from .fit import fit, SEED
from .geometry import SCHEMA
from .posterior import DirectionalField


def write_json(path, obj):
    path.write_text(json.dumps(obj, indent=2, allow_nan=False) + "\n")


def write_rows(path, rows):
    # Fix gzip header metadata so reruns produce the same artifact.
    import io

    with path.open("wb") as raw, gzip.GzipFile(fileobj=raw, mode="wb", filename="", mtime=0) as compressed:
        with io.TextIOWrapper(compressed, encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)


def predictions(field, data, design, fold, passed):
    marginal = field.distribution(data.xy)
    observed = field.distribution(data.xy, observation=True, log_sd=data.log_sd)
    median = marginal.quantile(0.5)
    scores = observed.score(data)
    cdf = observed.cdf(data.log_value)
    upper = observed.cdf(np.where(data.interval, data.interval_log_upper, data.log_value))
    return [
        dict(
            design=design,
            fold=fold,
            fit_passed=passed,
            row_id=int(r),
            evidence=(
                "interval"
                if data.interval[j]
                else "lower" if data.lower_censored[j] else "upper" if data.censored[j] else "exact"
            ),
            principal_log_score=float(scores[j]),
            principal_cdf=float(cdf[j]),
            upper_endpoint_cdf=float(upper[j]) if data.interval[j] else "",
            observed_log_cm=float(data.log_value[j]),
            median_log_cm=float(median[j]),
            median_absolute_log_error=(
                float(abs(median[j] - data.log_value[j])) if not data.censored[j] else ""
            ),
            event_probability=float(np.exp(scores[j])) if data.censored[j] else "",
        )
        for j, r in enumerate(data.row_ids)
    ]


def observations_digest(data):
    fields = (
        data.row_ids,
        data.xy,
        data.log_value,
        data.censored,
        data.lower_censored,
        data.interval_log_upper,
        data.log_sd,
    )
    return sha256(b"".join(np.ascontiguousarray(v).tobytes() for v in fields)).hexdigest()


def run_partition(data, latitude, work, kind, fold=None, *, resume=False, power=8):
    work = Path(work)
    work.mkdir(parents=True, exist_ok=True)
    data = data.canonical()
    if kind == "full":
        train = test = data
    else:
        _, train, test = next(s for s in splits(data, latitude, kind) if s[0] == fold)
    path = work / ("full" if kind == "full" else f"{kind}-{fold}")
    path.mkdir(exist_ok=True)
    record = path / "fit.json"
    if train is None or not len(test.xy) or not len(train.xy):
        write_json(
            record,
            dict(protocol=SCHEMA, status="empty", design=kind, fold=fold, test_row_ids=test.row_ids.tolist()),
        )
        return
    identity = dict(
        protocol=SCHEMA,
        engine_sha256=engine_sha256(),
        training_identity=observations_digest(train),
        test_identity=observations_digest(test),
        latitude=float(latitude),
        seed=SEED,
        design=kind,
        fold=fold,
        power=power,
    )
    attempts = []
    if resume and record.is_file():
        prev = json.loads(record.read_text())
        if (
            all(prev.get(k) == v for k, v in identity.items())
            and prev.get("refinement_complete")
            and all(
                (path / file).is_file() and sha256((path / file).read_bytes()).hexdigest() == prev.get(key)
                for file, key in (("field.npz", "packet_sha256"), ("scores.csv.gz", "scores_sha256"))
            )
        ):
            return
    for depth in (power, min(12, power + 2), min(12, power + 3)):
        if any(a.get("integration_power", a.get("power")) == depth for a in attempts):
            continue
        result, diagnostics = fit(train, latitude, seed=SEED, power=depth)
        attempts.append(
            dict(integration_power=depth, **{k: v for k, v in diagnostics.items() if k != "states"})
        )
        if diagnostics["passed"]:
            break
    with TemporaryDirectory(dir=path) as staging:
        stage = Path(staging)
        result.save(stage / "field.npz")
        rows = predictions(result, test, kind, fold, diagnostics["passed"])
        write_rows(stage / "scores.csv.gz", rows)
        write_json(
            stage / "fit.json",
            dict(
                **identity,
                status="estimated",
                train_row_ids=train.row_ids.tolist(),
                test_row_ids=test.row_ids.tolist(),
                packet_sha256=sha256((stage / "field.npz").read_bytes()).hexdigest(),
                scores_sha256=sha256((stage / "scores.csv.gz").read_bytes()).hexdigest(),
                attempts=attempts,
                refinement_complete=True,
                **diagnostics,
            ),
        )
        for file in ("field.npz", "scores.csv.gz", "fit.json"):
            os.replace(stage / file, path / file)


def read_rows(path):
    with gzip.open(path, "rt", newline="") as stream:
        return list(csv.DictReader(stream))


def summaries(rows, folds):
    def score(group):
        if not group:
            return None
        return -float(np.mean([float(row["principal_log_score"]) for row in group]))

    exact = [r for r in rows if r["evidence"] == "exact"]
    scores = [
        score([r for r in rows if str(r["fold"]) == str(f["fold"])])
        for f in folds
        if f["status"] == "estimated"
    ]
    coverage = float(np.mean([0.025 <= float(r["principal_cdf"]) <= 0.975 for r in exact])) if exact else None
    return dict(
        negative_log_score=float(np.mean(scores)) if scores else None,
        coverage95_exact=coverage,
        all_folds_passed=all(f["status"] == "estimated" and f["passed"] for f in folds),
        passed_folds=sum(f.get("passed", False) for f in folds),
        scored_rows=len(rows),
        fold_scores=scores,
        mean_absolute_log_error_exact=(
            float(np.mean([float(r["median_absolute_log_error"]) for r in exact])) if exact else None
        ),
        score_by_event={
            k: dict(rows=len(g), negative_log_score=score(g))
            for k in ("exact", "lower", "upper", "interval")
            if (g := [r for r in rows if r["evidence"] == k])
        },
    )
