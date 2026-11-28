"""Fit, validate and publish a wrapped angular covariance reconstruction."""

from contextlib import contextmanager
from tempfile import TemporaryDirectory
import os
import gzip
from hashlib import sha256
import json
from pathlib import Path
import numpy as np
import pandas as pd
from ...core.audit import audited_data
from ...core.fitting import model_input
from ...core.geometry import survey_bounds
from ...contour_support import audit_contours
from ...visualization.deposit import render_deposit
from ..directional.archive_readings import load_inventory, positions
from .fitting import fit, source_for, SourceBeds
from .integrate import integrate, joint_volumes
from .packet import FAMILY, PROTOCOL, write
from .validation import partitions, ContourView


def digest(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def engine_sha256():
    root = Path(__file__).parent
    empirical = root.parents[1]
    paths = [
        *root.glob("*.py"),
        empirical / "core/likelihood.py",
        empirical / "contour_support.py",
        empirical / "models/directional/multiscale/conditioning.py",
        empirical / "models/directional/multiscale/posterior.py",
        empirical.parent / "probabilistic_isopach/global_volume.py",
    ]
    return sha256(
        b"".join(p.relative_to(empirical.parent).as_posix().encode() + p.read_bytes() for p in sorted(paths))
    ).hexdigest()


@contextmanager
def publication_lock(folder):
    # Serialize writers for the same destination. A terminated process releases the OS lock.
    with (folder / ".wrapped-publication.lock").open("a+b") as stream:
        if os.name == "nt":
            import msvcrt

            stream.seek(0)
            msvcrt.locking(stream.fileno(), msvcrt.LK_LOCK, 1)
        else:
            import fcntl

            fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            if os.name == "nt":
                stream.seek(0)
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def numerical_checks(diag):
    receipts = diag["corrections"]
    ess = min(i["relative_ess"] for i in receipts)
    spread = np.ptp([i["log_evidence_correction"] for i in receipts])
    return dict(
        screen_retained=diag["screen_mass_retained"],
        max_ep_residual=diag["max_ep_residual"],
        min_relative_ess=ess,
        log_evidence_scramble_range=float(spread),
        passed=bool(
            diag["screen_mass_retained"] >= 0.99998
            and diag["max_ep_residual"] < 3.0e-6
            and ess >= 0.1
            and spread < 0.1
        ),
    )


def archive_predictions(path, archives, data, meta, deposit, posterior):
    inventory, origin = load_inventory(path, archives)
    sites = inventory["sites"]
    result = dict(inventory_sha256=digest(origin) if origin else None, sites=[], scenarios={})
    if not sites:
        return result
    xy = positions(sites, meta["vent_lonlat"])
    latent = np.exp(deposit.quantiles(xy, (0.025, 0.5, 0.975))).T
    for station, pt, q in zip(sites, xy, latent, strict=True):
        result["sites"].append(
            dict(
                id=station["id"],
                site=station["name"],
                reading=station["reading"],
                latent_cm=q.tolist(),
                coincident_row_ids=data.row_ids[np.linalg.norm(data.xy - pt, axis=1) < 1.0e-6].tolist(),
            )
        )
    for scenario in inventory["scenarios"]:
        limits = scenario["effective_limit_cm"]
        pts = xy[[next(i for i, s in enumerate(sites) if s["id"] == code) for code in limits]]
        if len(pts):
            log_limit = np.log(list(limits.values()))
            probability = posterior.distribution(pts, True).cdf(log_limit)
            near = deposit.local(pts)
            if near.any():
                probability[near] = SourceBeds(deposit, pts[near]).cdf(log_limit[near])
        else:
            probability = np.array([])
        result["scenarios"][scenario["id"]] = dict(
            assumption=scenario["assumption"],
            effective_limit_cm=limits,
            probability_below=probability.tolist(),
        )
    return result


def sensitivity(data, meta, source):
    """Change only the angular spectrum while retaining the same marginal variance and derivative."""
    _, _, predict, diag = fit(data, meta["vent_lonlat"][1], source, covariance="three-harmonic")
    return dict(
        covariance="three-harmonic",
        check=numerical_checks(diag),
        rows=predict["row_id"].tolist(),
        primary_median_cm=predict["primary_median_cm"].tolist(),
        bed_log_score=predict["log_score"].tolist(),
    )


def summarise_folds(events, folds):
    summaries = {}
    for design, part in events.groupby("design", sort=False):
        groups = []
        for k, rows in part.groupby("fold"):
            point = rows.loc[rows.kind.eq("point"), "cdf"]
            checks = numerical_checks(folds[f"{design}-{k}"]["fit"])
            groups.append(
                dict(
                    fold=int(k),
                    sites=len(rows),
                    negative_log_score=float(-rows.log_score.mean()),
                    coverage50_exact=float(point.between(0.25, 0.75).mean()) if len(point) else None,
                    coverage95_exact=float(point.between(0.025, 0.975).mean()) if len(point) else None,
                    checks=checks,
                )
            )

        def mean(key):
            values = [g[key] for g in groups if g[key] is not None]
            return float(np.mean(values)) if values else None

        summaries[design] = dict(
            folds=groups,
            negative_log_score=mean("negative_log_score"),
            coverage50_exact=mean("coverage50_exact"),
            coverage95_exact=mean("coverage95_exact"),
            all_folds_passed=all(g["checks"]["passed"] for g in groups),
        )
    return summaries


def assemble(
    folder,
    data,
    meta,
    name,
    source,
    diagnostics,
    folds,
    volume,
    prior,
    archive,
    *,
    sampled=None,
    near_half=None,
):
    write_json(folder / "archive-probes.json", archive)
    write_json(folder / "partitions.json", folds)
    write_json(folder / "earth-integrals.json", volume)
    events = pd.read_csv(folder / "heldout.csv.gz")
    checks = numerical_checks(diagnostics)
    joint = volume["joint_volume"]
    report = dict(
        protocol=PROTOCOL,
        family=FAMILY,
        engine_sha256=engine_sha256(),
        dataset=name,
        input_sha256=meta["input_sha256"],
        vent_lonlat=meta["vent_lonlat"],
        source_area=source,
        observations=dict(
            sites=len(data.xy),
            measured=int((~data.censored & ~data.reported_minimum).sum()),
            at_reported_minimum=int(data.reported_minimum.sum()),
            upper=int((data.censored & ~data.interval).sum()),
            interval=int(data.interval.sum()),
        ),
        fit=diagnostics,
        regional_numerics=checks,
        prior_sensitivity=prior,
        validation=summarise_folds(events, folds),
        independent_draw_integration="Positive annular Earth quadrature plus independent angular integration of the local Gaussian excess",
        conditional_interval95_km3=[joint["quantiles_km3"][0], joint["quantiles_km3"][-1]],
        joint_volume_refinement=joint["refinement"],
        draw_checks_passed=bool(checks["passed"] and volume["passed"] and joint["passed"]),
        artifacts_sha256={
            n: digest(folder / n)
            for n in (
                "posterior.npz",
                "measured-sites.csv.gz",
                "heldout.csv.gz",
                "support.json.gz",
                "archive-probes.json",
                "earth-integrals.json",
                "partitions.json",
            )
        },
    )
    write_json(folder / "report.json", report)
    if sampled is not None:
        sampled = (
            *sampled,
            {
                "report.json": digest(folder / "report.json"),
                "posterior.npz": digest(folder / "posterior.npz"),
            },
        )
    receipt = render_deposit(folder, FAMILY, data, meta, name, sampled=sampled, near_half=near_half)
    from .maps import three_scales

    three_scales(folder, data, meta, name, receipt)
    return report, receipt


def wrapped_dataset(path, output, name=None, resume=False, archives=None):
    folder = Path(output)
    folder.mkdir(parents=True, exist_ok=True)
    with publication_lock(folder):
        return _run_locked(path, folder, name, resume, archives)


def _run_locked(path, folder, name, resume, archives):
    path = model_input(path, folder)
    data, meta = audited_data(path)
    source = source_for(meta, data)
    _, inventory_path = load_inventory(path, archives)
    from ...core.catalog import dataset_record

    record = dataset_record(path)
    name = name or (record["name"] if record else Path(path).stem)
    identity = dict(
        protocol=PROTOCOL,
        engine_sha256=engine_sha256(),
        input_sha256=meta["input_sha256"],
        name=name,
        source=source,
        inventory_sha256=digest(inventory_path) if inventory_path else None,
    )
    if resume and verified_release(folder, identity):
        return json.loads((folder / "report.json").read_text())
    with TemporaryDirectory(dir=folder, prefix=".wrapped-fit-") as tmp:
        stage = Path(tmp)
        report = _calculate(path, stage, name, archives, data, meta, source)
        write_manifest(stage, identity)
        # Publish the checksummed receipt after every artifact is in place.
        paths = sorted(stage.iterdir(), key=lambda p: p.name == "artifacts.json")
        for p in paths:
            os.replace(p, folder / p.name)
    return report


def _calculate(path, folder, name, archives, data, meta, source):
    lat = meta["vent_lonlat"][1]
    deposit, posterior, rows, diag = fit(data, lat, source)
    write(folder / "posterior.npz", deposit, meta)
    pd.DataFrame(rows).to_csv(folder / "measured-sites.csv.gz", index=False)
    archive = archive_predictions(path, archives, data, meta, deposit, posterior)
    volume = integrate(deposit, data.xy, lat)
    volume["joint_volume"] = joint_volumes(deposit, data, lat)
    views = []
    events = []
    folds = {}
    for design in ("geographic", "quadrants", "distal"):
        for k, train, test in partitions(data, meta, path, design):
            if train is None or not len(train.xy) or not len(test.xy):
                continue
            deleted, _, pred, detail = fit(train, lat, source, target=test, refined_source=False)
            folds[f"{design}-{k}"] = dict(
                train=train.row_ids.tolist(), test=test.row_ids.tolist(), fit=detail
            )
            events.append(pd.DataFrame(pred).assign(design=design, fold=k))
            if design == "geographic":
                views.append(ContourView(deleted))
    if not events:
        raise ValueError("No populated buffered holdout retains training observations")
    pd.concat(events).to_csv(folder / "heldout.csv.gz", index=False)
    support, *_ = audit_contours(ContourView(deposit), data, survey_bounds(data.xy), views)
    (folder / "support.json.gz").write_bytes(
        gzip.compress((json.dumps(support, allow_nan=False) + "\n").encode(), mtime=0)
    )
    prior = sensitivity(data, meta, source)
    report, _ = assemble(folder, data, meta, name, source, diag, folds, volume, prior, archive)
    return report


RELEASE_FILES = (
    "posterior.npz",
    "measured-sites.csv.gz",
    "heldout.csv.gz",
    "support.json.gz",
    "archive-probes.json",
    "earth-integrals.json",
    "partitions.json",
    "report.json",
    "deposit.json",
    "deposit.png",
    "three-scales.png",
    "median-grid.npz",
    "thickness-area.csv",
    "README.md",
)


def verified_release(folder, identity):
    try:
        saved = json.loads((folder / "artifacts.json").read_text())
        if not isinstance(saved, dict) or saved.get("identity") != identity:
            return False
        hashes = saved.get("artifacts_sha256")
        return (
            isinstance(hashes, dict)
            and set(hashes) == set(RELEASE_FILES)
            and all(digest(folder / k) == hashes[k] for k in RELEASE_FILES)
        )
    except (OSError, ValueError):
        return False


def write_manifest(folder, identity):
    hashes = {k: digest(folder / k) for k in RELEASE_FILES}
    write_json(folder / "artifacts.json", dict(identity=identity, artifacts_sha256=hashes))
