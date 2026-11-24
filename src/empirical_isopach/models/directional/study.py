"""Immutable fit attempts, geographic refits and reproducible common policies."""
from dataclasses import asdict
from hashlib import sha256
from importlib.metadata import version
import json
from pathlib import Path
import platform

import numpy as np

from ...core.validation import spatial_folds, training_mask, interval_metrics
from ...validation import automatic_buffer
from .geometry import BASE, POLICIES, Geometry, polar
from .inference import _sampler, sample, predict


SEED = 20261117


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix+".tmp")
    temp.write_text(json.dumps(obj, indent=2, allow_nan=False)+"\n")
    temp.replace(path)


def digest(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def _source_hashes():
    root = Path(__file__).parent
    files = [root.parents[1]/"core"/"likelihood.py", root/"inference.py", root/"geometry.py", root/"study.py", root.parents[1]/"support.py",
             root.parent/"mass_mixture"/"geometry.py", root.parents[2]/"probabilistic_isopach"/"global_volume.py"]
    return {str(p.relative_to(root.parents[2])): digest(p) for p in files}


_SOURCE_HASHES = _source_hashes()


def fit(data, g, seed, folder):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    key = {"policy": asdict(g.policy), "axis": g.axis, "seed": seed, "python": platform.python_version(),
           "runtime": {k: version(k) for k in ("numpy", "numpyro", "jax", "jaxlib", "scipy", "shapely", "arviz")},
           "source_sha256": _SOURCE_HASHES,
           "arrays_sha256": {k: sha256(np.ascontiguousarray(getattr(data, k)).tobytes()).hexdigest()
                    for k in ("xy", "log_value", "log_sd", "censored", "lower_censored", "row_ids")}}
    if data.interval.any():
        key["arrays_sha256"]["interval_log_upper"] = sha256(data.interval_log_upper.tobytes()).hexdigest()
    manifest = folder/"design.json"
    design = {"signature": key, "geometry": g.describe()}
    if manifest.exists() and json.loads(manifest.read_text()) != design:
        raise ValueError(f"Changed scientific inputs in {folder}. Use a fresh destination.")
    write_json(manifest, design)
    attempts = []
    for attempt in range(2):
        path = folder/f"attempt-{attempt+1}.npz"
        info = path.with_suffix(".json")
        if info.exists():
            result = json.loads(info.read_text())
            if digest(path) != result["posterior_sha256"]:
                raise ValueError(f"Posterior checksum mismatch in {folder}")
            with np.load(path) as saved:
                draws = dict(saved)
        else:
            attempt_seed = seed+1009*attempt
            try:
                draws, diag = sample(data, g, attempt_seed, attempt)
            except ValueError as exc:
                if "vmap was requested to map its argument along axis 0" not in str(exc):
                    raise
                # NumPyro can vectorize an equal-shaped kernel a second time.
                # Reinitialize it and replay the identical seed and sampler settings.
                _sampler.cache_clear()
                draws, diag = sample(data, g, attempt_seed, attempt)
            with path.with_suffix(".tmp").open("wb") as f:
                np.savez_compressed(f, **draws)
            path.with_suffix(".tmp").replace(path)
            result = {"attempt": attempt+1, "seed": seed+1009*attempt, "diagnostics": diag,
                      "chains": 4, "warmup": 600*(attempt+1), "draws_per_chain": 1000*2**attempt,
                      "posterior_sha256": digest(path), "file": path.name}
            write_json(info, result)
            print(folder.name, folder.parent.name, result["attempt"], diag, flush=True)
        attempts.append(result)
        if result["diagnostics"]["passed"]:
            break
    return draws, {"attempts": attempts, "passed": attempts[-1]["diagnostics"]["passed"]}


def validate(data, latitude, output, directional=False):
    buffer, rule = automatic_buffer(data)
    if directional:
        _, azimuth = polar(data.xy, Geometry(data, latitude).axis)
        labels = np.floor(azimuth/(np.pi/2)).astype(int).clip(0, 3)
    else:
        labels = spatial_folds(data.xy, 4)
    folds, arrays = [], {k: np.full(len(data.xy), np.nan) for k in ("median", "lower", "upper", "log_score")}
    for i in range(4):
        test = labels == i
        entry = {"fold": i, "withheld_row_ids": data.row_ids[test].tolist()}
        if not test.any():
            folds.append({**entry, "status": "empty"})
            continue
        # training_mask also drops observations inside the fixed spatial buffer.
        train, test = training_mask(data, labels, i, buffer)
        training = data.subset(train)
        entry["training_row_ids"] = training.row_ids.tolist()
        try:
            g = Geometry(training, latitude)
        except ValueError as exc:
            folds.append({**entry, "status": "insufficient support", "reason": str(exc)})
            continue
        d, info = fit(training, g, SEED+(201 if directional else 101)+i, output/f"fold-{i}")
        preds = predict(d, g, data.subset(test))
        for k in arrays:
            arrays[k][test] = preds[k]
        folds.append({**entry, **info, "status": "estimated", "geometry": g.describe()})
    ok = np.isfinite(arrays["log_score"])
    scores = {str(i): -float(arrays["log_score"][(labels == i) & ok].mean())
              for i in range(4) if np.any((labels == i) & ok)}
    metrics = interval_metrics(data.subset(ok), *(arrays[k][ok] for k in ("median", "lower", "upper", "log_score"))) if ok.any() else {}
    return {"row_ids": data.row_ids.tolist(), "folds": labels.tolist(), "buffer_km": buffer,
            "buffer_rule": rule, "fits": folds, "fold_scores": scores,
            "equal_region_log_score": float(np.mean(list(scores.values()))) if scores else None,
            "metrics": metrics, "scored_rows": int(ok.sum()), "total_rows": len(ok),
            "all_estimable_passed": all(f.get("passed", True) for f in folds),
            "predictions": {k: [float(v) if np.isfinite(v) else None for v in a] for k, a in arrays.items()}}


def run_fits(data, latitude, output, *, all_policies=True, validation=True):
    output = Path(output)
    data = data.canonical()
    for i, p in enumerate(POLICIES if all_policies else (BASE,)):
        fit(data, Geometry(data, latitude, p), SEED+i, output/"fits"/p.name)
    if validation:
        for kind in ("geographic", "quadrants"):
            write_json(output/f"{kind}.json", validate(data, latitude, output/kind, kind == "quadrants"))

