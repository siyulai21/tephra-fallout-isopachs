"""Deterministic spatial validation; preprocessing is refitted in every fold."""
from dataclasses import asdict
from itertools import product

import numpy as np
from scipy.spatial.distance import cdist
from scipy.special import log_ndtr
from .model import Candidate, SpatialModel


DEFAULT_CANDIDATES = tuple(Candidate(*p) for p in product(
    ("exponential", "power"), (0.25, 0.75, 1.5), (0.25, 0.75, 1.5)))


def spatial_folds(xy, n_folds=4):
    """Recursive spatial median partitions on unique locations, no random state.

    Split the longest coordinate span, allocating unique sites approximately
    equally among leaves. Exact colocated samples always share a fold. There is
    no minimum separation across a cell boundary (use a buffer).
    """
    xy = np.asarray(xy, float)
    unique, inverse = np.unique(xy, axis=0, return_inverse=True)
    if n_folds < 2 or n_folds > len(unique):
        raise ValueError("Need 2 <= n_folds <= number of unique locations.")
    labels = np.empty(len(unique), dtype=int)
    def partition(indices, count, first_label):
        if count == 1:
            labels[indices] = first_label
            return
        axis = int(np.argmax(np.ptp(unique[indices], axis=0)))
        ordered = indices[np.lexsort((unique[indices, 1-axis], unique[indices, axis]))]
        left_count = count//2
        cut = int(len(indices)*left_count/count)
        partition(ordered[:cut], left_count, first_label)
        partition(ordered[cut:], count-left_count, first_label+left_count)
    partition(np.arange(len(unique)), n_folds, 0)
    return labels[inverse]


def training_mask(data, folds, fold, buffer_km):
    if not np.isfinite(buffer_km) or buffer_km < 0:
        raise ValueError("buffer_km must be nonnegative and finite.")
    test = folds == fold
    train = ~test
    if buffer_km:
        train &= cdist(data.xy, data.xy[test]).min(axis=1) > buffer_km
    return train, test


def score_predictions(data, mean, variance):
    variance = np.maximum(variance, 1e-12)
    z = (data.log_value-mean)/np.sqrt(variance)
    nll = .5*(np.log(2*np.pi*variance)+z*z)
    nll[data.censored] = -log_ndtr(z[data.censored])
    det = ~data.censored
    return {"mean_log_score": float(np.mean(nll)),
            "rmse_log": float(np.sqrt(np.mean((data.log_value[det]-mean[det])**2))) if det.any() else None,
            "coverage95_observations": float(np.mean(abs(z[det]) <= 1.959963984540054)) if det.any() else None}


def select_model(data, candidates=DEFAULT_CANDIDATES, n_folds=4, buffer_km=0.):
    data = data.canonical()
    folds = spatial_folds(data.xy, n_folds)
    records, predictions = [], []
    for candidate in candidates:
        mean, var = np.zeros(len(data.xy)), np.zeros(len(data.xy))
        scores, failures = [], []
        for fold in range(n_folds):
            train, test = training_mask(data, folds, fold, buffer_km)
            try:
                model = SpatialModel(candidate).fit(data.subset(train))
                mean[test], var[test] = model.predict_log(data.xy[test], observation=True,
                                                         log_sd=data.log_sd[test])
                scores.append(score_predictions(data.subset(test), mean[test], var[test])["mean_log_score"])
            except (ValueError, RuntimeError, np.linalg.LinAlgError) as exc:
                failures.append({"fold": fold, "error": str(exc)})
        record = {**asdict(candidate), "fold_scores": scores, "failures": failures,
                  "selection_score": None if failures else float(np.mean(scores))}
        if not failures:
            record.update(score_predictions(data, mean, var))
        records.append(record)
        predictions.append((mean, var))
    valid = [i for i, r in enumerate(records) if r["selection_score"] is not None]
    if not valid:
        raise ValueError(f"No valid candidate across all spatial folds: {records[0]['failures'] if records else 'empty candidate set'}")
    best = min(valid, key=lambda i: records[i]["selection_score"])
    model = SpatialModel(candidates[best]).fit(data)
    return model, {"selected_index": best, "candidates": records,
                   "folds": folds.tolist(), "row_ids": data.row_ids.tolist(),
                   "cv_mean_log": predictions[best][0].tolist(),
                   "cv_variance_log": predictions[best][1].tolist(),
                   "assessment": "Selection diagnostics; use nested CV for performance estimation.",
                   "buffer_km": buffer_km}


def nested_validate(data, candidates=DEFAULT_CANDIDATES, outer_folds=4, inner_folds=3, buffer_km=0.):
    data = data.canonical()
    folds = spatial_folds(data.xy, outer_folds)
    mean, var = np.zeros(len(data.xy)), np.zeros(len(data.xy))
    selections = []
    for fold in range(outer_folds):
        train, test = training_mask(data, folds, fold, buffer_km)
        model, selection = select_model(data.subset(train), candidates, inner_folds, buffer_km)
        mean[test], var[test] = model.predict_log(data.xy[test], observation=True, log_sd=data.log_sd[test])
        selections.append({"fold": fold, "n_train": int(train.sum()), "n_test": int(test.sum()),
                           "selected": asdict(model.candidate), "inner_selection": selection})
    metrics = score_predictions(data, mean, var)
    return {**metrics, "folds": folds.tolist(), "row_ids": data.row_ids.tolist(),
            "mean_log": mean.tolist(), "variance_log": var.tolist(), "selections": selections,
            "assessment": "Nested spatial validation; no evaluation on pseudo-points."}
