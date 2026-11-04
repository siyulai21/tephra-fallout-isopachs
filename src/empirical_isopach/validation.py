"""Identical outer folds and regional estimands for both empirical methods."""
import numpy as np
from scipy.spatial.distance import cdist

from src.probabilistic_isopach.distributions import predictive_distribution
from src.probabilistic_isopach.validation import spatial_folds, training_mask, select_model, interval_metrics
from .spline import survey_bounds
from .selection import select_spline
from .region import integrate_region


METHODS = ("stacked", "spline")
LABELS = {"stacked": "GP ensemble", "spline": "Trend-regularized cubic spline"}


def valid_training(data):
    points = data.xy[~data.censored]
    return (len(points) >= 6 and len(np.unique(points, axis=0)) >= 4
            and np.linalg.matrix_rank(points-points.mean(axis=0)) == 2)


def automatic_buffer(data, outer_folds=4, inner_folds=3):
    """Geometry-only buffer with a predeclared sample-size feasibility fallback."""
    xy = np.unique(data.xy, axis=0)
    distances = cdist(xy, xy)
    np.fill_diagonal(distances, np.inf)
    proposed = float(np.median(distances.min(axis=1)))
    labels = spatial_folds(data.xy, outer_folds)
    attempts = []
    for fraction in (1., .5, .25, 0.):
        buffer = proposed*fraction
        feasible = True
        for outer in range(outer_folds):
            train, _ = training_mask(data, labels, outer, buffer)
            subset = data.subset(train)
            if not valid_training(subset):
                feasible = False
                break
            inner_labels = spatial_folds(subset.xy, inner_folds)
            for inner in range(inner_folds):
                inner_train, _ = training_mask(subset, inner_labels, inner, buffer)
                if not valid_training(subset.subset(inner_train)):
                    feasible = False
                    break
            if not feasible:
                break
        attempts.append({"fraction": fraction, "buffer_km": buffer, "feasible": feasible})
        if feasible:
            return buffer, {"rule": "Median nearest-distinct-site distance; halve to 1/2, 1/4, then zero only if an inner training set lacks six exact values/four distinct non-collinear sites.",
                            "proposed_km": proposed, "attempts": attempts,
                            "uses": "Coordinates and observation kinds only; never thickness values."}
    raise ValueError("Not enough measurements for four outer / three inner geographic folds.")


def fit_method(method, data, bounds, folds, buffer):
    if method == "spline":
        return select_spline(data, bounds=bounds, n_folds=folds, buffer_km=buffer)
    if method == "stacked":
        return select_model(data, n_folds=folds, buffer_km=buffer, method="stacked")
    raise ValueError(f"Unknown empirical method: {method}")


def nested_comparison(data, region, bounds=None, buffer_km=None, outer_folds=4, inner_folds=3,
                      progress=None):
    data = data.canonical()
    bounds = survey_bounds(data.xy) if bounds is None else bounds
    if buffer_km is None:
        buffer_km, buffer_rule = automatic_buffer(data, outer_folds, inner_folds)
    else:
        buffer_rule = {"rule": "Explicit research override; not the automatic default."}
    labels = spatial_folds(data.xy, outer_folds)
    result = {"outer_folds": outer_folds, "inner_folds": inner_folds,
              "folds": labels.tolist(), "row_ids": data.row_ids.tolist(),
              "buffer_km": buffer_km, "buffer_rule": buffer_rule, "methods": {},
              "scope": "All response-dependent fitting and selection repeated in outer training sets. Full survey coordinates supply the common rectangle/hull.",
              "primary_metric": "equal_region_log_score; lower is better; no automatic preferred method"}
    for method in METHODS:
        arrays = {k: np.zeros(len(data.xy)) for k in ("median", "lower", "upper", "log_score")}
        selections = []
        for fold in range(outer_folds):
            train, test = training_mask(data, labels, fold, buffer_km)
            model, selection = fit_method(method, data.subset(train), bounds, inner_folds, buffer_km)
            heldout = data.subset(test)
            dist = predictive_distribution(model, heldout.xy, observation=True, log_sd=heldout.log_sd)
            for key, p in (("median", .5), ("lower", .025), ("upper", .975)):
                arrays[key][test] = dist.quantile(p)
            arrays["log_score"][test] = dist.log_score(heldout.log_value, heldout.censored, heldout.lower_censored)
            selections.append({"fold": fold, "n_train": int(train.sum()), "n_test": int(test.sum()),
                               "training_row_ids": data.row_ids[train].tolist(),
                               "withheld_row_ids": data.row_ids[test].tolist(),
                               "model": model.describe(), "selection": selection,
                               "regional_deletion_volume": integrate_region(model, region)})
            if progress:
                progress(f"{LABELS[method]}: geographic holdout {fold+1}/{outer_folds}")
        metrics = interval_metrics(data, arrays["median"], arrays["lower"], arrays["upper"], arrays["log_score"])
        fold_scores = [-float(np.mean(arrays["log_score"][labels == k])) for k in range(outer_folds)]
        metrics.update(equal_region_log_score=float(np.mean(fold_scores)), fold_log_scores=fold_scores)
        result["methods"][method] = {"metrics": metrics, "predictions": {k: v.tolist() for k, v in arrays.items()},
                                    "selections": selections,
                                    "deletion_interpretation": "Complete retuning after each geographic deletion; sensitivity, not a confidence interval."}
    result["paired_fold_score_difference_spline_minus_gp"] = (
        np.array(result["methods"]["spline"]["metrics"]["fold_log_scores"])
        - np.array(result["methods"]["stacked"]["metrics"]["fold_log_scores"])).tolist()
    return result
