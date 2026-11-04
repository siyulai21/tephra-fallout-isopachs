"""Deterministic, response-isolated geographic tuning for the cubic spline."""
from dataclasses import asdict
from itertools import product

import numpy as np

from ...core.distributions import predictive_distribution
from ...core.validation import spatial_folds, training_mask
from .model import SplineCandidate, SplineWorkspace, TrendRegularizedSpline, survey_bounds


SEARCH_AXES = {
    "smoothing": (1e-6, 1e-4, .01, 1., 100.),
    "trend_strength": (0., .01, 1., 100., 10000.),
    "noise_ratio": (.25, .75, 1.5, 3.),
    "tension": (0., .5, 1.),
    "knot_intervals": (2, 4, 6, 8),
}
EXTENSIONS = {
    "smoothing": (1e-8, 10000.),
    "trend_strength": (None, 1000000.),
    "noise_ratio": (.125, 6.),
    "knot_intervals": (1, 12),
}


def select_spline(data, bounds=None, n_folds=4, buffer_km=0., candidates=None):
    """Tune all five shape/noise choices and both trend families automatically.

    Exhaustive fixed grid for both families, followed by an exhaustive expanded
    grid when a family winner touches an extendable boundary. One expansion is
    allowed. This minimizes over the tested grid, not over continuous parameters.
Explicit candidates are supported for numerical tests/research, not required by
the normal interface. Coordinates define the survey design; responses are refit.
"""
    data = data.canonical()
    bounds = survey_bounds(data.xy) if bounds is None else bounds
    labels = spatial_folds(data.xy, n_folds)
    contexts = []
    for fold in range(n_folds):
        train, test = training_mask(data, labels, fold, buffer_km)
        contexts.append((SplineWorkspace(data.subset(train), bounds), data.subset(test)))
    records, cache = [], {}

    def evaluate(candidate):
        if candidate in cache:
            return cache[candidate]
        scores, failures = [], []
        for fold, (workspace, test) in enumerate(contexts):
            try:
                model = TrendRegularizedSpline(candidate).fit(workspace.data, workspace=workspace)
                distribution = predictive_distribution(model, test.xy, observation=True, log_sd=test.log_sd)
                score = float(-np.mean(distribution.log_score(test.log_value, test.censored, test.lower_censored)))
                if not np.isfinite(score):
                    raise RuntimeError("Nonfinite predictive score.")
                scores.append(score)
            except (ValueError, RuntimeError, np.linalg.LinAlgError) as exc:
                failures.append({"fold": fold, "error": str(exc)})
        index = len(records)
        records.append({"candidate": asdict(candidate), "fold_scores": scores, "failures": failures,
                        "selection_score": None if failures else float(np.mean(scores))})
        cache[candidate] = index
        return index

    def score(candidate):
        value = records[evaluate(candidate)]["selection_score"]
        return np.inf if value is None else value

    searches = []
    if candidates is not None:
        candidates = tuple(candidates)
        for candidate in candidates:
            evaluate(candidate)
    else:
        for tail in ("exponential", "power"):
            def grid(axes):
                values = [SplineCandidate(tail=tail, **dict(zip(axes, p)))
                          for p in product(*axes.values())]
                return min(values, key=score)
            current = grid(SEARCH_AXES)
            initial = evaluate(current)
            expanded = {}
            for key, extra in EXTENSIONS.items():
                values = SEARCH_AXES[key]
                value = getattr(current, key)
                if value == values[0] and extra[0] is not None:
                    expanded[key] = tuple(sorted((*values, extra[0])))
                elif value == values[-1] and extra[1] is not None:
                    expanded[key] = tuple(sorted((*values, extra[1])))
            if expanded:
                current = grid({**SEARCH_AXES, **expanded})
            searches.append({"tail": tail, "initial_winner_index": initial,
                             "expanded_winner_index": evaluate(current),
                             "expanded_axes": {k: list(v) for k, v in expanded.items()}})
    valid = [i for i, r in enumerate(records) if r["selection_score"] is not None]
    if not valid:
        raise ValueError("No spline candidate fitted every geographic training subset.")
    best = min(valid, key=lambda i: records[i]["selection_score"])
    candidate = SplineCandidate(**records[best]["candidate"])
    model = TrendRegularizedSpline(candidate).fit(data, bounds=bounds)
    edges = []
    for key, options in SEARCH_AXES.items():
        explored = sorted({r["candidate"][key] for r in records})
        value = getattr(candidate, key)
        # Zero trend penalty and pure first/second derivatives are legitimate ends.
        if key == "tension" or (key == "trend_strength" and value == 0):
            continue
        if value in (explored[0], explored[-1]):
            edges.append(key)
    return model, {"method": "spline", "protocol": "regional-spline-v1",
                   "selected_index": best, "candidates": records, "searches": searches,
                   "selected_at_explored_boundary": edges, "n_evaluated": len(records),
                   "folds": labels.tolist(), "row_ids": data.row_ids.tolist(),
                   "buffer_km": float(buffer_km), "bounds_km": list(map(float, bounds)),
                   "assessment": "Inner selection diagnostics, not estimates of generalization performance.",
                   "search_limitation": "Minimum over evaluated discrete grids; no continuous global-optimum guarantee."}
