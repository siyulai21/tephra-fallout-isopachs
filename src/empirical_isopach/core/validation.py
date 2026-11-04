"""Shared geographic partitions and predictive scoring; no model preference."""
import numpy as np
from scipy.spatial.distance import cdist
from scipy.special import log_ndtr

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
    nll[data.lower_censored] = -log_ndtr(-z[data.lower_censored])
    det = ~data.censored
    return {"mean_log_score": float(np.mean(nll)),
            "rmse_log": float(np.sqrt(np.mean((data.log_value[det]-mean[det])**2))) if det.any() else None,
            "coverage95_observations": float(np.mean(abs(z[det]) <= 1.959963984540054)) if det.any() else None}


def interval_metrics(data, median, lower, upper, log_scores):
    detected = ~data.censored
    return {"mean_log_score": float(-np.mean(log_scores)),
            "rmse_log": float(np.sqrt(np.mean((data.log_value[detected]-median[detected])**2))) if detected.any() else None,
            "coverage95_observations": float(np.mean((data.log_value[detected] >= lower[detected]) &
                                                     (data.log_value[detected] <= upper[detected]))) if detected.any() else None,
            "mean_interval_width_log": float(np.mean(upper[detected]-lower[detected])) if detected.any() else None}
