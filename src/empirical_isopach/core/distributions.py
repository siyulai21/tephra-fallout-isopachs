"""Exact marginal summaries for Gaussian mixtures on natural-log thickness.

Moment matching is useful for covariance export, but it is not used to compute
mixture quantiles, mean thickness, predictive scores or joint volume samples.
"""
from dataclasses import dataclass
import numpy as np
from scipy.special import logsumexp, ndtr, ndtri

from .likelihood import normal_log_event


@dataclass
class LogMixture:
    weights: np.ndarray
    means: np.ndarray  # component x location
    variances: np.ndarray

    def __post_init__(self):
        self.weights = np.asarray(self.weights, float)
        self.means = np.asarray(self.means, float)
        self.variances = np.asarray(self.variances, float)
        if (self.means.ndim != 2 or self.variances.shape != self.means.shape
                or self.weights.shape != (len(self.means),)
                or not np.isfinite(self.means).all() or not np.isfinite(self.variances).all()
                or np.any(self.variances < 0) or not np.isfinite(self.weights).all()
                or np.any(self.weights <= 0) or not np.isclose(self.weights.sum(), 1.)):
            raise ValueError("Mixture needs finite components, nonnegative variances and positive normalized weights.")

    def moments(self):
        mean = self.weights@self.means
        variance = self.weights@(self.variances+(self.means-mean)**2)
        return mean, variance

    def log_expected_thickness(self):
        return logsumexp(np.log(self.weights)[:, None]+self.means+self.variances/2, axis=0)

    def cdf(self, values):
        sd = np.sqrt(np.maximum(self.variances, 1e-16))
        return self.weights@ndtr((np.asarray(values)-self.means)/sd)

    def quantile(self, probability):
        if not 0 < probability < 1:
            raise ValueError("Quantile probability must be strictly between zero and one.")
        limits = self.means+ndtri(probability)*np.sqrt(np.maximum(self.variances, 1e-16))
        if len(self.weights) == 1:
            return limits[0]
        lo, hi = limits.min(axis=0), limits.max(axis=0)
        for _ in range(42):
            mid = (lo+hi)/2
            below = self.cdf(mid) < probability
            lo = np.where(below, mid, lo)
            hi = np.where(below, hi, mid)
        return (lo+hi)/2

    def score(self, data):
        return self.log_score(data.log_value, data.censored, data.lower_censored, data.interval_log_upper)

    def log_score(self, values, censored=None, lower_censored=None, interval_log_upper=None):
        censored = False if censored is None else np.asarray(censored, bool)
        lower = False if lower_censored is None else np.asarray(lower_censored, bool)
        if np.any(lower & ~np.asarray(censored)):
            raise ValueError("Lower bounds must also be marked censored.")
        if interval_log_upper is not None and np.any(np.isfinite(interval_log_upper) &
                (~np.asarray(lower) | (interval_log_upper <= np.asarray(values)))):
            raise ValueError("Intervals require ordered endpoints and a lower-bound flag.")
        sd = np.sqrt(np.maximum(self.variances, 1e-16))
        scores = normal_log_event(np.asarray(values), self.means, sd, censored, lower, interval_log_upper)
        return logsumexp(np.log(self.weights)[:, None]+scores, axis=0)


def predictive_distribution(model, xy, observation=False, log_sd=None, calibrated=False):
    if hasattr(model, "distribution"):
        return model.distribution(xy, observation, log_sd, calibrated)
    mean, variance = model.predict_log(xy, observation=observation, log_sd=log_sd) if observation else model.predict_log(xy)
    return LogMixture(np.ones(1), mean[None, :], variance[None, :])


def thickness_summaries(model, xy):
    distribution = predictive_distribution(model, xy)
    median = distribution.quantile(.5)
    _, variance = distribution.moments()
    return median, distribution.log_expected_thickness(), variance


class EmpiricalMarginals:
    def __init__(self, log_fields):
        self.values = log_fields

    def quantile(self, probability):
        return np.quantile(self.values, probability, axis=0)

    def log_expected_thickness(self):
        return logsumexp(self.values, axis=0)-np.log(len(self.values))
