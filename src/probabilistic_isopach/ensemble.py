"""Spatial predictive stacking with a separate observation-discrepancy layer.

Weights and discrepancy are estimated from out-of-fold predictive densities,
then assessed with outer geographic holdouts. They are not posterior model
probabilities, and this is not distribution-free conformal calibration.
"""
from dataclasses import dataclass, asdict
import numpy as np
from scipy.linalg import cholesky
from scipy.optimize import minimize, minimize_scalar
from scipy.special import log_ndtr

from .model import SpatialModel
from .distributions import LogMixture


def point_log_scores(data, means, variances):
    variance = np.maximum(variances, 1e-16)
    z = (data.log_value[:, None]-means)/np.sqrt(variance)
    scores = -.5*(np.log(2*np.pi*variance)+z*z)
    scores[data.censored] = log_ndtr(z[data.censored])
    return scores


def stack_weights(log_scores, row_weights):
    """Maximize a proper log predictive score over the probability simplex."""
    log_scores, row_weights = np.asarray(log_scores, float), np.asarray(row_weights, float)
    if (log_scores.ndim != 2 or not all(log_scores.shape) or not np.isfinite(log_scores).all()
            or row_weights.shape != (len(log_scores),) or not np.isfinite(row_weights).all()
            or np.any(row_weights < 0) or row_weights.sum() <= 0):
        raise ValueError("Stacking needs finite log densities and nonnegative row weights.")
    row_weights = row_weights/row_weights.sum()
    scaled = np.exp(log_scores-log_scores.max(axis=1, keepdims=True))
    def objective(weights):
        probability = np.maximum(scaled@weights, 1e-300)
        return -row_weights@np.log(probability), -(scaled.T@(row_weights/probability))
    n = log_scores.shape[1]
    result = minimize(objective, np.full(n, 1/n), jac=True, method="SLSQP",
                      bounds=[(0., 1.)]*n,
                      constraints={"type": "eq", "fun": lambda w: w.sum()-1,
                                   "jac": lambda w: np.ones_like(w)},
                      options={"ftol": 1e-11, "maxiter": 500})
    if not result.success:
        raise RuntimeError(f"Stacking did not converge: {result.message}")
    weights = np.maximum(result.x, 0.)
    return weights/weights.sum()


def observation_discrepancy(data, weights, means, variances, row_weights):
    """Extra new-observation log variance; never a latent-field/volume variance.

    The nonnegative variance maximizes an out-of-fold predictive score. The
    search cap is a declared numerical safeguard and boundary hits are reported.
    """
    center = weights@means.T
    detected = ~data.censored
    error_scale = np.mean((data.log_value[detected]-center[detected])**2)
    cap = max(4*error_scale, 1.)
    def loss(extra):
        distribution = LogMixture(weights, means.T, variances.T+extra)
        return -float(row_weights@distribution.log_score(data.log_value, data.censored))
    # Mixture scores need not be unimodal in variance: screen a fixed log grid
    # before refining its best bracket. Include both endpoints explicitly.
    grid = np.r_[0., np.geomspace(cap*1e-8, cap, 81)]
    losses = np.array([loss(value) for value in grid])
    best = int(np.argmin(losses))
    fitted = minimize_scalar(loss, bounds=(grid[max(0, best-1)], grid[min(len(grid)-1, best+1)]),
                             method="bounded", options={"xatol": 1e-9})
    if not fitted.success:
        raise RuntimeError("Observation discrepancy fit did not converge.")
    variance = float(fitted.x) if fitted.fun < losses[best] else float(grid[best])
    return variance, {"variance_log": variance, "sd_log": float(np.sqrt(variance)),
                      "search_upper_variance": cap, "at_search_boundary": bool(variance > .99*cap),
                      "training_cv_score_before": loss(0.), "training_cv_score_after": loss(variance),
                      "scope": "Additional discrepancy for new observations only; not included in latent maps or volume intervals.",
                      "coverage_guarantee": "None. Data-adaptive calibration must be evaluated in outer spatial holdouts."}


@dataclass(frozen=True)
class EnsembleSpecification:
    tail: str = "stacked"


class StackedModel:
    candidate = EnsembleSpecification()

    def __init__(self, models, weights, discrepancy_variance=0.):
        self.models = tuple(models)
        self.weights = np.asarray(weights, float)
        if (not self.models or self.weights.shape != (len(self.models),)
                or not np.isfinite(self.weights).all() or np.any(self.weights <= 0)
                or not np.isclose(self.weights.sum(), 1.) or not np.isfinite(discrepancy_variance)
                or discrepancy_variance < 0):
            raise ValueError("Invalid stacked model weights or discrepancy variance.")
        self.data = self.models[0].data
        if any(not np.array_equal(getattr(m.data, field), getattr(self.data, field))
               for m in self.models for field in ("xy", "log_value", "censored", "log_sd", "row_ids")):
            raise ValueError("Stacked models must use the same observations in the same order.")
        self.discrepancy_variance = discrepancy_variance
        self.amplitude = max(m.amplitude for m in self.models)
        self.trend_boundary = any(m.trend_boundary and w > .1 for m, w in zip(self.models, self.weights))
        family_weights = {tail: sum(w for m, w in zip(self.models, self.weights) if m.candidate.tail == tail)
                          for tail in ("exponential", "power")}
        self.dominant_tail = max(family_weights, key=family_weights.get)

    def distribution(self, xy, observation=False, log_sd=None, calibrated=False):
        predictions = [m.predict_log(xy, observation=observation, log_sd=log_sd) for m in self.models]
        means = np.array([p[0] for p in predictions])
        variances = np.array([p[1] for p in predictions])
        if observation and calibrated:
            variances += self.discrepancy_variance
        return LogMixture(self.weights, means, variances)

    def predict_log(self, xy, covariance=False, observation=False, log_sd=None):
        if not covariance:
            return self.distribution(xy, observation, log_sd).moments()
        # Law of total covariance. This is a moment export, not a Gaussian law.
        predictions = [m.predict_log(xy, covariance=True, observation=observation, log_sd=log_sd)
                       for m in self.models]
        mean = sum(w*p[0] for w, p in zip(self.weights, predictions))
        cov = sum(w*(p[1]+np.outer(p[0]-mean, p[0]-mean)) for w, p in zip(self.weights, predictions))
        return mean, (cov+cov.T)/2

    def sample_log_fields(self, xy, draws, seed):
        rng = np.random.default_rng(seed)
        labels = rng.choice(len(self.models), size=draws, p=self.weights)
        samples = np.empty((len(xy), draws))
        for index, model in enumerate(self.models):
            which = np.flatnonzero(labels == index)
            if not len(which):
                continue
            mean, covariance = model.predict_log(xy, covariance=True)
            jitter = max(model.amplitude**2, 1.)*1e-10
            L = cholesky(covariance+jitter*np.eye(len(xy)), lower=True)
            samples[:, which] = mean[:, None]+L@rng.standard_normal((len(xy), len(which)))
        return samples

    def describe(self):
        return {"candidate": asdict(self.candidate), "components": [
                    {"weight": float(w), "model": model.describe()} for w, model in zip(self.weights, self.models)],
                "dominant_tail": self.dominant_tail,
                "observation_discrepancy_variance_log": self.discrepancy_variance,
                "trend_on_constraint_boundary": self.trend_boundary,
                "uncertainty": "Predictive mixture; includes disagreement among retained candidates, conditional on each fit and estimated weights. Fitted-parameter uncertainty is not integrated."}


def build_stacked_model(data, candidates, records, predictions, folds, bootstrap_draws=64):
    valid = [i for i, record in enumerate(records) if record["selection_score"] is not None]
    # A small, declared screening rule limits weight overfitting and map cost.
    retained = sorted(i for tail in ("exponential", "power") for i in
                      sorted((i for i in valid if candidates[i].tail == tail),
                             key=lambda i: records[i]["selection_score"])[:3])
    means = np.column_stack([predictions[i][0] for i in retained])
    variances = np.column_stack([predictions[i][1] for i in retained])
    scores = point_log_scores(data, means, variances)
    counts = np.bincount(folds)
    row_weights = 1/(len(counts)*counts[folds])
    all_weights = stack_weights(scores, row_weights)
    # Numerically zero weights are omitted, without changing family choice by hand.
    active = all_weights > 1e-7
    weights = all_weights[active]/all_weights[active].sum()
    chosen = np.array(retained)[active]
    means, variances = means[:, active], variances[:, active]
    discrepancy, calibration = observation_discrepancy(data, weights, means, variances, row_weights)
    models = [SpatialModel(candidates[i]).fit(data) for i in chosen]
    model = StackedModel(models, weights, discrepancy)
    # Whole-fold Bayesian bootstrap is a stability diagnostic, not a posterior.
    rng = np.random.default_rng(1729)
    replicates = []
    for mass in rng.dirichlet(np.ones(len(counts)), size=bootstrap_draws):
        replicates.append(stack_weights(scores, mass[folds]/counts[folds]))
    stability = np.quantile(replicates, [.05, .95], axis=0).T.tolist() if bootstrap_draws else [None]*len(retained)
    diagnostic = {"screening": "Three lowest CV-score candidates per tail family; no screen uses outer holdout responses.",
                  "components": [{"candidate_index": int(i), "candidate": asdict(candidates[i]),
                                  "weight": float(w), "block_bootstrap_weight_5_95": interval}
                                 for i, w, interval in zip(retained, all_weights, stability)],
                  "block_bootstrap_draws": bootstrap_draws, "block_bootstrap_seed": 1729,
                  "calibration": calibration, "assessment": "Weight/calibration training diagnostics. Use outer nested validation for performance.",
                  "dropped_numerical_weight": float(all_weights[~active].sum())}
    return model, diagnostic, LogMixture(weights, means.T, variances.T)
