"""Finite-mixture inference for the regional log-thickness and preserved-bed fields."""

from collections import namedtuple
from dataclasses import dataclass

import numpy as np
from scipy.linalg import cho_solve, solve_triangular
from scipy.special import logsumexp
from src.probabilistic_isopach.global_volume import EarthDomain
from ...core.likelihood import normal_log_event
from ..directional.multiscale.posterior import Marginals
from .conditioning import BedConditioner, correction
from .geometry import SCALES, NOISES, LOGPRIOR, transported_mean, local_kernel, basis

GEOMETRIES = ((1.0, 1.0), (1.0, 2.0), (2.0, 1.0), (2.0, 2.0))
ARCHIVES = ((0.0, 1.0, 0.5), (0.5, 3.0, 1 / 6), (0.5, 30.0, 1 / 6), (0.5, 300.0, 1 / 6))
Group = namedtuple("Group", "scale amplitude power noise fraction length mean basis gram prior independent")


def screen(data, mean, gram, noise):
    """Exact point likelihood and an independent-marginal screen for bounds."""
    exact = ~data.censored
    bound = ~exact
    cov = gram + np.diag(noise**2 + data.log_sd**2 + 1.0e-9)
    if exact.any():
        chol = np.linalg.cholesky(cov[np.ix_(exact, exact)])
        res = solve_triangular(chol, (data.log_value[exact] - mean[:, exact]).T, lower=True)
        evidence = (
            -0.5 * (res * res).sum(axis=0)
            - np.log(np.diag(chol)).sum()
            - 0.5 * exact.sum() * np.log(2 * np.pi)
        )
    else:
        evidence = np.zeros(len(mean))
    if bound.any():
        if exact.any():
            proj = solve_triangular(chol, cov[np.ix_(exact, bound)], lower=True)
            mu = mean[:, bound] + res.T @ proj
            var = np.diag(cov)[bound] - (proj * proj).sum(axis=0)
        else:
            mu, var = mean[:, bound], np.diag(cov)[bound]
        evidence += normal_log_event(
            data.log_value[bound],
            mu,
            np.sqrt(np.maximum(var, 1.0e-12)),
            data.censored[bound],
            data.lower_censored[bound],
            data.interval_log_upper[bound],
        ).sum(axis=1)
    return evidence


class SourceField:
    def __init__(self, data, latitude, covariance):
        self.data = data.canonical()
        d = self.data
        self.axis = EarthDomain.wgs84(latitude).axis_scale
        physical = d.xy * [1 / self.axis, self.axis]
        distances = {l: local_kernel(physical, physical, l) for f, l, _ in ARCHIVES if f}
        groups = []
        proposals = []
        for scale in SCALES:
            mean = transported_mean(d.xy, self.axis, scale)
            for amplitude, power in GEOMETRIES:
                b = basis(d.xy, self.axis, scale, amplitude, power, covariance)
                signal = b @ b.T
                for n in NOISES:
                    for fraction, length, prior in ARCHIVES:
                        gram = signal + fraction * n * n * distances[length] if fraction else signal
                        independent = n * np.sqrt(1.0 - fraction)
                        evidence = screen(d, mean, gram, independent)
                        proposals.append(evidence + LOGPRIOR + np.log(prior))
                        groups.append(
                            Group(
                                scale,
                                amplitude,
                                power,
                                n,
                                fraction,
                                length,
                                mean,
                                b,
                                gram,
                                prior,
                                independent,
                            )
                        )
        preliminary = np.array(proposals)
        weights = np.exp(preliminary.ravel() - logsumexp(preliminary))
        order = np.argsort(-weights)
        count = min(len(order), max(96, 1 + np.searchsorted(weights[order].cumsum(), 0.99999)))
        self.parts = []
        scores = []
        conditioner = BedConditioner(d)
        for ix in order[:count]:
            gi, j = np.unravel_index(ix, preliminary.shape)
            g = groups[gi]
            st = conditioner.evaluate(gi, g.gram, g.mean[j], g.independent)
            scores.append(st["logz"] + np.log(g.prior) + LOGPRIOR[j])
            self.parts.append((gi, j, st))
        scores = np.array(scores)
        self.weights = np.exp(scores - logsumexp(scores))
        keep = self.weights > 1.0e-8
        self.weights = self.weights[keep]
        self.weights /= self.weights.sum()
        self.parts = [p for p, ok in zip(self.parts, keep) if ok]
        self.groups = {int(i): groups[i] for i in {p[0] for p in self.parts}}
        self.detail = dict(
            proposed=int(preliminary.size),
            refined=int(count),
            retained=len(self.parts),
            screen_mass_retained=float(weights[order[:count]].sum()),
            max_ep_residual=float(max(p[2]["ep_residual"] for p in self.parts)),
        )


@dataclass
class Posterior:
    axis: float
    data: object
    groups: dict
    parts: list
    weights: np.ndarray
    detail: dict
    covariance: str

    @classmethod
    def fit(cls, data, latitude, power=12, covariance="wrapped"):
        source = SourceField(data, latitude, covariance)
        populations = []
        receipts = []
        seeds = (176989, 817849) if data.censored.any() else (176989,)
        for seed in seeds:
            parts, weights, receipt = correction(source, power, seed)
            populations.append((parts, weights))
            receipts.append(receipt)
        z = np.array([r["log_evidence_correction"] for r in receipts])
        pool = np.exp(z - logsumexp(z))
        parts = [part for pp, _ in populations for part in pp]
        weights = np.concatenate([q * w for q, (_, w) in zip(pool, populations)])
        obj = cls(
            source.axis,
            source.data,
            source.groups,
            parts,
            weights,
            dict(**source.detail, corrections=receipts),
            covariance,
        )
        obj._prepare()
        return obj

    def _prepare(self):
        self.geometry = {}
        group_ids = np.array([part[0] for part in self.parts])
        for gid in np.unique(group_ids):
            ix = np.flatnonzero(group_ids == gid)
            g = self.groups[gid]
            j = np.array([self.parts[k][1] for k in ix])
            y = np.array([self.parts[k][2] for k in ix])
            chol = np.linalg.cholesky(g.gram + np.diag(g.independent**2 + self.data.log_sd**2 + 1.0e-9))
            alpha = cho_solve((chol, True), (y - g.mean[j]).T).T
            c = solve_triangular(chol, g.basis, lower=True)
            cov = np.eye(c.shape[1]) - c.T @ c
            self.geometry[gid] = (ix, j, chol, alpha, alpha @ g.basis, 0.5 * (cov + cov.T))

    def distribution(self, xy, noise=False, log_sd=0.0):
        xy = np.asarray(xy)
        mus = np.empty((len(self.parts), len(xy)))
        variance = np.empty_like(mus)
        means = {}
        bases = {}
        local = {}
        for gid, (ix, j, chol, alpha, beta, cov) in self.geometry.items():
            g = self.groups[gid]
            if g.scale not in means:
                means[g.scale] = transported_mean(xy, self.axis, g.scale)
            key = (g.scale, g.amplitude, g.power)
            if key not in bases:
                bases[key] = basis(xy, self.axis, *key, self.covariance)
            b = bases[key]
            if not noise or not g.fraction:
                mus[ix] = means[g.scale][j] + beta @ b.T
                v = np.einsum("ij,ij->i", b @ cov, b)
                if noise:
                    v = v + g.independent**2 + np.asarray(log_sd) ** 2
            else:
                if g.length not in local:
                    metric = np.array([1 / self.axis, self.axis])
                    local[g.length] = local_kernel(xy * metric, self.data.xy * metric, g.length)
                h = b @ g.basis.T + g.noise**2 * g.fraction * local[g.length]
                mus[ix] = means[g.scale][j] + alpha @ h.T
                t = solve_triangular(chol, h.T, lower=True)
                v = np.sum(b * b, axis=1) + g.noise**2 - (t * t).sum(axis=0) + np.asarray(log_sd) ** 2
            variance[ix] = np.maximum(v, 1.0e-12)
        return Marginals(self.weights, mus, variance)
