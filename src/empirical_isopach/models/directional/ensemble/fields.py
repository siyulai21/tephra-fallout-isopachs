"""A mixture of coherent directional fields and its pointwise empirical quantiles."""
from dataclasses import dataclass
from pathlib import Path
import json

import numpy as np
from scipy.special import logsumexp

from ....core.distributions import LogMixture
from ..geometry import Geometry
from ..inference import background, flat
from ..study import digest
from .members import MEMBERS, directory


PER_CHAIN = 64


def probabilities(weights):
    a = np.asarray(weights, dtype=float)
    if a.ndim != 1 or not len(a) or not np.isfinite(a).all() or np.any(a < 0) or a.sum() <= 0:
        raise ValueError("Supply nonnegative finite mixture masses with positive total")
    return a / a.sum()


def draw_weights(weights, sizes):
    if len(weights) != len(sizes) or min(sizes) <= 0:
        raise ValueError("Each structure needs a positive number of draws")
    return np.repeat(probabilities(weights)/np.asarray(sizes), sizes)


def inverse_cdf(values, weights, probs=(.5,), order=None):
    """Left inverse of a weighted empirical CDF. Last axis indexes coherent draws."""
    v = np.asarray(values)
    w = probabilities(weights)
    ps = np.asarray(probs, float)
    if len(w) != v.shape[-1] or np.any((ps <= 0) | (ps >= 1)):
        raise ValueError("Invalid quantile probabilities or sample size")
    if order is None:
        order = np.argsort(v, axis=-1, kind="stable")
    sorted_w = w[order]
    cdf = sorted_w.cumsum(axis=-1)
    # Normalizing by the observed cumulative sum controls binary rounding at ties.
    targets = (p*cdf[..., -1:] for p in ps)
    return np.stack([np.take_along_axis(v, np.take_along_axis(order,
                     np.argmax(cdf >= target-8*np.finfo(float).eps, axis=-1)[..., None], axis=-1),
                     axis=-1)[..., 0] for target in targets])


def subset_draws(draws, n):
    count = draws["source"].shape[1]
    if n is None or n >= count:
        idx = np.arange(count)
    else:
        idx = np.floor((np.arange(n)+.5)*count/n).astype(int)
    return {k: v[:, idx] for k, v in draws.items()}, idx


def read_fit(folder):
    records = [json.loads(f.read_text()) for f in sorted(Path(folder).glob("attempt-*.json"))]
    if not records:
        raise FileNotFoundError(f"Missing posterior in {folder}")
    path = Path(folder)/records[-1]["file"]
    if digest(path) != records[-1]["posterior_sha256"]:
        raise ValueError(f"Posterior checksum mismatch in {folder}")
    with np.load(path) as saved:
        return dict(saved), records


@dataclass
class Member:
    geometry: Geometry
    draws: dict
    index: np.ndarray
    records: list

    def __post_init__(self):
        self.samples = flat(self.draws)


def load_members(data, latitude, work, kind="full", fold=None, n=PER_CHAIN):
    members = []
    for policy in MEMBERS:
        folder = directory(work, policy, kind, fold)
        g = Geometry(data, latitude, policy)
        if json.loads((folder/"design.json").read_text())["geometry"] != g.describe():
            raise ValueError("Fit geometry differs from evaluation geometry")
        raw, records = read_fit(folder)
        d, idx = subset_draws(raw, n)
        members.append(Member(g, d, idx, records))
    return members


def latent_arrays(members, xy):
    """Reuse the common footprint. Conditional field evaluation matches inference.field."""
    _, ids, w = members[0].geometry.locate(xy)
    remaining = np.clip(1-w.sum(1), 0., 1.)
    outside = remaining > 1e-14
    local = w.sum(1) > 0.
    locations = np.asarray(xy)[outside]
    fields = []
    for member in members:
        p = member.samples
        values = np.zeros((len(p["source"]), len(w)))
        if outside.any():
            values[:, outside] = remaining[outside]*background(p, *member.geometry.coords(locations))
        if local.any():
            padded = np.pad(p["nodes"], ((0, 0), (1, 0)))
            inside = np.zeros((len(values), int(local.sum())))
            for k in range(3):
                inside += padded[:, ids[local, k]+1]*w[local, k]
            values[:, local] += inside
        fields.append(values)
    return fields, w


class Marginal:
    def __init__(self, values, weights):
        self.values = values
        self.weights = weights

    def quantile(self, probability):
        return inverse_cdf(self.values.T, self.weights, [probability])[0]

    def log_expected_thickness(self):
        return logsumexp(self.values, b=self.weights[:, None], axis=0)

    def moments(self):
        mean = self.weights @ self.values
        return mean, self.weights @ ((self.values-mean)**2)


class Ensemble:
    def __init__(self, members, weights):
        self.members = members
        self.model_weights = probabilities(weights)
        self.sizes = [len(m.samples["source"]) for m in members]
        self.weights = draw_weights(self.model_weights, self.sizes)

    def distribution(self, xy, observation=False, log_sd=None, calibrated=False):
        fields, w = latent_arrays(self.members, xy)
        means = np.concatenate(fields)
        if not observation:
            return Marginal(means, self.weights)
        known = 0. if log_sd is None else np.asarray(log_sd)**2
        variances = []
        for m in self.members:
            unresolved = 1+m.geometry.policy.local_sd_ratio**2*np.maximum(0., 1-(w**2).sum(1))
            variances.append(np.exp(2*m.samples["log_noise"][:, None])*unresolved + known)
        return LogMixture(self.weights, means, np.concatenate(variances))


def load_packet(folder, data, latitude, record):
    """Reconstruct the exact chain-balanced field packet used for all published maps."""
    folder = Path(folder)
    checksums = json.loads((folder/"artifacts.json").read_text())
    if digest(folder/"map-fields.npz") != checksums["map-fields.npz"]:
        raise ValueError("Published map-field checksum mismatch")
    members = []
    with np.load(folder/"map-fields.npz") as archive:
        for p in MEMBERS:
            params = {key.split("__")[1]:archive[key] for key in archive if key.startswith(p.name+"__")}
            indices = params.pop("draw_index")
            rows = record["coherent_fields"]["members"][p.name]["attempts"]
            chosen = np.floor((np.arange(len(indices))+.5)*rows[-1]["draws_per_chain"]/len(indices)).astype(int)
            if len(indices) != record["map_draws_per_chain"] or not np.array_equal(indices,chosen):
                raise ValueError("Published draw indices differ from the map design")
            members.append(Member(Geometry(data,latitude,p),params,indices,rows))
    return Ensemble(members,record["weighting"]["weights"])
