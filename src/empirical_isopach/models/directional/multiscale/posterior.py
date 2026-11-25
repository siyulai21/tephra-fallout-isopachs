"""Pointwise mixture quantiles and coherent complete-deposit draws from the directional fit."""

from dataclasses import dataclass

import numpy as np
from scipy.special import ndtr, ndtri
from scipy.stats import qmc

from src.probabilistic_isopach.global_volume import EarthDomain
from ....core.distributions import LogMixture
from .geometry import SCHEMA, LocalBasis, trend
from .spread import attenuation


@dataclass
class Component:
    weight: float
    scale: float
    evolving: int
    length: float
    tail: float
    noise: float
    spread: float
    mean: np.ndarray
    chol: np.ndarray


class Marginals(LogMixture):
    def quantile(self, probability):
        if not 0 < probability < 1:
            raise ValueError("Expected an interior probability")
        sd = np.sqrt(np.maximum(self.variances, 1e-16))
        q = ndtri(probability)
        single = self.means + q * sd
        if len(single) == 1:
            return single[0]
        lo, hi = single.min(axis=0), single.max(axis=0)
        x = np.clip(self.weights @ single, lo, hi)
        for _ in range(48):
            u = (x - self.means) / sd
            residual = self.weights @ ndtr(u) - probability
            if np.max(np.abs(residual)) < 2e-12:
                break
            lo = np.where(residual < 0, x, lo)
            hi = np.where(residual > 0, x, hi)
            pdf = self.weights @ (np.exp(-0.5 * u * u) / sd) / np.sqrt(2 * np.pi)
            proposal = x - residual / np.maximum(pdf, 1e-14)
            x = np.where((proposal >= lo) & (proposal <= hi), proposal, (lo + hi) / 2)
        return x


class DirectionalField:
    def __init__(self, latitude, local, components, reach):
        self.reach = np.asarray(reach)
        self.latitude = float(latitude)
        self.domain = EarthDomain.wgs84(latitude)
        self.local = local
        self.components = components
        self.weights = np.array([c.weight for c in components])
        self.weights /= self.weights.sum()

    def matrices(self, xy):
        xy = np.asarray(xy)
        radial, spatial, design = {}, {}, {}
        for c in self.components:
            key = (c.scale, c.evolving, c.length)
            if key in design:
                continue
            if key[:2] not in radial:
                radial[key[:2]] = trend(xy, self.domain.axis_scale, *key[:2])
            columns, d, rim = radial[key[:2]]
            if c.length not in spatial:
                spatial[c.length] = (
                    1.15 * rim[:, None] ** 2 * self.local[c.length].features(xy, self.domain.axis_scale)
                )
            design[key] = (np.column_stack((columns, spatial[c.length])), d)
        return design, attenuation(xy, self.domain.axis_scale, self.reach)

    def distribution(self, xy, observation=False, log_sd=None, calibrated=False):
        design, loss = self.matrices(xy)
        means, variances = [], []
        remainder = {}
        for c in self.components:
            B, d = design[(c.scale, c.evolving, c.length)]
            means.append(np.log(100.0) - c.tail * d - c.spread * loss + B @ c.mean)
            factor = B @ c.chol
            variance = np.sum(factor * factor, axis=1)
            if observation:
                if c.length not in remainder:
                    r = self.domain.rho(xy)
                    rim2 = np.maximum(0.0, 1.0 - (r / self.domain.rho_max) ** 2)
                    n = self.local[c.length].averaging.shape[0]
                    resolved = np.sum(B[:, -n:] ** 2, axis=1) if n else 0.0
                    remainder[c.length] = np.maximum(0.0, (1.15 * rim2) ** 2 - resolved)
                variance += c.noise**2 + remainder[c.length]
            variances.append(variance)
        values = np.asarray(variances)
        if observation and log_sd is not None:
            values += np.asarray(log_sd) ** 2
        return Marginals(self.weights, np.asarray(means), values)

    def values(self, xy, probabilities):
        xy = np.asarray(xy)
        values = np.empty((len(probabilities), len(xy)))
        for start in range(0, len(xy), 512):
            block = slice(start, start + 512)
            distribution = self.distribution(xy[block])
            values[:, block] = [distribution.quantile(p) for p in probabilities]
        return values / np.log(10.0)

    def central(self, xy):
        return self.values(xy, (0.5,))[0]

    def reduced(self, count):
        return CoherentDraws(self, self.sample_parameters(count, 41537))

    def sample_parameters(self, count, seed):
        dim = max(len(c.mean) for c in self.components)
        u = np.clip(
            qmc.Sobol(dim + 1, scramble=True, seed=seed).random_base2(int(np.ceil(np.log2(count))))[:count],
            1e-14,
            1.0 - 1e-14,
        )
        ids = np.searchsorted(self.weights.cumsum(), u[:, 0]).clip(max=len(self.weights) - 1)
        return [
            (
                i,
                ix,
                self.components[i].mean
                + ndtri(u[ix, 1 : 1 + len(self.components[i].mean)]) @ self.components[i].chol.T,
            )
            for i in np.unique(ids)
            if len(ix := np.flatnonzero(ids == i))
        ]

    def draw_fields(self, xy, parameters):
        matrices, loss = self.matrices(xy)
        vals = np.empty((sum(len(ix) for _, ix, _ in parameters), len(xy)))
        for i, ix, beta in parameters:
            c = self.components[i]
            B, d = matrices[(c.scale, c.evolving, c.length)]
            vals[ix] = np.log(100.0) - c.tail * d - c.spread * loss + beta @ B.T
        return vals

    def save(self, path):
        arrays = dict(
            protocol=SCHEMA,
            latitude=self.latitude,
            reach=self.reach,
            members=np.array(
                [
                    [c.weight, c.scale, c.evolving, c.length, c.tail, c.noise, c.spread]
                    for c in self.components
                ]
            ),
        )
        for k, local in self.local.items():
            arrays[f"locations_{k}"] = local.centers
            arrays[f"averages_{k}"] = local.averaging
        for i, c in enumerate(self.components):
            arrays[f"m{i}"] = c.mean
            arrays[f"l{i}"] = c.chol
        np.savez_compressed(path, **arrays)

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as z:
            if str(z["protocol"]) != SCHEMA:
                raise ValueError("Unrecognized directional posterior")
            parts = [
                Component(w, s, int(e), l, t, n, loss, z[f"m{i}"], z[f"l{i}"])
                for i, (w, s, e, l, t, n, loss) in enumerate(z["members"])
            ]
            local = {
                c.length: LocalBasis(z[f"locations_{c.length}"], c.length, z[f"averages_{c.length}"])
                for c in parts
            }
            return cls(float(z["latitude"]), local, parts, z["reach"])


@dataclass
class CoherentDraws:
    model: DirectionalField
    parameters: list

    def field(self, xy):
        return self.model.draw_fields(xy, self.parameters)
