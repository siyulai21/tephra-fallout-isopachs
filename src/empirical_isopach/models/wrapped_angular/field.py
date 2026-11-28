"""Joint Gaussian coefficient posterior and conditional source-area realizations."""

from dataclasses import dataclass
import numpy as np
from ..directional.multiscale.posterior import Marginals
from .geometry import basis, transported_mean
from .source import coordinates, log_shape, taper
from .source_prior import sample_prior, prior_quantile


@dataclass
class Coefficients:
    design: tuple
    index: np.ndarray
    shape: np.ndarray
    beta: np.ndarray
    covariance: np.ndarray


class RegionalField:
    """Finite mixture after marginalizing geometry, attenuation, noise and censored beds."""

    def __init__(self, axis, weights, groups, covariance="wrapped"):
        self.axis, self.weights, self.groups = axis, weights, groups
        self.covariance = covariance

    @classmethod
    def from_fit(cls, fit):
        groups = []
        for gid, (index, shape, _, _, beta, cov) in fit.geometry.items():
            g = fit.groups[gid]
            groups.append(Coefficients((g.scale, g.amplitude, g.power), index, shape, beta, cov))
        return cls(fit.axis, fit.weights, groups, fit.covariance)

    def distribution(self, xy):
        means, designs = {}, {}
        mu = np.empty((len(self.weights), len(xy)))
        var = np.empty_like(mu)
        for g in self.groups:
            scale = g.design[0]
            if scale not in means:
                means[scale] = transported_mean(xy, self.axis, scale)
            if g.design not in designs:
                designs[g.design] = basis(xy, self.axis, *g.design, self.covariance)
            design = designs[g.design]
            mu[g.index] = means[scale][g.shape] + g.beta @ design.T
            var[g.index] = np.maximum(np.sum((design @ g.covariance) * design, axis=1), 1.0e-12)
        return Marginals(self.weights, mu, var)

    def centre_cross(self, xy, centre):
        cross = np.empty((len(self.weights), len(xy)))
        designs = {}
        for g in self.groups:
            if g.design not in designs:
                designs[g.design] = (
                    basis(centre, self.axis, *g.design, self.covariance),
                    basis(xy, self.axis, *g.design, self.covariance),
                )
            a, b = designs[g.design]
            cross[g.index] = (a @ g.covariance) @ b.T
        return cross

    def coherent(self, count=32, seed=31657):
        """A draw shares its shape and every coefficient at all geographic locations."""
        fields = []
        for pop in range(2):
            rng = np.random.default_rng(np.random.SeedSequence([seed, pop]))
            idx = np.searchsorted(self.weights.cumsum(), (np.arange(count) + rng.random(count)) / count).clip(
                max=len(self.weights) - 1
            )
            for g in self.groups:
                selected = np.flatnonzero(np.isin(idx, g.index))
                if not len(selected):
                    continue
                rows = np.searchsorted(g.index, idx[selected])
                eig, vec = np.linalg.eigh(g.covariance)
                if eig.min() < -1.0e-8:
                    raise ArithmeticError("Indefinite regional coefficient covariance")
                noise = rng.normal(size=(len(rows), len(eig))) @ (vec * np.sqrt(np.maximum(eig, 0.0))).T
                fields.append((pop, selected, g.design, g.shape[rows], g.beta[rows] + noise))

        return CoherentFields(self.axis, self.covariance, count, fields)


@dataclass
class CoherentFields:
    axis: float
    covariance: str
    count: int
    groups: list

    def __call__(self, xy):
        result = np.empty((2, self.count, len(xy)))
        means, designs = {}, {}
        for pop, selected, design, shape, beta in self.groups:
            scale = design[0]
            if scale not in means:
                means[scale] = transported_mean(xy, self.axis, scale)
            if design not in designs:
                designs[design] = basis(xy, self.axis, *design, self.covariance)
            result[pop, selected] = means[scale][shape] + beta @ designs[design].T
        return result


class Deposit:
    """Marginal thickness quantiles and coherent posterior fields on registered LAEA coordinates."""

    def __init__(self, regional, source, bank, params=None, centres=None, prior_only=False):
        self.regional, self.source, self.bank = regional, source, bank
        self.params, self.centres, self.prior_only = params, centres, prior_only

    def local(self, xy):
        if self.source is None or self.source["area_km2"] is None:
            return np.zeros(len(xy), bool)
        return np.linalg.norm(coordinates(xy, self.source), axis=1) < 2.0

    def quantiles(self, xy, probabilities=(0.25, 0.5, 0.75), *, prior_power=12, prior_seed=67957):
        regional = self.regional
        output = np.empty((len(probabilities), len(xy)))
        inner = self.local(xy)
        centre = None
        if self.prior_only and inner.any():
            centre = np.array([self.source["centroid_km"]])
            center_law = regional.distribution(centre)
            sampled = sample_prior(regional.weights, prior_power, prior_seed)
        for start in range(0, len(xy), 96):
            sl = slice(start, start + 96)
            points = xy[sl]
            law = regional.distribution(points)
            output[:, sl] = [law.quantile(q) for q in probabilities]
            indices = np.flatnonzero(inner[sl])
            if not len(indices):
                continue
            points = points[indices]
            location = coordinates(points, self.source)
            if self.prior_only:
                near = Marginals(law.weights, law.means[:, indices], law.variances[:, indices])
                radius = np.linalg.norm(location, axis=1)
                cross = regional.centre_cross(points, centre)
                output[:, start + indices] = [
                    prior_quantile(near, center_law, cross, radius, taper(radius, 2.0), sampled, q)
                    for q in probabilities
                ]
            else:
                z = self.bank(points)
                base = z.reshape(-1, len(points))
                n = self.params.shape[2]
                local = (
                    log_shape(self.params.reshape(-1, 5), location, 2.0)
                    + np.repeat(self.centres.ravel(), n)[:, None]
                )
                complete = np.logaddexp(np.repeat(base, n, axis=0), local)
                output[:, start + indices] += np.quantile(complete, probabilities, axis=0) - np.quantile(
                    base, probabilities, axis=0
                )
        return output

    def draws(self, xy, local_index=0):
        z = self.bank(xy)
        inner = self.local(xy)
        if inner.any():
            params = self.params[:, :, local_index].reshape(-1, 5)
            shape = log_shape(params, coordinates(xy[inner], self.source), 2.0)
            z[:, :, inner] = np.logaddexp(
                z[:, :, inner], self.centres[:, :, None] + shape.reshape(*z.shape[:2], -1)
            )
        return z
