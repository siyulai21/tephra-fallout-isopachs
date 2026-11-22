"""Design-conditional geographic reach for posterior nested thickness fields.

Positive findings locate potential transport corridors. A finite upper limit
affects their posterior likelihood but never supplies a positive corridor.
"""
from dataclasses import dataclass

import numpy as np
from scipy.special import logsumexp

from ...core.distributions import EmpiricalMarginals
from .continuation import Continued, join
from .geometry import positions
from .inference import log_events, probabilities
from .profile import parameters

PROTOCOL = "nested-corridor-reach-v4"
WIDTHS = (12., 30., 60.)
EXTENSIONS = (.25, .5, 1.)
LAWS = ((None, None), *((w, e) for w in WIDTHS for e in EXTENSIONS))
# Equal odds for each finite width, independently equal odds for each length.
# Broad continuation is a separate structural possibility at every eruption.
PRIOR = np.r_[.25, np.full(9, .75 / 9)]
PROBS = (.025, .5, .975)


@dataclass
class Corridors:
    radii: np.ndarray
    bearings: np.ndarray
    proximal: float

    @classmethod
    def from_training(cls, data, latitude):
        positive = ~data.censored | data.lower_censored
        radius, theta = positions(data.xy[positive], latitude)
        # Repeated sections at one site must not increase its geometric weight.
        sites = np.unique(np.column_stack((radius, theta)), axis=0)
        sites = sites[sites[:, 0] > 0.]
        if not len(sites):
            raise ValueError("At least one positive finding away from the vent is needed for a corridor")
        # Resolution of the positive proximal survey. The median of up to the
        # three closest distinct sites prevents one vent-adjacent pit fixing it.
        nearest = np.sort(sites[:, 0])[:3]
        return cls(sites[:, 0], sites[:, 1], float(np.median(nearest)))

    def penalties(self, radius, bearing):
        angle = np.asarray(bearing)[:, None] - self.bearings
        r = np.asarray(radius)[:, None]
        along = r * np.cos(angle)
        projected = np.clip(along, 0., self.radii)
        cross = (r * np.sin(angle)) ** 2 + np.minimum(along, 0.) ** 2
        cross /= (self.proximal + projected) ** 2
        beyond = (np.maximum(along - self.radii, 0.) / (self.proximal + self.radii)) ** 2
        penalties = [np.zeros(len(r))]
        for width in WIDTHS:
            transverse = cross / np.tan(np.deg2rad(width)) ** 2
            for extension in EXTENSIONS:
                cost = np.min(transverse + beyond / extension ** 2, axis=1)
                # Half-thickness at unit scaled displacement, positive polynomial wings.
                penalties.append(-2 * np.log1p((np.sqrt(2.) - 1.) * cost))
        return np.array(penalties)


@dataclass
class ReachDraws:
    parent: Continued
    law: np.ndarray
    corridors: Corridors

    @property
    def latitude(self):
        return self.parent.latitude

    @property
    def z(self):
        return self.parent.z

    def subset(self, index):
        return ReachDraws(self.parent.subset(index), self.law[index], self.corridors)

    def reduced(self, count):
        size = len(self.law)
        ids = ((np.arange(min(count, size)) + .5) * size / min(count, size)).astype(int)
        return self.subset(ids)

    def field(self, xy):
        radius, bearing = positions(xy, self.latitude)
        return self.parent.field(xy) + self.corridors.penalties(radius, bearing)[self.law]

    def distribution(self, xy, observation=False, log_sd=None, calibrated=False):
        if observation:
            raise ValueError("Use score for the predictive observation model")
        return EmpiricalMarginals(self.field(xy))

    def score(self, data):
        logp = log_events(self.field(data.xy), parameters(self.z)[-1], data)
        return logsumexp(logp, axis=0) - np.log(len(self.law))

    def save(self, path):
        np.savez_compressed(path, **vars(self.parent), law=self.law,
                            corridor_radii=self.corridors.radii, corridor_bearings=self.corridors.bearings,
                            proximal=self.corridors.proximal)

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as f:
            model = Continued(*(f[k] for k in ("z", "geometry", "power_tail")), float(f["latitude"]),
                              *(f[k] for k in ("edge", "scale", "donor", "transfer")))
            return cls(model, f["law"], Corridors(f["corridor_radii"], f["corridor_bearings"], float(f["proximal"])))


def condition(proposal, train, *, positive_proposal=False, seed=71183, keep=4096):
    """Bayesian update on complete fields with all actual bounds, one event per row.

    A full-data parent is an importance proposal. Divide out its *intrinsic*
    upper-limit likelihood. Alternatively use a positive-sites-only proposal,
    for which the upper-limit likelihood has never entered the fit.
    """
    positive = ~train.censored | train.lower_censored
    observed = train.subset(positive)
    corridors = Corridors.from_training(train, proposal.latitude)
    parent = join(proposal, observed, donor=proposal.donor, transfer=proposal.transfer)
    # All positive reports, including lower limits and intervals, retain their
    # exact intrinsic likelihood by construction. Only upper events need updating.
    if not np.allclose(corridors.penalties(*positions(observed.xy, parent.latitude)), 0., atol=1e-12):
        raise ArithmeticError("A positive training site lost its corridor")
    n = len(parent.z)
    ratio = np.zeros((n, len(LAWS)))
    if not positive.all():
        from .inference import Deposit
        bounds = train.subset(~positive)
        base = parent.field(bounds.xy)
        noise = parameters(parent.z)[-1]
        penalties = corridors.penalties(*positions(bounds.xy, parent.latitude))
        if not positive_proposal:
            intrinsic = Deposit(proposal.z, proposal.geometry, proposal.power_tail, proposal.latitude)
            old = log_events(intrinsic.field(bounds.xy), noise, bounds).sum(axis=1)
        else:
            old = np.zeros(n)
        for j, adjustment in enumerate(penalties):
            ratio[:, j] = log_events(base + adjustment, noise, bounds).sum(axis=1) - old
    log_joint = ratio + np.log(PRIOR)
    indices, laws, checks = [], [], []
    for i, ids in enumerate(np.array_split(np.arange(n), 4)):
        log_weight = logsumexp(log_joint[ids], axis=1)
        w = probabilities(log_weight)
        ess = float(1. / (w @ w))
        weights = probabilities(log_joint[ids].ravel())
        rng = np.random.default_rng(np.random.SeedSequence([seed, i]))
        # Draw independently inside each stratum. A shared offset aliases the
        # law index when uniform parent weights repeat in regular blocks.
        count = min(keep // 4, len(ids))
        cdf = np.cumsum(weights)
        cdf[-1] = 1.
        chosen = np.searchsorted(cdf, (np.arange(count) + rng.random(count)) / count)
        chosen = rng.permutation(chosen)
        sample, which = np.divmod(chosen, len(LAWS))
        indices.append(ids[sample])
        laws.append(which)
        checks.append({"parent_particles":len(ids),"effective_parent_draws":ess,
                       "relative_ess":ess / len(ids),"maximum_weight":float(w.max()),
                       "log_mean_likelihood_ratio":float(logsumexp(log_weight) - np.log(len(ids))),
                       "law_probabilities":weights.reshape(len(ids), -1).sum(axis=0).tolist(),
                       "passed":bool(ess >= 64 and ess / len(ids) >= .02 and w.max() < .05)})
    model = ReachDraws(parent.subset(np.concatenate(indices)), np.concatenate(laws), corridors)
    return model, {"proposal":"positive_only" if positive_proposal else "all_accepted",
                   "positive_rows":train.row_ids[positive].tolist(), "upper_rows":train.row_ids[~positive].tolist(),
                   "proximal_resolution_km":corridors.proximal, "populations":checks,
                   "passed":all(c["passed"] for c in checks)}
