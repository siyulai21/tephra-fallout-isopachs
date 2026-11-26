"""Tempered sequential Monte Carlo for joint area profiles and geographic rankings."""
from dataclasses import dataclass

import numpy as np
from scipy.special import logsumexp

from ....core.likelihood import observation_log_event

from ....core.distributions import EmpiricalMarginals
from ..geometry import ANGLES, positions
from .growth import SHAPES, prior as geometry_probabilities, ranks
from ..profile import integrated, log_profile, parameters
from .flow import TrainingRanks
from . import numerics
from ..profile import budget_parameters


def probabilities(log_w):
    return np.exp(log_w-logsumexp(log_w))


def log_events(mu, noise, data):
    """One log-density or documented bound or interval event per accepted row."""
    scores = []
    for multiplier, proportion in ((1., .95), (3., .05)):
        sd = np.hypot(multiplier*noise[:, None], data.log_sd)
        event = observation_log_event(data, mu, sd)
        scores.append(np.log(proportion)+event)
    return np.logaddexp(*scores)


def systematic(weights, rng):
    cdf = weights.cumsum()
    cdf[-1] = 1.
    return np.searchsorted(cdf, (np.arange(len(weights))+rng.random())/len(weights))


@dataclass
class Deposit:
    z: np.ndarray
    geometry: np.ndarray
    power_tail: np.ndarray
    latitude: float

    def subset(self, idx):
        return Deposit(self.z[idx], self.geometry[idx], self.power_tail[idx], self.latitude)

    def field(self, xy):
        from .flow import ranks as flow_ranks
        r = flow_ranks(*positions(xy, self.latitude), self.geometry, self.z[:,6:])
        return log_profile(r, self.z[:,:6], self.power_tail)

    def distribution(self, xy, observation=False, log_sd=None, calibrated=False):
        if observation:
            raise ValueError("Use score() for the robust observation distribution")
        return EmpiricalMarginals(self.field(xy))

    def score(self, data):
        logp = log_events(self.field(data.xy), parameters(self.z[:,:6])[-1], data)
        return logsumexp(logp,axis=0)-np.log(len(self.z))

    def reduced(self, count):
        idx = np.minimum(((np.arange(count)+.5)*len(self.z)/count).astype(int),len(self.z)-1)
        return self.subset(idx)

    def volume(self):
        return integrated(self.z[:,:6], self.power_tail)


def population(data, latitude, radius, n, seed, power_tail, budget, *, geometry_prior=None):
    rng = np.random.default_rng(seed)
    geom_prior = geometry_probabilities() if geometry_prior is None else geometry_prior
    # Conditional catalogue updates change one coordinate at a time. The
    # uniform Cartesian products within each family make this reversible.
    neighbours = []
    for a in SHAPES:
        va = vars(a)
        neighbours.append(np.array([j for j,b in enumerate(SHAPES)
                           if a.family == b.family and set(va) == set(vars(b))
                           and sum(va[k] != vars(b)[k] for k in va) <= 1], dtype=int))
    degree = np.array([len(x) for x in neighbours])
    neighbour_table = np.zeros((len(neighbours),degree.max()),dtype=int)
    for i, adjacent in enumerate(neighbours):
        neighbour_table[i,:len(adjacent)] = adjacent
    dim = 8
    z = rng.normal(size=(n,dim))
    if budget:
        z[:,3] = rng.standard_t(3,size=n)
    geometry = rng.choice(len(geom_prior), size=n, p=geom_prior)
    tail = np.full(n,power_tail,dtype=bool)

    def likelihood(coords, geom, outer):
        return numerics.log_events(radius.at(coords,geom),np.array(budget_parameters(coords[:,:6],outer)),outer,
                                   data.log_value,data.log_sd,data.censored,data.lower_censored,data.interval_log_upper)

    # Training-only rank transport: a geometry change and its exact reverse
    # translate the profile length oppositely (unit Jacobian). This avoids
    # rejecting a new shape merely because its area scale differs.
    positive = np.any(radius.base > 0., axis=0)
    rank_scale = np.mean(np.log(np.maximum(radius.base[:, positive], 1e-10)), axis=1)
    transport_points = np.unique(np.linspace(0,len(data.xy)-1,min(32,len(data.xy))).astype(int))
    transport_radius = radius.subset(transport_points)
    ll = likelihood(z, geometry, tail)
    log_evidence = 0.
    beta = 0.
    history = []
    # Geometry modes have different thinning/angle correlations. Updating all
    # eight coordinates with one global covariance can almost freeze the chain.
    # Both blocks use symmetric proposals and the exact prior ratio. Adapt only
    # between tempering stages, then hold each proposal fixed for its MH moves.
    step = np.array([.65,.5])
    while beta < 1.:
        hi, lo = 1., beta
        for _ in range(24):
            prop = hi if _ == 0 else (lo+hi)/2
            w = probabilities((prop-beta)*ll)
            if 1/(w@w) >= .65*n:
                lo = prop
                if prop == 1.:
                    break
            else:
                hi = prop
        if lo <= beta:
            raise ArithmeticError("Particle tempering could not advance")
        log_increment = (lo-beta)*ll
        log_evidence += logsumexp(log_increment)-np.log(n)
        w = probabilities(log_increment)
        beta = lo
        idx = systematic(w, rng)
        z, geometry, tail, ll = z[idx],geometry[idx],tail[idx],ll[idx]
        chol = np.linalg.cholesky(np.cov(z.T)+.015*np.eye(dim))
        accepted = [0.,0.]
        blocks = np.zeros(2)
        counts = np.zeros(2)
        warp_chol = np.linalg.cholesky(np.cov(z[:,6:].T)+.015*np.eye(2))
        for move in range(24):
            block = int(move % 4 == 3)
            proposed = z.copy()
            if block:
                proposed[:,6:] += step[1]*rng.normal(size=(n,2))@warp_chol.T
            else:
                proposed[:,:6] += step[0]*rng.normal(size=(n,6))@chol[:6,:6].T
            lp = likelihood(proposed,geometry,tail)
            change = -.5*np.sum(proposed**2-z**2,axis=1)
            if budget:
                # For the distal log odds, four independent donor volcanoes
                # give a Student-t predictive prior with three degrees of freedom.
                change += .5*(proposed[:,3]**2-z[:,3]**2)-2*(np.log1p(proposed[:,3]**2/3)-np.log1p(z[:,3]**2/3))
            take = np.log(rng.random(n)) < beta*(lp-ll)+change
            z[take], ll[take] = proposed[take],lp[take]
            accepted[0] += take.mean()/24
            blocks[block] += take.mean()
            counts[block] += 1
            # Each of these kernels is reversible under the finite geometry prior.
            near = (geometry//ANGLES)*ANGLES+(geometry%ANGLES+rng.integers(-4,5,n))%ANGLES
            global_draw = rng.choice(len(geom_prior),size=n,p=geom_prior)
            mode = rng.random(n)
            same_bearing = (global_draw//ANGLES)*ANGLES+geometry%ANGLES
            shape_ids = geometry//ANGLES
            choice = (rng.random(n)*degree[shape_ids]).astype(int)
            local_shape = neighbour_table[shape_ids,choice]*ANGLES+geometry%ANGLES
            geom = np.where(mode<.25, near, np.where(mode<.75,local_shape,np.where(mode<.95,same_bearing,global_draw)))
            hastings = np.where((mode>=.25)&(mode<.75),
                                np.log(degree[geometry//ANGLES]/degree[geom//ANGLES])
                                +np.log(geom_prior[geom]/geom_prior[geometry]),0.)
            transported = z.copy()
            transported[:,1] += (rank_scale[geom]-rank_scale[geometry])/1.6
            old_profile = log_profile(transport_radius.at(z,geometry),z[:,:6],tail)
            new_profile = log_profile(transport_radius.at(transported,geom),transported[:,:6],tail)
            transported[:,0] += np.median(old_profile-new_profile,axis=1)/2.8
            lp = likelihood(transported,geom,tail)
            prior_change = -.5*np.sum(transported[:,:2]**2-z[:,:2]**2,axis=1)+hastings
            take = np.log(rng.random(n)) < beta*(lp-ll)+prior_change
            geometry[take],ll[take],z[take] = geom[take],lp[take],transported[take]
            accepted[1] += take.mean()/24
        rates = blocks/counts
        history.append(dict(beta=float(beta),continuous_acceptance=accepted[0],shape_acceptance=accepted[1],
                            block_acceptance=rates.tolist(),proposal_scale=step.tolist()))
        step = np.clip(step*np.exp(2*(rates-.28)),.03,1.2)
    # Random ordering makes display subsets independent of resampling ancestry order.
    idx = rng.permutation(n)
    return Deposit(z[idx],geometry[idx],tail[idx],latitude), history, log_evidence


def fit(data, latitude, *, initial=768, ceiling=6144, populations=4, seed=213091, budget=True,
        ranker=ranks, geometry_prior=None):
    geom = np.arange(ANGLES*len(SHAPES) if geometry_prior is None else len(geometry_prior))
    radius = TrainingRanks(*positions(data.xy,latitude))
    n = initial
    attempts = []
    while True:
        draws, histories, evidence = [], [], []
        for i in range(populations):
            branches = [population(data,latitude,radius,n//2,
                        np.random.SeedSequence([seed,n,i,b]),bool(b),budget,geometry_prior=geometry_prior) for b in (0,1)]
            log_z = np.array([b[2] for b in branches])
            branch_p = probabilities(log_z)
            if not np.isfinite(branch_p).all():
                raise ArithmeticError("Nonfinite marginal evidence for a distal thinning law")
            base = [b[0] for b in branches]
            both = Deposit(*(np.concatenate([getattr(d,k) for d in base]) for k in
                           ("z","geometry","power_tail")),latitude)
            rng = np.random.default_rng(np.random.SeedSequence([seed,n,i,2]))
            weights = np.repeat(branch_p/(n//2), n//2)
            idx = systematic(weights,rng)
            draws.append(both.subset(rng.permutation(idx)))
            evidence.append(log_z.tolist())
            histories.append([b[1] for b in branches])
        q = np.array([np.quantile(x.volume(),[.025,.5,.975]) for x in draws])
        spread = np.ptp(np.log(q),axis=0)
        passed = bool(spread[1] <= np.log(1.35) and max(spread[[0,2]]) <= np.log(2.))
        attempts.append({"particles_per_population":n,"independent_populations":populations,
                         "population_volume_quantiles_km3":q.tolist(),"passed":passed,
                         "log_evidence_exponential_power":evidence,"tempering":histories})
        if passed or n>=ceiling:
            break
        n = min(2*n, ceiling)
    merged = Deposit(*(np.concatenate([getattr(x,k) for x in draws]) for k in
                       ("z","geometry","power_tail")), latitude)
    return merged, {"seed":seed, "attempts":attempts, "passed":passed}
