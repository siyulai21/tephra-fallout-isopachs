"""Tempered sequential Monte Carlo for joint area profiles and geographic rankings."""
from dataclasses import dataclass

import numpy as np
from scipy.special import log_ndtr, logsumexp

from ...core.distributions import EmpiricalMarginals
from .geometry import ANGLES, SHAPES, geometry_probabilities, positions, ranks
from .profile import integrated, log_profile, parameters


def probabilities(log_w):
    return np.exp(log_w-logsumexp(log_w))


def log_events(mu, noise, data):
    """One exact log-density or documented one-sided event per accepted row."""
    scores = []
    for multiplier, proportion in ((1., .95), (3., .05)):
        sd = np.hypot(multiplier*noise[:, None], data.log_sd)
        residual = (data.log_value-mu)/sd
        event = -.5*residual**2-np.log(sd)-.5*np.log(2*np.pi)
        upper = data.censored & ~data.lower_censored
        event[:,upper] = log_ndtr(residual[:,upper])
        event[:,data.lower_censored] = log_ndtr(-residual[:,data.lower_censored])
        upper_values = getattr(data,"interval_log_upper",None)
        if upper_values is not None:
            mask = np.isfinite(upper_values)
            lo = residual[:,mask]
            hi = (upper_values[mask]-mu[:,mask])/sd[:,mask]
            # Subtract in the lower tail, reflecting positive intervals by symmetry.
            left = np.where(lo>0,-hi,lo)
            right = np.where(lo>0,-lo,hi)
            log_right = log_ndtr(right)
            with np.errstate(divide="ignore"):
                event[:,mask] = log_right+np.log(-np.expm1(log_ndtr(left)-log_right))
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
        unique, inverse = np.unique(self.geometry, return_inverse=True)
        r = ranks(*positions(xy, self.latitude), unique)
        return log_profile(r[inverse], self.z, self.power_tail)

    def distribution(self, xy, observation=False, log_sd=None, calibrated=False):
        if observation:
            raise ValueError("Use score() for the robust observation distribution")
        return EmpiricalMarginals(self.field(xy))

    def score(self, data):
        logp = log_events(self.field(data.xy), parameters(self.z)[-1], data)
        return logsumexp(logp,axis=0)-np.log(len(self.z))

    def reduced(self, count):
        idx = np.minimum(((np.arange(count)+.5)*len(self.z)/count).astype(int),len(self.z)-1)
        return self.subset(idx)

    def volume(self):
        return integrated(self.z, self.power_tail)


def population(data, latitude, radius, n, seed, power_tail, budget):
    rng = np.random.default_rng(seed)
    geom_prior = geometry_probabilities()
    dim = 6 if budget else 5
    z = rng.normal(size=(n,dim))
    if budget:
        z[:,3] = rng.standard_t(3,size=n)
    geometry = rng.choice(len(geom_prior), size=n, p=geom_prior)
    tail = np.full(n,power_tail,dtype=bool)

    def likelihood(coords, geom, outer):
        mu = log_profile(radius[geom], coords, outer)
        return log_events(mu, np.exp(np.log(.55)+.65*coords[:,4]), data).sum(axis=1)

    ll = likelihood(z, geometry, tail)
    log_evidence = 0.
    beta = 0.
    history = []
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
        for _ in range(7):
            proposed = z + .85*rng.normal(size=z.shape)@chol.T
            lp = likelihood(proposed,geometry,tail)
            change = -.5*np.sum(proposed**2-z**2,axis=1)
            if budget:
                # For the distal log odds, four independent donor volcanoes
                # give a Student-t predictive prior with three degrees of freedom.
                change += .5*(proposed[:,3]**2-z[:,3]**2)-2*(np.log1p(proposed[:,3]**2/3)-np.log1p(z[:,3]**2/3))
            take = np.log(rng.random(n)) < beta*(lp-ll)+change
            z[take], ll[take] = proposed[take],lp[take]
            accepted[0] += take.mean()/7
            # Each of these kernels is reversible under the finite geometry prior.
            near = (geometry//ANGLES)*ANGLES+(geometry%ANGLES+rng.integers(-4,5,n))%ANGLES
            global_draw = rng.choice(len(geom_prior),size=n,p=geom_prior)
            mode = rng.random(n)
            same_bearing = (global_draw//ANGLES)*ANGLES+geometry%ANGLES
            geom = np.where(mode<.45, near, np.where(mode<.80, same_bearing, global_draw))
            lp = likelihood(z,geom,tail)
            take = np.log(rng.random(n)) < beta*(lp-ll)
            geometry[take],ll[take] = geom[take],lp[take]
            accepted[1] += take.mean()/7
        history.append(dict(beta=float(beta),continuous_acceptance=accepted[0],shape_acceptance=accepted[1]))
    # Random ordering makes display subsets independent of resampling ancestry order.
    idx = rng.permutation(n)
    return Deposit(z[idx],geometry[idx],tail[idx],latitude), history, log_evidence


def fit(data, latitude, *, initial=768, ceiling=6144, populations=4, seed=213091, budget=True):
    geom = np.arange(ANGLES*len(SHAPES))
    radius = ranks(*positions(data.xy,latitude),geom)
    n = initial
    attempts = []
    while True:
        draws, histories, evidence = [], [], []
        for i in range(populations):
            branches = [population(data,latitude,radius,n//2,
                        np.random.SeedSequence([seed,n,i,b]),bool(b),budget) for b in (0,1)]
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
