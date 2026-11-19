"""A coherent azimuthal perturbation with conditional mean one on the sphere."""
import numpy as np
from scipy.special import i0e

from ..geometry import polar, TAU
from ..ensemble.fields import Ensemble, Marginal, draw_weights, latent_arrays, probabilities
from ....core.distributions import LogMixture


CONCENTRATIONS = np.array([0., 2., 4., 8.])
HALF_ANGLE = np.deg2rad(22.5)


class DirectionalSupport:
    """Distance to finite source-to-positive-site segments in the authalic chart."""
    def __init__(self, data, geometry):
        self.geometry = geometry
        scale = np.array([1/geometry.axis, geometry.axis])
        xy = data.xy*scale
        positive = np.unique(xy[~data.censored | data.lower_censored], axis=0)
        ranges = np.linalg.norm(positive, axis=1)
        self.lengths = ranges[ranges>0]
        if not len(self.lengths):
            raise ValueError("At least one positive thickness or lower-bound site away from the source is required")
        self.directions = positive[ranges>0]/self.lengths[:,None]
        self.upper_sites = np.unique(xy[data.censored & ~data.lower_censored], axis=0)
        from scipy.spatial import cKDTree
        unique = np.unique(xy[~data.censored],axis=0)
        distances = cKDTree(unique).query(unique,k=2)[0][:,1]
        self.spacing = float(np.median(distances))
        if not np.isfinite(self.spacing) or self.spacing<=0:
            raise ValueError("Directional support needs at least two distinct exact sites")
        self.outer = float(self.lengths.max())

    def evaluate(self, xy, bary=None):
        g = self.geometry
        radius, bearing = polar(xy,g.axis)
        radial = np.minimum(radius,self.outer)
        direction = np.column_stack((np.sin(bearing),np.cos(bearing)))
        target = radial[:,None]*direction
        along = np.clip(target@self.directions.T,0.,self.lengths)
        # Compute residuals directly to preserve zero distance on long station rays.
        diff = target[:,None,:]-along[:,:,None]*self.directions
        squared = np.min(np.sum(diff**2,axis=2),axis=1)
        if len(self.upper_sites):
            # An upper limit anchors its location. It supplies no positive-bearing ray.
            actual = radius[:,None]*direction
            off = actual[:,None,:]-self.upper_sites
            squared = np.minimum(squared,np.sum(off**2,axis=2).min(axis=1))
        width2 = self.spacing**2+(radial*np.sin(HALF_ANGLE))**2
        if bary is None:
            _,_,bary = g.locate(xy)
        free = np.clip(1.-bary.sum(axis=1),0.,1.)
        return free*squared/(squared+width2), bearing


def log_normalizer(a):
    return np.log(i0e(a))+np.abs(a)


def log_shift(k, exposure, theta, azimuth):
    a = np.asarray(k)*exposure
    return a*np.cos(theta-azimuth)-log_normalizer(a)


def stratified_states(members, concentration_weights, seed=20261119, replicate=0):
    """Randomized stratification separately within every chain and model.

    Each concentration occurs in each chain with its exact mass. Azimuth is
    stratified conditionally on concentration and shuffled independently of
    the MCMC order. One state describes the entire deposit.
    """
    state_k, phase, conditional_weights = [], [], []
    mix = probabilities(concentration_weights)
    for m, member in enumerate(members):
        chains,n = member.draws["source"].shape
        active = np.flatnonzero(mix>0)
        if n < 2*len(active):
            raise ValueError("Insufficient posterior draws to stratify all active concentrations")
        mass = (n-len(active))*mix[active]
        counts = 1+np.floor(mass).astype(int)
        rest = n-counts.sum()
        counts[np.argsort(-(mass-np.floor(mass)),kind="stable")[:rest]] += 1
        k = np.repeat(active,counts)
        w = np.repeat(mix[active]/(chains*counts),counts)
        azimuths,ks,weights = [],[],[]
        for c in range(chains):
            rng = np.random.default_rng(np.random.SeedSequence([seed,replicate,m,c]))
            angles = np.concatenate([TAU*(np.arange(count)+rng.random(count))/count for count in counts])
            order = rng.permutation(n)
            ks.append(k[order])
            azimuths.append(angles[order])
            weights.append(w[order])
        state_k.append(np.concatenate(ks))
        phase.append(np.concatenate(azimuths))
        conditional_weights.append(np.concatenate(weights))
    return np.concatenate(state_k),np.concatenate(phase),conditional_weights


class AzimuthField:
    def __init__(self, base, data, concentration_weights, *, replicate=0):
        self.base = base
        self.members = base.members
        self.support = DirectionalSupport(data,self.members[0].geometry)
        self.concentration_weights = probabilities(concentration_weights)
        self.k,self.phase,w = stratified_states(self.members,self.concentration_weights,replicate=replicate)
        self.weights = np.concatenate([x*y for x,y in zip(base.model_weights,w,strict=True)])
        self.sizes = base.sizes
        self.model_weights = base.model_weights
        self.median_grid_batch = max(32,min(512,500_000//sum(self.sizes)))

    def components(self,xy):
        arrays,bary = latent_arrays(self.members,xy)
        h,theta = self.support.evaluate(xy,bary)
        original = np.concatenate(arrays)
        revised = original + log_shift(CONCENTRATIONS[self.k,None],h,theta,self.phase[:,None])
        return original,revised,bary

    def distribution(self,xy,observation=False,log_sd=None,calibrated=False):
        _,values,bary = self.components(xy)
        if not observation:
            return Marginal(values,self.weights)
        local = np.maximum(0.,1-(bary*bary).sum(axis=1))
        variances = [np.exp(2*m.samples["log_noise"][:,None])*(1+m.geometry.policy.local_sd_ratio**2*local)
                     for m in self.members]
        known = 0. if log_sd is None else np.asarray(log_sd)**2
        return LogMixture(self.weights,values,np.concatenate(variances)+known)
