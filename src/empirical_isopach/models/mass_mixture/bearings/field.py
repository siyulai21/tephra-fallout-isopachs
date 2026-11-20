"""Training-design conditional continuation of a mass-normalized spherical mixture."""
import numpy as np
from scipy.spatial import cKDTree
import shapely

from ....support import SupportMesh
from ...directional.geometry import polar
from ..geometry import to_sphere
from ..inference import flatten
from ..geometry import log_field
from src.probabilistic_isopach.global_volume import EarthDomain


PROTOCOL = "spherical-mixture-bearing-continuation-v1"
POWERS = (0, 1, 4)
SEED = 204917
HALF_REACH = 22.5
TAU = 2*np.pi


class BearingSupport:
    """Finite positive-site reach, retained measurement triangles and local bound anchors."""
    def __init__(self, data, latitude, half_reach=HALF_REACH):
        self.latitude = latitude
        self.axis = EarthDomain.wgs84(latitude).axis_scale
        self.scale = np.array([1/self.axis, self.axis])
        exact = np.unique(data.xy[~data.censored]*self.scale, axis=0)
        self.spacing = float(np.median(cKDTree(exact).query(exact,k=2)[0][:,1]))
        if not 0 < self.spacing < np.inf or not 0 < half_reach < 90:
            raise ValueError("Bearing support needs distinct exact sites and a half-reach angle in (0,90)")
        upper = data.xy[data.censored & ~data.lower_censored]*self.scale
        self.upper = cKDTree(upper) if len(upper) else None
        positive = np.unique(data.xy[~data.censored | data.lower_censored]*self.scale,axis=0)
        radius = np.linalg.norm(positive,axis=1)
        positive,radius = positive[radius>0],radius[radius>0]
        self.outer = radius.max()
        self.half_reach = np.deg2rad(half_reach)
        self.power = np.log(.5)/np.log(np.cos(self.half_reach))
        self.vectors = np.vstack(([0.,0.],positive*(radius**(1/self.power-1))[:,None]))
        # The maximum projected length changes its winning vector only at a pairwise tie.
        i,j = np.triu_indices(len(self.vectors),1)
        delta = self.vectors[i]-self.vectors[j]
        cross = np.arctan2(-delta[:,1],delta[:,0])
        edge = np.unique(np.r_[0.,cross%TAU,(cross+np.pi)%TAU,TAU])
        mid = (edge[:-1]+edge[1:])/2
        winners = np.argmax(np.column_stack((np.sin(mid),np.cos(mid)))@self.vectors.T,axis=1)
        change = np.r_[True,winners[1:]!=winners[:-1]]
        self.edges,self.winners = edge[:-1][change],self.vectors[winners[change]]
        self.mesh = SupportMesh(data)
        self.polygon = self.mesh.geometry
        if not self.polygon.is_empty:
            self.bounds = np.asarray(self.polygon.bounds) + self.spacing*np.array([-1.,-1.,1.,1.])/min(self.scale)
            shapely.prepare(self.polygon)

    def reach(self, theta):
        theta = np.asarray(theta)%TAU
        v = self.winners[np.searchsorted(self.edges,theta,side="right")-1]
        return np.maximum(0.,v[...,0]*np.sin(theta)+v[...,1]*np.cos(theta))**self.power

    def evaluate(self, xy):
        xy = np.asarray(xy,dtype=float).reshape(-1,2)
        radius,theta = polar(xy,self.axis)
        r = np.minimum(radius,self.outer)
        deficit = np.maximum(r-self.reach(theta),0.)
        width2 = self.spacing**2+(r*np.sin(self.half_reach))**2
        exposure = deficit**2/(deficit**2+width2)
        if self.upper is not None:
            d = self.upper.query(xy*self.scale)[0]/self.spacing
            exposure *= np.minimum(1.,d)**2*(3.-2.*np.minimum(1.,d))
        if not self.polygon.is_empty:
            lo,hi = self.bounds[:2],self.bounds[2:]
            ids = np.flatnonzero((exposure>1e-15)&(xy>=lo).all(1)&(xy<=hi).all(1))
            if len(ids):
                d = shapely.distance(shapely.points(xy[ids]),self.polygon)/self.spacing
                t = np.minimum(1.,d)
                exposure[ids] *= t*t*(3.-2*t)
        return exposure,theta


def selected(samples, per_chain):
    chains,n = samples["log_volume"].shape
    index = np.unique(np.linspace(0,n-1,min(per_chain,n)).round().astype(int))
    indices = (np.arange(chains)[:,None]*n+index).ravel()
    flat = flatten(samples)
    return {k:v[indices] for k,v in flat.items()},indices,(chains,len(index))


def phases(shape, seed=SEED):
    """Stratification within each chain keeps orientation independent of MCMC order."""
    chains,n = shape
    phi = []
    for c in range(chains):
        rng = np.random.default_rng(np.random.SeedSequence([seed,c]))
        phi.extend(rng.permutation((np.arange(n)+rng.random(n))*TAU/n))
    return np.array(phi)


def multiplier(h, theta, phi, power, *, enhanced=False):
    if power == 0:
        return np.ones(np.broadcast_shapes(np.shape(h),np.shape(phi)))
    fan = (.5+.5*np.cos(theta-phi))**power
    if enhanced:
        from math import comb
        fan *= 4**power/comb(2*power,power)
    return 1-h+h*fan


class BearingField:
    def __init__(self, draws, support, phi):
        self.draws,self.support,self.phi = draws,support,np.asarray(phi)
        if len(phi)!=len(draws["log_volume"]):
            raise ValueError("Every retained field needs one coherent orientation")

    def components(self,xy):
        base = log_field(self.draws,to_sphere(xy,self.support.latitude))
        h,theta = self.support.evaluate(xy)
        return [base if p==0 else base+np.log(multiplier(h,theta,self.phi[:,None],p)) for p in POWERS]

    def distribution(self,xy,observation=False,log_sd=None,calibrated=False):
        from ....core.distributions import EmpiricalMarginals
        if observation:
            raise ValueError("Use the observation mixture for held-out site predictions")
        return EmpiricalMarginals(np.concatenate(self.components(xy)))
