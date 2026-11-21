"""Transfer the remaining mass of independently mapped falls beyond the observed area ranks."""
from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
from scipy.special import gammainc, betainc, ndtri

from .geometry import DIAMETER, positions, ranks
from .inference import Deposit, fit
from .profile import ASSET, budget_parameters, contour_radius, integrated, log_profile
from .study import digest, engine_digest, load_draws, write_json

PROTOCOL = "nested-observed-range-v3"
QUANTILES = (.025,.5,.975)


def donor_curves():
    groups = {}
    for name,unit,k,k1,B in json.loads(ASSET.read_text())["rows"]:
        groups.setdefault(name,[]).append(np.log([k*B,k1*B]))
    return tuple(groups),np.array([np.exp(np.mean(x,axis=0)) for x in groups.values()])


NAMES,CURVES = donor_curves()


def residual_scaled(loss,b,c):
    """Ratio of remaining to interior donor volume, divided by relative edge thickness."""
    u = np.where(loss<=b,loss/b,1+(loss-b)/c)
    first = gammainc(2.,b*np.minimum(u,1.))/b**2
    w = np.maximum(u-1.,0.)
    inside = first+np.exp(-b)*(-np.expm1(-c*w)/c+gammainc(2.,c*w)/c**2)
    w = np.maximum(1-u,0.)
    rem = u*(-np.expm1(-b*w))/b+gammainc(2.,b*w)/b**2+np.exp(-b*w)*(1/c+1/c**2)
    return np.where(u>=1,u/c+1/c**2,rem)/np.maximum(inside,1e-300)


def scaled_moments(width,s,p,tail):
    """Finite radial moments about the edge, with distances in units of edge radius."""
    w = np.maximum(width,0.)
    a = s*(p-1)
    u = w/s
    first = -s*np.expm1(-(p-1)*np.log1p(w/a))
    second = s**2*((p-1)/(p-2))*betainc(2.,p-2,w/(a+w))
    exp_m = -s*np.expm1(-u)+s**2*gammainc(2.,u)
    return np.where(u<1e-10,w+w*w/2,np.where(tail,first+second,exp_m))


@dataclass
class Continued(Deposit):
    edge: np.ndarray
    scale: np.ndarray
    donor: np.ndarray
    transfer: np.ndarray

    def subset(self,idx):
        return Continued(*(getattr(self,k)[idx] for k in ("z","geometry","power_tail")),self.latitude,
                         *(getattr(self,k)[idx] for k in ("edge","scale","donor","transfer")))

    def log_at(self,r):
        inside = log_profile(r,self.z,self.power_tail)
        edge_h = log_profile(self.edge[:,None],self.z,self.power_tail)
        p = 2+np.exp(self.z[:,5,None])
        w = np.maximum(r/self.edge[:,None]-1,0.)/self.scale[:,None]
        outer = edge_h-np.where(self.power_tail[:,None],p*np.log1p(w/(p-1)),w)
        return np.where(r<=self.edge[:,None],inside,outer)

    def field(self,xy):
        geom,ids = np.unique(self.geometry,return_inverse=True)
        r = ranks(*positions(xy,self.latitude),geom)
        return self.log_at(r[ids])

    def volume_at(self,r):
        core = integrated(self.z,self.power_tail,np.minimum(r,self.edge))
        edge_h = log_profile(self.edge[:,None],self.z,self.power_tail)[:,0]
        p = 2+np.exp(self.z[:,5])
        outside = scaled_moments(np.asarray(r)/self.edge-1,self.scale,p,self.power_tail)
        return core+2*np.pi*1e-5*self.edge**2*np.exp(edge_h)*outside

    def volume(self):
        return self.volume_at(DIAMETER)

    def radius_at(self,cm):
        inner = contour_radius(self.z,self.power_tail,cm)
        edge_h = log_profile(self.edge[:,None],self.z,self.power_tail)[:,0]
        loss = np.maximum(edge_h-np.log(cm),0.)
        p = 2+np.exp(self.z[:,5])
        a = self.scale*(p-1)
        w = np.where(self.power_tail,a*np.expm1(np.minimum(loss/p,np.log1p(DIAMETER/(self.edge*a)))),self.scale*loss)
        return np.where(np.log(cm)>=edge_h,inner,np.minimum(DIAMETER,self.edge*(1+w)))


def join(model,training,seed=97231,tilt=1.,donor=None,transfer=None):
    geom,ids = np.unique(model.geometry,return_inverse=True)
    edge = ranks(*positions(training.xy,model.latitude),geom).max(axis=1)[ids]
    n = len(edge)
    if donor is None:
        rng = np.random.default_rng(seed)
        donor = rng.permutation(np.arange(n)%len(CURVES))
        transfer = np.exp(np.log(2.)*ndtri((rng.permutation(n)+.5)/n))
    b,c = CURVES[donor].T
    h = np.log(100.)+2.8*model.z[:,0]
    at = log_profile(edge[:,None],model.z,model.power_tail)[:,0]
    inside = integrated(model.z,model.power_tail,edge)
    scaled = residual_scaled(np.maximum(h-at,0.),b,c)
    logu = np.log(scaled)+np.log(inside)+np.log(transfer*tilt)-np.log(2*np.pi*1e-5)-2*np.log(edge)-h
    p = 2.+np.exp(model.z[:,5])
    logc = np.where(model.power_tail,np.log1p(1/(p-2)),0.)
    logs = np.log(2.)+logu-np.logaddexp(0.,.5*np.logaddexp(0.,np.log(4.)+logc+logu))
    scale = np.exp(np.clip(logs,-300.,300.))
    return Continued(model.z,model.geometry,model.power_tail,model.latitude,edge,scale,donor,transfer)


def load_packet(path):
    with np.load(path,allow_pickle=False) as a:
        if "edge" in a:
            return Continued(a["z"],a["geometry"],a["power_tail"],float(a["latitude"]),
                             a["edge"],a["scale"],a["donor"],a["transfer"])
    return load_draws(path)


def outer_status(model):
    q = np.array([np.quantile(x,QUANTILES) for x in np.array_split(model.volume(),4)])
    span = np.ptp(np.log(q),axis=0)
    return {"population_volume_quantiles_km3":q.tolist(),
            "passed":bool(span[1]<=np.log(1.35) and max(span[[0,2]])<=np.log(2.))}


def check_and_refine(raw,train,numerics,*,seed=97231,ceiling=12288):
    """Refine both the fitted and continued volume functional under one catalogue rule."""
    attempts = list(numerics["attempts"])
    checks = []
    while True:
        n = attempts[-1]["particles_per_population"]
        model = join(raw,train,seed)
        checked = outer_status(model)
        checks.append(dict(particles_per_population=n,**checked))
        parent_passed = attempts[-1]["passed"]
        if (checked["passed"] and parent_passed) or n>=ceiling:
            break
        n = min(2*n,ceiling)
        raw,more = fit(train,raw.latitude,initial=n,ceiling=n,seed=numerics["seed"])
        attempts.extend(more["attempts"])
    status = dict(numerics,attempts=attempts,parent_passed=parent_passed,continuation=checked,
                  continued_attempts=checks,ceiling=ceiling,passed=parent_passed and checked["passed"])
    return raw,model,status


def continuation_digest():
    return sha256((engine_digest()+digest(__file__)).encode()).hexdigest()


def from_parent(data,meta,base,home):
    """The piecewise continuation is exactly the parent at every training location."""
    home.mkdir(parents=True,exist_ok=True)
    geo = json.loads((base/"geographic.json").read_text())
    revised = []
    signature = continuation_digest()
    for i in [None,*[f["fold"] for f in geo["folds"]]]:
        key = "full" if i is None else f"fold-{i}"
        src = json.loads((base/(key+".json")).read_text())
        identity = src["identity"]
        if identity["input_sha256"]!=meta["input_sha256"] or identity["engine_sha256"]!=engine_digest():
            raise ValueError("Parent profile or accepted data differ from this continuation")
        p = base/(key+".npz")
        if digest(p)!=src["posterior_sha256"]:
            raise ValueError("Parent posterior digest differs")
        train = data.subset(np.isin(data.row_ids,identity["training_rows"]))
        test = data.subset(np.isin(data.row_ids,identity["test_rows"])) if i is not None else None
        raw = load_draws(p)
        raw,model,num = check_and_refine(raw,train,src["numerics"],seed=97231+(i or 0))
        if not np.allclose(raw.field(train.xy),model.field(train.xy),rtol=0,atol=1e-10):
            raise ArithmeticError("Terminal continuation altered a training observation")
        outfile = home/(key+".npz")
        np.savez_compressed(outfile,**vars(model))
        row = {"identity":dict(identity,protocol=PROTOCOL,engine_sha256=signature),"numerics":num,
               "posterior_sha256":digest(outfile),"starting_parent_posterior_sha256":digest(p),
               "unrestricted_km3":np.quantile(raw.volume(),QUANTILES).tolist()}
        if test is not None:
            scores = model.score(test)
            row.update(row_log_scores=scores.tolist(),negative_log_score=float(-scores.mean()),
                       median_log_cm=np.median(model.field(test.xy),axis=0).tolist(),
                       unrestricted_negative_log_score=float(-raw.score(test).mean()))
            revised.append(dict(fold=i,**row))
        write_json(home/(key+".json"),row)
    result = dict(geo,folds=revised,equal_region_negative_log_score=float(np.mean([f["negative_log_score"] for f in revised])))
    write_json(home/"geographic.json",result)
    return result


def radius_for(model,cm):
    return model.radius_at(cm) if isinstance(model,Continued) else contour_radius(model.z,model.power_tail,cm)


def volume_for(model,r):
    return model.volume_at(r) if isinstance(model,Continued) else integrated(model.z,model.power_tail,r)


def log_for(model,r):
    return model.log_at(r) if isinstance(model,Continued) else log_profile(r,model.z,model.power_tail)


def slab_for(model,cm):
    r = radius_for(model,cm)
    return np.maximum(0.,volume_for(model,r)-1e-5*cm*np.pi*r**2)
