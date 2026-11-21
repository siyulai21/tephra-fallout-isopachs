"""Nested thinning with an empirical prior on total distal ash relative to the core."""
import json

import numpy as np
from scipy.special import betainc, gammainc

from .geometry import DIAMETER


def analogue_budget(asset):
    groups = {}
    for name, unit, k, k1, B in json.loads(asset.read_text())["rows"]:
        b, c = k*B, k1*B
        core = gammainc(2.,b)/b**2
        odds = np.exp(-b)*(1/c+1/c**2)/core
        groups.setdefault(name,[]).append(np.log(odds))
    odds = np.array([np.mean(x) for x in groups.values()])
    return dict(zip(groups,np.exp(odds).tolist())),float(odds.mean()),float(odds.std(ddof=1)*np.sqrt(1+1/len(odds)))


def geometry(z, tail, core_centre, odds_centre, odds_spread):
    h = np.log(100.) + 2.8*z[:,0]
    length = np.exp(np.log(100.) + 1.6*z[:,1])
    drop = np.exp(core_centre + .9*z[:,2])
    exponent = 2.+np.exp(z[:,5])
    noise = np.exp(np.log(.55)+.65*z[:,4])
    # A single e-folding length is s. A Lomax tail has scale (p-1)*s.
    # In the reference area plane the two radial moments give s + C*s².
    log_c = np.where(tail,np.log1p(1/(exponent-2)),0.)
    log_u = odds_centre+odds_spread*z[:,3]+drop+np.log(gammainc(2.,drop))-2*np.log(drop)
    log_s = np.log(2.)+log_u-np.logaddexp(0.,.5*np.logaddexp(0.,np.log(4.)+log_c+log_u))
    # At these representational extremes the entire physical disk has either
    # constant distal thickness or negligible distal thickness.
    s = np.exp(np.clip(log_s,-300.,300.))
    return h,length,drop,s,exponent,noise


def log_thickness(radius, tail, pars):
    h,length,drop,s,p,_ = pars
    x = radius/length[:,None]
    w = np.maximum(x-1,0.)/s[:,None]
    distal = np.where(tail[:,None],p[:,None]*np.log1p(w/(p-1)[:,None]),w)
    return h[:,None]-drop[:,None]*np.minimum(x,1.)-distal


def volume_at(outer, tail, pars):
    h,length,drop,s,p,_ = pars
    x = np.asarray(outer)/length
    core = gammainc(2.,drop*np.minimum(x,1.))/drop**2
    w = np.maximum(x-1,0.)
    u = w/s
    a = s*(p-1)
    fraction = w/(a+w)
    first = -s*np.expm1(-(p-1)*np.log1p(w/a))
    second = s**2*((p-1)/(p-2))*betainc(2.,p-2,fraction)
    exponential = -s*np.expm1(-u)+s**2*gammainc(2.,u)
    distal = np.where(tail,first+second,exponential)
    # Below double-precision resolution of the slope, use the local integral.
    distal = np.where(u<1e-10,w+w*w/2,distal)
    return 2*np.pi*1e-5*np.exp(h)*length**2*(core+np.exp(-drop)*distal)


def radius_at(cm, tail, pars):
    h,length,drop,s,p,_ = pars
    loss = np.maximum(h-np.log(cm),0.)
    outer = np.maximum(loss-drop,0.)
    a = s*(p-1)
    factor = np.minimum(outer/p,np.log1p(DIAMETER/(length*a)))
    w = np.where(tail,a*np.expm1(factor),s*outer)
    return np.minimum(DIAMETER,length*np.where(loss<=drop,loss/drop,1+w))
