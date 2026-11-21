"""Monotone area profiles, external analogue slopes, and closed-form slab volumes."""
import json
from pathlib import Path

import numpy as np
from scipy.special import gammainc, exprel

from .geometry import DIAMETER
from . import budget_profile

ASSET = Path(__file__).parents[2]/"assets/nested-thinning.json"


def analogue_centres():
    rows = json.loads(ASSET.read_text())["rows"]
    groups = {}
    for eruption, _, proximal, distal, transition in rows:
        groups.setdefault(eruption, []).append([np.log(proximal*transition),
                                                 np.log(distal/proximal)])
    return np.median([np.mean(group, axis=0) for group in groups.values()], axis=0)


CORE, RATIO = analogue_centres()
DONOR_ODDS, ODDS_CENTRE, ODDS_SPREAD = budget_profile.analogue_budget(ASSET)


def budget_parameters(z,tail):
    return budget_profile.geometry(z,tail,CORE,ODDS_CENTRE,ODDS_SPREAD)
# Scale uncertainties are explicit transfer priors in natural-log units.
LOCATION = np.array([np.log(100.), np.log(100.), CORE, RATIO, np.log(.55)])
SPREAD = np.array([2.8, 1.6, .9, 1., .65])


def parameters(z):
    """Peak log cm, transition radius km, core log drop, two distal slopes, noise."""
    if z.shape[1]==6:
        return budget_parameters(z,np.zeros(len(z),dtype=bool))
    h, log_l, log_drop, log_ratio, log_noise = (LOCATION+SPREAD*z).T
    drop = np.exp(log_drop)
    # The power exponent has a separate structural centre of 3. The same latent
    # normal coordinate spans weak and steep tails in both candidate laws.
    power = np.exp(log_ratio-RATIO)*3.
    return h, np.exp(log_l), drop, drop*np.exp(log_ratio), power, np.exp(log_noise)


def log_profile(radius, z, tail):
    if z.shape[1]==6:
        return budget_profile.log_thickness(radius,tail,budget_parameters(z,tail))
    h, length, drop, rate, power, _ = parameters(z)
    x = radius / length[:, None]
    outer = np.maximum(x-1., 0.)
    distal = np.where(tail[:, None], power[:, None]*np.log1p(outer), rate[:, None]*outer)
    return h[:, None]-drop[:, None]*np.minimum(x, 1.)-distal


def integrated(z, tail, outer=DIAMETER):
    """Integral over ranks from zero through outer (equivalent radius, km)."""
    if z.shape[1]==6:
        return budget_profile.volume_at(outer,tail,budget_parameters(z,tail))
    h, length, drop, rate, power, _ = parameters(z)
    b = np.asarray(outer)/length
    core = gammainc(2., drop*np.minimum(b,1.))/drop**2
    width = np.maximum(b-1., 0.)
    log_b = np.log1p(width)
    power_part = log_b*exprel((2.-power)*log_b)
    exp_part = -np.expm1(-rate*width)/rate + gammainc(2., rate*width)/rate**2
    return 2*np.pi*1e-5*np.exp(h)*length**2*(core+np.exp(-drop)*np.where(tail,power_part,exp_part))


def contour_radius(z, tail, cm):
    if z.shape[1]==6:
        return budget_profile.radius_at(cm,tail,budget_parameters(z,tail))
    h, length, drop, rate, power, _ = parameters(z)
    loss = np.maximum(h-np.log(cm), 0.)
    outer = np.maximum(loss-drop, 0.)
    power_radius = np.exp(np.minimum(outer/power, np.log(np.maximum(1.,DIAMETER/length))))
    r = length*np.where(loss<=drop, loss/drop, np.where(tail, power_radius, 1.+outer/rate))
    return np.minimum(r, DIAMETER)


def above_slab(z, tail, cm):
    """Bulk volume in excess of the first cm at each location, integral (T-cm)+."""
    r = contour_radius(z, tail, cm)
    return np.maximum(0., integrated(z, tail, r)-1e-5*cm*np.pi*r**2)
