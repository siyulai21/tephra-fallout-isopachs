"""Paired whole-Earth volume integrals and uncertainty for complete deposit realizations."""
from pathlib import Path
import json

import numpy as np
from scipy.stats import qmc

from .directional.ensemble.fields import inverse_cdf
from .directional.ensemble.integrals import polar_nodes
from .extent_review import ExtentFields, STATES
from .mass_mixture.bearings.field import BearingField, BearingSupport, phases, selected
from .nested.reach import ReachDraws
from src.probabilistic_isopach.global_volume import EarthDomain

SEEDS = (32147, 81701)
PROBS = (.025,.5,.975)


def parent_fields(folder, family, data, latitude, *, states=STATES, spherical=None, count=None):
    folder = Path(folder)
    if family=="spherical-mixture":
        path = Path(spherical or folder/"posterior.npz")
        with np.load(path) as a:
            draws,_,shape = selected(dict(a),count or 256)
        parent = BearingField(draws,BearingSupport(data,latitude),phases(shape))
    elif family=="directional":
        from .directional.azimuth.primary import read_field
        report = json.loads((folder/"azimuth/report.json").read_text())
        parent = read_field(folder/"azimuth",data,report)
    else:
        parent = ReachDraws.load(folder/"full.npz").reduced(count or 2048)
    return ExtentFields(family,parent,data,latitude,states)


def quantiles(v,w):
    return np.quantile(v,PROBS) if np.ptp(w)<1e-15 else inverse_cdf(v,w,PROBS)


def integrate(fields, latitude, orders=(11,12,13,14,15)):
    axis = EarthDomain.wgs84(latitude).axis_scale
    weights = fields.weights
    k = len(fields.states)
    totals = np.zeros((len(SEEDS),k,len(weights)))
    history = []
    for step, order in enumerate(orders):
        previous = totals.copy()
        for j, seed in enumerate(SEEDS):
            nodes = qmc.Sobol(2,scramble=True,seed=seed).random_base2(order)
            xy, area, _, _ = polar_nodes(nodes,axis)
            start = 2**orders[step-1] if step else 0
            for i in range(start,len(xy),192):
                pts = slice(i,i+192)
                for v,f in enumerate(fields.variants(xy[pts])):
                    totals[j,v] += np.exp(f) @ (area[pts]*1e-5)
        volume = totals.mean(0)/len(xy)
        q = np.array([quantiles(row,weights) for row in volume])
        each = np.array([[quantiles(row/len(xy),weights) for row in t] for t in totals])
        scramble = np.max(np.abs(each[0]-each[1])/q,axis=1)
        if step:
            old_q = np.array([quantiles(row/start,weights) for row in previous.mean(0)])
            change = np.max(np.abs(old_q/q-1.),axis=1)
        else:
            change = np.full(k,np.inf)
        passed = bool(np.all(scramble<.04) and np.all(change<.05))
        history.append(dict(nodes_per_scramble=len(xy),scrambles=list(SEEDS),conditional_q025_q50_q975=q.tolist(),
                            max_relative_scramble_by_state=scramble.tolist(),
                            max_relative_refinement_by_state=change.tolist() if step else None,passed=passed))
        if passed:
            break
    return volume,history
