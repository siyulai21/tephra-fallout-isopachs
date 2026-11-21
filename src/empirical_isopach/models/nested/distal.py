"""Smooth, finite-range transport sensitivities outside every training-site area rank."""
import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.special import logsumexp

from .geometry import DIAMETER, positions, ranks
from .inference import log_events
from .profile import integrated, log_profile, parameters
from .continuation import load_packet as load_draws, log_for, volume_for
from .study import write_json

FACTORS = (1., 2., 4.)
PROBS = (.025, .5, .975)


def ranked_sites(model, xy):
    geometries, ids = np.unique(model.geometry,return_inverse=True)
    by_shape = ranks(*positions(xy,model.latitude),geometries)
    return by_shape, ids


def edges(model, training_xy):
    site_ranks, ids = ranked_sites(model,training_xy)
    return np.maximum(site_ranks.max(axis=1)[ids],np.finfo(float).tiny)


def tapered_volumes(model, edge, factor, order):
    """Exact untapered interior and Gauss–Legendre quadrature over two log-radius panels."""
    length = parameters(model.z)[1]
    split = np.clip(length,edge,DIAMETER)
    nodes, weights = leggauss(order)
    total = volume_for(model,edge)
    for left, right in ((edge,split),(split,np.full_like(edge,DIAMETER))):
        # Splitting at the deposit's slope change preserves quadrature accuracy.
        for start in range(0,len(edge),1024):
            take = slice(start,start+1024)
            a,b = np.log(left[take,None]),np.log(right[take,None])
            logr = (a+b)/2+(b-a)*nodes/2
            radius = np.exp(logr)
            logt = log_for(model.subset(take),radius)
            decay = ((radius-edge[take,None])/(factor*edge[take,None]))**2
            mass = np.exp(logt+2*logr-decay) @ weights
            total[take] += np.pi*1e-5*(b-a)[:,0]*mass
    return total


def evaluate(home, data, folds):
    model = load_draws(home/"full.npz")
    edge = edges(model,data.xy)
    baseline = model.volume()
    scenarios = {}
    for factor in FACTORS:
        coarse = tapered_volumes(model,edge,factor,24)
        for order in (48,96,192):
            fine = tapered_volumes(model,edge,factor,order)
            error = float(np.max(abs(fine/coarse-1)))
            if error<1e-6:
                break
            coarse = fine
        if np.any(fine>baseline*(1+1e-8)) or np.any(fine<=0):
            raise ArithmeticError("A smooth protective taper must conserve positivity and cannot add fallout")
        scenarios[str(factor)] = {"volume_km3":np.quantile(fine,PROBS).tolist(),
                 "retained_volume_fraction":np.quantile(fine/baseline,PROBS).tolist(),
                 "order":order,"maximum_doubling_relative_error":error,
                 "quadrature_passed":error<1e-6,"geographic_folds":[]}
    for f in folds["folds"]:
        model = load_draws(home/f"fold-{f['fold']}.npz")
        train = data.subset(np.isin(data.row_ids,f["identity"]["training_rows"]))
        test = data.subset(np.isin(data.row_ids,f["identity"]["test_rows"]))
        edge = edges(model,train.xy)
        r, ids = ranked_sites(model,test.xy)
        relative = np.maximum(r[ids]/edge[:,None]-1,0.)
        mu = model.field(test.xy)
        scatter = parameters(model.z)[-1]
        for factor in FACTORS:
            event = log_events(mu-(relative/factor)**2,scatter,test)
            score = logsumexp(event,axis=0)-np.log(len(mu))
            scenarios[str(factor)]["geographic_folds"].append({"fold":f["fold"],
                 "training_rows":train.row_ids.tolist(),"test_rows":test.row_ids.tolist(),
                 "log_scores":score.tolist(),"negative_log_score":float(-score.mean()),
                 "posterior_fraction_beyond_training_rank":np.mean(relative>0,axis=0).tolist(),
                 "parent_fit_passed":f["numerics"]["passed"]})
    for s in scenarios.values():
        s["equal_region_negative_log_score"] = np.mean([f["negative_log_score"] for f in s["geographic_folds"]]).item()
    report = {"protocol":"nested-distal-rolloff-v1","scenarios":scenarios,"baseline_km3":np.quantile(baseline,PROBS).tolist(),
              "definition":"Each draw uses the largest equal-area radius of any accepted training site. For radius r beyond that value m, multiply thickness by exp(-((r-m)/(factor*m))^2). Every training observation keeps its original likelihood. Integrate on the entire Earth."}
    write_json(home/"distal-taper.json",report)
    return report
