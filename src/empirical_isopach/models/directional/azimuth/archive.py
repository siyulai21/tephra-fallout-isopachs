"""Conditional visibility readings, evaluated on complete joint directional fields."""
import numpy as np
from scipy.special import log_ndtr, logsumexp

from ..archive_readings import positions
from ..ensemble.fields import inverse_cdf


def assess(model, data, inventory, vent):
    sites = inventory["sites"]
    if not sites:
        return {},None
    xy = positions(sites,vent)
    used = {key for scenario in inventory["scenarios"] for key in scenario["effective_limit_cm"]}
    selected_xy = xy[[s["id"] in used for s in sites]]
    if np.any(np.linalg.norm(data.xy[:,None,:]-selected_xy,axis=2)<1e-6):
        raise ValueError("An archive location coincides with quantitative data. Review dependence first.")
    local = model.distribution(xy,observation=True)
    original = model.base.distribution(xy)
    h,_ = model.support.evaluate(xy)
    sd = np.sqrt(local.variances)
    ids = [s["id"] for s in sites]
    logw = np.log(model.weights)
    selected = None
    summaries = {}
    for label,limits in [("quantitative",{})]+[(a["id"],a["effective_limit_cm"]) for a in inventory["scenarios"]]:
        ll = sum((log_ndtr((np.log(v)-local.means[:,ids.index(k)])/sd[:,ids.index(k)]) for k,v in limits.items()),
                 start=np.zeros(len(logw)))
        logz = logsumexp(logw+ll)
        w = np.exp(logw+ll-logz)
        if label==inventory.get("validation_scenario"):
            selected = w
        q = np.exp(inverse_cdf(local.means.T,w,(.025,.5,.975))).T
        measures = {}
        for i,s in enumerate(sites):
            measures[s["id"]]={"name":s["name"],"reading":s["reading"],"in_likelihood":s["id"] in limits,
                        "exposure":float(h[i]),"latent_cm_p025_p50_p975":q[i].tolist(),
                        "predictive_probability_below_1cm":float(w@np.exp(log_ndtr(-local.means[:,i]/sd[:,i])))}
        summaries[label]={"assumed_effective_scales_cm":limits,"log_predictive_probability":float(logz),
                          "effective_draws":float(1/(w@w)),"maximum_weight":float(max(w)),"sites":measures}
    return {"scenarios":summaries,"matched_control_median_cm":dict(zip(ids,np.exp(original.quantile(.5)).tolist(),strict=True)),
            "mapped_scenario":inventory.get("validation_scenario")},selected

def phase_medians(model,xy):
    """Integrate out azimuth exactly at each site, including the atom at zero concentration."""
    from .field import CONCENTRATIONS,log_normalizer
    base = model.base.distribution(xy)
    h,_ = model.support.evaluate(xy)
    answer = []
    for site,spread in enumerate(h):
        a = CONCENTRATIONS*spread
        mu = base.values[:,site,None]-log_normalizer(a)
        lo,hi = np.min(mu-a),np.max(mu+a)
        for _ in range(52):
            mid = (lo+hi)/2
            cdf = np.broadcast_to(mid>=mu,mu.shape).astype(float)
            positive = a>0
            cdf[:,positive]=1-np.arccos(np.clip((mid-mu[:,positive])/a[positive],-1,1))/np.pi
            if base.weights@(cdf@model.concentration_weights)<.5:
                lo=mid
            else:
                hi=mid
        answer.append(float(np.exp((hi+lo)/2)))
    return answer
