"""Geographic coverage, thickness-slice attribution and matched predictive intervals."""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import ndtr

from .inference import log_events
from .profile import parameters
from .continuation import load_packet as load_draws
from .geometry import positions, ranks


def edge_sites(model, data, source):
    """Attribute the last measured rank to actual rows, splitting exact geographic ties."""
    shapes,ids = np.unique(model.geometry,return_inverse=True)
    radii = ranks(*positions(data.xy,model.latitude),shapes)
    outermost = radii.max(axis=1)
    if not np.allclose(outermost[ids],model.edge,rtol=0,atol=1e-9):
        raise ValueError("Stored continuation edge differs from training-site geography")
    boundary = np.isclose(radii,outermost[:,None],rtol=0,atol=1e-9)
    shares = boundary / boundary.sum(axis=1,keepdims=True)
    volumes = model.volume()
    extreme = volumes >= np.quantile(volumes,.95)
    frequency = np.bincount(ids,minlength=len(shapes)) @ shares / len(ids)
    above = np.bincount(ids[extreme],minlength=len(shapes)) @ shares / extreme.sum()
    records = pd.read_csv(source).iloc[data.row_ids-2]
    sites = records.Site.astype("string").fillna("") if "Site" in records else ""
    kinds = records.Kind.fillna("measurement") if "Kind" in records else "measurement"
    result = pd.DataFrame({"row_id":data.row_ids,"site":sites,
                           "longitude":records["Lon"],"latitude":records["Lat"],
                           "kind":kinds,"edge_probability":frequency,
                           "edge_probability_in_largest_5pct_volumes":above})
    return result.sort_values(["edge_probability","row_id"],ascending=[False,True])


def classes(mesh,xy,r,theta,limits):
    tri = mesh.triangulation.find_simplex(xy)
    inside = (tri>=0)&mesh.accepted[np.maximum(tri,0)]
    cls = np.where(inside,0,np.where(r<limits[0],1,np.where(r>limits[1],2,3)))
    sector = np.floor((theta+np.pi/8)/(np.pi/4)).astype(int)%8
    return cls*8+sector


def allocation_info(mass, volumes):
    parts = mass.reshape(len(mass),4,8)
    fractions = parts/parts.sum(axis=(1,2))[:,None,None]
    highest = volumes >= np.quantile(volumes,.95)
    return {"parts":["retained_exact_site_triangles","unmeasured_proximal","beyond_farthest_exact_site","other_gaps"],
            "sectors":["N","NE","E","SE","S","SW","W","NW"],
            "fraction_by_part":np.quantile(fractions.sum(axis=2),[.025,.5,.975],axis=0).T.tolist(),
            "fraction_by_sector":np.quantile(fractions.sum(axis=1),[.025,.5,.975],axis=0).T.tolist(),
            "highest_5pct_by_total_median_fraction":np.median(fractions[highest].sum(axis=2),axis=0).tolist(),
            "sampled_fields":len(volumes),"highest_5pct_fields":int(highest.sum())}


def predictions(output,data,folds):
    rows = []
    for f in folds["folds"]:
        i = f["fold"]
        test = data.subset(np.isin(data.row_ids,f["identity"]["test_rows"]))
        model = load_draws(Path(output)/f"fold-{i}.npz")
        mu = model.field(test.xy)
        scatter = parameters(model.z)[-1][:,None]
        sd = [np.hypot(c*scatter,test.log_sd) for c in (1.,3.)]
        def quantile(p):
            lo,hi = (mu-12*sd[1]).min(axis=0),(mu+12*sd[1]).max(axis=0)
            for _ in range(48):
                mid = (hi+lo)/2
                cdf = (.95*ndtr((mid-mu)/sd[0])+.05*ndtr((mid-mu)/sd[1])).mean(axis=0)
                lo,hi = np.where(cdf<p,mid,lo),np.where(cdf<p,hi,mid)
            return (hi+lo)/2
        lower,upper = quantile(.025),quantile(.975)
        endpoints = getattr(test,"interval_log_upper",None)
        for j,row_id in enumerate(test.row_ids):
            rows.append({"fold":i,"row_id":int(row_id),"censored":bool(test.censored[j]),
                         "lower_bound":bool(test.lower_censored[j]),"observed_or_bound_log_cm":test.log_value[j],
                         "latent_median_log_cm":f["median_log_cm"][j],"predictive_lower_log_cm":lower[j],
                         "predictive_upper_log_cm":upper[j],"log_score":f["row_log_scores"][j],
                         "fit_passed":f["numerics"]["passed"]})
            if endpoints is not None:
                rows[-1]["interval_upper_log_cm"] = endpoints[j]
    t = pd.DataFrame(rows)
    t.to_csv(Path(output)/"heldout.csv",index=False)
    exact = t.loc[~t.censored]
    covered = exact.observed_or_bound_log_cm.between(exact.predictive_lower_log_cm,exact.predictive_upper_log_cm)
    return {"exact_sites":len(exact),"covered":int(covered.sum()),
            "fraction":float(covered.mean()) if len(exact) else None}


def check_bounds(data, meta, model):
    """Posterior compatibility of each actual, accepted one- or two-sided observation."""
    if not data.censored.any():
        return []
    rows = data.subset(data.censored)
    mu = model.field(rows.xy)
    prob = np.exp(log_events(mu,parameters(model.z)[-1],rows)).mean(axis=0)
    names = {x["row_id"]:x["name"] for x in meta.get("interpreted_upper_limits",[])}
    upper = getattr(rows,"interval_log_upper",np.full(len(rows.xy),np.nan))
    result = []
    for j,i in enumerate(rows.row_ids):
        interval = np.isfinite(upper[j])
        lo = rows.log_value[j] if rows.lower_censored[j] else -np.inf
        hi = upper[j] if interval else (np.inf if rows.lower_censored[j] else rows.log_value[j])
        values = mu[:,j]
        result.append({"row_id":int(i),"name":names.get(int(i)),"event":"interval" if interval else
                       ("lower" if rows.lower_censored[j] else "upper"),
                       "lower_cm":float(np.exp(lo)) if np.isfinite(lo) else None,
                       "upper_cm":float(np.exp(hi)) if np.isfinite(hi) else None,
                       "latent_cm":np.exp(np.quantile(values,[.025,.5,.975])).tolist(),
                       "latent_satisfies":float(np.mean((values>=lo)&(values<=hi))),
                       "predictive_probability_of_event":float(prob[j])})
    return result
