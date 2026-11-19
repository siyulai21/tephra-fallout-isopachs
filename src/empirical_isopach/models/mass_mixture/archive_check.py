"""Independent spherical-posterior checks at audited archive horizons."""
from pathlib import Path
import json

import arviz as az
import numpy as np
from scipy.special import log_ndtr,logsumexp
from ...region import SurveyRegion

from ...core.distributions import LogMixture
from ..directional.archive_readings import positions
from ..directional.ensemble.fields import inverse_cdf
from ..directional.study import digest,write_json
from .geometry import log_field,to_sphere
from .inference import flatten


def archive_checks(samples, data, vent, inventory):
    sites = inventory["sites"]
    xy = positions(sites,vent)
    p = flatten(samples)
    unit = to_sphere(xy,vent[1])
    n = len(p["noise"])
    values = [log_field({k:v[i:i+128] for k,v in p.items()},unit) for i in range(0,n,128)]
    mu = np.concatenate(values)
    noise,eps = p["noise"][:,None],p["contamination"][:,None]
    shape = samples["log_volume"].shape
    all_models,archive_weights = {},{}
    maps = {s["id"]:i for i,s in enumerate(sites)}
    within = SurveyRegion(data.xy).contains(xy)
    for scenario in [None,*inventory["scenarios"]]:
        key = scenario["id"] if scenario else "quantitative"
        score = np.zeros(n)
        if scenario:
            for sid,limit in scenario["effective_limit_cm"].items():
                z = (np.log(limit)-mu[:,maps[sid]])/noise[:,0]
                with np.errstate(divide="ignore"):
                    score += np.logaddexp(np.log1p(-eps[:,0])+log_ndtr(z),np.log(eps[:,0])+log_ndtr(z/4))
        normalized = score-logsumexp(score)
        weights = np.exp(normalized)
        volume = np.exp(p["log_volume"])
        q = inverse_cdf(volume,weights,(.025,.5,.975)).tolist()
        ess = float(1/(weights@weights))
        pareto = float(az.psislw(score)[1]) if np.ptp(score)>1e-12 else 0.
        chain_scores = score.reshape(shape)
        chain_p = np.exp(logsumexp(chain_scores,axis=1)-np.log(shape[1]))
        chain_volume = [float(inverse_cdf(v,np.exp(s-logsumexp(s)))[0])
                        for v,s in zip(volume.reshape(shape),chain_scores,strict=True)]
        # Archive probabilities are bounded above by one. Check concentration, chain influence
        # and actual volume stability before using a self-normalized update as an estimate.
        chain_mass = weights.reshape(shape).sum(axis=1)
        stable = (ess>=400 and pareto<.7 and min(chain_mass)>.10 and max(chain_mass)<.40
                  and max(abs(np.asarray(chain_volume)/q[1]-1))<.15)
        observation = LogMixture(np.r_[weights*(1-eps[:,0]),weights*eps[:,0]],np.vstack([mu,mu]),
                                 np.tile(np.r_[noise[:,0]**2,16*noise[:,0]**2][:,None],(1,len(sites))))
        pred=[{"id":s["id"],"name":s["name"],"in_regional_hull":bool(inside),"latent_cm":{},"new_site_cm":{},
               "p_local_below_scale":{}} for s,inside in zip(sites,within,strict=True)]
        for prob,label in ((.025,"lower95"),(.05,"lower90"),(.5,"median"),(.975,"upper95")):
            latent=np.exp(inverse_cdf(mu.T,weights,[prob])[0]);loc=np.exp(observation.quantile(prob))
            for i,row in enumerate(pred):
                row["latent_cm"][label] = float(latent[i]);row["new_site_cm"][label] = float(loc[i])
        for scale in (.1,1.,10.):
            for row,v in zip(pred,observation.cdf(np.log(scale)),strict=True):
                row["p_local_below_scale"][str(scale)] = float(v)
        all_models[key]={"sites":pred,"scenario":scenario,"full_field_volume_km3":dict(zip(("lower95","median","upper95"),q,strict=True)),
                         "joint_reading_probability":float(np.exp(logsumexp(score)-np.log(n))),
                         "chain_reading_probabilities":chain_p.tolist(),"raw_weight_ess":ess,"pareto_k":pareto,
                         "chain_mass":chain_mass.tolist(),"chain_volume_medians_km3":chain_volume,
                         "importance_checks_passed":bool(stable)}
        archive_weights[key]=weights.reshape(shape)
    return all_models,archive_weights


def publish(posterior, data, meta, inventory, output):
    if not inventory["sites"]:
        return None
    output=Path(output)
    output.mkdir(parents=True,exist_ok=True)
    with np.load(posterior) as source:
        info,weights=archive_checks(dict(source),data,meta["vent_lonlat"],inventory)
    path=output/"spherical-archive-weights.npz"
    np.savez_compressed(path,**weights)
    report={"posterior_sha256":digest(posterior),"weights_sha256":digest(path),"input_sha256":meta["input_sha256"],
            "method":"Self-normalized archive-likelihood update of complete mass-normalized posterior draws",
            "diagnostic_policy":"ESS at least 400, Pareto k under 0.7, each chain supplies 10–40% of mass, each chain volume median within 15% of pooled. Raw weights used; instability is retained.",
            "scenarios":info}
    write_json(output/"spherical-archive-checks.json",report)
    return report
