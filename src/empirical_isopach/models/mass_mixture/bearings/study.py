"""Verified posterior transport and buffered geographic tests for the spherical family."""
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
import pandas as pd

from ....core.validation import interval_metrics
from ....core.distributions import LogMixture
from ..geometry import log_field, to_sphere
from ..inference import flatten
from .field import BearingSupport, POWERS, phases, multiplier
from .volume import checked_volumes


def digest(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def save(path, obj):
    Path(path).write_text(json.dumps(obj,indent=2,allow_nan=False)+"\n")


def verified(path,fit):
    if digest(path)!=fit["posterior_sha256"]:
        raise ValueError(f"Spherical posterior checksum mismatch at {path}")
    with np.load(path) as ar:
        return dict(ar)


def fit_volume(data,latitude,run,fit,cache,output,part):
    samples = verified(cache/part/fit["posterior_file"],fit)
    support = BearingSupport(data,latitude)
    result,packet = checked_volumes(samples,support,parts=part=="baseline",
                                    initial=512 if part=="baseline" else 256,ceiling=2048)
    result["posterior_sha256"] = fit["posterior_sha256"]
    result["input_sha256"] = run["input"]["input_sha256"]
    result["parent_sampler_passed"] = fit["diagnostics"]["passed"]
    folder = output/"assumptions"
    folder.mkdir(exist_ok=True,parents=True)
    np.savez_compressed(folder/(part+".npz"),**packet)
    result["packet_sha256"] = digest(folder/(part+".npz"))
    save(folder/(part+".json"),result)
    return result


def predictive(samples,train,test,latitude,*,with_components=False):
    """Use every retained parent draw and one stratified angle per chain/draw, across all widths."""
    support = BearingSupport(train,latitude)
    h,theta = support.evaluate(test.xy)
    if np.max(support.evaluate(train.xy)[0])>1e-12:
        raise ArithmeticError("Continuation changed a training-site likelihood")
    flat = flatten(samples)
    mu = log_field(flat,to_sphere(test.xy,latitude))
    phi = phases(samples["log_volume"].shape)[:,None]
    sd2 = flat["noise"][:,None]**2
    eps = flat["contamination"]
    ws = np.r_[1-eps,eps]/len(eps)
    mask = ws>0
    var = np.vstack((sd2+test.log_sd**2,16*sd2+test.log_sd**2))[mask]
    vals,by_power = [],[]
    for p in POWERS:
        mean = mu if p==0 else mu+np.log(multiplier(h,theta,phi,p))
        means = np.vstack((mean,mean))[mask]
        dist = LogMixture(ws[mask],means,var)
        by_power.append(dist.score(test))
        vals.append(means)
    dist = LogMixture(np.tile(ws[mask],len(vals))/len(vals),np.concatenate(vals),np.tile(var,(len(vals),1)))
    # Slice sites to limit transient CDF memory for long chains.
    from scipy.special import logsumexp
    out = {"log_score":logsumexp(np.stack(by_power),axis=0)-np.log(len(vals))}
    for name,p in (("median",.5),("lower",.025),("upper",.975)):
        out[name] = dist.quantile(p)
    return out,h,by_power


def validation(data,latitude,parent,cache):
    data = data.canonical()
    identity = data.row_ids.tolist()
    val = parent["validation"]
    if identity != val["row_ids"]:
        raise ValueError("Stored spherical validation order differs from the accepted rows")
    result = {key:np.full(len(identity),np.nan) for key in ("median","lower","upper","log_score")}
    exposures = np.full(len(identity),np.nan)
    columns = np.full((len(POWERS),len(identity)),np.nan)
    for fold in val["fits"]:
        train = np.isin(data.row_ids,fold["training_row_ids"])
        test = np.isin(data.row_ids,fold["withheld_row_ids"])
        if np.any(train&test) or not np.array_equal(data.row_ids[train],fold["training_row_ids"]) or not np.array_equal(data.row_ids[test],fold["withheld_row_ids"]):
            raise ValueError("Saved training and test identifiers do not reconstruct the matched fold")
        d = verified(cache/"validation"/f"fold-{fold['fold']}"/fold["fit"]["posterior_file"],fold["fit"])
        predictions,h,by_power = predictive(d,data.subset(train),data.subset(test),latitude)
        for k,v in predictions.items():
            result[k][test] = v
        columns[:,test] = by_power
        exposures[test] = h
    if not all(np.isfinite(a).all() for a in (*result.values(),columns,exposures)):
        raise ValueError("Incomplete spherical holdout predictions")
    np.testing.assert_allclose(columns[0],val["predictions"]["log_score"],rtol=1e-8,atol=1e-8)
    labels = np.asarray(val["folds"])
    metrics = interval_metrics(data,result["median"],result["lower"],result["upper"],result["log_score"])
    metrics["fold_log_scores"] = [-float(result["log_score"][labels==f].mean()) for f in range(4)]
    metrics["equal_region_log_score"] = float(np.mean(metrics["fold_log_scores"]))
    out = {**val,"predictions":{k:v.tolist() for k,v in result.items()},"metrics":metrics,
           "interpretation":"Full matched training-fold posterior, fixed bearing prior and training-only support. Scores include exact readings and limit events."}
    table = pd.DataFrame({"row_id":data.row_ids,"fold":labels,"observed_log_value":data.log_value,"censored":data.censored,
        "lower_censored":data.lower_censored,
        **({"interval_upper_log_cm":data.interval_log_upper} if data.interval.any() else {}),
        **result,"gap_exposure":exposures,
        **{f"log_score_power_{p}":v for p,v in zip(POWERS,columns,strict=True)}})
    return out,table
