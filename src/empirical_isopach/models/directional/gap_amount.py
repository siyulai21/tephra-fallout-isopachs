"""A coherent sensitivity for the amount of ash carried into an unmeasured bearing."""
import numpy as np
from scipy.special import logsumexp
from scipy.stats import qmc

from .ensemble.fields import inverse_cdf
from .ensemble.integrals import polar_nodes
from .azimuth.report import model_for
from .ensemble.members import splits

FACTORS = np.array([.25,1.,4.])
PRIOR = np.array([.25,.5,.25])


def weight_summary(values, field_weights):
    weights = PRIOR[:,None]*field_weights
    return inverse_cdf(values.ravel(),weights.ravel(),(.025,.5,.975))


def integrated_amount(model):
    estimates = np.zeros((2,3,len(model.weights)))
    history = []
    generators = [qmc.Sobol(2,scramble=True,seed=s) for s in (41791,80627)]
    for order in range(11,16):
        for i,gen in enumerate(generators):
            uv = gen.random_base2(order if order==11 else order-1)
            for batch in np.array_split(uv, len(uv)//128):
                xy, area, _, _ = polar_nodes(batch,model.members[0].geometry.axis)
                _,field,bary = model.components(xy)
                exposure,_ = model.support.evaluate(xy,bary)
                weights = FACTORS[:,None]**exposure * area * 1e-5
                estimates[i] += weights @ np.exp(field).T
        combined = estimates.mean(axis=0)/2**order
        summaries = np.array([weight_summary(e/2**order,model.weights) for e in estimates])
        central = weight_summary(combined,model.weights)
        discrepancy = float(np.max(np.ptp(summaries,axis=0)/central))
        change = None if not history else float(np.max(abs(central/history[-1]["km3"]-1)))
        history.append({"nodes_per_scramble":2**order,"scramble_change":discrepancy,
                        "doubling_change":change,"km3":central.tolist()})
        if change is not None and max(change,discrepancy)<.025:
            break
    return combined,history


def holdouts(data,latitude,work,weights,reference):
    rows = []
    for i,train,test in splits(data,latitude,"geographic"):
        record = reference["validation"]["geographic"]["folds"][i]
        if record["train"]!=train.row_ids.tolist() or record["test"]!=test.row_ids.tolist():
            raise ValueError("Directional training or test rows differ from the release")
        if record["status"]!="estimated":
            rows.append({"fold":i,"status":record["status"]})
            continue
        model = model_for(train,latitude,work,weights,128,"geographic",i)
        for member in model.members:
            declared = record["member_attempts"][member.geometry.policy.name]
            if not declared or member.records[-1]["posterior_sha256"] != declared[-1]["posterior_sha256"]:
                raise ValueError("Directional training posterior differs from the published attempt")
        dist = model.distribution(test.xy,True,test.log_sd)
        exposure,_ = model.support.evaluate(test.xy)
        scores = np.stack([dist.log_score(test.log_value-np.log(f)*exposure,test.censored,
                                         test.lower_censored, test.interval_log_upper-np.log(f)*exposure) for f in FACTORS])
        mixed = logsumexp(scores,axis=0,b=PRIOR[:,None])
        rows.append({"fold":i,"status":"estimated","parent_passed":record["passed"],
                     "training_rows":train.row_ids.tolist(),"test_rows":test.row_ids.tolist(),
                     "reference_score":float(-scores[1].mean()),"sensitivity_score":float(-mixed.mean()),
                     "per_site_reference":scores[1].tolist(),"per_site_sensitivity":mixed.tolist()})
    return rows
