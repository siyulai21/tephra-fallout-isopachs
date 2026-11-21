"""A bounded soft prior on displacement relative to the fitted component width."""
import numpy as np
from scipy.special import logsumexp

from ...core.distributions import LogMixture
from .bearings.field import BearingField, BearingSupport, phases


def prior_ratio(draws):
    """Multiply the parent prior by 1/(1+sum w_j |offset_j|²/4)."""
    if "angular" in draws:
        return np.ones_like(draws["log_volume"])
    displacement2 = np.sum(draws["offset"]**2,axis=-1)
    return 1./(1.+.25*np.sum(draws["weights"]*displacement2,axis=-1))


def heldout(draws, train, test, latitude, chains):
    n = len(draws["log_volume"])
    field = BearingField(draws,BearingSupport(train,latitude),phases((chains,n//chains)))
    components = field.components(test.xy)
    eps = draws["contamination"]
    noise = draws["noise"][:,None]
    variance = np.vstack((noise**2+test.log_sd**2,16*noise**2+test.log_sd**2))
    def score(weight):
        w = np.r_[weight*(1-eps),weight*eps]/weight.sum()
        keep = w>0
        logs = []
        for mu in components:
            dist = LogMixture(w[keep],np.vstack((mu,mu))[keep],variance[keep])
            logs.append(dist.log_score(test.log_value,test.censored,test.lower_censored))
        return logsumexp(logs,axis=0)-np.log(len(components))
    ratio = prior_ratio(draws)
    return score(np.ones(n)),score(ratio),float(ratio.sum()**2/(ratio@ratio))
