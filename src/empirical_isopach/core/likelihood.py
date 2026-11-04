"""Upper/lower-bound likelihood and derivatives shared by all compatible models."""
import numpy as np
from scipy.special import log_ndtr

def censored_terms(f, log_limit, sd, lower=False):
    """Negative log CDF (upper bound) or survival (lower), with derivatives."""
    sign = np.where(lower, -1., 1.)
    z = sign * (log_limit - f) / sd
    lp = log_ndtr(z)
    mills = np.exp(-0.5*z*z - 0.5*np.log(2*np.pi) - lp)
    return -lp, sign*mills/sd, np.maximum(mills*(mills+z)/(sd*sd), 0.)
