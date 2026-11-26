import numpy as np
from scipy.integrate import quad
from scipy.special import ndtr
from scipy.stats import norm

from src.empirical_isopach.core.observations import Observations
from src.empirical_isopach.models.directional.multiscale.residual import W, V, condition, corrected, compression


def test_scale_mixture_events_and_control_variates_against_quadrature():
    data = Observations([[0, 0], [2, 1], [4, 2]], [.8, -.1, -.4],
                        [False, True, True], [.1, .2, .15], [2, 3, 4],
                        [False, False, True], [np.nan, np.nan, .6])
    B = np.array([[.8], [1.3], [1.1]])
    noise = np.array([.7, .35, .4])
    state = condition(B, np.zeros(3), data, noise)
    sims = [corrected(state, 15, s) for s in (751, 876)]
    parts, error, minimum = compression(state, sims, np.array([[1.]]), max_groups=1)

    def density(beta):
        sd = np.sqrt(noise[:, None]**2*V + data.log_sd[:, None]**2)
        event = (W@norm.pdf(.8, .8*beta, sd[0])) * (W@norm.cdf((-.1-1.3*beta)/sd[1]))
        event *= W@(norm.cdf((.6-1.1*beta)/sd[2])-norm.cdf((-.4-1.1*beta)/sd[2]))
        return norm.pdf(beta)*event

    evidence = quad(density, -10, 10, epsabs=1e-11)[0]
    mean = quad(lambda b:b*density(b), -10, 10)[0]/evidence
    variance = quad(lambda b:(b-mean)**2*density(b), -10, 10)[0]/evidence
    assert state['ep_residual'] < 1e-5
    assert minimum > 0
    np.testing.assert_allclose(parts[0][1][0], mean, atol=2e-4)
    np.testing.assert_allclose(parts[0][2][0, 0], variance, atol=2e-4)
    for beta, weights, correction, ess in sims:
        assert ess > .5
        assert abs(state['logz']+correction-np.log(evidence)) < 2e-4
        cdf = (ndtr((.1-state['mean'][0])/np.sqrt(state['cov'][0, 0]))
               + (beta[:, 0] <= .1)@(weights-1/len(weights)))
        assert abs(cdf-quad(density, -10, .1)[0]/evidence) < 5e-4


def test_unit_variance_and_heavy_tails_are_common_to_every_site():
    np.testing.assert_allclose(W.sum(), 1.)
    np.testing.assert_allclose(W@V, 1.)
    assert np.all(W > 0) and np.all(V > 0)
    assert W@norm.sf(4/np.sqrt(V)) > 10*norm.sf(4)
