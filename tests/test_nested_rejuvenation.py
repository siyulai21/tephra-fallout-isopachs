"""Check proposal normalization and exact reversible target updates."""
from types import SimpleNamespace

import numpy as np
from scipy.stats import multivariate_t, norm, t

from src.empirical_isopach.models.nested.footprints.rejuvenation import PilotProposal, Chain, log_prior, tail_transport


def test_pilot_density_includes_discrete_mass_and_student_scale():
    rng=np.random.default_rng(18)
    z=rng.normal(size=(80,8))
    model=SimpleNamespace(z=z[:,:6],warp=z[:,6:],geometry=np.repeat([0,1],[60,20]),power_tail=np.zeros(80,bool))
    q=PilotProposal(model)
    sample,g,tail=q.draw(100,rng)
    expected=[]
    for value,geom in zip(sample,g):
        group=q.lookup[2*geom]
        expected.append(np.log(q.weights[group])+multivariate_t.logpdf(value,loc=q.means[group],shape=q.chols[group]@q.chols[group].T,df=5))
    np.testing.assert_allclose(q.log_density(sample,g,tail),expected,atol=1e-12)
    assert np.isneginf(q.log_density(sample,np.full(100,2),tail)).all()


def test_tail_transport_is_volume_preserving_in_latent_coordinates():
    class Ranks:
        def at(self,z,g):return np.tile([3.,10.,100.,1000.],(len(z),1))
    rng=np.random.default_rng(8)
    z=rng.normal(size=(4,8));g=np.zeros(4,int);tail=np.array([False,True,True,False]);rank=Ranks()
    moved=tail_transport(z,g,tail,rank)
    np.testing.assert_allclose(tail_transport(moved,g,~tail,rank),z,atol=1e-12)
    columns=[]
    for k in range(8):
        change=np.zeros_like(z);change[:,k]=1e-6
        columns.append((tail_transport(z+change,g,tail,rank)-tail_transport(z-change,g,tail,rank))/(2e-6))
    jacobian=np.stack(columns,axis=2)
    np.testing.assert_allclose(np.linalg.det(jacobian),1.,atol=1e-6)


def test_exact_metropolis_update_recovers_known_geometry_and_tail_probabilities():
    n=6000
    chain=object.__new__(Chain);chain.rng=np.random.default_rng(312)
    def prior_sample():
        z=chain.rng.normal(size=(n,8));z[:,3]=chain.rng.standard_t(3,n);return z
    chain.z=prior_sample();chain.g=np.zeros(n,int);chain.tail=np.zeros(n,bool)
    chain.likelihood=lambda z,g,tail: np.log(np.where(g==0,.2,.8))+np.log(np.where(tail,.7,.3))
    chain.lp=log_prior(chain.z);chain.ll=chain.likelihood(chain.z,chain.g,chain.tail)
    chain.accepted=np.zeros(5);chain.proposed=np.zeros(5)
    for _ in range(32):
        z=prior_sample();g=chain.rng.integers(0,2,n);tail=chain.rng.random(n)<.5
        chain.accept(z,g,tail,log_prior(chain.z)-log_prior(z),4)
    assert abs(chain.g.mean()-.8)<.025
    assert abs(chain.tail.mean()-.7)<.025
    assert abs(np.mean(chain.g*chain.tail)-.56)<.025
    # The Student-t donor coordinate retains its intended heavy-tailed prior.
    expected=np.sum(norm.logpdf(chain.z[:,[0,1,2,4,5,6,7]]),axis=1)+t.logpdf(chain.z[:,3],3)
    np.testing.assert_allclose(log_prior(chain.z)-log_prior(chain.z[:1]),expected-expected[0],atol=1e-12)
