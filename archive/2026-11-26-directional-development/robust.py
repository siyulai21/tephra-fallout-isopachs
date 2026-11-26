"""Exploratory EP only; exact correction is required before any promotion."""
import numpy as np
from scipy.special import logsumexp
from scipy.stats import kstwobign
from numpy.polynomial.legendre import leggauss
from src.empirical_isopach.core.likelihood import censored_terms
from src.empirical_isopach.models.directional.multiscale.conditioning import gaussian
from src.empirical_isopach.models.directional.multiscale.posterior import DirectionalField, Marginals

U,W=leggauss(24)
W=W/2
V=12/np.pi**2*kstwobign.ppf((U+1)/2)**2
V=V/(W@V)

def condition(B,offset,data,noise):
    y=data.log_value-offset
    hi=data.interval_log_upper-offset
    error2=noise[:,None]**2*V+data.log_sd[:,None]**2
    tau=np.where(data.censored,0.,1/(noise**2+data.log_sd**2))
    eta=tau*y
    cs=np.where(data.censored,0.,-.5*(np.log(2*np.pi*(noise**2+data.log_sd**2))+y*y*tau))
    for iteration in range(128):
        precision=np.eye(B.shape[1])+(B.T*tau)@B
        linear=B.T@eta
        mean,cov,logdet=gaussian(precision,linear)
        mu=B@mean
        variance=np.maximum(1e-10,np.sum((B@cov)*B,axis=1))
        cavity=1/np.maximum(1e-10,1/variance-tau)
        center=cavity*(mu/variance-eta)
        total=cavity[:,None]+error2
        logz=-.5*(np.log(2*np.pi*total)+(y-center)[:,None]**2/total)
        cm=center[:,None]+cavity[:,None]/total*(y-center)[:,None]
        cv=cavity[:,None]*error2/total
        bounded=data.censored
        if bounded.any():
            loss,grad,curv=censored_terms(center[bounded,None],y[bounded,None],np.sqrt(total[bounded]),data.lower_censored[bounded,None],hi[bounded,None])
            logz[bounded]=-loss
            cm[bounded]=center[bounded,None]-cavity[bounded,None]*grad
            cv[bounded]=cavity[bounded,None]-cavity[bounded,None]**2*curv
        normalizer=logsumexp(logz+np.log(W),axis=1)
        probs=np.exp(logz+np.log(W)-normalizer[:,None])
        target_mean=np.sum(probs*cm,axis=1)
        target_var=np.maximum(1e-10,np.sum(probs*(cv+(cm-target_mean[:,None])**2),axis=1))
        nt=np.maximum(0,1/target_var-1/cavity)
        ne=target_mean/target_var-center/cavity
        nc=normalizer-.5*np.log(target_var/cavity)-.5*(target_mean**2/target_var-center**2/cavity)
        residual=max(np.max(abs(nt-tau)/(1+abs(nt))),np.max(abs(ne-eta)/(1+abs(ne))),np.max(abs(nc-cs)/(1+abs(nc))))
        if residual<3e-6 or iteration==127:break
        tau+=.5*(nt-tau);eta+=.5*(ne-eta);cs+=.5*(nc-cs)
    return dict(mean=mean,cov=cov,complete_cov=cov,logz=float(.5*linear@mean-logdet+cs.sum()),X=B,
                ep_residual=float(residual),iterations=iteration+1,tau=tau,eta=eta,cs=cs,
                target=y,hi=hi,lower=data.lower_censored,censored=data.censored,error2=error2)

def corrected(state,power,seed):
    return state['mean'][None,:],np.ones(1),0.,1.

class RobustField(DirectionalField):
    def distribution(self,xy,observation=False,log_sd=None,calibrated=False):
        latent=super().distribution(xy)
        if not observation:return latent
        noisevar=super().distribution(xy,observation=True).variances-latent.variances
        variance=latent.variances[:,None,:]+noisevar[:,None,:]*V[None,:,None]
        if log_sd is not None:variance+=np.asarray(log_sd)**2
        return Marginals((self.weights[:,None]*W).ravel(),np.repeat(latent.means,len(W),axis=0),variance.reshape(-1,len(xy)))
