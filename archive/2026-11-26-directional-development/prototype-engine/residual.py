"""Heavy-tailed site variability, with independent corrections to a Gaussian EP proposal.

The residual is a fixed positive normal-scale quadrature of a unit-variance
logistic distribution. Documented measurement error remains Gaussian. Neither
outliers nor limits are discarded. All corrections use the original mixture
density or bounded-event probability, rather than the EP site factors.
"""
import numpy as np
from scipy.special import logsumexp, ndtr, ndtri
from scipy.stats import kstwobign, qmc, norm
from scipy.linalg import solve_triangular
from numpy.polynomial.legendre import leggauss
from src.empirical_isopach.core.likelihood import censored_terms, normal_log_event
from src.empirical_isopach.models.directional.multiscale.conditioning import gaussian

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
    return dict(mean=mean,cov=cov,complete_cov=np.zeros_like(cov),logz=float(.5*linear@mean-logdet+cs.sum()),X=B,
                ep_residual=float(residual),iterations=iteration+1,tau=tau,eta=eta,cs=cs,
                target=y,hi=hi,lower=data.lower_censored,censored=data.censored,error2=error2,
                control_variate=True)

def corrected(state,power,seed):
    mean=state['mean']
    u=np.clip(qmc.Sobol(len(mean),scramble=True,seed=seed).random_base2(power),1e-14,1-1e-14)
    beta=mean+ndtri(u)@np.linalg.cholesky(state['cov']).T
    ratios=[]
    for first in range(0,len(beta),256):
        f=beta[first:first+256]@state['X'].T
        logp=np.full(f.shape,-np.inf)
        for j,weight in enumerate(W):
            event=normal_log_event(state['target'],f,np.sqrt(state['error2'][:,j]),
                                   state['censored'],state['lower'],state['hi'])
            logp=np.logaddexp(logp,np.log(weight)+event)
        ratios.append((logp+.5*f*f*state['tau']-f*state['eta']-state['cs']).sum(axis=1))
    ratio=np.concatenate(ratios)
    norm=logsumexp(ratio)
    weights=np.exp(ratio-norm)
    return beta,weights,float(norm-np.log(len(beta))),float(1/(len(beta)*(weights@weights)))


def compression(state,sims,checks,max_groups=4):
    """Correct proposal moments, using its analytic moments as control variates.

    When the correction weights are constant the result is exactly the Gaussian
    proposal, independent of Monte Carlo sample covariance error. A separate CDF
    correction checks the Gaussian approximation at off-survey probes as well.
    """
    mixing=np.exp(np.array([s[2] for s in sims])-logsumexp([s[2] for s in sims]))
    beta=np.concatenate([s[0] for s in sims])
    delta=np.concatenate([w*(s[1]-1/len(s[1])) for w,s in zip(mixing,sims,strict=True)])
    if state.get('control_variate') == 'bounded':
        factor=state['control_factor']
        z=(beta-state['mean'])@state['control_recovery']
        completion=state['complete_cov']
    else:
        factor=np.linalg.cholesky(state['cov'])
        z=solve_triangular(factor,(beta-state['mean']).T,lower=True,check_finite=False).T
        completion=np.zeros_like(state['cov'])
    shift=delta@z
    relative=np.eye(len(shift))+(z.T*delta)@z-np.outer(shift,shift)
    ev,vectors=np.linalg.eigh((relative+relative.T)/2)
    minimum=float(ev.min())
    # A nonpositive estimate is retained as a failed numerical attempt. The
    # caller must refine it; this floor only makes that attempt serializable.
    covariance=completion+factor@((vectors*np.maximum(ev,1e-8))@vectors.T)@factor.T
    mean=state['mean']+factor@shift
    projected=checks@mean
    sd=np.sqrt(np.maximum(1e-12,((checks@covariance)*checks).sum(1)))
    reference_sd=np.sqrt(np.maximum(1e-12,((checks@state['cov'])*checks).sum(1)))
    reference_mean=checks@state['mean']
    mapped=checks@beta.T
    quantiles=np.array([.025,.25,.5,.75,.975])
    values=projected+ndtri(quantiles)[:,None]*sd
    conditional_sd=np.sqrt(np.maximum(1e-20,((checks@completion)*checks).sum(1)))
    reference=np.array([ndtr((at-reference_mean)/reference_sd)
                        +ndtr((at[:,None]-mapped)/conditional_sd[:,None])@delta for at in values])
    weights=np.concatenate([w*s[1] for w,s in zip(mixing,sims,strict=True)])
    transformed=checks@factor
    axes=np.empty((0,len(shift)))

    def group(ix,lo,hi):
        mass0=float(np.prod(ndtr(hi)-ndtr(lo)))
        if len(lo):
            normz=ndtr(hi)-ndtr(lo)
            first=(norm.pdf(lo)-norm.pdf(hi))/normz
            lp=np.zeros_like(lo);hp=np.zeros_like(hi)
            np.multiply(lo,norm.pdf(lo),out=lp,where=np.isfinite(lo))
            np.multiply(hi,norm.pdf(hi),out=hp,where=np.isfinite(hi))
            second=1+(lp-hp)/normz-first*first
            m0=first@axes
            c0=np.eye(len(shift))+(axes.T*(second-1))@axes
        else:
            m0=np.zeros(len(shift));c0=np.eye(len(shift))
        correction=delta[ix]
        mass=mass0+correction.sum()
        if mass<=1e-8:return None
        m=(mass0*m0+correction@z[ix])/mass
        c=(mass0*(c0+np.outer(m0,m0))+(z[ix].T*correction)@z[ix])/mass-np.outer(m,m)
        c=(c+c.T)/2
        smallest=float(np.linalg.eigvalsh(c).min())
        if smallest<=1e-8:return None
        cm=reference_mean+transformed@m
        csd=np.sqrt(np.maximum(1e-12,((transformed@c)*transformed).sum(1)+conditional_sd**2))
        cdf=mass*ndtr((values-cm)/csd)
        return dict(mass=mass,m=m,cov=c,minimum=smallest,ix=ix,lo=lo,hi=hi,cdf=cdf)

    first=group(np.arange(len(beta)),np.empty(0),np.empty(0))
    if first is None:
        return [(1.,mean,covariance)],1.,minimum
    groups=[first]
    for _ in range(max_groups):
        discrepancy=sum(g['cdf'] for g in groups)-reference
        probe=np.unravel_index(abs(discrepancy).argmax(),discrepancy.shape)
        error=float(abs(discrepancy[probe]))
        if error<=.012 or len(groups)>=max_groups:break
        eligible=[j for j,g in enumerate(groups) if len(g['ix'])>=32 and g['mass']>.002]
        if not eligible:break
        sign=np.sign(discrepancy[probe])
        def mismatch(j):
            g=groups[j];ix=g['ix']
            actual=weights[ix]@ndtr((values[probe]-mapped[probe[1],ix])/conditional_sd[probe[1]])
            return sign*(g['cdf'][probe]-actual)
        which=max(eligible,key=mismatch)
        direction=transformed[probe[1]].copy()
        direction/=max(np.linalg.norm(direction),1e-15)
        orthogonal=direction-axes.T@(axes@direction)
        if np.linalg.norm(orthogonal)>.2:
            axes=np.vstack((axes,orthogonal/np.linalg.norm(orthogonal)))
            for g in groups:
                g['lo']=np.r_[g['lo'],-np.inf];g['hi']=np.r_[g['hi'],np.inf]
            dim=len(axes)-1
        else:
            dim=int(np.argmax(abs(axes@direction)))
        parent=groups[which]
        split=ndtri((ndtr(parent['lo'][dim])+ndtr(parent['hi'][dim]))/2)
        ix=parent['ix'];lower=(z[ix]@axes[dim])<=split
        hi=parent['hi'].copy();hi[dim]=split
        lo=parent['lo'].copy();lo[dim]=split
        children=[group(ix[lower],parent['lo'].copy(),hi),group(ix[~lower],lo,parent['hi'].copy())]
        if any(c is None for c in children):break
        groups[which:which+1]=children
    discrepancy=sum(g['cdf'] for g in groups)-reference
    error=float(abs(discrepancy).max())
    parts=[(g['mass'],state['mean']+factor@g['m'],completion+factor@g['cov']@factor.T) for g in groups]
    return parts,error,min(g['minimum'] for g in groups)
