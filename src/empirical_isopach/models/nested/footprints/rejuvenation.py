"""Fixed-target posterior moves after unsuccessful particle refinements.

The pilot supplies proposals, never target probabilities. Each population keeps
its own states; there is no resampling between populations. A Student-t proposal
conditioned on geometry and thinning law crosses correlated posterior modes.
Its full density enters the Metropolis ratio. Local kernels retain access to
geometries absent from the finite pilot. Adaptation stops after 256 moves.
"""
import json
from pathlib import Path

import numpy as np
from scipy.special import gammaln

from ..geometry import ANGLES, positions
from ..profile import budget_parameters, integrated, log_profile
from ..continuation import outer_status
from ..study import digest, write_json
from . import growth, numerics, diagnostics
from .flow import TrainingRanks
from .continuous import FlowField, PROTOCOL, continued, engine, predictions

SAMPLER = 'fixed-target-rejuvenation-v1'
PARTICLES = 3072
SEED = 149724
CHECKPOINTS = (0, 256, 512, 1024, 2048, 4096)


def log_prior(z):
    return -.5*np.sum(z[:,[0,1,2,4,5,6,7]]**2,axis=1)-2*np.log1p(z[:,3]**2/3)


class PilotProposal:
    """Normalized joint proposal q(z, geometry, tail), fixed before any moves."""
    def __init__(self, model):
        pilot=np.column_stack((model.z,model.warp))
        codes=2*model.geometry+model.power_tail
        self.labels,counts=np.unique(codes,return_counts=True)
        self.weights=counts/counts.sum()
        family=np.array([s.family for s in growth.SHAPES])[model.geometry//ANGLES]
        common={f:(np.cov(pilot[family==f].T) if (family==f).sum()>8 else np.eye(8))+.03*np.eye(8)
                for f in np.unique(family)}
        means,chols=[],[]
        for code,size in zip(self.labels,counts,strict=True):
            sample=pilot[codes==code]
            means.append(sample.mean(axis=0))
            pooled=common[growth.SHAPES[(code//2)//ANGLES].family]
            covariance=np.cov(sample.T) if size>8 else pooled
            alpha=size/(size+32.)
            # df=5 has covariance 5/3 times its scale matrix.
            chols.append(np.linalg.cholesky((alpha*covariance+(1-alpha)*pooled+.01*np.eye(8))*.6))
        self.means,self.chols=np.asarray(means),np.asarray(chols)
        self.inverses=np.linalg.inv(self.chols)
        self.logdet=np.log(np.diagonal(self.chols,axis1=1,axis2=2)).sum(axis=1)
        self.lookup=np.full(2*len(growth.prior()),-1,int)
        self.lookup[self.labels]=np.arange(len(self.labels))
        self.constant=gammaln(6.5)-gammaln(2.5)-4*np.log(5*np.pi)

    def log_density(self,z,g,tail):
        group=self.lookup[2*g+tail]
        valid=group>=0
        out=np.full(len(z),-np.inf)
        j=group[valid]
        delta=np.einsum('nij,nj->ni',self.inverses[j],z[valid]-self.means[j])
        out[valid]=(np.log(self.weights[j])+self.constant-self.logdet[j]
                    -6.5*np.log1p(np.sum(delta**2,axis=1)/5))
        return out

    def draw(self,n,rng):
        group=rng.choice(len(self.labels),n,p=self.weights)
        noise=rng.normal(size=(n,8))/np.sqrt(rng.chisquare(5,n)[:,None]/5)
        z=self.means[group]+np.einsum('nij,nj->ni',self.chols[group],noise)
        return z,self.labels[group]//2,(self.labels[group]%2).astype(bool)


def tail_transport(z,g,tail,radius):
    """An involution with unit Jacobian; the amplitude shift is independent of h."""
    moved=z.copy()
    rank=radius.at(z,g)
    moved[:,0]+=np.median(log_profile(rank,z[:,:6],tail)-log_profile(rank,z[:,:6],~tail),axis=1)/2.8
    return moved


class Chain:
    def __init__(self,model,train,n=PARTICLES,seed=SEED):
        self.rng=np.random.default_rng(seed)
        self.n=n
        groups=np.array_split(np.arange(len(model.z)),4)
        if min(map(len,groups))<n:
            raise ValueError('The pilot must contain four sufficiently large populations')
        idx=np.concatenate([self.rng.choice(p,n,replace=False) for p in groups])
        self.z=np.column_stack((model.z[idx],model.warp[idx]))
        self.g=model.geometry[idx].copy()
        self.tail=model.power_tail[idx].copy()
        self.train=train
        self.latitude=model.latitude
        self.radius=TrainingRanks(*positions(train.xy,model.latitude))
        self.proposal=PilotProposal(model)
        self.prior=growth.prior()
        neighbours=[]
        for a in growth.SHAPES:
            va=vars(a)
            neighbours.append(np.array([j for j,b in enumerate(growth.SHAPES) if a.family==b.family
                and set(va)==set(vars(b)) and sum(va[k]!=vars(b)[k] for k in va)<=1]))
        self.degree=np.array([len(x) for x in neighbours])
        self.table=np.zeros((len(neighbours),self.degree.max()),int)
        for i,v in enumerate(neighbours):self.table[i,:len(v)]=v
        positive=np.any(self.radius.base>0,axis=0)
        self.scale=np.mean(np.log(np.maximum(self.radius.base[:,positive],1e-10)),axis=1)
        selected=np.unique(np.linspace(0,len(train.xy)-1,min(32,len(train.xy))).astype(int))
        self.small=self.radius.subset(selected)
        self.ll=self.likelihood(self.z,self.g,self.tail)
        self.lp=log_prior(self.z)
        self.accepted=np.zeros(5)
        self.proposed=np.zeros(5)
        self.iteration=0

    def likelihood(self,z,g,tail):
        t=self.train
        return numerics.log_events(self.radius.at(z,g),np.array(budget_parameters(z[:,:6],tail)),tail,
                                   t.log_value,t.log_sd,t.censored,t.lower_censored,t.interval_log_upper)

    def accept(self,z,g,tail,hastings,block):
        ll=self.likelihood(z,g,tail)
        lp=log_prior(z)
        take=np.log(self.rng.random(len(z)))<ll-self.ll+lp-self.lp+hastings
        self.z[take],self.g[take],self.tail[take]=z[take],g[take],tail[take]
        self.ll[take],self.lp[take]=ll[take],lp[take]
        self.accepted[block]+=take.mean()
        self.proposed[block]+=1

    def move(self):
        self.iteration+=1
        i,n,rng=self.iteration,len(self.z),self.rng
        if i==1 or (i<=256 and i%32==1):
            self.chols=[np.linalg.cholesky(np.cov(part.T)+.015*np.eye(8)) for part in np.array_split(self.z,4)]
            self.warps=[np.linalg.cholesky(np.cov(part[:,6:].T)+.015*np.eye(2)) for part in np.array_split(self.z,4)]
        proposed=self.z.copy()
        warp=i%4==0
        for p,(chol,wc) in enumerate(zip(self.chols,self.warps,strict=True)):
            sl=slice(p*self.n,(p+1)*self.n)
            if warp:proposed[sl,6:]+=.2*rng.normal(size=(self.n,2))@wc.T
            else:proposed[sl,:6]+=.2*rng.normal(size=(self.n,6))@chol[:6,:6].T
        self.accept(proposed,self.g.copy(),self.tail.copy(),0.,int(warp))
        z,g,tail=self.z,self.g,self.tail
        near=(g//ANGLES)*ANGLES+(g%ANGLES+rng.integers(-4,5,n))%ANGLES
        global_draw=rng.choice(len(self.prior),n,p=self.prior)
        mode=rng.random(n)
        same=(global_draw//ANGLES)*ANGLES+g%ANGLES
        ids=g//ANGLES
        local=self.table[ids,(rng.random(n)*self.degree[ids]).astype(int)]*ANGLES+g%ANGLES
        geom=np.where(mode<.25,near,np.where(mode<.75,local,np.where(mode<.95,same,global_draw)))
        hastings=np.where((mode>=.25)&(mode<.75),np.log(self.degree[g//ANGLES]/self.degree[geom//ANGLES])
                          +np.log(self.prior[geom]/self.prior[g]),0.)
        moved=z.copy()
        moved[:,1]+=(self.scale[geom]-self.scale[g])/1.6
        moved[:,0]+=np.median(log_profile(self.small.at(z,g),z[:,:6],tail)
                              -log_profile(self.small.at(moved,geom),moved[:,:6],tail),axis=1)/2.8
        self.accept(moved,geom,tail.copy(),hastings,2)
        if i%4==0:
            self.accept(tail_transport(self.z,self.g,self.tail,self.small),self.g.copy(),~self.tail,0.,3)
            z,g,tail=self.proposal.draw(n,rng)
            hastings=(np.log(self.prior[g]/self.prior[self.g])
                      +self.proposal.log_density(self.z,self.g,self.tail)-self.proposal.log_density(z,g,tail))
            self.accept(z,g,tail,hastings,4)

    def field(self):
        return continued(self.z[:,:6],self.g,self.tail,self.latitude,self.z[:,6:],self.train,seed=97231)


def numerical_check(chain,model,full,previous=None):
    groups=np.array_split(np.arange(len(chain.z)),4)
    volumes=np.array([np.quantile(integrated(chain.z[p,:6],chain.tail[p]),[.025,.5,.975]) for p in groups])
    spread=np.ptp(np.log(volumes),axis=0)
    med=np.array([np.median(model.subset(p).field(chain.train.xy),axis=0) for p in groups])
    diff=np.ptp(med,axis=0)
    middle,high=np.quantile(diff,[.5,.95])
    intrinsic=bool(spread[1]<=np.log(1.35) and max(spread[[0,2]])<=np.log(2))
    outer=outer_status(model)
    spatial=dict(median_log_cm=float(middle),p95_log_cm=float(high),passed=bool(middle<.15 and high<.4))
    field=diagnostics.population_field_check(model,chain.train) if full else None
    pooled=np.quantile(model.volume(),[.025,.5,.975])
    change=None if previous is None else np.abs(pooled/np.asarray(previous['pooled_volume_quantiles_km3'])-1.)
    stationary=bool(change is not None and max(change)<.1)
    base_pass=intrinsic and outer['passed'] and spatial['passed'] and (field is None or field['passed'])
    return dict(particles_per_population=chain.n,rejuvenation_moves=chain.iteration,intrinsic_passed=intrinsic,
                intrinsic_volume_quantiles=volumes.tolist(),continuation=outer,spatial_populations=spatial,
                field_populations=field,pooled_volume_quantiles_km3=pooled.tolist(),
                temporal_volume_relative_change=None if change is None else change.tolist(),stationary_passed=stationary,
                base_passed=bool(base_pass),passed=bool(chain.iteration>=1024 and base_pass and stationary
                    and previous is not None and previous['base_passed']),
                tail_probabilities=[float(x.mean()) for x in np.array_split(chain.tail,4)],
                mean_log_likelihood=[float(x.mean()) for x in np.array_split(chain.ll,4)],
                acceptance=dict(zip(('profile','warp','geometry','tail','pilot'),
                    (chain.accepted/np.maximum(1,chain.proposed)).tolist(),strict=True)))


def write_fit(folder,key,train,test,meta,source,receipt):
    """Uniform fallback, retaining each checkpoint and the exact pilot identity."""
    folder,source=Path(folder),Path(source)
    folder.mkdir(parents=True,exist_ok=True)
    if digest(source)!=receipt['posterior_sha256'] or receipt['numerics']['passed']:
        raise ValueError('Rejuvenation needs the verified unsuccessful production packet')
    identity=dict(protocol=PROTOCOL,engine_sha256=engine(),input_sha256=meta['input_sha256'],
                  train=train.row_ids.tolist(),test=[] if test is None else test.row_ids.tolist(),
                  sampler=SAMPLER,sampler_sha256=digest(__file__),pilot_sha256=digest(source),
                  initial=PARTICLES,ceiling=PARTICLES,seed=SEED,checkpoint_moves=list(CHECKPOINTS))
    info,packet=folder/(key+'.json'),folder/(key+'.npz')
    if info.exists():
        saved=json.loads(info.read_text())
        if saved['identity']!=identity or digest(packet)!=saved['posterior_sha256']:
            raise ValueError(f'Rejuvenation identity differs at {info}')
        return saved
    progress=folder/(key+'-progress.json')
    if progress.exists() and json.loads(progress.read_text())['identity']!=identity:
        raise ValueError(f'Rejuvenation checkpoint identity differs at {progress}')
    chain=Chain(FlowField.load(source),train)
    attempts,checkpoints=[],[]
    # A deterministic restart verifies any retained checkpoint before advancing.
    for moves in CHECKPOINTS:
        while chain.iteration<moves:chain.move()
        model=chain.field()
        check=numerical_check(chain,model,key=='full',attempts[-1] if attempts else None)
        dest=folder/'attempts'/f'{key}-moves-{moves}.npz'
        dest.parent.mkdir(exist_ok=True)
        if dest.exists():
            old=FlowField.load(dest)
            if any(not np.array_equal(np.asarray(getattr(old,k)),np.asarray(v)) for k,v in vars(model).items()):
                raise ValueError(f'Rejuvenation restart differs at {dest}')
        else:np.savez_compressed(dest,**vars(model))
        attempts.append(check)
        checkpoints.append(dict(packet=dest.relative_to(folder).as_posix(),posterior_sha256=digest(dest),numerics=check))
        write_json(folder/(key+'-progress.json'),dict(identity=identity,pilot_outcome=receipt,checkpoints=checkpoints))
        print(folder.parent.name,key,'posterior moves',moves,'passed',check['passed'],flush=True)
        if check['passed']:break
    np.savez_compressed(packet,**vars(model))
    family=np.array([s.family for s in growth.SHAPES])[model.geometry//ANGLES]
    saved=dict(identity=identity,posterior_sha256=digest(packet),draws=len(model.z),checkpoints=checkpoints,
               numerics=dict(seed=SEED,initial=PARTICLES,ceiling=PARTICLES,attempts=attempts,passed=check['passed']),
               volume_quantiles_km3=np.quantile(model.volume(),[.025,.5,.975]).tolist(),
               posterior_family={f:float(np.mean(family==f)) for f in growth.FAMILY_PRIOR},
               heldout=predictions(model,test) if test is not None else [],pilot_outcome=receipt)
    write_json(info,saved)
    return saved
