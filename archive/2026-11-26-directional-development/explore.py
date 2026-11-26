import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import json, time
import importlib
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from scipy.linalg import solve_triangular
from src.empirical_isopach.core.catalog import datasets
from src.empirical_isopach.core.audit import audited_data
from src.empirical_isopach.models.directional.multiscale import geometry as g, posterior as p
from src.empirical_isopach.models.directional.multiscale.fit import fit
from src.empirical_isopach.models.directional.multiscale.validation import predictions
from src.empirical_isopach.models.directional.ensemble.members import splits

ORIGINAL_TREND = g.trend
ORIGINAL_KERNEL = g.matern
ORIGINAL_BASIS = g.LocalBasis.fit.__func__
ORIGINAL_CLASS = g.LocalBasis
FIT_MODULE=importlib.import_module('src.empirical_isopach.models.directional.multiscale.fit')
ORIGINAL_CONDITION=FIT_MODULE.condition
ORIGINAL_CORRECTED=FIT_MODULE.corrected
ORIGINAL_FIELD=FIT_MODULE.DirectionalField
ORIGINAL_ATTENUATION=FIT_MODULE.attenuation
MICRO_LENGTH=35.0
SMOOTH_BROAD=False
SMOOTH_FINE=False

class HarmonicBasis(ORIGINAL_CLASS):
    @classmethod
    def fit(cls,xy,axis,length,shared_fraction=.5):
        return cls(np.empty((0,2)),length,np.empty((8*17,0)))
    def features(self,xy,axis):
        x=g.chart(xy,axis)
        r=np.linalg.norm(x,axis=1); t=np.arctan2(x[:,0],x[:,1])
        h=r/np.hypot(g.WARP_KM,r);z=np.log(np.hypot(g.WARP_KM,r)/g.WARP_KM)
        rb=np.exp(-.5*((z[:,None]-np.arange(8))/.7)**2)
        cols=[rb]
        for k in range(1,9):
            a=np.sqrt(2)*h**k/(1+(k*self.length)**2)
            cols.extend(rb*(a*fn(k*t))[:,None] for fn in (np.cos,np.sin))
        B=np.column_stack(cols)
        return B/np.linalg.norm(B,axis=1)[:,None]

def separate_covariance(a,b,length):
    n=len(a)//2
    ra=np.hypot(g.WARP_KM,np.linalg.norm(a[:n],axis=1))[:,None]
    rb=np.hypot(g.WARP_KM,np.linalg.norm(b,axis=1))[None,:]
    square=(ra*ra+rb*rb)/2
    d=cdist(a[:n],b)*np.sqrt(5/square)/length
    fine=cdist(a[n:],b)*np.sqrt(5)/(MICRO_LENGTH*length)
    broad=np.exp(-.5*d*d/5) if SMOOTH_BROAD else (1+d+d*d/3)*np.exp(-d)
    return np.vstack((.8*ra*rb/square*broad, .2*(1+fine+fine*fine/3)*np.exp(-fine)))

class SeparateBasis(ORIGINAL_CLASS):
    @classmethod
    def fit(cls,xy,axis,length,shared_fraction=.5):
        from scipy.linalg import block_diag
        points=np.unique(g.chart(xy,axis),axis=0);n=len(points)
        k=separate_covariance(np.vstack((points,points)),points,length)
        broad,fine=k[:n],k[n:]
        ids=g.independent_modes(broad,points,n)
        factor=np.linalg.cholesky(broad[np.ix_(ids,ids)]+1e-10*np.eye(len(ids)))
        broad_projector=solve_triangular(factor,np.eye(n)[ids],lower=True)
        groups=np.argsort(-fine,axis=1,kind='stable')[:,:3]
        eligible=(fine[np.arange(n),groups[:,-1]]>=.2*.2)&(groups.shape[1]==3)
        groups=np.unique(np.sort(groups[eligible],axis=1),axis=0)
        average=np.zeros((len(groups),n));average[np.arange(len(groups))[:,None],groups]=1/3
        averaged=average@fine@average.T
        ids=g.independent_modes(averaged,average@points,n//3)
        factor=np.linalg.cholesky(averaged[np.ix_(ids,ids)]+1e-10*np.eye(len(ids)))
        regional=solve_triangular(factor,average[ids],lower=True)
        mapped=fine@regional.T;remainder=fine-mapped@mapped.T
        ids=g.independent_modes(remainder,points,n) if shared_fraction else np.array([],int)
        if len(ids):
            factor=np.linalg.cholesky(remainder[np.ix_(ids,ids)]+1e-10*np.eye(len(ids)))
            contrasts=solve_triangular(factor,np.eye(n)[ids]-mapped[ids]@regional,lower=True)
            fine_projector=np.vstack((regional,np.sqrt(shared_fraction)*contrasts))
        else:fine_projector=regional
        return cls(np.vstack((points,points)),length,block_diag(broad_projector,fine_projector))
    def cross_covariance(self,xy,axis):
        return separate_covariance(self.centers,g.chart(xy,axis),self.length)

def gaussian(a,b,length):
    ra=np.hypot(g.WARP_KM,np.linalg.norm(a,axis=1))[:,None]
    rb=np.hypot(g.WARP_KM,np.linalg.norm(b,axis=1))[None,:]
    square=(ra*ra+rb*rb)/2
    return ra*rb/square*np.exp(-0.5*cdist(a,b,'sqeuclidean')/square/length**2)

def transport(a,b,length):
    def tensor(x):
        r2=np.sum(x*x,axis=1)
        return ((g.WARP_KM**2+r2)[:,None,None]*np.eye(2)+3*x[:,:,None]*x[:,None,:])
    sa,sb=tensor(a),tensor(b)
    s=(sa[:,None]+sb[None,:])/2
    det=s[:,:,0,0]*s[:,:,1,1]-s[:,:,0,1]**2
    delta=a[:,None]-b[None,:]
    d2=(s[:,:,1,1]*delta[:,:,0]**2+s[:,:,0,0]*delta[:,:,1]**2-2*s[:,:,0,1]*delta[:,:,0]*delta[:,:,1])/det/length**2
    gain=(np.linalg.det(sa)[:,None]*np.linalg.det(sb)[None,:])**.25/np.sqrt(det)
    d=np.sqrt(np.maximum(0,5*d2))
    return gain*(1+d+d*d/3)*np.exp(-d)

def gaussmix(a,b,length):
    return .8*gaussian(a,b,length)+.2*gaussian(a,b,length/3)

def diffusive(a,b,length):
    ra=np.hypot(g.WARP_KM,np.linalg.norm(a,axis=1))[:,None]
    rb=np.hypot(g.WARP_KM,np.linalg.norm(b,axis=1))[None,:]
    distance=cdist(a,b)
    result=np.zeros_like(distance)
    for weight,sa,sb in [(.8,ra,rb),(.2,np.sqrt(g.WARP_KM*ra),np.sqrt(g.WARP_KM*rb))]:
        square=(sa*sa+sb*sb)/2
        d=distance*np.sqrt(5/square)/length
        result+=weight*sa*sb/square*(1+d+d*d/3)*np.exp(-d)
    return result

def physical(a,b,length):
    ra=np.hypot(g.WARP_KM,np.linalg.norm(a,axis=1))[:,None]
    rb=np.hypot(g.WARP_KM,np.linalg.norm(b,axis=1))[None,:]
    square=(ra*ra+rb*rb)/2
    distance=cdist(a,b)
    d=distance*np.sqrt(5/square)/length
    fine=distance*np.sqrt(5)/(g.WARP_KM*length)
    broad=np.exp(-.5*d*d/5) if SMOOTH_BROAD else (1+d+d*d/3)*np.exp(-d)
    local=np.exp(-.5*fine*fine/5) if SMOOTH_FINE else (1+fine+fine*fine/3)*np.exp(-fine)
    return .8*ra*rb/square*broad+.2*local

def rational(a,b,length):
    ra=np.hypot(g.WARP_KM,np.linalg.norm(a,axis=1))[:,None]
    rb=np.hypot(g.WARP_KM,np.linalg.norm(b,axis=1))[None,:]
    square=(ra*ra+rb*rb)/2
    d2=cdist(a,b,'sqeuclidean')/square/length**2
    return ra*rb/square/np.sqrt(1+d2)

def cylinder(a,b,length):
    def warp(x):
        radius=np.hypot(g.WARP_KM,np.linalg.norm(x,axis=1))
        return np.column_stack((x/radius[:,None]/length,np.log(radius/g.WARP_KM)))
    d=np.sqrt(5)*cdist(warp(a),warp(b))
    return (1+d+d*d/3)*np.exp(-d)

def polar_kernel(a,b,length):
    def parts(x):
        r=np.linalg.norm(x,axis=1)
        h=r/np.hypot(8.,r)
        return h,np.arctan2(x[:,0],x[:,1]),np.log(np.hypot(8.,r)/8.)
    ha,ta,za=parts(a);hb,tb,zb=parts(b)
    k=np.arange(1,33)
    weights=2*np.exp(-.5*(k*length)**2)
    ga=ha[:,None]**k;gb=hb[:,None]**k
    angular=1+(ga*np.cos(ta[:,None]*k)*weights)@(gb*np.cos(tb[:,None]*k)).T+(ga*np.sin(ta[:,None]*k)*weights)@(gb*np.sin(tb[:,None]*k)).T
    diag_a=1+(ga*ga)@weights;diag_b=1+(gb*gb)@weights
    d=np.sqrt(5)*abs(za[:,None]-zb[None,:])/(1.5*length)
    return angular/np.sqrt(diag_a[:,None]*diag_b[None,:])*(1+d+d*d/3)*np.exp(-d)

def slope_trend(xy,axis,scale,complexity):
    cols,d,rim=ORIGINAL_TREND(xy,axis,scale,complexity)
    r,t=g.polar(xy,axis); x=r/scale; h=x/np.hypot(1,x)
    more=[]
    for k,amplitude in enumerate((.9,.45,.2) if complexity else (.9,.45),1):
        gate=(h*rim)**k*h*d
        for func in (np.cos,np.sin):more.append(amplitude*gate*func(k*t))
    return np.column_stack([cols,*more]),d,rim

def solid_trend(xy,axis,scale,complexity):
    cols,d,rim=ORIGINAL_TREND(xy,axis,scale,complexity)
    r,_=g.polar(xy,axis);x=r/scale;h=x/np.hypot(1,x)
    cols[:,3:]*=h[:,None]
    return cols,d,rim

def broad_trend(xy,axis,scale,complexity):
    cols,d,rim=ORIGINAL_TREND(xy,axis,scale,complexity)
    r,t=g.polar(xy,axis); x=r/scale
    z=np.log1p(x*x)/2
    rb=np.exp(-0.5*((z[:,None]-np.array([0,1.5,3,4.5,6]))/1.2)**2)
    h=x/np.hypot(1,x)
    more=[1.2*rb]
    for k,amplitude in enumerate((1.1,.6,.3) if complexity else (1.1,.6),1):
        gate=(h*rim)**k*h
        for func in (np.cos,np.sin):
            more.append(amplitude*rb*(gate*func(k*t))[:,None])
    return np.column_stack([cols,*more]),d,rim

def setup(variant):
    global MICRO_LENGTH,SMOOTH_BROAD,SMOOTH_FINE
    SMOOTH_BROAD='smoothbroad' in variant
    SMOOTH_FINE='smoothfine' in variant
    MICRO_LENGTH=35/3 if "micro" in variant else 35.
    g.LOCAL_CAP=384 if "cap384" in variant else 192 if "cap192" in variant else 96
    g.matern=(diffusive if 'diffusive' in variant else physical if 'physical' in variant else polar_kernel if 'polar' in variant else cylinder if 'cylinder' in variant else gaussmix if 'gaussmix' in variant else rational if 'rational' in variant
              else transport if 'transport' in variant else gaussian if 'gaussian' in variant else ORIGINAL_KERNEL)
    g.trend=p.trend=solid_trend if 'solid' in variant else slope_trend if 'slope' in variant else broad_trend if 'broad' in variant else ORIGINAL_TREND
    f=importlib.import_module('src.empirical_isopach.models.directional.multiscale.fit')
    f.attenuation=p.attenuation=(lambda xy,axis,reach:np.zeros(len(xy))) if 'free' in variant else ORIGINAL_ATTENUATION
    if 'robustexact' in variant:
        from src.empirical_isopach.models.directional.multiscale import residual
        f.condition,f.corrected,f.DirectionalField=residual.condition,ORIGINAL_CORRECTED,ORIGINAL_FIELD
    elif 'robust' in variant:
        import robust
        f.condition,f.corrected,f.DirectionalField=robust.condition,robust.corrected,robust.RobustField
    else:f.condition,f.corrected,f.DirectionalField=ORIGINAL_CONDITION,ORIGINAL_CORRECTED,ORIGINAL_FIELD
    g.LOCAL_SD=p.LOCAL_SD=f.LOCAL_SD=4.6 if 'amp4' in variant else 2.3 if 'amp2' in variant else 1.15
    g.LocalBasis=p.LocalBasis=f.LocalBasis=SeparateBasis if 'separate' in variant else HarmonicBasis if 'harmonic' in variant else ORIGINAL_CLASS
    if 'analytic' in variant:
        from src.empirical_isopach.models.directional.multiscale.curvature import regularize
        def basis(cls,xy,axis,length,shared_fraction=.5):
            kernel='gaussian-mixture' if 'gaussmix' in variant else 'matern-v5'
            base=ORIGINAL_BASIS(cls,xy,axis,length,shared_fraction,kernel=kernel)
            return regularize(base, .01 if 'gentle' in variant else .05,
                              amplitude_precision=1. if 'amp1' in variant else .5 if 'amp05' in variant else .2, tension='tension' in variant)
        g.LocalBasis.fit=classmethod(basis)
    elif 'spectral' in variant:
        def basis(cls,xy,axis,length,shared_fraction=.5):
            points=np.unique(g.chart(xy,axis),axis=0)
            k=g.matern(points,points,length)
            ev,vec=np.linalg.eigh(k)
            ids=np.argsort(ev)[::-1][:g.LOCAL_CAP]
            ids=ids[ev[ids]>1e-8]
            ridge=.25 if 'soft' in variant else 1.0
            return cls(points,length,(vec[:,ids]/np.sqrt(ev[ids]+ridge)).T)
        g.LocalBasis.fit=classmethod(basis)
    elif 'bending' in variant:
        def basis(cls,xy,axis,length,shared_fraction=.5):
            base=ORIGINAL_BASIS(cls,xy,axis,length,shared_fraction)
            refined='refined' in variant
            nr,na=(65,96) if refined else (33,48)
            radii=g.WARP_KM*np.sqrt(np.expm1(np.linspace(0,np.log1p((12000/g.WARP_KM)**2),nr)))
            theta=np.arange(na)*2*np.pi/na
            points=np.stack((np.outer(radii,np.sin(theta))*axis,np.outer(radii,np.cos(theta))/axis),axis=-1).reshape(-1,2)
            r=np.repeat(radii,len(theta));h=.025*np.hypot(g.WARP_KM,r)
            def features(dx,dy):
                pts=points+np.column_stack((h*dx*axis,h*dy/axis))
                rim2=np.maximum(0,1-(np.linalg.norm(g.chart(pts,axis),axis=1)/(2*g.RADIUS))**2)
                return base.features(pts,axis)*rim2[:,None]
            center=features(0,0)
            xx=(features(1,0)+features(-1,0)-2*center)/.025**2
            yy=(features(0,1)+features(0,-1)-2*center)/.025**2
            cross=(features(1,1)+features(-1,-1)-features(1,-1)-features(-1,1))/(4*.025**2)
            energy=(xx.T@xx+yy.T@yy+2*cross.T@cross)/len(points)
            strength=5. if 'hard' in variant else 1. if 'very' in variant else .2 if 'strong' in variant else .05
            factor=np.linalg.cholesky(.2*np.eye(len(base.averaging))+strength*energy)
            return cls(base.centers,length,solve_triangular(factor,base.averaging,lower=True),
                       base.averaging if 'proper' in variant else None)
        g.LocalBasis.fit=classmethod(basis)
    elif 'harmonic' not in variant and 'separate' not in variant:g.LocalBasis.fit=classmethod(ORIGINAL_BASIS)

def job(args):
    variant,slug,kind,fold,root,grid=args
    setup(variant)
    r=next(r for r in datasets() if r['slug']==slug)
    data,meta=audited_data(r['path']); lat=meta['vent_lonlat'][1]
    if kind=='full': train=test=data
    else: _,train,test=next(s for s in splits(data,lat,kind) if s[0]==fold)
    if train is None or not len(train.xy) or not len(test.xy): return variant,slug,kind,fold,'empty'
    out=root/variant/slug/(kind if fold is None else f'{kind}-{fold}')
    out.mkdir(parents=True,exist_ok=True)
    if (out/'scores.csv').exists(): return variant,slug,kind,fold,'cached'
    t=time.time()
    model,info=fit(train,lat,shared_fraction=.25 if 'quarter' in variant else 0. if 'regional' in variant else 1.0 if 'full' in variant else .5)
    attempts=[{k:v for k,v in info.items() if k!='states'}]
    if 'robustexacttree' in variant or 'cv_' in variant:
        for power in (10,11):
            if info['passed']:break
            model,info=fit(train,lat,shared_fraction=.25 if 'quarter' in variant else 0. if 'regional' in variant else 1.0 if 'full' in variant else .5,power=power)
            attempts.append({k:v for k,v in info.items() if k!='states'})
    info['attempts']=attempts
    if 'robust' in variant and 'robustexact' not in variant:info['approximation_only']='EP; no independent exact posterior correction yet'
    pd.DataFrame(predictions(model,test,kind,fold,info['passed'])).to_csv(out/'scores.csv',index=False)
    (out/'fit.json').write_text(json.dumps(info))
    if kind=='full':
        model.save(out/'field.npz')
    if kind=='full' and grid:
        radii=np.r_[0,np.geomspace(.05,12750,289)]
        angles=np.arange(240)*2*np.pi/240
        xy=np.stack((model.domain.axis_scale*np.outer(radii,np.sin(angles)),np.outer(radii,np.cos(angles))/model.domain.axis_scale),axis=-1).reshape(-1,2)
        vals=model.central(xy).reshape(len(radii),len(angles))
        np.savez_compressed(out/'grid.npz',radii_km=radii,log10_cm=vals,axis=model.domain.axis_scale)
    return variant,slug,kind,fold,round(time.time()-t,1),info['passed']

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--variants',nargs='+',default=['gaussian','gaussian_full','gaussian_broad_full']); a.add_argument('--designs',nargs='+',default=['full','geographic']);a.add_argument('--jobs',type=int,default=4)
    a.add_argument('--slugs',nargs='*');a.add_argument('--no-grids',action='store_true')
    opt=a.parse_args(); root=Path('results/directional-geography-v6/experiments')
    jobs=[(v,r['slug'],k,f,root,not opt.no_grids) for k in opt.designs for f in ([None] if k=='full' else range(4)) for r in datasets() if not opt.slugs or r['slug'] in opt.slugs for v in opt.variants]
    with ProcessPoolExecutor(opt.jobs) as pool:
        for fut in as_completed([pool.submit(job,j) for j in jobs]):print(fut.result(),flush=True)
