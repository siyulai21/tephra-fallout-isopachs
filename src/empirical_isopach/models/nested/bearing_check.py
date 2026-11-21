"""Optional test of how much nested volume is assigned to poorly observed bearings."""
from pathlib import Path
import json

import numpy as np
from scipy.special import logsumexp
from scipy.stats import qmc

from ..continuation_review import paired_map
from ..directional.ensemble.integrals import polar_nodes
from ..mass_mixture.bearings.field import BearingSupport, multiplier, phases
from .continuation import load_packet
from .inference import log_events
from .study import write_json

POWERS = (0,1,4)
PROBS = (.025,.5,.975)


def fields(model,support,phi,xy):
    base = model.field(xy)
    h,theta = support.evaluate(xy)
    return [base+np.log(multiplier(h,theta,phi[:,None],p)) for p in POWERS]


def integrate(model,support,phi,order,seed):
    uv = qmc.Sobol(2,scramble=True,seed=seed).random_base2(order)
    sums = np.zeros((len(phi),4,8))
    for chunk in np.array_split(uv,len(uv)//128):
        xy,area,_,_ = polar_nodes(chunk,support.axis)
        h,theta = support.evaluate(xy)
        sector = np.floor((theta+np.pi/8)/(np.pi/4)).astype(int)%8
        base = np.exp(model.field(xy))*area*1e-5
        for i,(p,enhanced) in enumerate(((0,False),(1,False),(4,False),(4,True))):
            mass = base*multiplier(h,theta,phi[:,None],p,enhanced=enhanced)
            for k in np.unique(sector):
                sums[:,i,k] += mass[:,sector==k].sum(axis=1)
    return sums/len(uv)


def review(data,meta,home,name):
    home = Path(home)
    out = home/"bearing-study"
    out.mkdir(exist_ok=True,parents=True)
    full = load_packet(home/"full.npz")
    support = BearingSupport(data,full.latitude)
    if np.max(support.evaluate(data.xy)[0])>1e-10:
        raise ArithmeticError("A fitted training location lost its bearing anchor")
    history = []
    last = None
    for n,order in ((256,12),(512,13),(1024,14),(2048,15)):
        model = full.reduced(n)
        phi = phases((4,n//4))
        true = model.volume()
        trials = np.stack([integrate(model,support,phi,order,k) for k in (28439,95717)])
        base = trials[:,:,0].sum(axis=-1)
        scaled = trials*(true[None,:]/base)[:,:,None,None]
        total = scaled.sum(axis=-1)
        q = np.stack((np.quantile(total[:,:,:3].mean(axis=0),PROBS),
                      np.quantile(total[:,:,3].mean(axis=0),PROBS)))
        by_scramble = np.stack((np.quantile(total[:,:,:3],PROBS,axis=(1,2)),
                                np.quantile(total[:,:,3],PROBS,axis=1)))
        scramble_errors = np.max(np.ptp(by_scramble,axis=2)/q,axis=1)
        changes = None if last is None else np.max(abs(q/last-1),axis=1)
        row = {"draws":n,"nodes_per_scramble":2**order,"unmodified_p95_relative_integration_error":
               float(np.quantile(abs(base/true-1),.95)),
               "ordinary_scramble_relative_difference":float(scramble_errors[0]),
               "enhanced_scramble_relative_difference":float(scramble_errors[1]),
               "scramble_relative_quantile_difference":float(max(scramble_errors)),
               "doubling_relative_difference":None if changes is None else float(max(changes))}
        ok = last is not None and row["unmodified_p95_relative_integration_error"]<.05 and max(row["scramble_relative_quantile_difference"],row["doubling_relative_difference"])<.1
        history.append(dict(row,passed=bool(ok)))
        last=q
        if ok:break
    avg = scaled.mean(axis=0)
    total = avg.sum(axis=-1)
    by_sector = avg[:,:3].reshape(-1,8)
    fractions = by_sector/by_sector.sum(axis=1)[:,None]
    geo=json.loads((home/"geographic.json").read_text())
    folds=[]
    for f in geo["folds"]:
        train=data.subset(np.isin(data.row_ids,f["identity"]["training_rows"]))
        test=data.subset(np.isin(data.row_ids,f["identity"]["test_rows"]))
        part=load_packet(home/f"fold-{f['fold']}.npz").reduced(2048)
        fold_support=BearingSupport(train,part.latitude)
        values=fields(part,fold_support,phases((4,512)),test.xy)
        noise=np.exp(np.log(.55)+.65*part.z[:,4])
        scores=np.stack([logsumexp(log_events(v,noise,test),axis=0)-np.log(len(noise)) for v in values])
        combined=logsumexp(scores,axis=0)-np.log(3)
        folds.append({"fold":f["fold"],"training_rows":train.row_ids.tolist(),"test_rows":test.row_ids.tolist(),
                      "reference_score":float(-scores[0].mean()),"bearing_score":float(-combined.mean()),
                      "parent_passed":f["numerics"]["passed"]})
    shown=full.reduced(256)
    phase=phases((4,64))
    def compare(xy):
        sims=fields(shown,support,phase,xy)
        return np.stack([np.median(sims[0],axis=0),np.median(np.concatenate(sims),axis=0)])
    paired_map(out,data,meta,name,compare,("Principal nested isopachs","Uncertain continuation in bearing gaps"))
    q = np.quantile(total[:,:3],PROBS).tolist()
    enhanced=np.quantile(total[:,3],PROBS).tolist()
    result={"protocol":"nested-bearing-sensitivity-v1","input_sha256":meta["input_sha256"],"history":history,
            "baseline_km3":np.quantile(full.volume(),PROBS).tolist(),"retained_volume_km3":q,"enhanced_hidden_fan_km3":enhanced,
            "sector_fraction":np.quantile(fractions,PROBS,axis=0).T.tolist(),"folds":folds,
            "paired_scores":{k:float(np.mean([f[k] for f in folds])) for k in ("reference_score","bearing_score")}}
    write_json(out/"report.json",result)
    scores=result["paired_scores"]
    page=[f"# {name} · How much ash is assigned to bearings with limited evidence?","",
          f"The uncertain-bearing median is **{q[1]:,.1f} km³** with a conditional 95% interval of **{q[0]:,.1f}–{q[2]:,.1f} km³**. Allowing a concentrated hidden fan to carry extra ash gives {enhanced[1]:,.1f} km³ [{enhanced[0]:,.1f}–{enhanced[2]:,.1f}]. The principal nested median is {result['baseline_km3'][1]:,.1f} km³.","",
          "Every training location keeps its fitted thickness. The existing spherical bearing policy progressively allows an unseen fan to turn beyond the reach of measured bearings. The same policy includes broad continuation as a possibility. The extra-fallout experiment normalizes the rotating fan to unit mean over an unknown direction.","",
          f"The paired geographic scores are {scores['reference_score']:.3f} for the nested control and {scores['bearing_score']:.3f} for uncertain bearings. The finite-Earth integration and posterior-draw refinement {'passed' if history[-1]['passed'] else 'retained a warning'}. The scenario depends on its assigned angular widths and hidden-fan probabilities.","",
          "![Nested contours with broad and uncertain continuation](isopachs.png)","",
          "[Sector volume fractions, matched holdouts and integration checks](report.json)"]
    (out/"README.md").write_text("\n".join(page)+"\n")
    return result
