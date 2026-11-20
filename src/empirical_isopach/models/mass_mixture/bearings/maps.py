"""Principal spherical fields, extent checks, and independent known-truth map errors."""
import numpy as np
from scipy.stats import qmc

from ....visualization.global_maps import PosteriorView
from ...directional.ensemble.integrals import polar_nodes
from ..geometry import RADIUS
from .field import BearingField, BearingSupport, phases, selected
from .volume import SECTORS

LEVELS = (1.,10.,100.)


class BearingView(PosteriorView):
    def __init__(self,samples,data,latitude,per_chain=96):
        from src.probabilistic_isopach.global_volume import EarthDomain
        self.g = None
        self.latitude,self.domain = latitude,EarthDomain.wgs84(latitude)
        self.samples,self.indices,shape = selected(samples,per_chain)
        self.n = len(self.indices)
        self.field = BearingField(self.samples,BearingSupport(data,latitude),phases(shape))

    def evaluate(self,xy):
        xy = np.asarray(xy)
        out = np.full((3,len(xy)),np.nan)
        valid = np.flatnonzero(self.domain.rho(xy)<self.domain.rho_max)
        for start in range(0,len(valid),256):
            ix = valid[start:start+256]
            out[:,ix] = np.quantile(np.concatenate(self.field.components(xy[ix])),[.025,.5,.975],axis=0)/np.log(10.)
        return out

    @property
    def statistic(self):
        return "Bearing-conditioned spherical median"


def medians(view,xy,*,split=False):
    result = []
    for start in range(0,len(xy),256):
        fields = view.field.components(xy[start:start+256])
        all_fields = np.stack(fields)
        values = [np.median(fields[0],axis=0),np.median(np.concatenate(fields),axis=0)]
        if split:
            for sl in (slice(None,None,2),slice(1,None,2)):
                subset=all_fields[:,sl]
                values.append(np.median(subset.reshape(-1,subset.shape[-1]),axis=0))
        result.append(np.stack(values))
    return np.concatenate(result,axis=1)


def extent(view,truths=None,power=13):
    """Area and false/missed area are measured over the physical Earth in km²."""
    summary,errors = [],[]
    true_fields = truths or {}
    for seed in (48529,29111):
        uv = qmc.Sobol(2,scramble=True,seed=seed).random_base2(power)
        xy,area,_,angle = polar_nodes(uv,view.field.support.axis)
        area = area/len(xy)
        h,_ = view.field.support.evaluate(xy)
        sectors = np.floor((angle+np.pi/8)/(np.pi/4)).astype(int)%8
        fields = medians(view,xy,split=True)
        truth = {key:f(xy) for key,f in true_fields.items()}
        for cm in LEVELS:
            old,new,a,b = fields>=np.log(cm)
            if np.any(new&~old):
                raise ArithmeticError("A median continuation exceeded its paired reference")
            for i,name in enumerate(SECTORS):
                m = sectors==i
                summary.append({"scramble":seed,"sector":name,"level_cm":cm,"reference_km2":float(area[m&old].sum()),
                    "principal_km2":float(area[m&new].sum()),"less_extent_km2":float(area[m&old&~new].sum()),
                    "high_exposure_reference_km2":float(area[m&old&(h>=.5)].sum()),
                    "high_exposure_principal_km2":float(area[m&new&(h>=.5)].sum()),
                    "split_disagreement_km2":float(area[m&(a!=b)].sum()),"split_union_km2":float(area[m&(a|b)].sum()),
                    "split_union_area_squared_weights":float((area[m&(a|b)]**2).sum())})
            for target,z in truth.items():
                true=z>=np.log(cm)
                for mask,label in ((np.ones(len(xy),bool),"Earth"),(h>=.5,"gap exposure >= 0.5")):
                    for pred,model in ((old,"reference"),(new,"principal")):
                        union = mask&(pred|true)
                        errors.append({"truth":target,"model":model,"region":label,"scramble":seed,"level_cm":cm,
                             "false_area_km2":float(area[mask&pred&~true].sum()),"missed_area_km2":float(area[mask&~pred&true].sum()),
                             "intersection_km2":float(area[mask&pred&true].sum()),"union_km2":float(area[union].sum()),
                             "union_area_squared_weights":float((area[union]**2).sum())})
    return summary,errors


def ray_crossings(view):
    out = []
    r = np.r_[0.,np.geomspace(.05,2*RADIUS*(1-1e-10),480)]
    for i,name in enumerate(SECTORS):
        theta = i*np.pi/4
        axis = view.field.support.axis
        xy = r[:,None]*np.array([np.sin(theta)*axis,np.cos(theta)/axis])
        fields = medians(view,xy)
        for z,kind in zip(fields,("reference","principal"),strict=True):
            for cm in LEVELS:
                inside = z>=np.log(cm)
                changes = np.flatnonzero(inside[:-1]!=inside[1:])
                cross = [(r[j]+(r[j+1]-r[j])*(np.log(cm)-z[j])/(z[j+1]-z[j])) for j in changes]
                dist = [float(2*RADIUS*np.arcsin(np.clip(v/(2*RADIUS),0,1))) for v in cross]
                out.append({"sector":name,"level_cm":cm,"method":kind,"source_above_level":bool(inside[0]),
                            "reaches_antipode":bool(inside[-1]),"great_circle_crossings_km":dist})
    return out


def extent_diagnostics(rows):
    """Check map splits only where the contour union has enough area-weighted nodes."""
    from collections import defaultdict
    groups=defaultdict(list)
    for row in rows:
        for key in ((row["sector"],row["level_cm"]),("Earth",row["level_cm"])):
            groups[key].append(row)
    checks=[]
    for (sector,cm),part in groups.items():
        union=sum(r["split_union_km2"] for r in part)
        squared=sum(r["split_union_area_squared_weights"] for r in part)
        effective=union**2/squared if squared else 0.
        split=sum(r["split_disagreement_km2"] for r in part)/union if union else None
        checks.append({"sector":sector,"level_cm":cm,"union_effective_nodes":effective,"resolved":bool(effective>=64),
                       "split_symmetric_difference_fraction":split})
    return checks


def select_extent(samples,data,latitude,*,ceiling=384):
    history=[]
    for count in (96,192,384):
        view=BearingView(samples,data,latitude,per_chain=count)
        rows,_=extent(view)
        checks=extent_diagnostics(rows)
        warnings=[r for r in checks if r["resolved"] and r["split_symmetric_difference_fraction"]>=.10]
        history.append({"per_chain":min(count,samples["log_volume"].shape[1]),"warnings":warnings,"checks":checks})
        if not warnings or count>=min(ceiling,samples["log_volume"].shape[1]):
            break
    return view,rows,{"draws_per_chain":history[-1]["per_chain"],"passed":not bool(warnings),"history":history}
