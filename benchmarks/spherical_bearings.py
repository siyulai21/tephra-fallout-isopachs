"""Test spherical bearing continuation on known deposits from either kernel geometry."""
import argparse
from pathlib import Path
import numpy as np
from numpy.polynomial.legendre import leggauss

from .mass_mixture import case
from src.empirical_isopach.models.mass_mixture.bearings.field import BearingSupport
from src.empirical_isopach.models.mass_mixture.bearings.maps import BearingView, extent
from src.empirical_isopach.models.mass_mixture.bearings.study import read, save, verified
from src.empirical_isopach.models.mass_mixture.bearings.volume import checked_volumes


def weak_gap(points, original):
    r2 = ((points-[110.,110.])**2).sum(1)/80.**2
    return original(points)+np.log1p(-.99*np.maximum(1-r2,0.)**3)


def removed_volume(truth):
    # Independent Gaussian integration on the compact disk where the alternative differs.
    def rule(n):
        u,w=leggauss(n)
        r=(u+1)*40.; theta=(u+1)*np.pi
        x,y=np.meshgrid(r,theta)
        xy=np.column_stack((110.+x.ravel()*np.cos(y.ravel()),110.+x.ravel()*np.sin(y.ravel())))
        density=(np.exp(truth(xy))-np.exp(weak_gap(xy,truth))).reshape(x.shape)
        return float(1e-5*np.pi*40*(w[:,None]*w[None,:]*x*density).sum())
    a,b=rule(48),rule(96)
    np.testing.assert_allclose(a,b,rtol=1e-10)
    return b


def stress(cache,summary,output):
    output.mkdir(parents=True,exist_ok=True)
    for run in read(summary)["runs"]:
        s,k=run["scenario"],run["seed"]
        dest=output/f"{s}-{k}.json"
        data,total,truth=case(s,k,return_field=True)
        fit=run["fits"]["baseline"]
        samples=verified(cache/f"{s}-{k}"/"baseline"/fit["posterior_file"],fit)
        support=BearingSupport(data,0.)
        vol,_=checked_volumes(samples,support,parts=False,initial=256,ceiling=1024)
        targets={"original":truth}
        totals={"original":total}
        if s=="missing_sector":
            targets["weaker_hidden_fan"]=lambda xy:weak_gap(xy,truth)
            np.testing.assert_array_equal(targets["weaker_hidden_fan"](data.xy),truth(data.xy))
            totals["weaker_hidden_fan"]=total-removed_volume(truth)
        view=BearingView(samples,data,0.)
        _,areas=extent(view,targets,power=12)
        save(dest,{"scenario":s,"seed":k,"posterior_sha256":fit["posterior_sha256"],
             "sampler_passed":fit["diagnostics"]["passed"],"volumes_km3":totals,"reference":fit["volume"],
             "principal":vol,"areas":areas})
        print(s,k,flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cache",type=Path,required=True)
    p.add_argument("--summary",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    a=p.parse_args()
    stress(a.cache,a.summary,a.output)
