"""Run the principal spherical continuation on the exact directional challenge sites."""
import argparse

import numpy as np
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from .directional import trial
from .mass_mixture import SCENARIOS
from src.empirical_isopach.models.mass_mixture.bearings.field import BearingSupport
from src.empirical_isopach.models.mass_mixture.bearings.volume import checked_volumes
from src.empirical_isopach.models.directional.ensemble.fields import inverse_cdf
from src.empirical_isopach.models.mass_mixture.displacement import prior_ratio
from src.empirical_isopach.models.mass_mixture.inference import Sampling
from src.empirical_isopach.models.mass_mixture.prior import BASELINE
from src.empirical_isopach.models.mass_mixture.study import fit_cached
from src.empirical_isopach.models.nested.study import write_json


def run(job):
    name,seed,output = job
    folder = output/f"{name}-{seed}"
    data,totals = trial(name,seed)
    data = data.canonical()
    samples,fit = fit_cached(data,0.,BASELINE,folder,Sampling(seed=20261104+seed))
    vol,packet = checked_volumes(samples,BearingSupport(data,0.),parts=False,initial=256,ceiling=1024)
    mass = packet["volume_km3"]
    ratio = prior_ratio(samples).ravel()[packet["flat_index"]]
    candidate = inverse_cdf(mass.ravel(),np.repeat(ratio,mass.shape[1]),(.025,.5,.975))
    write_json(folder/"spherical.json",{"scenario":name,"seed":seed,"truths":totals,"volume":vol["volume"],
                                      "candidate_quantiles_km3":candidate.tolist(),"candidate_effective_draws":float(ratio.sum()**2/(ratio@ratio)),
                                      "numerical":vol["checks"],"parent":fit["diagnostics"],
                                      "posterior_sha256":fit["posterior_sha256"]})
    print(name,seed,vol["volume"],flush=True)


if __name__=="__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--workers",type=int,default=1)
    a = p.parse_args()
    with ProcessPoolExecutor(a.workers) as pool:
        list(pool.map(run,[(s,k,a.output) for k in (90,91) for s in (*SCENARIOS,"invisible_sector")]))
