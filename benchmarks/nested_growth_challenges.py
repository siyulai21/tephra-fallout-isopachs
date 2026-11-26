"""Known deposits and actual withheld surveys for positive angular area growth."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .footprint_challenges import rotated_spokes, synthetic_id
from .mass_mixture import SCENARIOS
from .nested_reach_challenge import ONE_SIDE, known_design
from .nested_reach_field import partition, FIELD_CASES
from src.empirical_isopach.core.audit import audited_data
from src.empirical_isopach.core.catalog import resolve_dataset
from src.empirical_isopach.core.fitting import model_input
from src.empirical_isopach.models.nested.footprints import legacy_posterior
from src.empirical_isopach.models.nested.footprints.continuous import FlowField as FootprintField
from .nested_growth_validation import fit_part
from src.empirical_isopach.models.nested.reach_integrals import nodes
from src.empirical_isopach.models.nested.report import MapView
from src.empirical_isopach.models.nested.study import digest, write_json


def selected(folder, receipt, key):
    return folder/('refinement' if receipt['identity']['initial']==12288 else '')/(key+'.npz')


def known(job):
    name, seed, output, previous_only = job
    folder=output/f'{name}-{seed}'
    folder.mkdir(parents=True,exist_ok=True)
    data,truths=rotated_spokes(name) if name.startswith('survey_rotation') else known_design(name,seed)
    assert not (data.lower_censored & ~data.interval).any(), 'Translate lower-only observations before fitting'
    meta=dict(input_sha256=synthetic_id(data),vent_lonlat=[0.,0.])
    if previous_only:
        receipt=legacy_posterior.write_fit(folder,'previous',data,None,meta)
        if not receipt['numerics']['passed']:
            receipt=legacy_posterior.write_fit(folder/'refinement','previous',data,None,meta,initial=12288,ceiling=12288)
        model=legacy_posterior.FootprintField.load(selected(folder,receipt,'previous'))
    else:
        receipt=fit_part(folder,'growth',data,None,meta)
        model=FootprintField.load(selected(folder,receipt,'growth'))
    view=MapView(model.reduced(1024))
    quantiles=np.quantile(model.volume(),[.025,.5,.975])
    rows=[]
    for scramble in (42983,84779):
        xy,weight,_,theta=nodes(14,scramble,0.)
        weight/=len(weight)
        fitted=view.central(xy)*np.log(10.)
        volume=1e-5*weight@np.exp(fitted)
        for key,(truth,total) in truths.items():
            actual=truth(xy)
            for cm in (.5,1.,10.,100.):
                present,predicted=actual>=np.log(cm),fitted>=np.log(cm)
                rows.append(dict(scenario=name,seed=seed,input_sha256=meta['input_sha256'],truth=key,
                                 method='previous footprints' if previous_only else 'area growth',scramble=scramble,level_cm=cm,true_km3=total,
                                 median_map_km3=volume,true_area_km2=weight[present].sum(),
                                 false_km2=weight[predicted & ~present].sum(),missed_km2=weight[present & ~predicted].sum(),
                                 west_false_km2=weight[predicted & ~present & (np.sin(theta)<0)].sum(),
                                 west_missed_km2=weight[present & ~predicted & (np.sin(theta)<0)].sum(),
                                 lower95=quantiles[0],upper95=quantiles[-1],fit_passed=receipt['numerics']['passed']))
    pd.DataFrame(rows).to_csv(folder/('previous-recovery.csv' if previous_only else 'recovery.csv'),index=False)
    print(name,seed,'previous' if previous_only else 'growth',receipt['numerics']['passed'],flush=True)


def survey(job):
    name,output,previous_only=job
    folder=output/name
    folder.mkdir(parents=True,exist_ok=True)
    if name=='pinatubo-all-offshore':
        path=resolve_dataset('pinatubo-1991-layer-c')
        data,meta=audited_data(path)
        audit=pd.read_csv('data/audits/Pinatubo-1991/full-evidence.csv')
        registered=pd.read_csv(path)
        marine=audit.loc[audit.evidence_id.str.startswith(('WW-','W04-','H01-','S99-')),'evidence_id']
        held=registered.EvidenceID.isin(marine).to_numpy()[data.row_ids-2]
        # The current reviewed edition also contains WW-SCS-C: 35 accepted
        # marine records. Identify the campaign through the source ledger,
        # rather than carrying the previous edition's 34-row count forward.
        assert held.any() and (~held).any()
        train,test=data.subset(~held),data.subset(held)
        write_json(folder/'partition.json',dict(input_sha256=meta['input_sha256'],
                   source_audit_sha256=digest('data/audits/Pinatubo-1991/full-evidence.csv'),
                   train=train.row_ids.tolist(),test=test.row_ids.tolist(),
                   held_evidence_ids=registered.iloc[test.row_ids-2].EvidenceID.tolist()))
    else:
        train,test,meta,path=partition(name,folder)
        if (train.lower_censored & ~train.interval).any() or (test.lower_censored & ~test.interval).any():
            # Historical/external campaigns are not necessarily registered
            # fitting editions. Preserve their CSV and stable row IDs while
            # applying exactly the same minimum-as-point policy as new cases.
            fitted=model_input(path,folder/'fitting')
            data,meta=audited_data(fitted)
            train_ids,test_ids=train.row_ids,test.row_ids
            train=data.subset(np.isin(data.row_ids,train_ids))
            test=data.subset(np.isin(data.row_ids,test_ids))
            write_json(folder/'partition.json',dict(input_sha256=meta['input_sha256'],
                       reported_source_sha256=digest(path),fitting_table=fitted.relative_to(folder).as_posix(),
                       train=train.row_ids.tolist(),test=test.row_ids.tolist(),
                       reported_minima_fitted_as_points=int(data.reported_minimum.sum())))
    assert not (train.lower_censored & ~train.interval).any()
    assert not (test.lower_censored & ~test.interval).any()
    # The preceding model is actually refitted to the current training rows;
    # older source editions cannot stand in for this matched comparison.
    old=legacy_posterior.write_fit(folder,'previous',train,test,meta)
    if not old['numerics']['passed']:
        old=legacy_posterior.write_fit(folder/'refinement','previous',train,test,meta,initial=12288,ceiling=12288)
    if previous_only:
        pd.DataFrame(old['heldout']).to_csv(folder/'previous-predictions.csv',index=False)
        print(name,'previous footprints',old['numerics']['passed'],flush=True)
        return
    new=fit_part(folder,'growth',train,test,meta)
    rows=[dict(survey=name,method=label,input_sha256=meta['input_sha256'],fit_passed=r['numerics']['passed'],**v)
          for label,r in [('previous footprints',old),('area growth',new)] for v in r['heldout']]
    pd.DataFrame(rows).to_csv(folder/'predictions.csv',index=False)
    print(name,old['numerics']['passed'],new['numerics']['passed'],flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--workers',type=int,default=2)
    p.add_argument('--design',choices=('known','surveys'),required=True)
    p.add_argument('--previous-only',action='store_true')
    p.add_argument('--case',action='append',help='Known-deposit design or actual survey to include')
    p.add_argument('--shard-index',type=int,default=0)
    p.add_argument('--shard-count',type=int,default=1)
    a=p.parse_args()
    if a.design=='known':
        jobs=[(s,k,a.output) for k in (90,91) for s in (*SCENARIOS,'invisible_sector')]
        jobs += [(s,k,a.output) for k in (1201,1202) for s in ONE_SIDE]
        jobs += [(s,56421,a.output) for s in ('survey_rotation_0','survey_rotation_23')]
        jobs = [(*job,a.previous_only) for job in jobs]
    else:
        jobs=[(s,a.output,a.previous_only) for s in (*FIELD_CASES,'pinatubo-all-offshore')]
    if not 0<=a.shard_index<a.shard_count:
        p.error('Require 0 <= shard-index < shard-count')
    jobs=[job for job in jobs if not a.case or job[0] in a.case]
    jobs=jobs[a.shard_index::a.shard_count]
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        list(pool.map(known if a.design=='known' else survey,jobs))
