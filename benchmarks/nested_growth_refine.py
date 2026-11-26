"""Run the uniform larger population attempt for a recorded unsuccessful partition."""
import argparse
import json
from pathlib import Path

from src.empirical_isopach.core.audit import audited_data
from src.empirical_isopach.core.catalog import datasets
from src.empirical_isopach.models.directional.ensemble.members import splits
from src.empirical_isopach.models.nested.footprints.continuous import write_fit, engine
from src.empirical_isopach.models.nested.study import write_json


def refine(slug,key,output,plan):
    rec=next(r for r in datasets() if r['slug']==slug)
    data,meta=audited_data(rec['path'])
    assert not (data.lower_censored & ~data.interval).any()
    source=next(p for p in json.loads(plan.read_text()) if p['case']==slug and p['part']==key)
    assert not source['source_numerics_passed']
    train,test=data,None
    if key!='full':
        index=int(key.removeprefix('fold-'))
        _,train,test=next(p for p in splits(data,meta['vent_lonlat'][1],'geographic') if p[0]==index)
    identity=source['source_identity']
    assert identity['input_sha256']==meta['input_sha256'] and identity['engine_sha256']==engine()
    assert identity['train']==train.row_ids.tolist() and identity['test']==([] if test is None else test.row_ids.tolist())
    folder=output/slug/'refinement'
    result=write_fit(folder,key,train,test,meta,initial=12288,ceiling=12288)
    write_json(folder/(key+'-refinement-source.json'),source)
    print(slug,key,'uniform 12288-particle refinement',result['numerics']['passed'],flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('case')
    p.add_argument('part',choices=('full','fold-0','fold-1','fold-2','fold-3'))
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--plan',type=Path,default=Path('benchmarks/nested_growth_refinement_plan.json'))
    a=p.parse_args()
    refine(a.case,a.part,a.output,a.plan)
