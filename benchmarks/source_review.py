"""Compare fixed source-constrained candidates without choosing by total volume."""
import argparse
from dataclasses import replace
import json
from pathlib import Path
from src.empirical_isopach.core.catalog import datasets
from src.empirical_isopach.models.mass_mixture.prior import BASELINE as DISPLACED
from src.empirical_isopach.models.mass_mixture.source_protocol import BASELINE as FOCUSED
from src.empirical_isopach.models.mass_mixture.study import audited_data,fit_checked,cross_validate
from src.empirical_isopach.models.mass_mixture.source_checks import source_checks


CANDIDATES = {
    "source-fans":FOCUSED,
    "six-bases":replace(DISPLACED,components=6,vent_envelope=True,supported_basis=True,fan_concentration=1.),
    "twelve-bases":replace(DISPLACED,components=12,vent_envelope=True,supported_basis=True,fan_concentration=1.),
}


def run(output,slug):
    record=next(r for r in datasets() if r['slug']==slug)
    data,meta=audited_data(record['path'])
    output=Path(output)/slug
    result={'input':{k:v for k,v in meta.items() if k!='transform_inv'},'candidates':{},
        'interpretation':'Development comparison of fixed geometries. No volume-based selection and no automatic probability weights. These holdouts test observations, not unseen mass; post-comparison selection would need separate validation.'}
    for name,prior in CANDIDATES.items():
        samples,fit=fit_checked(data,meta['vent_lonlat'][1],prior,output/name/'baseline',progress=lambda s:print(s,flush=True))
        validation=cross_validate(data,meta['vent_lonlat'][1],output/name/'validation',baseline_prior=prior,progress=lambda s:print(s,flush=True))
        result['candidates'][name]={'fit':fit,'validation':validation,'source_checks':source_checks(samples)}
        (output/'review.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--dataset',choices=[r['slug'] for r in datasets()],required=True)
    args=parser.parse_args()
    run(args.output,args.dataset)
