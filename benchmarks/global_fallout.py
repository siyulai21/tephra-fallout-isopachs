"""Reproduce complete-Earth fallout for every catalogued eruption and method.

python -m benchmarks.global_fallout --output results/global-fallout
Retains accepted inputs, refits nested validation, compares alternate tails,
and records four geographic delete-block volumes per selected model.
"""
import argparse
from dataclasses import asdict
from hashlib import sha256
import json
from pathlib import Path
import numpy as np
import pandas as pd

from src.probabilistic_isopach.catalog import datasets
from src.probabilistic_isopach.cli import main as fit
from src.probabilistic_isopach.data import Observations, read_observations
from src.probabilistic_isopach.global_volume import global_volume
from src.probabilistic_isopach.model import Candidate, SpatialModel
from src.probabilistic_isopach.validation import select_model, training_mask


def synthetic(repeats):
    """Fixed global truths; hidden-lobe failures remain visible. No tuning here."""
    rows=[]
    for scenario in ('smooth','secondary_lobe','missing_sector'):
        def truth(xy):
            z=5-.018*np.linalg.norm(xy,axis=1)+.005*xy[:,0]
            if scenario!='smooth': z+=2.5*np.exp(-((xy[:,0]-130)**2+(xy[:,1]-100)**2)/(2*45**2))
            return z
        class Oracle:
            data=Observations([[-250,-250],[250,-250],[250,250],[-250,250]],np.zeros(4),np.zeros(4,bool),np.zeros(4),np.arange(4))
            def predict_log(self,xy):return truth(xy),np.zeros(len(xy))
        actual=global_volume(Oracle(),0.,tolerance=1e-7,orders=((48,128),(96,256),(192,512),(384,1024),(768,2048)))
        if not actual['converged']:raise RuntimeError('Synthetic truth integral did not converge.')
        for seed in range(repeats):
            rng=np.random.default_rng(seed+200);xy=rng.uniform(-250,250,(120,2))
            if scenario=='missing_sector':xy=xy[~((xy[:,0]>0)&(xy[:,1]>0))]
            xy=xy[:60]
            observed=truth(xy)+rng.normal(0,.25,len(xy))
            data=Observations(xy,observed,np.zeros(len(xy),bool),np.zeros(len(xy)),np.arange(len(xy)))
            model,_=select_model(data)
            result=global_volume(model,0.)
            rows.append({'scenario':scenario,'seed':seed,'truth_global_km3':actual['mean_km3'],
                         'estimated_global_km3':result['mean_km3'],
                         'relative_error':result['mean_km3']/actual['mean_km3']-1,
                         'converged':result['converged'],'candidate':asdict(model.candidate),
                         'interpretation':'Model recovery under a known simulated surface; not validation of real global deposits.'})
    return rows


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--synthetic-repeats',type=int,default=10)
    args=parser.parse_args(argv)
    if args.synthetic_repeats<1 or (args.output.exists() and any(args.output.iterdir())):
        parser.error('Use positive repeats and a new output directory.')
    args.output.mkdir(parents=True,exist_ok=True)
    comparison=[];deleted=[]
    for record in datasets():
        slug=record['slug'];data,metadata=read_observations(record['path'])
        for method in ('selected','stacked'):
            output=args.output/slug/method
            fit([record['name'],'--output',str(output),'--name',record['name'],'--method',method,
                 '--nested-cv','--grid','128','--volume-grid','48','--draws','1500','--seed','1729'])
            report=json.loads((output/'report.json').read_text())
            report['reproducibility']['benchmark_sha256']=sha256(Path(__file__).read_bytes()).hexdigest()
            (output/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
            globe=report['global_fallout'];g=globe['selected'];alt=globe['alternative'];nv=report['nested_validation']
            comparison.append({'dataset':record['name'],'slug':slug,'method':method,'n_measurements':metadata['n_measurements'],
                'n_lower_limits':metadata['n_lower_limits'],'expected_global_km3':g['mean_km3'],
                'global_median_surface_km3':g['median_surface_km3'],
                'alternate_family_global_km3':None if alt is None else alt['mean_km3'],
                'regional_km3':report['volume_analysis']['regional']['mean_km3'],
                'fraction_beyond_farthest_observation':g['fraction_beyond_farthest_observation'],
                'fraction_on_opposite_hemisphere':g['fraction_on_opposite_hemisphere'],
                'global_converged':g['converged'],
                **{k:nv[k] for k in ('mean_log_score','rmse_log','coverage95_observations')}})
            if method=='selected':
                for selection in nv['selections']:
                    fold=selection['fold'];train,test=training_mask(data,np.array(nv['folds']),fold,report['settings']['buffer_km'])
                    # Candidate learned on that outer training split, not on held-out responses.
                    candidate=Candidate(**selection['selected'])
                    model=SpatialModel(candidate).fit(data.subset(train))
                    value=global_volume(model,metadata['vent_lonlat'][1])
                    deleted.append({'dataset':record['name'],'fold':fold,'n_train':int(train.sum()),
                                    'n_withheld':int(test.sum()),'expected_global_km3':value['mean_km3'],
                                    'converged':value['converged'],'candidate':asdict(candidate),
                                    'fraction_beyond_farthest_training_observation':value['fraction_beyond_farthest_observation'],
                                    'interpretation':'Geographic deletion/refitting sensitivity, not a confidence interval.'})
        print('Completed global fits:',record['name'],flush=True)
        pd.DataFrame(comparison).to_csv(args.output/'comparison.csv',index=False)
        (args.output/'geographic_deletions.json').write_text(json.dumps(deleted,indent=2,allow_nan=False)+'\n')
    simulation=synthetic(args.synthetic_repeats)
    (args.output/'synthetic.json').write_text(json.dumps(simulation,indent=2,allow_nan=False)+'\n')
    print('Completed synthetic global-volume trials:',len(simulation),flush=True)


if __name__=='__main__':main()
