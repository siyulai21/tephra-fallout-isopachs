"""Replay the principal continuous-growth packets and their actual geographic events."""
import argparse
import gzip
import json
from pathlib import Path
from hashlib import sha256
import tarfile

import numpy as np
import pandas as pd

from .publication import result_home
from .verify_deposit_maps import verify_global
from src.empirical_isopach.core.audit import audited_data
from src.empirical_isopach.core.catalog import datasets
from src.empirical_isopach.models.directional.ensemble.members import splits
from src.empirical_isopach.models.nested.footprints.continuous import FlowField,PROTOCOL,engine,predictions
from src.empirical_isopach.models.nested.footprints.diagnostics import population_field_check
from src.empirical_isopach.models.nested.study import digest


def read(path):
    return json.loads(path.read_text())


def verify_case(record,home):
    data,meta=audited_data(record['path'])
    folder=home/'footprints'
    report=read(folder/'report.json')
    assert report['protocol']==PROTOCOL and report['engine_sha256']==engine()
    assert report['input_sha256']==meta['input_sha256'] and report['accepted_rows']==len(data.xy)
    assert {p:digest(folder/p) for p in report['posterior_sha256']}==report['posterior_sha256']
    assert digest(folder/'heldout.csv.gz')==report['heldout_sha256']
    assert digest(folder/'support.json.gz')==report['support_sha256']
    held=pd.read_csv(folder/'heldout.csv.gz')
    assert held.row_id.is_unique and set(held.row_id)==set(data.row_ids)
    signature=engine()
    for index,train,test in [(None,data,None),*splits(data,meta['vent_lonlat'][1],'geographic')]:
        key='full' if index is None else f'fold-{index}'
        fit=report['fits'][key]
        assert fit['train']==train.row_ids.tolist()
        assert fit['test']==([] if test is None else test.row_ids.tolist())
        assert not set(fit['train']) & set(fit['test'])
        for trial in fit['attempts']:
            identity=trial['identity']
            assert identity['protocol']==PROTOCOL and identity['engine_sha256']==signature
            assert identity['input_sha256']==meta['input_sha256']
            assert identity['train']==fit['train'] and identity['test']==fit['test']
            for stage in trial['numerics']['attempts']:
                for label,quantiles in (('intrinsic_passed',stage['intrinsic_volume_quantiles']),
                                         ('continuation',stage['continuation']['population_volume_quantiles_km3'])):
                    span=np.ptp(np.log(quantiles),axis=0)
                    passed=bool(span[1]<=np.log(1.35) and max(span[[0,2]])<=np.log(2.))
                    assert passed==(stage[label]['passed'] if label=='continuation' else stage[label])
                spatial=stage['spatial_populations']
                assert spatial['passed']==(spatial['median_log_cm']<.15 and spatial['p95_log_cm']<.4)
        for checkpoint in fit['checkpoints']:
            assert digest(folder/checkpoint['packet'])==checkpoint['posterior_sha256']
            assert checkpoint['identity']['engine_sha256']==signature
            with np.load(folder/checkpoint['packet'],allow_pickle=False) as a:
                assert len(a['z'])==4*checkpoint['numerics']['particles_per_population']
        retained={fit['posterior_sha256'],*(c['posterior_sha256'] for c in fit['checkpoints'])}
        assert all(a['posterior_sha256'] in retained for a in fit['attempts'])
        model=FlowField.load(folder/(key+'-selected.npz'))
        assert len(model.z)==fit['draws'] and model.warp.shape==(len(model.z),2)
        np.testing.assert_allclose(np.quantile(model.volume(),[.025,.5,.975]),fit['volume_quantiles_km3'])
        final=fit['attempts'][-1]['numerics']['attempts'][-1]
        raw_pass=final['intrinsic_passed'] and final['continuation']['passed'] and final['spatial_populations']['passed']
        checks=fit.get('field_population_checks',[])
        assert fit['passed']==bool(raw_pass and (not checks or checks[-1]['check']['passed']))
        if index is None:
            assert checks and checks[-1]['identity']['posterior_sha256']==fit['posterior_sha256']
            actual=population_field_check(model,train)
            for name,value in actual.items():
                np.testing.assert_allclose(value,checks[-1]['check'][name],atol=1e-10,rtol=1e-10)
            np.testing.assert_allclose(np.quantile(model.volume(),[.025,.5,.975]),
                                      [report['conditional_interval95_km3'][0],report['complete_draw_median_km3'],report['conditional_interval95_km3'][1]])
        else:
            expected=pd.DataFrame(predictions(model,test))
            actual=held.loc[held.fold==index]
            assert actual.row_id.tolist()==expected.row_id.tolist()
            for column in ('exact','evidence'):
                assert actual[column].tolist()==expected[column].tolist()
            for column in ('log_score','cdf','median_log_cm','latent_lower95_log_cm','latent_upper95_log_cm'):
                np.testing.assert_allclose(actual[column],expected[column],rtol=1e-9,atol=1e-9)
    scores=-held.groupby('fold').log_score.mean()
    np.testing.assert_allclose(scores.to_numpy(),report['validation']['fold_scores'])
    np.testing.assert_allclose(scores.mean(),report['validation']['negative_log_score'])
    exact=held.loc[held.exact]
    np.testing.assert_allclose(exact.cdf.between(.025,.975).mean(),report['validation']['coverage95_exact'])
    assert report['validation']['passed_folds']==sum(report['fits'][f'fold-{i}']['passed'] for i in range(4))
    assert report['validation']['all_folds_passed']==(report['validation']['passed_folds']==4)
    for trial in report['independent_draw_integration']:
        assert trial['passed']==(trial['median_error']<.01 and trial['p95_error']<.05)
    assert report['draw_checks_passed']==bool(report['fits']['full']['passed'] and report['independent_draw_integration'][-1]['passed'])
    screen=json.loads(gzip.decompress((folder/'support.json.gz').read_bytes()))
    assert screen['status']!='completed' or set(screen['policies'])=={'strict','standard','permissive'}
    sectors=pd.read_csv(folder/'sector-volumes.csv')
    sector=read(folder/'sector-checks.json')
    assert sector['table_sha256']==digest(folder/'sector-volumes.csv')
    assert sector['input_sha256']==meta['input_sha256']
    assert int(sectors[['exact','lower','upper','interval']].sum().sum())+sector['unassigned_vent_sites']==len(data.xy)
    np.testing.assert_allclose(sectors.median_map_fraction.sum(),1.)
    np.testing.assert_allclose(sectors.median_map_km3.sum(),sector['median_map_km3'])
    verify_global(home,'nested-isopachs',data,meta)
    print(record['slug'],'continuous geometry, checkpoints, geographic predictions and map replay verified',flush=True)


def main(release=None):
    archive=Path('archive/2026-11-25-nested-growth')
    for name,expected in read(archive/'previous-files.json').items():
        assert digest(archive/name)==expected, name
    for name,record in read(archive/'development/files.json').items():
        path=archive/'development'/name
        assert digest(path)==record['sha256'] and path.stat().st_size==record['bytes']
        with tarfile.open(path,'r:gz') as packed:
            assert set(packed.getnames())==set(record['files'])
            for member in packed:
                assert member.isfile() and not Path(member.name).is_absolute() and '..' not in Path(member.name).parts
                assert sha256(packed.extractfile(member).read()).hexdigest()==record['files'][member.name]
    for record in datasets():
        home=release/record['slug'] if release else result_home(record)/'nested-isopachs'
        verify_case(record,home)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--release',type=Path)
    main(parser.parse_args().release)
