"""Run independent parts of the full registered Mazama calculation on separate machines."""
import argparse
import csv
from dataclasses import replace
import json
from pathlib import Path
import subprocess
import sys

from benchmarks.mazama import INPUT, sha, write_csv

ROOT = Path('results/mazama-recompute')
PRINCIPAL = ROOT/'principal'
STUDIES = ROOT/'sources'
PHASE = ROOT/'phase'
FAMILIES = ('regional','directional','spherical-mixture','nested-isopachs')


def execute(*args):
    subprocess.run([sys.executable,'-m',*map(str,args)],check=True)


def spherical_cache(key):
    from src.empirical_isopach.core.audit import audited_data
    from src.empirical_isopach.core.catalog import dataset_record
    from src.empirical_isopach.core.validation import spatial_folds, training_mask
    from src.empirical_isopach.models.mass_mixture.inference import Sampling
    from src.empirical_isopach.models.mass_mixture.source_protocol import protocol
    from src.empirical_isopach.models.mass_mixture.study import fit_checked
    from src.empirical_isopach.validation import automatic_buffer
    data,meta=audited_data(INPUT)
    prior,alternatives=protocol(dataset_record(INPUT).get('global_geometry','displaced'))
    settings=Sampling()
    destination=PRINCIPAL/'spherical-mixture/posteriors'
    if key.startswith('fold-'):
        fold=int(key[5:])
        data=data.canonical()
        labels=spatial_folds(data.xy,4)
        train,_=training_mask(data,labels,fold,automatic_buffer(data)[0])
        data=data.subset(train)
        settings=replace(settings,seed=settings.seed+fold+1)
        directory=destination/'validation'/key
    else:
        prior=next(p for p in (prior,*alternatives) if p.name==key)
        directory=destination/key
    fit_checked(data,meta['vent_lonlat'][1],prior,directory,settings)


def principal(family):
    from src.empirical_isopach.core.catalog import resolve_dataset
    from src.empirical_isopach.models.continuation_review import directional_review, spherical_review
    path=resolve_dataset('Mazama')
    folder=PRINCIPAL/family
    if family=='regional':
        execute('src.empirical_isopach.cli','compare','Mazama','--regional-only','--output',PRINCIPAL)
    elif family=='spherical-mixture':
        execute('src.empirical_isopach.cli','global','Mazama','--resume','--output',folder)
        spherical_review(path,folder,folder/'sensitivity-study')
    elif family=='directional':
        execute('src.empirical_isopach.cli','global','Mazama','--geometry','directional','--output',folder)
        directional_review(path,folder,folder/'sensitivity-study',folder/'work/ensemble')
    elif family=='nested-isopachs':
        from src.empirical_isopach.models.nested.report import nested_dataset,whole_area_dataset
        whole=PRINCIPAL/'nested-whole-area'
        whole_area_dataset(path,whole)
        nested_dataset(path,folder,parent=whole)
    else:
        raise ValueError(family)


def source_case(case,family=None):
    from benchmarks import mazama_sensitivity as study
    from benchmarks.mazama_release import regional_details
    root=STUDIES
    study.prepare(root)
    for item in (FAMILIES if family is None else (family,)):
        study.run(root,case,item,PRINCIPAL)
    if family is None or family=='regional':
        regional_details(PRINCIPAL,root,only=case)


def pack(kind,name):
    """Keep source runs and method attempt receipts together without storing principal scratch twice."""
    import tarfile
    dest=Path('results/mazama-packages');dest.mkdir(parents=True,exist_ok=True)
    with tarfile.open(dest/f'{kind}-{name}.tar.gz','w:gz',compresslevel=3) as stream:
        def include(path):
            stream.add(path,arcname=path.relative_to(ROOT))
        if kind=='source':
            include(STUDIES/name)
        elif kind=='cache':
            p=PRINCIPAL/'spherical-mixture/posteriors'
            include(p/'validation'/name if name.startswith('fold-') else p/name)
        elif name=='regional':
            for path in sorted(PRINCIPAL.iterdir()):
                if path.name not in {'directional','spherical-mixture','nested-whole-area','nested-isopachs'}:
                    include(path)
        else:
            def publication(member):
                return None if 'work' in Path(member.name).parts or 'posteriors' in Path(member.name).parts else member
            families=('nested-whole-area','nested-isopachs') if name=='nested-isopachs' else (name,)
            for family in families:
                path=PRINCIPAL/family
                stream.add(path,arcname=path.relative_to(ROOT),filter=publication)


def list_jobs():
    from benchmarks.mazama_sensitivity import CASES
    from src.empirical_isopach.core.catalog import dataset_record
    from src.empirical_isopach.models.mass_mixture.source_protocol import protocol
    prior,alts=protocol(dataset_record(INPUT).get('global_geometry','displaced'))
    print('sources='+json.dumps(list(CASES)))
    print('caches='+json.dumps([p.name for p in (prior,*alts)]+[f'fold-{i}' for i in range(4)]))


if __name__=='__main__':
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument('operation',choices=('list','cache','principal','source','pack'))
    cli.add_argument('target',nargs='?')
    cli.add_argument('part',nargs='?')
    args=cli.parse_args()
    if args.operation=='list':
        list_jobs()
    elif args.operation=='cache':
        spherical_cache(args.target)
    elif args.operation=='principal':
        principal(args.target)
    elif args.operation=='source':
        source_case(args.target,args.part)
    else:
        pack(args.target,args.part)
