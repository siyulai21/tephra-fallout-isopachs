from pathlib import Path
import json,platform,importlib.metadata as md
from benchmarks.publication import result_home
from src.empirical_isopach.core.catalog import datasets
from src.empirical_isopach.models.directional.multiscale import engine_sha256
from src.empirical_isopach.models.directional.multiscale.geometry import SCHEMA
from src.empirical_isopach.visualization.deposit_field import digest
study=Path('comparisons/directional/geographic-regularity')
cases={}
for r in datasets():
 home=result_home(r)/'directional'
 paths={'input_sha256':r['path'],'fit_report_sha256':home/'multiscale/report.json','posterior_sha256':home/'multiscale/posterior.npz','figure_sha256':home/'deposit.png','map_receipt_sha256':home/'deposit.json','median_grid_sha256':home/'median-grid.npz','probability_figure_sha256':home/'exceedance.png','probability_receipt_sha256':home/'exceedance.json','fold_marginals_sha256':study/'fold-marginals'/(r['slug']+'.npz'),'paired_figure_sha256':study/(r['slug']+'.png'),'paired_figure_receipt_sha256':study/(r['slug']+'.json')}
 cases[r['slug']]={k:digest(p) for k,p in paths.items()}
 assert json.loads((home/'multiscale/report.json').read_text())['fit']['engine_sha256']==engine_sha256()
 sources=list(Path('src/empirical_isopach/models/directional/multiscale').glob('*.py'))
sources+=list(Path('src/empirical_isopach/visualization').glob('*.py'))
sources+=list(Path('benchmarks').glob('directional_*.py'))
sources+=[Path('benchmarks/verify_directional_geography.py'),Path('benchmarks/verify_directional_multiscale.py'),Path('benchmarks/regional.py'),Path('benchmarks/footprint_challenges.py'),Path('benchmarks/nested_reach_challenge.py'),Path('benchmarks/nested_reach_field.py'),Path('benchmarks/azimuth_summary.py'),Path('benchmarks/principal_volumes.py'),Path('benchmarks/spherical_bearing_figure.py'),Path('data/datasets.json')]
record=dict(protocol=SCHEMA,numerical_engine_sha256=engine_sha256(),states_per_partition=864,baseline_revision='beee82b7020d62aa2c3dc847d74e45dcf651084f',baseline='archive/2026-11-25-directional-v5/current-cases',baseline_provenance_sha256=digest(Path('archive/2026-11-25-directional-v5/current-cases/provenance.json')),observation_policy='lower-only-at-reported-minimum-v1',cases=cases,software={'python':platform.python_version(),**{k:md.version(k) for k in ('numpy','scipy','pandas','matplotlib','pyproj','shapely')}},seeds=dict(fit=173071,complete_parameter=[34141,79031],complete_spatial=[51071,68389],known_deposits=[90,91],turning_truth=85613),source_files_sha256={str(p):digest(p) for p in sorted(set(sources))},study_files_sha256={str(p.relative_to(study)):digest(p) for p in sorted(study.rglob('*')) if p.is_file() and p.name!='provenance.json'},qualification=dict(estimable_partitions=78,empty_test_partitions=3,accepted_rows=1091,point_rows=883,interval_rows=61,upper_rows=147,synthetic_fits=16,survey_fits=8,coherent_fraction_fits=27),selection_scope='Common method selected using development visual reviews, geographic deletions and synthetic challenges; no eruption-specific settings or volume targets.',commands=['python -m benchmarks.directional_multiscale --work results/directional-geography-v6/principal-fits --jobs 3','python -m benchmarks.directional_release --work results/directional-geography-v6/principal-fits --output results/directional-geography-v6/principal-release --jobs 2','python -m benchmarks.directional_scale_maps --work results/directional-geography-v6/principal-fits --output results/directional-geography-v6/principal-study --baseline archive/2026-11-25-directional-v5/current-cases --jobs 2','See docs/methods/directional-multiscale.md for matched evaluation, morphology, local-feature, sensitivity, campaign and synthetic commands.'])
(study/'provenance.json').write_text(json.dumps(record,indent=2,allow_nan=False)+'\n')
print(engine_sha256(),len(record['study_files_sha256']),'study artifacts identified')
