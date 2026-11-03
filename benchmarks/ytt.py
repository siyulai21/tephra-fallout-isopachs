"""Reproduce the audited YTT inputs, matched method comparison and reports.

python -m benchmarks.ytt --output results/ytt
Source decisions are explicit in selection_rules.json and record_review.csv.
"""
import argparse
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
import pandas as pd
from src.probabilistic_isopach.cli import main as fit
from src.probabilistic_isopach.publication import write_short_report

ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT/'data/audits/YTT'
from benchmarks.ytt_inputs import prepare_inputs, verify_inputs

LEVELS = [.01, .03, .1, .3, 1, 3, 10, 30, 100, 300]
BOUNDS = [-6900, -1400, 2400, 2900]
DEFINITIONS = [('principal/selected', 'principal', 'selected'),
               ('principal/stacked', 'principal', 'stacked'),
               ('sensitivity/without_maninjau', 'without_maninjau', 'selected'),
               ('sensitivity/without_u1445b', 'without_u1445b', 'selected'),
               ('sensitivity/without_md161', 'without_md161', 'selected'),
               ('sensitivity/with_u1448b', 'with_u1448b', 'selected')]


def summarize(output):
    definitions = DEFINITIONS
    reports = {key: json.loads((output/key/'report.json').read_text()) for key, _, _ in definitions}
    manifest = pd.read_csv(AUDIT/'selection_manifest.csv')
    comparisons, sites, folds = [], [], []
    for key, case, method in definitions:
        r = reports[key]; v = r['nested_validation']; regional = r['volume_analysis']['regional']
        comparisons.append({'run': key, 'measurements': r['input']['n_measurements'], 'method': method,
                            **{metric: v[metric] for metric in ('mean_log_score', 'equal_region_log_score', 'rmse_log', 'coverage95_observations', 'mean_interval_width_log')},
                            'expected_regional_km3': regional['mean_km3'], 'median_surface_km3': regional['median_surface_km3'],
                            'adjusted_log_score': v['discrepancy_adjusted']['mean_log_score'],
                            'adjusted_coverage95': v['discrepancy_adjusted']['coverage95_observations']})
        lookup = manifest[manifest[f'included_{case}'] == 1].set_index(f'{case}_csv_row')
        run_sites = []
        for i, rowid in enumerate(v['row_ids']):
            source = lookup.loc[rowid]; observed = float(source[f'{case}_thickness_cm'])
            lo, median, hi = np.exp([v['lower95_log'][i], v['median_log'][i], v['upper95_log'][i]])
            run_sites.append({'run': key, 'record_id': source.record_id, 'site': source.site, 'source_row': rowid,
                              'fold': v['folds'][i], 'observed_cm': observed, 'predicted_median_cm': median,
                              'lower95_cm': lo, 'upper95_cm': hi, 'covered': bool(lo <= observed <= hi),
                              'negative_log_score': -v['point_log_scores'][i]})
        sites.extend(run_sites)
        for label in sorted(set(v['folds'])):
            region = [x for x in run_sites if x['fold'] == label]
            folds.append({'run': key, 'fold': label, 'observations': len(region),
                          'covered': sum(x['covered'] for x in region),
                          'negative_log_score': np.mean([x['negative_log_score'] for x in region])})
    pd.DataFrame(comparisons).to_csv(output/'comparison.csv', index=False)
    pd.DataFrame(sites).to_csv(output/'site_holdouts.csv', index=False)
    pd.DataFrame(folds).to_csv(output/'fold_comparison.csv', index=False)
    a, b = reports['principal/selected'], reports['principal/stacked']
    assert a['nested_validation']['row_ids'] == b['nested_validation']['row_ids']
    assert a['nested_validation']['folds'] == b['nested_validation']['folds']
    same_fit = len(b['model']['components']) == 1 and b['model']['components'][0]['model'] == a['model']
    same_grids = np.allclose(pd.read_csv(output/'principal/selected/predictions.csv')['conditional_mean_cm'],
                            pd.read_csv(output/'principal/stacked/predictions.csv')['conditional_mean_cm'], rtol=1e-12, atol=1e-12)
    av, bv = a['nested_validation'], b['nested_validation']
    # Changing accepted observations must not encode the previous score ordering
    # as a success condition. Report the new comparison for scientific review.
    decision = {'default': 'selected', 'promote_stacked': False,
                'reason': 'This revision promotes documented primary-fallout observations into the principal input. The established method remains the default; assess the new matched holdout metrics before any separate method replacement.',
                'principal_case': 'principal',
                'principal_same_final_component': same_fit, 'principal_same_expected_thickness_grid': bool(same_grids),
                'metrics': {name: {k: v[k] for k in ('mean_log_score', 'equal_region_log_score', 'rmse_log', 'coverage95_observations', 'mean_interval_width_log')}
                            for name, v in [('selected', av), ('stacked', bv)]},
                'negative_log_score_difference_stacked_minus_selected': b['nested_validation']['mean_log_score']-a['nested_validation']['mean_log_score'],
                'criterion': 'Proper predictive score is primary; also inspect coverage, width and point error. A small point-error gain alone does not establish superiority.',
                'input_verification': verify_inputs(),
                'reproduce': 'python -m benchmarks.ytt --output results/ytt'}
    (output/'method_decision.json').write_text(json.dumps(decision, indent=2, allow_nan=False)+'\n')
    return comparisons, decision


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--prepare-only', action='store_true', help='Rebuild inputs and provenance from explicit reviewed rules.')
    args = parser.parse_args(argv)
    if args.prepare_only:
        prepare_inputs()
        print(json.dumps(verify_inputs(), indent=2))
        return
    if args.output is None:
        parser.error('--output is required unless --prepare-only is used')
    if args.output.exists() and any(args.output.iterdir()):
        parser.error('Choose a new output directory to preserve earlier runs.')
    verify_inputs()
    provenance = json.loads((AUDIT/'provenance.json').read_text())
    cases = provenance['cases']
    for name, case, method in DEFINITIONS:
        titles = {'principal': 'principal 46', 'without_maninjau': 'omit Maninjau',
                  'without_u1445b': 'omit U1445B', 'without_md161': 'omit MD161',
                  'with_u1448b': 'include U1448B'}
        command = [str(ROOT/cases[case]['file']), '--output', str(args.output/name), '--name', f'YTT · {titles[case]} · {method}',
                   '--method', method, '--nested-cv', '--folds', '4', '--buffer-km', '50',
                   '--bounds', *map(str, BOUNDS), '--grid', '128', '--map-grid', '481',
                   '--volume-grid', '48', '--draws', '1500', '--seed', '1729', '--max-extent-km', '8000',
                   '--levels', *map(str, LEVELS)]
        fit(command)
        report_path = args.output/name/'report.json'
        report = json.loads(report_path.read_text())
        report['input']['audit'] = {
            'provenance_file': 'data/audits/YTT/provenance.json',
            'provenance_sha256': sha256((AUDIT/'provenance.json').read_bytes()).hexdigest(),
            'source_workbook_sha256': provenance['source']['sha256'],
            'selection': cases[case]['definition'], 'preserved_semantics': provenance['preserved_semantics']}
        report['reproducibility']['benchmark_sha256'] = sha256(Path(__file__).read_bytes()).hexdigest()
        report_path.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
        write_short_report(args.output/name, report['dataset'], report)
    summarize(args.output)


if __name__ == '__main__':
    main()
