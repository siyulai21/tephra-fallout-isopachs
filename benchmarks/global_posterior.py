"""Run and summarize the fixed global posterior protocol for accepted eruptions."""
import argparse
from hashlib import sha256
import json
from pathlib import Path
import numpy as np
import pandas as pd

from src.empirical_isopach.core.catalog import datasets
from src.empirical_isopach.global_reporting import global_dataset
from .spherical_bearing_summary import build_validation
from .publication import published_methods, released_datasets, result_home


def summarize(output, regional_root=Path('eruptions'), published=False, trials=None):
    output, regional_root = Path(output), Path(regional_root)
    rows, cases, validation_rows = [], [], []
    case_dirs = {}
    records = released_datasets("spherical-mixture") if published else datasets()
    for record in sorted(records, key=lambda row: row["slug"]):
        slug, name = record['slug'], record['name']
        available = published_methods(record)
        if published and 'spherical-mixture' not in available:
            continue
        folder = result_home(record)/'spherical-mixture' if published else output/slug
        case_dirs[slug] = folder
        path = folder/'report.json'
        if not path.exists():
            raise ValueError(f'Missing completed global report: {path}')
        report = json.loads(path.read_text())
        regional_path = result_home(record)/'report.json' if published else regional_root/slug/'report.json'
        if not regional_path.exists():
            regional_path = regional_root/slug/'results/report.json'
        val = report['validation']
        regional_methods = {}
        regional_hash = None
        if published and report['input']['input_sha256'] != sha256(record['path'].read_bytes()).hexdigest():
            raise ValueError(f'{slug}: published spherical fit and accepted input differ')
        if not published or {'stacked','spline'}.intersection(available):
            regional = json.loads(regional_path.read_text())
            regional_hash = sha256(regional_path.read_bytes()).hexdigest()
            assert report['input']['input_sha256'] == regional['input']['input_sha256']
            other = regional['validation']
            assert val['row_ids'] == other['row_ids'] and val['folds'] == other['folds']
            # Projection libraries can differ below a micrometre across platforms.
            np.testing.assert_allclose(val['buffer_km'], other['buffer_km'], atol=1e-9, rtol=0)
            regional_methods = other['methods']
            for model in regional_methods.values():
                for a, b in zip(val['fits'], model['selections'], strict=True):
                    assert a['training_row_ids'] == b['training_row_ids']
                    assert a['withheld_row_ids'] == b['withheld_row_ids']
        deposit_path = folder/'deposit.json'
        deposit = json.loads(deposit_path.read_text())
        if deposit['input_sha256'] != report['input']['input_sha256']:
            raise ValueError(f'{slug}: median map and spherical fit differ')
        band = deposit['thickness_band']['volumes_km3']
        base = report['fits']['baseline']
        row = {'eruption': name, 'slug': slug, 'geometry':report.get('geometry','displaced'),
            'review_status':record.get('global_review_status','conditional'),
            'median_map_volume_km3': deposit['median_map_volume_km3'],
            'thickness_q25_volume_km3': band[0], 'thickness_q75_volume_km3': band[2],
            'sensitivity_lower': report['sensitivity_envelope_km3'][0],
            'sensitivity_upper': report['sensitivity_envelope_km3'][1],
            'baseline_sampling_passed': base['diagnostics']['passed'],
            'all_sensitivity_sampling_passed': report['all_assumption_fits_passed_diagnostics'],
            'all_outer_sampling_passed': val['all_fits_passed_diagnostics'],
            'median_fraction_outside_local_footprint': 1-report['mass_partition']['parts']['local_footprint']['fraction']['median'],
            'global_equal_region_negative_log_score': val['metrics']['equal_region_log_score'],
            'global_observation_coverage95': val['metrics']['coverage95_observations']}
        for key, model in regional_methods.items():
            row[key+'_equal_region_negative_log_score'] = model['metrics']['equal_region_log_score']
        metrics = dict(val['metrics'])
        extent_path = folder/'extent/report.json'
        sources = {'deposit_sha256': sha256(deposit_path.read_bytes()).hexdigest()}
        if extent_path.is_file():
            extent = json.loads(extent_path.read_text())
            assert extent['input_sha256'] == deposit['input_sha256']
            geo = extent['validation']['geographic']
            alternatives = json.loads(extent_path.with_name('prior-sensitivity.json').read_text())
            assert set(alternatives)==set(report['fits'])
            row.update(sensitivity_lower=min(v['conditional_interval'][0] for v in alternatives.values()),
                       sensitivity_upper=max(v['conditional_interval'][-1] for v in alternatives.values()),
                       all_outer_sampling_passed=geo['all_folds_passed'],
                       global_equal_region_negative_log_score=geo['negative_log_score'],
                       global_observation_coverage95=geo['coverage95_exact'])
            metrics = dict(equal_region_log_score=geo['negative_log_score'],coverage95_observations=geo['coverage95_exact'],
                           fold_log_scores=geo['fold_scores'])
            sources['extent_report_sha256'] = sha256(extent_path.read_bytes()).hexdigest()
        rows.append(row)
        validation_rows.append({'eruption':name,'global':metrics,
            **{key:model['metrics'] for key,model in regional_methods.items()},
            'matched_input_folds_buffer_and_training_rows':True if regional_methods else None,'global_outer_sampling_passed':row['all_outer_sampling_passed']})
        cases.append({'name':name,'slug':slug,'input_sha256':report['input']['input_sha256'],
            'global_report_sha256':sha256(path.read_bytes()).hexdigest(),
            'regional_report_sha256':regional_hash,**sources})
    output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output/'comparison.csv', index=False)
    (output/'report.json').write_text(json.dumps({'protocol': 'spherical-median-thickness-and-finite-spread-v2',
        'summary_source_sha256': sha256(Path(__file__).read_bytes()).hexdigest(),
        'cases': cases, 'estimates': rows, 'matched_validation': validation_rows,
        'scope': 'Conditional whole-Earth bulk fallout. The principal estimate and band integrate pointwise Q50, Q25 and Q75 thickness. The prior envelope spans total-volume draw endpoints across eight named priors. Geographic scores measure prediction at withheld sites.'}, indent=2, allow_nan=False)+'\n')
    figures(output, rows)
    build_validation(trials, output / "bearings", case_dirs, link_prefix="bearings/")
    lines = ['# Spherical-mixture maps and validation','',
        'Each estimate integrates the pointwise median thickness over Earth. The range integrates the pointwise Q25 and Q75 thickness maps. '
        'The separate prior sensitivity envelope summarizes total-volume draws under eight alternative assumptions. '
        '[Model and assumptions](../../docs/methods/spherical-bearings.md)','',
        '| Eruption | Median map (km³) | Q25–Q75 thickness maps (km³) | Geographic score ↓ | Fitted-point coverage |',
        '|---|---:|---:|---:|---:|']
    for row in rows:
        case = f"[{row['eruption']}](../../eruptions/{row['slug']}/README.md)"
        flag = '' if row['baseline_sampling_passed'] and row['all_outer_sampling_passed'] else ' †'
        lines.append(f"| {case} | {row['median_map_volume_km3']:,.3g}{flag} | {row['thickness_q25_volume_km3']:,.3g}–{row['thickness_q75_volume_km3']:,.3g} | {row['global_equal_region_negative_log_score']:.3f} | {row['global_observation_coverage95']:.1%} |")
    lines += ['', 'A dagger marks a sampling or geographic validation warning. [Full diagnostics](comparison.csv) · '
              '[Inputs, source hashes and matched folds](report.json) · [Known-deposit experiments](bearings/known-volumes.csv)', '']
    (output/'README.md').write_text('\n'.join(lines))


def figures(output, rows):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from src.empirical_isopach.visualization.style import STYLE
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(1, 2, figsize=(13, 6.5), layout='constrained')
        for i, row in enumerate(rows):
            axes[0].plot([row['sensitivity_lower'], row['sensitivity_upper']], [i, i], ':', color='#8F9698', lw=1.5)
            axes[0].plot([row['thickness_q25_volume_km3'], row['thickness_q75_volume_km3']], [i, i], color='#157F88', lw=3)
            axes[0].scatter(row['median_map_volume_km3'], i, color='#157F88', marker='o' if row['baseline_sampling_passed'] else 'x', s=40)
            for offset, key, color, label in ((-.15, 'global', '#157F88', 'Global mixture'),
                    (0, 'stacked', '#C57B3D', 'GP ensemble'), (.15, 'spline', '#78639F', 'Cubic spline')):
                if key+'_equal_region_negative_log_score' not in row:
                    continue
                axes[1].scatter(row[key+'_equal_region_negative_log_score'], i+offset,
                                c=color, s=35, marker='x' if key == 'global' and not row['all_outer_sampling_passed'] else 'o',
                                label=label if i == 0 else None)
        for ax in axes:
            ax.set(yticks=range(len(rows)), yticklabels=[r['eruption']+(' (questioned)' if r.get('review_status') == 'questioned' else '') for r in rows])
            ax.invert_yaxis(); ax.grid(axis='x', alpha=.2)
        axes[0].set(xscale='log', xlabel='Whole-deposit volume (km³, log scale)',
            title='Whole-deposit volumes\nQ25–Q75 thickness (solid) and prior envelope (dotted)')
        axes[1].set(xlabel='Equal-region negative log score (lower is better)',
            title='Matched geographic holdouts\n× marks provisional sampling')
        axes[1].legend(fontsize=9)
        fig.suptitle('Whole-Earth fallout · explicit assumptions and uncertainty', fontsize=16)
        for suffix in ('png', 'svg'):
            fig.savefig(output/f'comparison.{suffix}', dpi=180)
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--dataset')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--summarize-only', action='store_true')
    parser.add_argument('--published', action='store_true')
    parser.add_argument('--trials', type=Path, help='Optional fresh displaced and source known-deposit trials for the comparison')
    parser.add_argument('--regional-root', type=Path, default=Path('eruptions'))
    args = parser.parse_args()
    if args.published and not args.summarize_only:
        parser.error('--published is only for summarizing already reviewed published reports')
    if not args.summarize_only:
        chosen = [r for r in datasets() if args.dataset is None or args.dataset.lower() in (r['slug'], r['name'].lower())]
        if not chosen:
            parser.error('Unknown eruption')
        for record in chosen:
            global_dataset(record['path'], args.output/record['slug'], name=record['name'], resume=args.resume,
                           progress=lambda s: print(s, flush=True))
    if args.dataset is None or args.summarize_only:
        summarize(args.output, args.regional_root, args.published, args.trials)


if __name__ == '__main__':
    main()
