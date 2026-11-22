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


def summarize(output, regional_root=Path('eruptions'), published=False, trials=None):
    output, regional_root = Path(output), Path(regional_root)
    rows, cases, validation_rows, sensitivity_warnings = [], [], [], []
    case_dirs = {}
    for record in sorted(datasets(), key=lambda row: row["slug"]):
        slug, name = record['slug'], record['name']
        folder = Path('eruptions')/slug/'results'/'spherical-mixture' if published else output/slug
        case_dirs[slug] = folder
        path = folder/'report.json'
        if not path.exists():
            raise ValueError(f'Missing completed global report: {path}')
        report = json.loads(path.read_text())
        regional_path = regional_root/slug/'report.json'
        if not regional_path.exists():
            regional_path = regional_root/slug/'results/report.json'
        regional = json.loads(regional_path.read_text())
        val = report['validation']
        other = regional['validation']
        assert report['input']['input_sha256'] == regional['input']['input_sha256']
        if published and report['input']['input_sha256'] != sha256(record['path'].read_bytes()).hexdigest():
            raise ValueError(f'{slug}: published spherical fit and accepted input differ')
        assert val['row_ids'] == other['row_ids'] and val['folds'] == other['folds']
        assert val['buffer_km'] == other['buffer_km']
        for model in other['methods'].values():
            for a, b in zip(val['fits'], model['selections']):
                assert a['training_row_ids'] == b['training_row_ids']
                assert a['withheld_row_ids'] == b['withheld_row_ids']
        base = report['fits']['baseline']
        failed = [key.replace('_', ' ') for key, fit in report['fits'].items() if not fit['diagnostics']['passed']]
        if failed:
            sensitivity_warnings.append(f"{name} ({', '.join(failed)})")
        row = {'eruption': name, 'slug': slug, 'geometry':report.get('geometry','displaced'),
            'review_status':record.get('global_review_status','conditional'),
            'median_map_volume_km3': json.loads((folder/'deposit.json').read_text())['median_map_volume_km3'],
            **{k: base['volume'][k] for k in ('lower95', 'upper95')},
            'sensitivity_lower': report['sensitivity_envelope_km3'][0],
            'sensitivity_upper': report['sensitivity_envelope_km3'][1],
            'baseline_sampling_passed': base['diagnostics']['passed'],
            'all_sensitivity_sampling_passed': report['all_assumption_fits_passed_diagnostics'],
            'all_outer_sampling_passed': val['all_fits_passed_diagnostics'],
            'median_fraction_outside_local_footprint': 1-report['mass_partition']['parts']['local_footprint']['fraction']['median'],
            'global_equal_region_negative_log_score': val['metrics']['equal_region_log_score'],
            'global_observation_coverage95': val['metrics']['coverage95_observations']}
        for model in ('stacked', 'spline'):
            row[model+'_equal_region_negative_log_score'] = other['methods'][model]['metrics']['equal_region_log_score']
        rows.append(row)
        validation_rows.append({'eruption': name, 'global': val['metrics'],
            **{key: model['metrics'] for key, model in other['methods'].items()},
            'matched_input_folds_buffer_and_training_rows': True,
            'global_outer_sampling_passed': val['all_fits_passed_diagnostics']})
        cases.append({'name': name, 'slug': slug, 'input_sha256': report['input']['input_sha256'],
            'global_report_sha256': sha256(path.read_bytes()).hexdigest(),
            'regional_report_sha256': sha256(regional_path.read_bytes()).hexdigest()})
    output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output/'comparison.csv', index=False)
    (output/'report.json').write_text(json.dumps({'protocol': 'reviewed-spherical-bearing-mixtures-v1',
        'summary_source_sha256': sha256(Path(__file__).read_bytes()).hexdigest(),
        'cases': cases, 'estimates': rows, 'matched_validation': validation_rows,
        'scope': 'Conditional global bulk fallout, not ignimbrite or DRE. Each geometry uses fixed generic assumptions; the reviewed geometry is recorded per case. This is not a uniform-model experiment. Prior/model envelopes are not probability intervals. Heldout prediction does not validate unobserved volume.'}, indent=2, allow_nan=False)+'\n')
    figures(output, rows)
    paired, gap_scores, maps, known_deposits = build_validation(trials, output / "bearings", case_dirs, link_prefix="bearings/")
    volumes, forecasts = [], []
    for row in rows:
        slug = row["slug"]
        earlier = paired.loc[slug]
        label = row["geometry"] + (" · **questioned**" if row["review_status"] == "questioned" else "")
        case = f"[{row['eruption']}](../../eruptions/{slug}/README.md)"
        baseline_warning = "" if row["baseline_sampling_passed"] else " · **provisional**"
        mapped = json.loads((case_dirs[slug]/"deposit.json").read_text())
        volumes.append(
            f"| {case} | {label} | **{mapped['median_map_volume_km3']:,.3g}**{baseline_warning} "
            f"| {row['lower95']:,.1f}–{row['upper95']:,.1f} | {row['sensitivity_lower']:.3g}–{row['sensitivity_upper']:.3g} |"
        )
        status = "pass" if row["all_outer_sampling_passed"] else "**provisional**"
        forecasts.append(
            f"| {row['eruption']} | {earlier.parent_score:.3f} → {row['global_equal_region_negative_log_score']:.3f} "
            f"| {row['stacked_equal_region_negative_log_score']:.3f} | {row['spline_equal_region_negative_log_score']:.3f} "
            f"| {row['global_observation_coverage95']:.1%} | {status} |"
        )
    table = "\n".join(volumes)
    predictions = "\n".join(forecasts)
    n_pass = sum(r['baseline_sampling_passed'] for r in rows)
    warnings = ("Numerical warnings occurred in " + ", ".join(sensitivity_warnings) +
                ". Their results contribute to the envelope." if sensitivity_warnings
                else "All alternative refits passed their numerical checks.")
    (output/'README.md').write_text(f"""# Spherical-mixture maps and volumes

Each eruption uses the same observations for the broad control and the principal spherical fit. In one third of simulated deposits the kernels extend broadly into unsampled directions. In the others, the thickness depends on the orientation of an unseen fan. Each simulated deposit gives one whole-Earth bulk fallout volume. The principal volume integrates the median-thickness map. The interval describes uncertainty in total fallout across complete posterior draws. The last column gives the envelope across all eight prior fits. [Model and assumptions](../../docs/methods/spherical-bearings.md)

| Eruption | Kernel geometry | Median-thickness map (km³) | Conditional 95% interval (km³) | Prior-sensitivity envelope (km³) |
|---|---|---:|---:|---:|
{table}

{n_pass} of {len(rows)} baseline fits passed the sampling and integration checks. {warnings} Aso-4’s source audit questions the displaced peak and its large upper tail. B-Tm’s source-focused fit includes three interpreted upper bounds. [Aso-4 source audit](../aso-4-audit/README.md) · [B-Tm evidence](../../data/audits/B-Tm/README.md)

{maps}
## Predictions in withheld regions

All methods faced the same held-out observations, bounds, regions and buffers for each eruption. Lower scores indicate better prediction. Coverage is the fraction of exact thickness readings inside the principal spherical 95% predictive interval.

| Eruption | Spherical score, broad → principal | GP score | Spline score | Spherical coverage | Spherical holdout checks |
|---|---:|---:|---:|---:|---|
{predictions}

{gap_scores}

![Conditional spherical volumes, prior envelopes and withheld-region scores](comparison.png)

{known_deposits}
## Data and reproduction

[Case estimates and prior diagnostics](comparison.csv) · [Input identities and matched validation](report.json) · [Kernel-geometry experiment](../source-revision/README.md) · [Displaced-kernel trials](stress-tests.md) · [Source-focused trials](../source-revision/stress-tests.md) · [Reproduction commands](../../docs/methods/spherical-bearings.md#integration-validation-and-reproducibility) · [Preceding release](../../archive/2026-11-20-spherical-continuation/README.md)
""")
    return rows


def figures(output, rows):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from src.empirical_isopach.visualization.style import STYLE
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(1, 2, figsize=(13, 6.5), layout='constrained')
        for i, row in enumerate(rows):
            axes[0].plot([row['sensitivity_lower'], row['sensitivity_upper']], [i, i], ':', color='#8F9698', lw=1.5)
            axes[0].plot([row['lower95'], row['upper95']], [i, i], color='#157F88', lw=3)
            axes[0].scatter(row['median_map_volume_km3'], i, color='#157F88', marker='o' if row['baseline_sampling_passed'] else 'x', s=40)
            for offset, key, color, label in ((-.15, 'global', '#157F88', 'Global mixture'),
                    (0, 'stacked', '#C57B3D', 'GP ensemble'), (.15, 'spline', '#78639F', 'Cubic spline')):
                axes[1].scatter(row[key+'_equal_region_negative_log_score'], i+offset,
                                c=color, s=35, marker='x' if key == 'global' and not row['all_outer_sampling_passed'] else 'o',
                                label=label if i == 0 else None)
        for ax in axes:
            ax.set(yticks=range(len(rows)), yticklabels=[r['eruption']+(' (questioned)' if r.get('review_status') == 'questioned' else '') for r in rows])
            ax.invert_yaxis(); ax.grid(axis='x', alpha=.2)
        axes[0].set(xscale='log', xlabel='Whole-deposit volume (km³, log scale)',
            title='Whole-deposit volumes\nConditional 95% (solid) and prior envelope (dotted)')
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
