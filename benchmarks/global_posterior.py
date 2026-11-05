"""Run and summarize the fixed global posterior protocol for accepted eruptions."""
import argparse
from hashlib import sha256
import json
from pathlib import Path
import numpy as np
import pandas as pd

from src.empirical_isopach.core.catalog import datasets
from src.empirical_isopach.global_reporting import global_dataset


def summarize(output, regional_root=Path('eruptions'), published=False):
    output, regional_root = Path(output), Path(regional_root)
    rows, cases, validation_rows, sensitivity_warnings = [], [], [], []
    for record in datasets():
        slug, name = record['slug'], record['name']
        folder = Path('eruptions')/slug/'results'/'global' if published else output/slug
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
        assert val['row_ids'] == other['row_ids'] and val['folds'] == other['folds']
        assert val['buffer_km'] == other['buffer_km']
        for model in other['methods'].values():
            for a, b in zip(val['fits'], model['selections']):
                assert a['training_row_ids'] == b['training_row_ids']
                assert a['withheld_row_ids'] == b['withheld_row_ids']
        base = report['fits']['baseline']
        failed = [key.replace('_', ' ') for key, fit in report['fits'].items() if not fit['diagnostics']['passed']]
        if failed:
            sensitivity_warnings.append(name+': '+', '.join(failed))
        row = {'eruption': name, 'slug': slug, **{k: base['volume'][k] for k in ('median', 'lower95', 'upper95')},
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
    (output/'report.json').write_text(json.dumps({'protocol': 'mass-normalized-spherical-mixture-v1',
        'summary_source_sha256': sha256(Path(__file__).read_bytes()).hexdigest(),
        'cases': cases, 'estimates': rows, 'matched_validation': validation_rows,
        'scope': 'Conditional global bulk fallout, not ignimbrite or DRE. All assumptions fixed across eruptions. Prior/model envelopes are not probability intervals. Heldout prediction does not validate unobserved volume.'}, indent=2, allow_nan=False)+'\n')
    figures(output, rows)
    table = '\n'.join(f"| [{r['eruption']}](../../eruptions/{r['slug']}/results/global/README.md) | {r['median']:.3g} | {r['lower95']:.3g}–{r['upper95']:.3g} | {r['sensitivity_lower']:.3g}–{r['sensitivity_upper']:.3g} | {'pass' if r['baseline_sampling_passed'] else '**provisional**'} |" for r in rows)
    predictions = '\n'.join(f"| {r['eruption']} | {r['global_equal_region_negative_log_score']:.3f} | {r['stacked_equal_region_negative_log_score']:.3f} | {r['spline_equal_region_negative_log_score']:.3f} | {r['global_observation_coverage95']:.1%} | {'pass' if r['all_outer_sampling_passed'] else '**provisional**'} |" for r in rows)
    n_pass = sum(r['baseline_sampling_passed'] for r in rows)
    (output/'README.md').write_text(f"""# Global fallout estimates across six eruptions

[Project home](../../README.md) · [Method, equations and priors](../../docs/methods/probabilistic-global.md) · [Regional peers](../regional/README.md) · [Local support diagnostic](../support/README.md)

The complete-contour screen was too conservative as a prerequisite for conditional global estimation. It remains useful for local evidence checks. The replacement global calculation fits positive, finite-mass fallout fans jointly on the whole Earth. **No closing pseudo-data, chosen calculation rectangle or distal cutoff determines these totals.**

## Global estimates and uncertainty

| Eruption | Median bulk km³ | Conditional 95% credible interval | Assumption-sensitivity envelope km³ | Baseline sampling |
|---|---:|---:|---:|---|
{table}

The interval in column three is conditional on the baseline field, likelihood and priors. The envelope in column four spans the endpoints from seven alternative refits plus the baseline; **it is not a 95% interval or an exhaustive bound**. All accepted fans, including Minoan's Plinian and co-ignimbrite fallout, are fitted together. Ignimbrite deposits and dense-rock-equivalent conversion are excluded.

**Assumption refits with remaining numerical warnings:** {'; '.join(sensitivity_warnings) if sensitivity_warnings else 'none'}. Their provisional summaries remain in the envelope and are marked in the individual reports; the envelope's endpoints therefore need the corresponding numerical qualification. Holdout-fit warnings are reported separately below.

Some restrictive alternatives allow geologically extreme totals, particularly the one-fan Aso-4 model. Those outcomes expose weak identification or an inadequate shape assumption; **the envelope is not a range of certified geologically plausible volumes**. Numerical convergence alone does not rescue an unsuitable geological model. The baseline remains the jointly fitted multi-fan specification, with every alternative visible in its eruption report.

![Global estimates and matched predictive checks](comparison.png)

Every individual posterior field integrates exactly to its fitted volume parameter over WGS84. This removes an arbitrary calculation region, not the need for geological assumptions. Sparse measurements cannot exclude arbitrarily large deposits in unsampled places. General smoothness, source-related fan structure, finite mass, alternative thinning shapes and proper priors make the conditional inference possible. The same specifications apply to all six eruptions, without fitting to published volume targets.

Each eruption report includes the baseline field, its uncertainty, all assumption refits, whole-field samples, mass in unsampled sectors/proximal/distal areas, and all numerical attempts. **{n_pass}/6 baseline fits pass the declared sampling checks.** Numerical warnings on any sensitivity or withheld fit remain visible. Passing those checks does not establish geological correctness.

## Same-data geographic prediction

| Eruption | Global-model negative log score ↓ | GP score ↓ | Spline score ↓ | Global heldout 95% coverage | Holdout sampling |
|---|---:|---:|---:|---:|---|
{predictions}

Scores use identical accepted inputs, four geographic folds, buffers, training rows and heldout observation/bound semantics. Lower score is better. Global-model fitting is repeated inside every training fold with the fixed priors. Better local prediction does not prove an accurate global total; the regional peers retain their own purpose and no automatic winner is selected by volume.

## Evidence and reproducibility

- [Known-truth stress tests and retained failures](stress-tests.md)
- [Development decisions and numerical equivalence checks](development/README.md)
- [Machine-readable estimates](comparison.csv) and [matched comparison record](report.json)
- [Regional rerun and input-preservation verification](verification.json)

The original four datasets retain their supplied source flags; YTT and Minoan retain their documented audits. No accepted values, bounds, source decisions or input hashes changed. Global uncertainty does not automatically cover unrecorded erosion, reworking, preferential sampling or source misattribution.

For new data, upload measurements and source notes in chat. The assistant audits them, runs the regional and global protocols, and adds the data and illustrated reports to GitHub. No browser workflow is needed.
""")
    return rows


def figures(output, rows):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from src.probabilistic_isopach.figures import STYLE
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(1, 2, figsize=(13, 6.5), layout='constrained')
        for i, row in enumerate(rows):
            axes[0].plot([row['sensitivity_lower'], row['sensitivity_upper']], [i, i], ':', color='#8F9698', lw=1.5)
            axes[0].plot([row['lower95'], row['upper95']], [i, i], color='#157F88', lw=3)
            axes[0].scatter(row['median'], i, color='#157F88', marker='o' if row['baseline_sampling_passed'] else 'x', s=40)
            for offset, key, color, label in ((-.15, 'global', '#157F88', 'Global mixture'),
                    (0, 'stacked', '#C57B3D', 'GP ensemble'), (.15, 'spline', '#78639F', 'Cubic spline')):
                axes[1].scatter(row[key+'_equal_region_negative_log_score'], i+offset,
                                c=color, s=35, marker='x' if key == 'global' and not row['all_outer_sampling_passed'] else 'o',
                                label=label if i == 0 else None)
        for ax in axes:
            ax.set(yticks=range(len(rows)), yticklabels=[r['eruption'] for r in rows])
            ax.invert_yaxis(); ax.grid(axis='x', alpha=.2)
        axes[0].set(xscale='log', xlabel='Global bulk fallout (km³; log scale)',
            title='Global volumes\nSolid: conditional 95% · dotted: assumption envelope')
        axes[1].set(xlabel='Equal-region negative log score (lower is better)',
            title='Matched geographic holdouts\nLocal prediction, not global truth · × = provisional sampling')
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
        summarize(args.output, args.regional_root, args.published)


if __name__ == '__main__':
    main()
