"""Summarize a completed benchmark without refitting or changing its record."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('benchmark', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    result = json.loads(args.benchmark.read_text())
    args.output.mkdir(parents=True, exist_ok=True)
    real = []
    for row in result['real']:
        for method in ('selected', 'stacked'):
            real.append({'dataset': row['dataset'], 'method': method,
                         **{k: v for k, v in row[method].items() if isinstance(v, (float, int))}})
    pd.DataFrame(real).to_csv(args.output/'real_comparison.csv', index=False)
    scenarios = ('smooth', 'secondary_lobe', 'missing_sector')
    synthetic = []
    for scenario in scenarios:
        rows = [row for row in result['synthetic'] if row['scenario'] == scenario]
        for method in ('selected', 'stacked'):
            values = {key: np.mean([r[method][key] for r in rows]) for key in
                      ('latent_rmse_log', 'latent_coverage95', 'volume_relative_error_matched_mc_grid')}
            values.update({key: np.mean([r[method]['observations'][key] for r in rows]) for key in
                           ('coverage95_observations', 'mean_log_score', 'mean_interval_width_log')})
            synthetic.append({'scenario': scenario, 'method': method, 'replicates': len(rows), **values})
    frame = pd.DataFrame(synthetic)
    frame.to_csv(args.output/'synthetic_comparison.csv', index=False)
    with plt.rc_context({'font.family': 'DejaVu Sans', 'font.size': 10, 'axes.spines.top': False,
                         'axes.spines.right': False, 'figure.facecolor': '#fafbf8', 'axes.facecolor': '#fafbf8'}):
        fig, axes = plt.subplots(1, 3, figsize=(13, 4.8))
        fig.subplots_adjust(top=.76, bottom=.24, wspace=.4, left=.065, right=.98)
        fig.text(.065, .92, 'Empirical ensemble / measured gains and remaining gaps', fontsize=18, weight='bold', color='#183843')
        fig.text(.065, .85, 'Matched geographic holdouts for real data · 10 fixed-seed replicates per synthetic scenario', color='#5c7075', fontsize=11)
        for method, color, offset, label in (('selected', '#b67936', -.10, 'Established single model'),
                                             ('stacked', '#087e8b', .10, 'Experimental ensemble')):
            axes[0].plot([r[method]['mean_log_score'] for r in result['real']], np.arange(len(result['real']))+offset,
                         'o', color=color, label=label)
            subset = frame[frame.method == method]
            axes[1].plot(subset.coverage95_observations*100, np.arange(3)+offset, 'o', color=color)
            axes[2].plot(subset.volume_relative_error_matched_mc_grid*100, np.arange(3)+offset, 'o', color=color)
        axes[0].set(yticks=np.arange(len(result['real'])), yticklabels=[r['dataset'] for r in result['real']],
                    xlabel='Negative log predictive score ↓', title='Real eruptions / predictive score')
        for ax in axes[1:]:
            ax.set(yticks=np.arange(3), yticklabels=['Smooth', 'Secondary lobe', 'Missing sector'])
        axes[1].axvline(95, ls='--', lw=1, c='#7e8b88')
        axes[1].set(xlabel='95% observation interval coverage (%)', title='Synthetic / predictive coverage', xlim=(72, 101))
        axes[2].axvline(0, ls='--', lw=1, c='#7e8b88')
        axes[2].set(xlabel='Mean relative volume error (%)', title='Synthetic / finite-region volume')
        for ax in axes:
            title = ax.get_title()
            ax.set_title('')
            ax.invert_yaxis(); ax.grid(axis='x', alpha=.2); ax.set_title(title, fontsize=11, loc='left', pad=14)
        axes[0].legend(loc='upper left', bbox_to_anchor=(-.05, -.25), frameon=False, ncol=2)
        fig.text(.065, .035, 'No invented observations. Missing-sector volume is still underestimated by about 23%. These tests do not validate total erupted volume.',
                 fontsize=9, color='#5c7075')
        for suffix in ('png', 'svg'):
            fig.savefig(args.output/f'comparison.{suffix}', dpi=180)
        plt.close(fig)


if __name__ == '__main__':
    main()
