"""Matched evaluation of selected, stacked and discrepancy-adjusted predictions.

Run: python -m benchmarks.empirical --output results/empirical --repeats 10
All synthetic scenarios/seeds match benchmarks.run; no benchmark responses are
used to fit a candidate, weight, discrepancy or tune the candidate grid.
"""
import argparse
from hashlib import sha256
from importlib.metadata import version
import json
from pathlib import Path
import platform
import numpy as np

from src.probabilistic_isopach import Observations, read_observations
from src.probabilistic_isopach.distributions import predictive_distribution
from src.probabilistic_isopach.validation import select_model, nested_validate, interval_metrics


def assess(data, distribution):
    return interval_metrics(data, distribution.quantile(.5), distribution.quantile(.025),
                            distribution.quantile(.975), distribution.log_score(data.log_value, data.censored, data.lower_censored))


def synthetic(repeats):
    for scenario in ('smooth', 'secondary_lobe', 'missing_sector'):
        for seed in range(repeats):
            rng = np.random.default_rng(seed+200)
            xy = rng.uniform(-250, 250, (120, 2))
            if scenario == 'missing_sector':
                xy = xy[~((xy[:, 0] > 0) & (xy[:, 1] > 0))]
            xy = xy[:60]
            test = rng.uniform(-250, 250, (400, 2))
            def truth(points):
                values = 5-.018*np.linalg.norm(points, axis=1)+.005*points[:, 0]
                if scenario != 'smooth':
                    values += 2.5*np.exp(-((points[:, 0]-130)**2+(points[:, 1]-100)**2)/(2*45**2))
                return values
            observed = truth(xy)+rng.normal(0, .25, len(xy))
            data = Observations(xy, observed, np.zeros(len(xy), bool), np.zeros(len(xy)), np.arange(len(xy)))
            actual = truth(test)
            measured = actual+rng.normal(0, .25, len(test))
            heldout = Observations(test, measured, np.zeros(len(test), bool), np.zeros(len(test)), np.arange(len(test)))
            row = {'scenario': scenario, 'seed': seed}
            for method in ('selected', 'stacked'):
                model, selection = select_model(data, method=method)
                distribution = predictive_distribution(model, test)
                median = distribution.quantile(.5)
                expected = np.exp(distribution.log_expected_thickness()).mean()
                row[method] = {
                    'latent_rmse_log': float(np.sqrt(np.mean((median-actual)**2))),
                    'latent_coverage95': float(np.mean((actual >= distribution.quantile(.025)) & (actual <= distribution.quantile(.975)))),
                    'volume_relative_error_matched_mc_grid': float(expected/np.exp(actual).mean()-1),
                    'observations': assess(heldout, predictive_distribution(model, test, observation=True)),
                    'discrepancy_adjusted': assess(heldout, predictive_distribution(model, test, observation=True, calibrated=True)),
                    'components': selection.get('stacking', {}).get('components'),
                    'discrepancy_variance_log': getattr(model, 'discrepancy_variance', 0.)}
            yield row


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--repeats', type=int, default=10)
    args = parser.parse_args(argv)
    if args.repeats < 1 or (args.output.exists() and any(args.output.iterdir())):
        parser.error('Use positive repeats and a new output directory.')
    args.output.mkdir(parents=True, exist_ok=True)
    result = {'design': {'outer_folds': 4, 'inner_folds': 3, 'buffer_km': 0,
                         'synthetic_repeats': args.repeats, 'test_sites_per_run': 400,
                         'synthetic_noise_sd_log': .25,
                         'python': platform.python_version(),
                         'packages': {name: version(name) for name in ('numpy', 'scipy', 'pandas', 'matplotlib', 'pyproj')},
                         'scope': 'Matched diagnostic benchmark, not external validation or evidence of correct total eruption volume.',
                         'source_sha256': {str(p): sha256(p.read_bytes()).hexdigest() for p in
                                           sorted(Path('src/probabilistic_isopach').glob('*.py'))},
                         'benchmark_sha256': sha256(Path(__file__).read_bytes()).hexdigest()},
              'real': [], 'synthetic': []}
    def save():
        (args.output/'benchmark.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    for name in ('Mazama', 'Tambora', 'Aso-4', 'B-Tm'):
        path = Path('data/eruptions')/(name+'_tephra.csv')
        data, _ = read_observations(path)
        row = {'dataset': name, 'input_sha256': sha256(path.read_bytes()).hexdigest(), 'n_real': len(data.xy)}
        for method in ('selected', 'stacked'):
            validation = nested_validate(data, method=method)
            (args.output/(name+'-'+method+'.json')).write_text(json.dumps(validation, indent=2, allow_nan=False)+'\n')
            row[method] = {k: validation[k] for k in ('mean_log_score', 'equal_region_log_score', 'rmse_log',
                                                    'coverage95_observations', 'mean_interval_width_log', 'discrepancy_adjusted')}
        result['real'].append(row)
        save()
        print('Finished geographic holdouts:', name, flush=True)
    for row in synthetic(args.repeats):
        result['synthetic'].append(row)
        save()
        print('Finished synthetic:', row['scenario'], row['seed'], flush=True)


if __name__ == '__main__':
    main()
