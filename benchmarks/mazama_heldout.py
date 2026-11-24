"""Expose geographic held-out predictions by Mazama evidence ID, without refitting."""
import argparse
import csv
import gzip
import json
import math
from pathlib import Path

from benchmarks.mazama import INPUT, sha, write_csv


def read_csv(path):
    with (gzip.open(path, 'rt') if path.suffix == '.gz' else path.open()) as stream:
        return list(csv.DictReader(stream))


def summarize(home, output):
    accepted = read_csv(INPUT)
    regional_path = home/'heldout_predictions.csv'
    regional = {int(r['row_id']): r for r in read_csv(regional_path)}
    sources = {str(regional_path): sha(regional_path), str(INPUT): sha(INPUT)}
    results = []
    for family in ('stacked', 'spline', 'directional', 'spherical-mixture', 'nested-isopachs'):
        if family in ('stacked', 'spline'):
            predictions = regional
        else:
            path = home/family/'extent/heldout.csv.gz'
            predictions = {int(r['row_id']): r for r in read_csv(path) if r['design'] == 'geographic'}
            sources[str(path)] = sha(path)
        for row_id, observation in enumerate(accepted, start=2):
            bound = observation['Kind'] != 'measurement'
            pred = predictions.get(row_id)
            result = dict(method=family, evidence_id=observation['EvidenceID'],
                site=observation['Site'], row_id=row_id, latitude=observation['Lat'],
                longitude=observation['Lon'], kind=observation['Kind'],
                reported_value_cm=observation['Limit_cm'] if bound else observation['Thickness'],
                fold=regional[row_id]['fold'], status='estimated' if pred else 'unscored',
                fit_passed='', log_score='', bound_probability='', predictive_cdf='',
                predictive_lower95_cm='', predictive_median_cm='', predictive_upper95_cm='',
                point_covered95='')
            if pred:
                assert int(pred['fold']) == int(result['fold'])
                log_score = float(pred[family+'_log_score'] if family in ('stacked', 'spline') else pred['principal_log_score'])
                result['log_score'] = log_score
                if bound:
                    result['bound_probability'] = math.exp(log_score)
                    assert 0 <= result['bound_probability'] <= 1 + 1e-12
                if family in ('stacked', 'spline'):
                    result['fit_passed'] = True
                    q = [math.exp(float(pred[family+'_'+suffix])) for suffix in ('lower', 'median', 'upper')]
                    result.update(zip(('predictive_lower95_cm', 'predictive_median_cm', 'predictive_upper95_cm'), q))
                    if not bound:
                        result['point_covered95'] = q[0] <= float(observation['Thickness']) <= q[2]
                else:
                    result['fit_passed'] = pred['fit_passed'] == 'True'
                    result['predictive_cdf'] = float(pred['principal_cdf'])
                    if bound:
                        p = result['predictive_cdf']
                        if observation['Kind'] == 'lower_limit':
                            p = 1-p
                        assert abs(result['bound_probability']-p) < 1e-10
                    else:
                        result['point_covered95'] = .025 <= result['predictive_cdf'] <= .975
            results.append(result)
    assert len(results) == len({(r['method'],r['evidence_id']) for r in results}) == 5 * len(accepted)
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output/'heldout-source-observations.csv', results)
    bounds = [r for r in results if r['kind'] != 'measurement']
    bound_count = sum(r['Kind'] != 'measurement' for r in accepted)
    assert len(bounds) == 5 * bound_count
    write_csv(output/'heldout-bound-probabilities.csv', bounds)
    report = dict(input_sha256=sha(INPUT), script_sha256=sha(Path(__file__)),
        source_sha256=sources, observations_per_family=len(accepted), bound_readings_per_family=bound_count,
        methods={family:dict(scored=sum(r['status']=='estimated' for r in results if r['method']==family),
            covered_points=sum(r['point_covered95'] is True for r in results if r['method']==family),
            point_count=sum(r['kind']=='measurement' and r['status']=='estimated' for r in results if r['method']==family))
            for family in ('stacked','spline','directional','spherical-mixture','nested-isopachs')},
        meaning='Each site is predicted with its buffered geographic fold withheld. Bound probabilities use the observation likelihood, including its fitted discrepancy; latent isopach intervals are separate. A point log score is a density in natural-log thickness, not an event probability. No probability for the geological correctness of a bound is inferred. Unscored or numerically warned folds remain explicit.')
    (output/'heldout-source-observations.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report['methods'], indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--home', type=Path, default=Path('eruptions/mazama/results'))
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    summarize(args.home, args.output)
