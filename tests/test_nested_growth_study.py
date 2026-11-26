"""A hidden second truth must not give its observed input a second vote."""
import numpy as np
import pandas as pd

from benchmarks.nested_growth_study import summary


def test_summary_balances_hidden_truths_and_keeps_failed_inputs():
    methods = ['Previous footprints', 'Continuous area growth']
    paired = pd.DataFrame([dict(method=m, geographic_negative_log_score=1., passed_folds=4,
        full_passed=True, map_draws_passed=True, map_integration_passed=True) for m in methods])
    measured = pd.DataFrame([dict(case='example', method=m, measured_log_rmse=1.,
        measured_median_abs_log_error=1., bounded_negative_log_probability=1.) for m in methods])
    rows = []
    for index in range(20):
        for truth, error in ([('small', 0.), ('large', 2.)] if index == 0 else [('single', 0.)]):
            for scramble in (1, 2):
                for level in (1., 10.):
                    rows.append(dict(method='candidate', input_sha256=str(index), truth=truth,
                        scramble=scramble, level_cm=level, true_km3=np.exp(error), median_map_km3=1.,
                        fit_passed=index!=0, false_km2=error, missed_km2=0.,
                        west_false_km2=0., west_missed_km2=0.))
    surveys = pd.DataFrame([dict(survey='campaign', method='candidate', log_score=-1., fit_passed=False)])
    quadrants = pd.DataFrame([dict(status='estimated', fit_passed=False),
                              dict(status='insufficient_training_or_empty_sector', fit_passed=None)])
    numerical = {'example':dict(previous=dict(field_population_check=dict(passed=False),
        mapped_quantile_checks=[dict(passed=False)]), current=dict(
        field_population_check=dict(check=dict(passed=True)), mapped_quantile_checks=[dict(passed=True)]))}
    result = summary(paired, measured, pd.DataFrame(rows), surveys, quadrants, numerical)
    known = result['known_deposits']['candidate']
    assert known['inputs'] == 20 and known['compatible_truths'] == 21 and known['passed_inputs'] == 19
    # The one ambiguous input contributes the mean of errors 0 and 2, once.
    np.testing.assert_allclose(known['mean_absolute_log_volume_error'], 1./20)
    assert known['median_absolute_log_volume_error'] == 0.
    assert result['surveys'][0]['fit_passed'] is False
    assert result['quadrants'] == dict(requested=2, estimated=1, passed=0, insufficient=1)
    assert result['matched_map_draw_checks'] == dict(previous=0, current=1)
