"""Audit arithmetic and right-censoring checks independent of the implementation."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.optimize._numdiff import approx_derivative
from scipy.special import log_ndtr

from benchmarks.minoan import prepare_inputs, source_rows, classify
from src.probabilistic_isopach.data import Observations, read_observations
from src.probabilistic_isopach.model import SpatialModel, Candidate, censored_terms
from src.probabilistic_isopach.distributions import LogMixture
from src.probabilistic_isopach.validation import nested_validate, score_predictions
from src.probabilistic_isopach.ensemble import point_log_scores


@pytest.mark.parametrize('z', [-30., -3., 0., 4.])
def test_lower_bound_likelihood_derivatives_and_survival_probability(z):
    f, limit, sd = np.array([z*.7]), np.array([0.]), np.array([.7])
    loss, gradient, hessian = censored_terms(f, limit, sd, lower=True)
    np.testing.assert_allclose(loss, -log_ndtr(z))
    numeric_g = approx_derivative(lambda a: censored_terms(a, limit, sd, True)[0], f).ravel()
    numeric_h = approx_derivative(lambda a: censored_terms(a, limit, sd, True)[1], f).ravel()
    np.testing.assert_allclose(gradient, numeric_g, rtol=1e-5, atol=1e-7)
    np.testing.assert_allclose(hessian, numeric_h, rtol=2e-4, atol=1e-7)


def test_lower_limit_pushes_prediction_up_and_loose_limit_has_no_effect():
    rng = np.random.default_rng(5)
    xy = rng.uniform(-100,100,(40,2))
    values = 4-.02*np.linalg.norm(xy,axis=1)+rng.normal(0,.3,40)
    data = Observations(xy,values,np.zeros(40,bool),np.zeros(40),np.arange(40))
    query = np.array([[10.,20.]])
    before = SpatialModel().fit(data).predict_log(query)[0][0]
    augmented = Observations(np.vstack((xy,query)),np.r_[values,before+2],
        np.r_[data.censored,True],np.zeros(41),np.arange(41),np.r_[data.lower_censored,True])
    after = SpatialModel().fit(augmented)
    assert after.predict_log(query)[0][0] > before
    augmented.log_value[-1] = -40
    np.testing.assert_allclose(SpatialModel().fit(augmented).predict_log(query)[0],before,atol=2e-5)


def test_censor_scores_use_correct_tail_in_selection_stacking_and_mixture():
    data = Observations([[0,0],[1,1],[2,2]],[2,2,2],[False,True,True],[0,0,0],[1,2,3],[False,False,True])
    mean, variance = np.zeros(3), np.ones(3)
    expected = np.array([-.5*(np.log(2*np.pi)+4),log_ndtr(2),log_ndtr(-2)])
    distribution = LogMixture([1],mean[None,:],variance[None,:])
    np.testing.assert_allclose(distribution.log_score(data.log_value,data.censored,data.lower_censored),expected)
    np.testing.assert_allclose(point_log_scores(data,mean[:,None],variance[:,None])[:,0],expected)
    assert score_predictions(data,mean,variance)['mean_log_score']==pytest.approx(-expected.mean())
    assert data.subset([2,0]).lower_censored.tolist()==[True,False]
    with pytest.raises(ValueError,match='also be marked censored'):
        Observations([[0,0]],[0],[False],[0],[1],[True])


def test_combined_audit_sums_fallout_removes_flow_and_keeps_bounds():
    provenance = prepare_inputs()
    assert provenance['cases']['combined']['n_measurements']==92
    assert provenance['cases']['combined']['n_lower_limits']==9
    rows = {r['map_id']:r for r in source_rows()}
    assert classify(rows[11])[:2]==('measurement',34.)  # 25 + 9, not 49 incl. flow
    assert classify(rows[14])[:2]==('measurement',16.)  # 12 + 4, not 25
    assert classify(rows[10])[:2]==('lower_limit',31.) # incomplete, not exact
    assert classify(rows[13])[:2]==('lower_limit',46.) # 40 + 6 recovered fallout
    assert classify(rows[63])[:2]==('measurement',2.)  # explicit total, uncertain partition
    assert classify(rows[83])[0] is None
    assert classify(rows[85])[0] is None
    assert classify(rows[89])[0] is None
    assert classify(rows[9])[0] is None                # not reached, not zero
    assert classify(rows[85],'component_conflicts')[:2]==('measurement',5.)
    assert classify(rows[85],'total_conflicts')[:2]==('measurement',15.)
    data,meta=read_observations('data/eruptions/Minoan_tephra.csv')
    assert meta['n_measurements']==92 and meta['n_lower_limits']==9 and meta['n_upper_limits']==0
    assert data.lower_censored.sum()==9


def test_nested_holdouts_score_bounds_but_cover_only_measured_values():
    data,_=read_observations('data/eruptions/Minoan_tephra.csv')
    report=nested_validate(data,(Candidate('exponential',.75,1.5),),outer_folds=4,inner_folds=3,buffer_km=10)
    canonical=data.canonical();measured=~canonical.censored
    covered=(canonical.log_value>=report['lower95_log'])&(canonical.log_value<=report['upper95_log'])
    assert report['coverage95_observations']==pytest.approx(covered[measured].mean())
    assert len(report['point_log_scores'])==101 and np.isfinite(report['point_log_scores']).all()


def test_old_ytt_selected_fit_remains_numerically_unchanged():
    data,_=read_observations('data/audits/YTT/history/2026-11-02/without_maninjau_43.csv')
    previous=json.loads(Path('reports/ytt/archive/2026-11-02/main43_selected/report.json').read_text())['model']
    model=SpatialModel(Candidate(**previous['candidate'])).fit(data).describe()
    assert model.keys()==previous.keys()
    for key in ('candidate','trend_on_constraint_boundary','uncertainty'):
        assert model[key]==previous[key]
    # BLAS/projection builds can differ in their last floating-point bits.
    # Keep a strict relative tolerance, including for the tiny jitter term.
    for key in ('beta','radius_scale_km','length_km','amplitude_log','noise_log','jitter'):
        np.testing.assert_allclose(model[key],previous[key],rtol=1e-12,atol=0)


def test_minoan_publication_preserves_bound_types_and_writes_no_html(tmp_path):
    from src.probabilistic_isopach.cli import main
    main(['Minoan','--output',str(tmp_path/'run'),'--grid','8',
          '--volume-grid','8','--draws','20','--map-grid','65'])
    output=tmp_path/'run'
    report=json.loads((output/'report.json').read_text())
    assert report['settings']['buffer_km']==10
    assert report['settings']['bounds']==[-400,-850,1200,1100]
    assert report['input']['n_lower_limits']==9
    assert not (output/'index.html').exists()
    assert (output/'README.md').is_file()
    observations=pd.read_csv(output/'observations_used.csv')
    bounds=observations[observations.Kind=='lower_limit']
    assert len(bounds)==9 and bounds.Thickness_cm.isna().all() and (bounds.Limit_cm>0).all()
