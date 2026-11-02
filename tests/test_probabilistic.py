import numpy as np
import pandas as pd
import pytest
from scipy.optimize._numdiff import approx_derivative

from src.probabilistic_isopach import Observations, Candidate, SpatialModel, read_observations
from src.probabilistic_isopach.model import matern32, censored_terms, condition_independent_evidence
from src.probabilistic_isopach.validation import spatial_folds, select_model, nested_validate
from src.probabilistic_isopach.report import integrate, grid_cells


def synthetic(n=40, seed=7):
    rng = np.random.default_rng(seed)
    xy = rng.uniform(-200, 200, (n, 2))
    r = np.linalg.norm(xy, axis=1)
    y = 5-.018*r+.005*xy[:, 0]+rng.normal(0, .2, n)
    return Observations(xy, y, np.zeros(n, bool), np.zeros(n), np.arange(n))


def test_gaussian_posterior_matches_independent_dense_equations():
    d = synthetic()
    model = SpatialModel().fit(d)
    q = np.array([[0, 0], [55, -30], [400, 500]])
    m, C = model.predict_log(q, covariance=True)
    train = model.data
    K = matern32(train.xy, train.xy, model.length, model.amplitude)
    K += np.diag(model.sd**2+model.jitter)
    k = matern32(train.xy, q, model.length, model.amplitude)
    exact_m = model._features(q)@model.beta + k.T@np.linalg.solve(K, train.log_value-model.mean_train)
    exact_C = matern32(q, q, model.length, model.amplitude)-k.T@np.linalg.solve(K, k)
    np.testing.assert_allclose(m, exact_m, atol=1e-10)
    np.testing.assert_allclose(C, exact_C, atol=1e-10)
    assert np.linalg.eigvalsh(C).min() >= -1e-10
    np.testing.assert_allclose(np.diag(C), model.predict_log(q)[1])


@pytest.mark.parametrize("z", [-30., -3., 0., 4.])
def test_censored_likelihood_derivatives(z):
    f = np.array([-z*.7])
    limit, sd = np.array([0.]), np.array([.7])
    _, g, h = censored_terms(f, limit, sd)
    numeric_g = approx_derivative(lambda a: censored_terms(a, limit, sd)[0], f).ravel()
    numeric_h = approx_derivative(lambda a: censored_terms(a, limit, sd)[1], f).ravel()
    np.testing.assert_allclose(g, numeric_g, rtol=1e-5, atol=1e-7)
    np.testing.assert_allclose(h, numeric_h, rtol=2e-4, atol=1e-7)


def test_actual_upper_limit_reduces_local_prediction():
    d = synthetic()
    base = SpatialModel().fit(d)
    q = np.array([[100., 20.]])
    before = base.predict_log(q)[0][0]
    augmented = Observations(np.vstack((d.xy, q)), np.r_[d.log_value, before-2],
                             np.r_[d.censored, True], np.r_[d.log_sd, 0.], np.r_[d.row_ids, 99])
    after = SpatialModel().fit(augmented)
    mean, C = after.predict_log(q, covariance=True)
    assert mean[0] < before
    assert C[0, 0] > 0
    # No fake zeros: a very loose upper limit is almost uninformative.
    augmented.log_value[-1] = 40
    loose = SpatialModel().fit(augmented)
    np.testing.assert_allclose(loose.predict_log(q)[0], before, atol=2e-5)


def test_correlated_external_evidence_duplicate_invariance():
    m, C = np.array([0., 0.]), np.array([[1., .4], [.4, 1.]])
    one = condition_independent_evidence(m, C, [0], [2], [[.5]])
    repeated = condition_independent_evidence(m, C, [0, 0], [2, 2], np.full((2, 2), .5))
    for a, b in zip(one, repeated):
        np.testing.assert_allclose(a, b, atol=1e-12)
    independent = condition_independent_evidence(m, C, [0, 0], [2, 2], np.eye(2)*.5)
    assert independent[1][0, 0] < one[1][0, 0]
    with pytest.raises(ValueError, match="Contradictory"):
        condition_independent_evidence(m, C, [0, 0], [2, 3], np.full((2, 2), .5))


def test_csv_pseudo_invariance_and_vent_not_first_row(tmp_path):
    df = pd.read_csv("data/eruptions/Mazama_tephra.csv")
    original = tmp_path/"original.csv"
    df.to_csv(original, index=False)
    d, meta = read_observations(original)
    # Preserve known vent metadata independently of the fake rows.
    df.loc[:, ["Vent Lon", "Vent Lat"]] = np.nan
    df.loc[12, ["Vent Lon", "Vent Lat"]] = [-122.1, 42.95]
    fake = df.Pseudo.eq("Yes")
    df.loc[fake, ["Lon", "Lat", "Thickness"]] = [0, 0, 1e100]
    df = pd.concat([df, df.loc[fake]], ignore_index=True)
    changed = tmp_path/"changed.csv"
    df.to_csv(changed, index=False)
    changed_data, changed_meta = read_observations(changed)
    np.testing.assert_array_equal(d.xy, changed_data.xy)
    np.testing.assert_array_equal(d.log_value, changed_data.log_value)
    candidates = (Candidate("exponential", .75, .75), Candidate("power", .75, .75))
    a, va = select_model(d, candidates)
    b, vb = select_model(changed_data, candidates)
    assert va["selected_index"] == vb["selected_index"]
    np.testing.assert_allclose(a.predict_log(d.xy)[0], b.predict_log(d.xy)[0], atol=1e-12)
    assert len(changed_meta["ignored_pseudo_rows"]) == 2*len(meta["ignored_pseudo_rows"])


def test_zero_is_not_silently_a_measurement(tmp_path):
    p = tmp_path/"data.csv"
    p.write_text("Lon,Lat,Thickness,Vent Lon,Vent Lat\n1,1,0,0,0\n")
    with pytest.raises(ValueError, match="non-detections"):
        read_observations(p)
    p.write_text("Lon,Lat,Thickness,Vent Lon,Vent Lat,Kind,Limit_cm\n1,1,0,0,0,upper_limit,0.3\n")
    d, _ = read_observations(p)
    assert d.censored[0]
    assert np.exp(d.log_value[0]) == pytest.approx(.3)


def test_spatial_partition_is_permutation_invariant_and_groups_duplicates():
    xy = np.vstack((synthetic().xy, synthetic().xy[:3]))
    f = spatial_folds(xy)
    order = np.random.default_rng(0).permutation(len(xy))
    np.testing.assert_array_equal(spatial_folds(xy[order]), f[order])
    np.testing.assert_array_equal(f[:3], f[-3:])


def test_training_fits_do_not_use_heldout_values(monkeypatch):
    d = synthetic()
    candidates = (Candidate(),)
    records = []
    actual_fit = SpatialModel.fit
    def traced(self, obs):
        records.append(set(obs.row_ids))
        return actual_fit(self, obs)
    monkeypatch.setattr(SpatialModel, "fit", traced)
    _, report = select_model(d, candidates, n_folds=3)
    canonical = d.canonical()
    folds = np.array(report["folds"])
    for fold in range(3):
        assert records[fold].isdisjoint(canonical.row_ids[folds == fold])
    assert records[-1] == set(d.row_ids)


def test_reproducible_joint_volume_and_unit_conversion():
    model = SpatialModel().fit(synthetic())
    bounds = [-100, -100, 100, 100]
    a = integrate(model, bounds, resolution=8, draws=100, seed=11)
    b = integrate(model, bounds, resolution=8, draws=100, seed=11)
    assert a == b
    xy, area = grid_cells(bounds, 8)
    m, v = model.predict_log(xy)
    assert a["mean_km3"] == pytest.approx(np.exp(m+v/2).sum()*area*1e-5)
    assert a["conditional_quantiles_km3"]["0.025"] < a["conditional_quantiles_km3"]["0.975"]


@pytest.mark.parametrize("tail", ["exponential", "power"])
def test_trend_decays_in_all_directions(tail):
    model = SpatialModel(Candidate(tail)).fit(synthetic())
    angle = np.linspace(0, 2*np.pi, 100)
    direction = np.column_stack((np.cos(angle), np.sin(angle)))
    assert np.all((model._features(direction*1000)-model._features(direction*500))@model.beta < 0)
    if tail == "power":
        assert model.beta[1]-np.linalg.norm(model.beta[2:]) > 2


def test_nested_validation_assigns_each_real_row_once():
    d = synthetic(65)
    result = nested_validate(d, (Candidate(),), outer_folds=3, inner_folds=3)
    assert sorted(result["row_ids"]) == sorted(d.row_ids)
    assert np.isfinite(result["mean_log"]).all()
    assert sum(r["n_test"] for r in result["selections"]) == len(d.xy)
