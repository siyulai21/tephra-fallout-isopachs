"""Regression checks on synthetic thickness data and the frozen Mazama input."""
from dataclasses import replace

import numpy as np
import pytest
from scipy.integrate import quad
from scipy.special import ndtr

from src.empirical_isopach.core.audit import audited_data
from src.empirical_isopach.core.catalog import resolve_dataset
from src.empirical_isopach.core.distributions import LogMixture
from src.empirical_isopach.core.likelihood import censored_terms, interval_log_probability, normal_log_event
from src.empirical_isopach.core.observations import Observations, read_observations
from src.empirical_isopach.models.cubic_spline.model import TrendRegularizedSpline
from src.empirical_isopach.models.gaussian_process.model import SpatialModel


@pytest.mark.parametrize("lo,hi", [(-52.,-51.), (40.,40.04), (-.7,.8), (7.,7.000000001)])
def test_interval_mass_and_laplace_derivatives(lo, hi):
    # Independent integration, rescaled at the point of highest density.
    mode = np.clip(0., lo, hi)
    width = hi-lo
    integral = quad(lambda t: np.exp(-.5*((lo+t*width)**2-mode**2)), 0., 1.)[0]
    expected = np.log(integral*width)-.5*(mode**2+np.log(2*np.pi))
    assert interval_log_probability(lo,hi) == pytest.approx(expected, abs=1e-10)
    def terms(f):
        return np.ravel(censored_terms(np.array([f]), np.array([lo]), np.ones(1), True, np.array([hi])))
    value, grad, curvature = terms(0.)
    assert value == pytest.approx(-expected, abs=1e-10)
    delta = 1e-4
    minus, plus = terms(-delta), terms(delta)
    assert grad == pytest.approx((plus[0]-minus[0])/(2*delta), rel=2e-7, abs=1e-6)
    assert curvature == pytest.approx((plus[1]-minus[1])/(2*delta), rel=2e-5, abs=2e-6)


def test_interval_score_in_mixture_and_jax_gradients():
    import jax
    import jax.numpy as jnp
    from jax.scipy.special import log_ndtr
    jax.config.update("jax_enable_x64", True)
    y = np.array([.1, -.6, .3, -.5])
    upper = np.array([np.nan, np.nan, np.nan, .4])
    cens = np.array([False, True, True, True])
    lower = np.array([False, False, True, True])
    means = np.array([[-1.]*4, [1.]*4])
    sd = np.array([[.7]*4, [1.1]*4])
    weights = np.array([.25, .75])
    mix = LogMixture(weights, means, sd**2)
    expected = weights @ (ndtr((upper[3]-means[:,3])/sd[:,3])-ndtr((y[3]-means[:,3])/sd[:,3]))
    assert np.exp(mix.log_score(y, cens, lower, upper)[3]) == pytest.approx(expected)
    def objective(mu):
        return normal_log_event(y, mu, .7, cens, lower, upper, xp=jnp, logcdf=log_ndtr).sum()
    np.testing.assert_allclose(objective(-1.), normal_log_event(y, -1., .7, cens, lower, upper).sum())
    assert float(jax.grad(objective)(-1.)) == pytest.approx((objective(-.99999)-objective(-1.00001))/.00002)


@pytest.mark.parametrize("model", [SpatialModel, TrendRegularizedSpline])
def test_regional_training_responds_to_both_endpoints(model):
    x, y = np.meshgrid([10., 30., 50., 70.], [20., 40., 60.])
    xy = np.c_[x.ravel(), y.ravel()]
    values = 2.2-.004*x.ravel()-.003*y.ravel()+.18*np.cos(x.ravel())
    i = 5
    values[i] = .3
    bound = np.arange(len(xy)) == i
    upper = np.where(bound, .6, np.nan)
    data = Observations(xy, values, bound, np.zeros(len(xy)), np.arange(len(xy)), bound, upper)
    wide = replace(data, interval_log_upper=np.where(bound, 2.6, np.nan))
    narrow_fit = model().fit(data).predict_log(xy[[i]])[0][0]
    wide_fit = model().fit(wide).predict_log(xy[[i]])[0][0]
    assert narrow_fit < wide_fit - 1e-4


def test_registered_mazama_is_new_input_and_legacy_rows_stay_available():
    data, meta = audited_data(resolve_dataset("Mazama"))
    assert (len(data.xy), meta["n_measurements"], meta["n_intervals"],
            meta["n_lower_limits"], meta["n_upper_limits"]) == (300,175,29,0,96)
    assert meta["n_reported_measurements"] == 46 and data.reported_minimum.sum() == 129
    previous, old = audited_data(resolve_dataset("Mazama").with_name("Mazama_reviewed.csv"))
    assert len(previous.xy) == 271 and old["input_sha256"] != meta["input_sha256"]
    assert old["input_sha256"] == "1554141b67af566055595624bb7295c40c6da4500f95bc78c912e9d8b07a1b29"


@pytest.mark.parametrize("kind,point,limit,lo,hi", [
    ("interval","","",2,2), ("interval","","",0,4), ("interval","","",2,""),
    ("interval",3,"",2,4), ("lower_limit","",2,2,4)])
def test_inconsistent_range_schema_is_rejected(tmp_path, kind, point, limit, lo, hi):
    path = tmp_path/"input.csv"
    path.write_text(f"Lon,Lat,Vent Lon,Vent Lat,Kind,Thickness,Limit_cm,Lower_cm,Upper_cm\n1,1,0,0,{kind},{point},{limit},{lo},{hi}\n")
    with pytest.raises(ValueError):
        read_observations(path)
