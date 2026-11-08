"""Independent checks for source semantics and approximate deletion inference."""
from dataclasses import asdict
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm

from benchmarks.aso_diagnosis import weighted_quantiles, deletion_summary
from benchmarks.aso_evidence import case_data, RAW
from benchmarks.aso_revision import revised_table
from benchmarks.aso_marine import marine_rows
from src.empirical_isopach.core.observations import read_observations
from src.empirical_isopach.models.mass_mixture.prior import BASELINE


def test_weighted_quantiles_ignore_zero_mass_and_validate_probabilities():
    np.testing.assert_equal(weighted_quantiles([0.,1.,2.,3.],[0.,1.,2.,1.],[0.,.25,.5,1.]),[1.,1.,2.,3.])
    for p in ([-.1],[1.1],[np.nan],[[.5]]):
        with pytest.raises(ValueError): weighted_quantiles([1.],[1.],p)


def test_psis_deletion_matches_conjugate_normal_reference():
    # theta ~ N(0,4), yi | theta ~ N(theta,1). Delete the most extreme datum.
    y = np.array([.4,.6,.7,1.,1.1,1.3,1.4,2.])
    var = 1/(.25+len(y)); mean = var*y.sum()
    theta = norm.ppf((np.arange(80000)+.5)/80000,loc=mean,scale=np.sqrt(var))
    likelihood = norm.logpdf(y[-1],loc=theta,scale=1.)
    result = deletion_summary(theta,likelihood,1.)
    deleted_var = 1/(.25+len(y)-1); deleted_mean = deleted_var*y[:-1].sum()
    exact = np.exp(norm.ppf([.025,.5,.975],loc=deleted_mean,scale=np.sqrt(deleted_var)))
    actual = [result[k] for k in ("lower95_km3","median_km3","upper95_km3")]
    np.testing.assert_allclose(actual,exact,rtol=.015)
    assert result["psis_passed"]
    assert abs(result["d_mean_log_volume_d_likelihood_power"]-var*(y[-1]-mean)) < 1e-4


def test_source_bounds_and_deletions_keep_raw_row_identity(tmp_path):
    original = pd.read_csv(RAW)
    unchanged = original.copy(deep=True)
    table,excluded,prior = case_data("bounds_only",original)
    path=tmp_path/"input.csv"; table.to_csv(path,index=False)
    data,meta=read_observations(path)
    assert meta["n_measurements"] == 84
    assert meta["n_upper_limits"] == 1 and meta["n_lower_limits"] == 2
    for row,lower,value in ((43,False,10),(52,True,3),(55,True,7)):
        i = np.flatnonzero(data.row_ids == row)[0]
        assert data.censored[i] and data.lower_censored[i] == lower
        assert np.exp(data.log_value[i]) == pytest.approx(value)
    assert sorted(meta["ignored_pseudo_rows"]) == list(range(89,95))
    assert excluded == [] and prior == BASELINE
    pd.testing.assert_frame_equal(original,unchanged)
    _,excluded,_ = case_data("omit_ambiguous",original)
    assert excluded == [4,6,14,20]
    retained=data.subset(~np.isin(data.row_ids,excluded))
    assert len(retained.xy) == 83 and not np.isin(retained.row_ids,excluded).any()
    assert 37 in retained.row_ids and 49 in retained.row_ids


def test_source_conflict_and_prior_sensitivities_are_separate():
    original=pd.read_csv(RAW)
    lower,_,_=case_data("s2_bounds",original)
    assert lower.loc[51,"Kind"] == "lower_limit" and lower.loc[51,"Limit_cm"] == 6.7
    for case,value in (("primary_6p5",6.5),("primary_7",7.)):
        table,_,_=case_data(case,original)
        assert table.loc[51,"Kind"] == "measurement" and table.loc[51,"Thickness"] == value
    for case,sd in (("offset_sd1",1.),("offset_sd6",6.)):
        table,excluded,prior=case_data(case,original)
        assert table.Kind.eq("measurement").all() and excluded == []
        expected=asdict(BASELINE); expected["offset_sd"]=sd
        assert asdict(prior) == expected
        pd.testing.assert_frame_equal(table[original.columns],original)


def test_cruise_corrections_preserve_measured_thickness_and_row_order():
    original = pd.read_csv(RAW)
    ledger = pd.DataFrame(columns=["csv_row", "s2_reference", "s2_sheet_row"])
    revised = revised_table(ledger, coordinate_only=True)
    expected = original.copy()
    expected.loc[3, "Lat"] = 34.5
    expected.loc[6, "Lon"] = 146.
    pd.testing.assert_frame_equal(revised[original.columns], expected)
    assert revised.loc[3, "Thickness"] == 7. and revised.loc[6, "Thickness"] == 6.
    bounded = revised_table(ledger)
    assert bounded.Kind.ne("measurement").sum() == 4
    assert list(bounded.index) == list(original.index)


def test_setting_stress_test_does_not_infer_terrestrial_from_missing_setting():
    ledger = pd.DataFrame({"csv_row": [2,3,4,5,6,7],
        "setting": ["Submarine", " Submarine ", "Submarine (location estimated)", "Lake Biwa", None, "Subaerial"]})
    assert marine_rows(ledger) == [2,3,4]


def test_registered_source_revision_reproduces_from_immutable_raw_and_ledger():
    ledger = pd.read_csv("data/audits/Aso-4/row-ledger.csv")
    path = Path("data/eruptions/Aso-4_reviewed.csv")
    assert revised_table(ledger).to_csv(index=False) == path.read_text()
    audit = json.loads(Path("data/audits/Aso-4/provenance.json").read_text())
    assert sha256(path.read_bytes()).hexdigest() == audit["cases"]["principal"]["sha256"]
    assert len(marine_rows(ledger)) == 28
