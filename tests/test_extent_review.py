"""Regressions for off-track peaks, supported holes and data leakage across eruptions."""
import numpy as np

from src.empirical_isopach.core.observations import Observations
from src.empirical_isopach.models.continuation_geometry import SampleLayout
from src.empirical_isopach.models.extent_release import predictive_summary
from src.empirical_isopach.models import extent_validation
from src.empirical_isopach.models.extent_weights import calibration
from src.empirical_isopach.models.mass_mixture.bearings.flank import FlankKernels
from src.empirical_isopach.models.mass_mixture.geometry import centers,from_sphere,log_field,to_sphere


def survey():
    xy=np.array([[0.,5.],[1.,8.],[-2.,10.],[80.,8.],[150.,9.],[220.,10.],[60.,-24.]])
    bounded=np.array([False]*6+[True])
    return Observations(xy,np.log([45.,42.,41.,12.,3.,1.,2.]),bounded,np.zeros(len(xy)),np.arange(len(xy))).canonical()


def test_observed_flanks_preserve_the_likelihood_and_reduce_a_concealed_peak():
    data=survey()
    layout=SampleLayout(data,0.)
    d=dict(log_volume=np.array([2.]),log_scale=np.log([[18.]]),offset=np.array([[[7.,-6.]]]),
           exponent=np.array([1.75]),tail_fraction=np.array([.12]),weights=np.ones((1,1)))
    model=FlankKernels(d,data,0.)
    xy=np.vstack((data.xy,from_sphere(centers(d["log_scale"],d["offset"])[0],0.)))
    original,broad,resolved=model.variants(xy,layout,("parent","broad","resolved"))
    np.testing.assert_allclose(original,log_field(d,to_sphere(xy,0.)),atol=2e-12)
    for corrected in (broad,resolved):
        np.testing.assert_allclose(corrected[:,:-1],original[:,:-1],atol=2e-12)
        assert corrected[0,-1]<original[0,-1]
    assert resolved[0,-1]<broad[0,-1]
    assert len(layout.lengths)==len(data.xy)-1


def test_other_eruptions_set_weights_and_exclude_failed_blocks():
    rows=[dict(slug="target",input_sha256="a",folds=[dict(passed=True,broad=[0.],resolved=[99.])]),
          dict(slug="donor",input_sha256="b",folds=[dict(passed=True,broad=[0.,0.],resolved=[-.8,-.5]),
                                                   dict(passed=False,broad=[0.],resolved=[500.])])]
    result=calibration("directional","new-hash",exclude_slug="target",ledger={"directional":rows})
    assert result["donors"]==["donor"] and result["eligible_folds"]==1
    assert result["weights"][0]>.5 and np.isclose(sum(result["weights"]),1.)
    assert result==calibration("directional","a",ledger={"directional":rows})


def test_empty_deletion_design_remains_visible_and_unscored(monkeypatch):
    folds = [("quadrants", n, None, None, None, None, "insufficient_training") for n in range(4)]
    monkeypatch.setattr(extent_validation, "fold_models", lambda *a, **kw: iter(folds))
    frame, fits = extent_validation.validate("directional", None, None, 0., None)
    rows, summary = predictive_summary(frame, fits, {"states": ["broad", "resolved"], "weights": [.5, .5]})
    assert rows.empty
    assert summary["quadrants"]["attempted_folds"] == 4
    assert summary["quadrants"]["negative_log_score"] is None
    assert not summary["quadrants"]["all_folds_passed"]
