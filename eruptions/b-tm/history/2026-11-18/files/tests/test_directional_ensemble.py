from types import SimpleNamespace

import numpy as np

from src.empirical_isopach.core.catalog import datasets
from src.empirical_isopach.models.directional.inference import field
from src.empirical_isopach.models.directional.geometry import Geometry, RADIUS
from src.empirical_isopach.models.directional.ensemble.fields import Member, Ensemble, draw_weights, inverse_cdf, latent_arrays
from src.empirical_isopach.models.directional.ensemble.integrals import median_integral
from src.empirical_isopach.models.directional.ensemble.members import MEMBERS, NAMES
from src.empirical_isopach.models.directional.ensemble.weights import calibrate, solve
from src.empirical_isopach.models.mass_mixture.study import audited_data


def test_pooled_field_quantile_and_complete_field_volume_are_distinct():
    values = np.array([[0,0,100,50,50,50.]])
    weights = draw_weights([.6,.4],[3,3])
    assert inverse_cdf(values,weights)[0,0] == 50
    assert inverse_cdf(np.array([0,50]),[.6,.4])[0] == 0
    fields = 1+9*np.eye(3)
    np.testing.assert_allclose(inverse_cdf(fields,np.ones(3)).sum(),3)
    assert inverse_cdf(fields.sum(1),np.ones(3))[0] == 12
    np.testing.assert_array_equal(inverse_cdf(np.arange(2),[.5,.5]),[0])
    np.testing.assert_array_equal(inverse_cdf(np.arange(3),[0,.5,.5]),[1])
    np.testing.assert_allclose(draw_weights([.5,.5],[2,4]),[.25,.25,.125,.125,.125,.125])


def test_exclude_target_all_copies_and_balance_calibration_cases():
    fold = {"usable":True, "row_ids":[3,4], "log_scores":[[0,-1,-2,-3,-4,-5],[-4,-2,-3,0,-4,-2]]}
    def case(slug, sha, lon, f):
        return {"slug":slug,"input_sha256":sha,"vent_lonlat":[lon,0],"folds":[f]}
    packet = {"members":NAMES,"cases":[case("alpha","aaa",0,fold),case("beta","bbb",25,fold),
                                     case("bad","ccc",55,{**fold,"usable":False})]}
    m = {"input_sha256":"aaa","vent_lonlat":[25,0]}
    assert not calibrate(m,packet=packet)["calibration_cases"]
    m["vent_lonlat"] = [-60,0]
    a = calibrate(m,packet=packet)
    assert a["calibration_cases"] == ["beta"]
    # Mutating the target's holdout values cannot change the fitted weights.
    packet["cases"][0]["folds"] = [{**fold,"log_scores":[[-1000,0,1000,0,-11,21]]*2}]
    np.testing.assert_array_equal(a["weights"],calibrate(m,packet=packet)["weights"])
    ls = np.array(fold["log_scores"])
    np.testing.assert_allclose(solve(ls,[.5,.5]),solve(np.repeat(ls,5,axis=0),np.full(10,.1)))
    assert np.isclose(sum(a["weights"]),1.) and min(a["weights"])>0


def fake_draws(g, n=4):
    rng = np.random.default_rng(67)
    shape = (2,n)
    return {"source":rng.normal(3,.2,shape),"log_range":rng.normal(3,.1,shape),
            "fan_log_range":rng.normal(3,.3,(*shape,len(g.centers))),
            "fan_logit":rng.normal(0,.1,shape),"tail_shape":rng.uniform(.1,.4,shape),
            "log_noise":rng.normal(-1,.05,shape),"nodes":rng.normal(3,.4,(*shape,len(g.vertices)))}


def test_fast_shared_geometry_matches_member_fields_and_observation_likelihood():
    data,meta = audited_data(datasets()[0]["path"])
    gs = [Geometry(data,meta["vent_lonlat"][1],p) for p in MEMBERS]
    models = [Member(g,fake_draws(g),np.arange(4),[]) for g in gs]
    xy = np.vstack([data.xy[:5],[[0,0],[200,300],[5000,-2500]]])
    arrays,_ = latent_arrays(models,xy)
    for m,values in zip(models,arrays,strict=True):
        np.testing.assert_allclose(field(m.samples,m.geometry,xy),values,rtol=2e-13,atol=2e-13)
    model=Ensemble(models,[1,3,2,1,6,1])
    dist=model.distribution(xy,observation=True,log_sd=np.full(len(xy),.2))
    value = dist.quantile(.15)
    np.testing.assert_allclose(dist.cdf(value),.15,atol=1e-11)
    np.testing.assert_allclose(np.exp(dist.log_score(value,np.ones(len(xy),bool))),.15,atol=1e-11)
    np.testing.assert_allclose(np.exp(dist.log_score(value,np.ones(len(xy),bool),np.ones(len(xy),bool))),.85,atol=1e-11)


def test_reference_free_whole_earth_median_integral_against_radial_solution():
    class Radial:
        axis=1.
        mesh=SimpleNamespace(xy=np.array([[20.,0],[25.,0]]))
        inside=np.empty(0,bool)
        centers=vertices=[]
        def __init__(self, policy):
            self.policy=policy
        def locate(self,xy):
            return np.full(len(xy),-1),np.full((len(xy),3),-1),np.zeros((len(xy),3))
        def coords(self,xy):
            return np.linalg.norm(xy,axis=1),np.empty((len(xy),0))
    members=[]
    for p in MEMBERS:
        g=Radial(p)
        d=fake_draws(g)
        d["source"].fill(np.log(51.))
        d["log_range"].fill(np.log(43.))
        d["tail_shape"].fill(0.)
        members.append(Member(g,d,np.arange(4),[]))
    expected=2*np.pi*51.*43.**2*(1-(1+2*RADIUS/43.)*np.exp(-2*RADIUS/43.))*1e-5
    answer=median_integral(Ensemble(members,np.arange(1,7)),expected)
    assert answer["passed"]
    np.testing.assert_allclose(answer["volume_km3"],expected,rtol=.001)
    np.testing.assert_allclose(np.sum(answer["parts_sectors_km3"],axis=0),expected/8,rtol=.012)


def test_assistant_led_directional_run_publishes_full_ensemble_at_root(tmp_path, monkeypatch):
    from pathlib import Path
    import json
    import pytest
    from src.empirical_isopach.models.directional import reporting
    from src.empirical_isopach.models.directional.ensemble import members, report

    calls = []
    def single(path, work, dest, name):
        from src.empirical_isopach.models.directional.study import digest
        page = dest/"README.md"
        page.write_text("Preserved reference study")
        (dest/"artifacts.json").write_text(json.dumps({"README.md": digest(page)}))

    # Exercise the real narrative on a checked-in packet without refitting the posterior.
    root = Path(__file__).resolve().parents[1]
    published = root/"eruptions/minoan/results/directional"
    result = json.loads((published/"ensemble/report.json").read_text())

    def ensemble(path, work, dest, name, *, family_page):
        dest.mkdir()
        report.narrative(dest, result, family_page=family_page)
        return result

    monkeypatch.setattr(reporting, "run_fits", lambda *args: calls.append("reference fits"))
    monkeypatch.setattr(reporting, "render", single)
    monkeypatch.setattr(report, "reuse_reference_fits", lambda *args: calls.append("verified cache import"))
    monkeypatch.setattr(members, "run_members", lambda *args: calls.append("ensemble fits"))
    monkeypatch.setattr(report, "render", ensemble)
    assert reporting.directional_dataset(datasets()[0]["path"], tmp_path) == result
    assert calls == ["reference fits", "verified cache import", "ensemble fits"]
    assert (tmp_path/"README.md").read_bytes() == (published/"README.md").read_bytes()
    assert not (tmp_path/"ensemble/README.md").exists()
    assert (tmp_path/"single-structure.md").read_text() == "Preserved reference study"
    with pytest.raises(ValueError, match="fresh"):
        reporting.directional_dataset(datasets()[0]["path"], tmp_path)
