import json
from pathlib import Path

import numpy as np
import pytest

from src.empirical_isopach.models.directional.archive_readings import bridge_log_probability,load_inventory,positions
from src.empirical_isopach.models.directional.bearing_calibration import PACKET,calibrate
from src.empirical_isopach.models.directional.bearing_support import BearingGeometry,include_visibility_readings
from src.empirical_isopach.models.directional.bearing_validation import independent_archive_prediction, training_readings
from src.empirical_isopach.models.directional.geometry import Geometry,polar
from src.empirical_isopach.models.directional.integration import integrate,make_rule
from src.empirical_isopach.models.mass_mixture.study import audited_data

REVIEWED = Path("data/eruptions/B-Tm_reviewed.csv")
INVENTORY = Path("data/audits/B-Tm/archive-observations.json")


@pytest.mark.parametrize("scale_key", ["5", "5.0"])
def test_spherical_archive_summary_matches_numeric_detection_scales(tmp_path, scale_key):
    from src.empirical_isopach.models.mass_mixture.archive_check import write_summary

    site = {"id": "core", "name": "Core", "latent_cm": {"median": 2, "lower95": 1, "upper95": 4},
            "p_local_below_scale": {scale_key: 0.8}}
    scenario = {"scenario": None, "full_field_volume_km3": {"lower95": 10, "upper95": 20},
                "importance_checks_passed": True, "sites": [site]}
    report = {"scenarios": {"quantitative": scenario}, "accepted_archive_bounds_cm": {"core": 5.0},
              "bearing_packet_sha256": "test"}
    write_summary(report, tmp_path)
    assert "| Core | 2 | 1–4 | 5 | 80.0% |" in (tmp_path / "README.md").read_text()


def test_bearing_geometry_respects_the_likelihood_and_physical_area():
    data,meta=audited_data(REVIEWED)
    g=Geometry(data,meta["vent_lonlat"][1])
    narrow=BearingGeometry(g,data,.25)
    angles=np.linspace(0,2*np.pi,501)
    factor=narrow.stretch(angles)
    assert (factor>=1).all() and factor.max()>3.8 and factor.max()<=4+1e-12
    np.testing.assert_allclose(narrow.stretch(angles),narrow.stretch(angles-2*np.pi))
    np.testing.assert_allclose(narrow.coords(data.xy)[0],g.coords(data.xy)[0],rtol=1e-14)
    np.testing.assert_allclose(narrow.coords(g.vertices)[0],g.coords(g.vertices)[0],rtol=1e-14)
    shape=(2,2)
    d={"source":np.full(shape,np.log(20.)),"log_range":np.full(shape,np.log(25.)),
       "fan_log_range":np.zeros((*shape,len(g.centers))),"fan_logit":np.zeros(shape),
       "tail_shape":np.full(shape,.12),"nodes":np.full((*shape,len(g.vertices)),np.log(12.))}
    rule=make_rule(g,2,4)
    identity=BearingGeometry(g,data,1.)
    old=integrate(d,g,rule)
    np.testing.assert_array_equal(integrate(d,g,identity.quadrature(2,4)),old)
    changed=integrate(d,g,narrow.quadrature(2,4))
    np.testing.assert_allclose(changed[...,0,:],old[...,0,:],rtol=1e-12,atol=1e-12)
    assert np.all(changed.sum(axis=(-1,-2))<old.sum(axis=(-1,-2)))


def test_archive_transfer_requires_an_inspected_secure_independent_horizon(tmp_path):
    path=REVIEWED
    inv,_=load_inventory(path,INVENTORY)
    test=tmp_path/"archives.json"
    inv["scenarios"][0]["effective_limit_cm"]["gushantun"]=1
    test.write_text(json.dumps(inv))
    with pytest.raises(ValueError,match="Unresolved"):
        load_inventory(path,test)
    inv["scenarios"][0]["effective_limit_cm"].pop("gushantun")
    inv["sites"][0]["dependency_group"]=inv["sites"][1]["id"]
    test.write_text(json.dumps(inv))
    with pytest.raises(ValueError,match="independent"):
        load_inventory(path,test)


def test_validation_needs_a_mapped_scenario_and_retains_empty_archive_holdouts(tmp_path):
    path = REVIEWED
    inventory,_ = load_inventory(path,INVENTORY)
    inventory["validation_scenario"] = "missing_scenario"
    supplied = tmp_path/"inventory.json"
    supplied.write_text(json.dumps(inventory))
    with pytest.raises(ValueError,match="mapped deposit"):
        load_inventory(path,supplied)
    data,meta = audited_data(path)
    site = inventory["sites"][0]["id"]
    result = independent_archive_prediction(data,None,inventory["sites"],meta["vent_lonlat"],
        inventory["scenarios"][0],{site:1.},{},tmp_path,np.inf,None,False)
    assert result["status"]=="insufficient buffered support"
    assert result["quantitative_training_row_ids"]==[]


def test_bridge_recovers_exact_event_probability_with_a_small_fraction_of_overlapping_states():
    # Under the two equally likely states the observation probabilities are .01 and .99.
    # The updated distribution assigns the high-probability state mass .99.
    before=np.log(np.tile([.01,.99],1000))
    after=np.log(np.repeat([.01,.99],[20,1980]))
    assert np.exp(bridge_log_probability(before,after))==pytest.approx(.5,rel=1e-10)


def test_conditional_sites_cannot_seed_a_fan_or_leak_into_heldout_areas():
    path=REVIEWED;data,meta=audited_data(path)
    inv,_=load_inventory(path,INVENTORY);vent=meta["vent_lonlat"];g=Geometry(data,vent[1])
    arch=inv["sites"][:2];xy=positions(arch,vent)
    extended=include_visibility_readings(data,xy,[1.,2.])
    assert len(extended.xy)==len(data.xy)+2 and np.all(extended.censored[-2:])
    assert np.all(extended.row_ids[-2:]<0)
    assert not np.any((g.vertices[:,None,:]==xy).all(axis=2))
    limits=inv["scenarios"][0]
    test=data.subset([int(np.linalg.norm(data.xy-xy[0],axis=1).argmin())])
    kept=training_readings(data,test,arch,limits,vent,"geographic",0,g,0)
    assert arch[0]["id"] not in kept["effective_limit_cm"]
    w,external=calibrate(None,"new-name",meta["input_sha256"],vent)
    assert len(w)==12 and np.isclose(w.sum(),1.) and "b-tm" not in external
    for rec in json.loads(PACKET.read_text())["cases"]:
        for fold in rec["folds"]:
            assert not set(fold["training_row_ids"]) & set(fold["withheld_row_ids"])
