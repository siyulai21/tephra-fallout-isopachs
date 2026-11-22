"""Replay the released corridor fields, independent survey thresholds and matched partitions."""
import csv
import gzip
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from .nested_reach import case_home
from .nested_reach_challenge import records
from .nested_reach_field import FIELD_CASES, assess, partition
from .nested_reach_transects import crossings
from src.empirical_isopach.core.audit import audited_data
from src.empirical_isopach.models.directional.ensemble.members import splits
from src.empirical_isopach.models.nested.continuation import load_packet
from src.empirical_isopach.models.nested.reach import Corridors, PROBS, ReachDraws
from src.empirical_isopach.models.nested.reach_fit import signature
from src.empirical_isopach.models.nested.reach_report import prior_sensitivity
from src.empirical_isopach.models.nested.study import digest


def read(path):
    return json.loads(Path(path).read_text())


def verify_case(rec):
    slug = rec["slug"]
    home = case_home(slug) / "nested-reach"
    old = case_home(slug) / "nested-isopachs"
    j = read(home / "report.json")
    data, meta = audited_data(rec["path"], allow_intervals=True)
    assert j["input_sha256"] == meta["input_sha256"] and j["engine_sha256"] == signature()
    lat = meta["vent_lonlat"][1]
    for index,train,test in [(None,data,None),*splits(data,lat,"geographic")]:
        key = "full" if index is None else f"fold-{index}"
        fit = read(home / (key + ".json"))
        file = home / (key + ".npz")
        assert fit["posterior_sha256"] == digest(file)
        assert fit["identity"]["parent_posterior_sha256"] == digest(old / (key + ".npz"))
        assert fit["identity"]["engine_sha256"] == signature()
        assert fit["identity"]["training_rows"] == train.row_ids.tolist()
        model = ReachDraws.load(file)
        original = Corridors.from_training(train, lat)
        # PROJ and libm differ by a few nanometres in projected distances on arm64 and
        # x86_64. Input IDs and packet digests above still match byte for byte.
        np.testing.assert_allclose(original.radii, model.corridors.radii, rtol=2e-12, atol=1e-9)
        np.testing.assert_allclose(np.sin(original.bearings-model.corridors.bearings), 0., atol=2e-12)
        np.testing.assert_allclose(np.cos(original.bearings-model.corridors.bearings), 1., atol=2e-12)
        np.testing.assert_allclose(original.proximal, model.corridors.proximal, rtol=2e-12, atol=1e-9)
        if test is not None:
            assert fit["identity"]["test_rows"] == test.row_ids.tolist()
            np.testing.assert_allclose(model.score(test), fit["row_log_scores"], atol=1e-8)
            np.testing.assert_allclose(fit["row_log_scores"], j["folds"][index]["row_log_scores"])
        else:
            assert j["full_packet_sha256"] == digest(file)
            np.testing.assert_allclose(j["reference_volume_km3"],np.quantile(load_packet(old/"full.npz").volume(),PROBS))
    with np.load(home/"volume-budget.npz",allow_pickle=False) as b:
        np.testing.assert_allclose(b["sector_km3"].sum(axis=1), b["volume_km3"])
        np.testing.assert_allclose(np.quantile(b["volume_km3"],PROBS),j["volume_km3"])
        scenarios=prior_sensitivity(b)["broad_prior_scenarios"]
        for p,s in scenarios.items():
            np.testing.assert_allclose(s["volume_km3"],j["prior_sensitivity"]["broad_prior_scenarios"][p]["volume_km3"])
    mesh=read(home/"support.json")
    assert j["supported_band_policies"] == [k for k,v in mesh["policies"].items() if v["selected_band"]]
    with gzip.open(home/"local-brackets.csv.gz","rt") as stream:
        for row in csv.DictReader(stream):
            good = int(row["spacing_accepted"]) and int(row["measured_bracket"]) and float(row["posterior_crossing_probability"])>=.9
            assert bool(int(row["display_solid"])) == bool(good)
    return j


def main():
    comp=Path("comparisons/nested-reach")
    for rec in records():
        verify_case(rec)
    known=json.loads(gzip.decompress((comp/"validation.json.gz").read_bytes()))
    assert len(known["quadrants"])==36 and len(known["known"])==18
    assert sum(len(r["truths"]) for r in known["known"])==20
    for j in known["quadrants"]:
        assert not (set(j["test_rows"]) & set(j["training_rows"]))
        if j["status"]=="estimated":
            assert len(j["test_rows"])==len(j["previous_log_score"])==len(j["corridor_log_score"])
    for v in read(comp/"regional-parity.json").values():
        assert digest(v["path"])==v["sha256"]
    survey=comp/"field-surveys"
    with TemporaryDirectory() as scratch:
        for slug in FIELD_CASES:
            home=survey/slug
            train,test,meta,_=partition(slug,Path(scratch))
            val=read(home/"validation.json")
            assert val["identity"]["training_rows"]==train.row_ids.tolist()
            assert val["identity"]["test_rows"]==test.row_ids.tolist()
            assert val["identity"]["input_sha256"]==meta["input_sha256"]
            assert val["identity"]["engine"]==signature()
            assert digest(home/"training-parent.npz")==read(home/"training-parent.json")["sha256"]
            assert digest(home/"corridor.npz")==val["corridor_packet_sha256"]
            for key,m in (("whole_area",load_packet(home/"training-parent.npz")),("corridor",ReachDraws.load(home/"corridor.npz"))):
                result=assess(m,test)
                np.testing.assert_allclose(result["row_log_score"],val["tested_models"][key]["row_log_score"],atol=1e-8)
                for lev,items in result["thresholds"].items():
                    for k,x in items.items():
                        expected=val["tested_models"][key]["thresholds"][lev][k]
                        if x is None:
                            assert expected is None
                        else:
                            np.testing.assert_allclose(x,expected,atol=1e-8)
            if slug=="calbuco-2015":
                assert digest(Path(scratch)/"row-decisions.csv")==digest(home/"row-decisions.csv")
                assert len(train.xy)==159 and len(test.xy)==43
    _,references,_,_=crossings()
    with (survey/"calbuco-2015/mapped-01mm-transects.csv").open() as stream:
        for expected,printed in zip(references,csv.DictReader(stream),strict=True):
            for k in expected:
                np.testing.assert_allclose(expected[k],float(printed[k]))
    print("Verified nine geographic comparisons, 36 compass deletions, 20 known truths and four withheld field surveys.")


if __name__=="__main__":
    main()
