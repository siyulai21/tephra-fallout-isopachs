"""Replay the released corridor fields, independent survey thresholds and matched partitions."""
import csv
import gzip
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from PIL import Image

import numpy as np

from .nested_reach import case_home
from .nested_reach_challenge import records
from .nested_reach_field import FIELD_CASES, assess, partition
from .nested_reach_transects import crossings
from src.empirical_isopach.core.audit import audited_data
from src.empirical_isopach.models.directional.ensemble.members import splits
from src.empirical_isopach.models.nested.continuation import load_packet
from src.empirical_isopach.models.nested.reach import Corridors, PROBS, PROTOCOL, ReachDraws
from src.empirical_isopach.models.nested.reach_fit import signature
from src.empirical_isopach.models.nested.reach_report import prior_sensitivity
from src.empirical_isopach.models.nested.study import digest


def read(path):
    return json.loads(Path(path).read_text())


def verify_case(rec):
    slug = rec["slug"]
    home = case_home(slug) / "nested-isopachs"
    old = case_home(slug) / "nested-whole-area"
    j = read(home / "report.json")
    data, meta = audited_data(rec["path"], allow_intervals=True)
    assert j["input_sha256"] == meta["input_sha256"] and j["engine_sha256"] == signature()
    assert j["protocol"] == PROTOCOL and j["model_role"] == "principal"
    assert set(j).isdisjoint({"volume_km3","conditional_volume_km3"})
    assert j["estimate_source"] == "deposit.json#median_map_volume_km3"
    lat = meta["vent_lonlat"][1]
    covered = exact = 0
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
            coverage = j["folds"][index]["exact_site_coverage95"]
            replay = assess(model, test)
            assert coverage["exact_rows"] == test.row_ids[~test.censored].tolist()
            np.testing.assert_allclose(coverage["predictive_cdf"], np.asarray(replay["predictive_cdf_at_value"])[~test.censored], atol=1e-8)
            assert coverage["covered"] == replay["covered95_exact"]
            covered += coverage["covered"]
            exact += len(coverage["exact_rows"])
            np.testing.assert_allclose(model.score(test), fit["row_log_scores"], atol=1e-8)
            np.testing.assert_allclose(fit["row_log_scores"], j["folds"][index]["row_log_scores"])
        else:
            assert j["full_packet_sha256"] == digest(file)
            with Image.open(home/"isopachs.png") as image:
                shown = json.loads(image.info["Description"])
            assert shown["protocol"] == PROTOCOL and shown["posterior_sha256"] == digest(file)
            assert shown["input_sha256"] == meta["input_sha256"]
            np.testing.assert_allclose(shown["conditional_volume_interval95_km3"], j["conditional_volume_interval95_km3"])
            np.testing.assert_allclose(shown["median_map_volume_km3"], j["median_map_volume_km3"])
            points = np.asarray(shown["x_km_y_km_log10_median_cm"])
            expected = np.median(model.reduced(shown["map_draws"]).field(points[:,:2]), axis=0)/np.log(10.)
            np.testing.assert_allclose(points[:,2], expected, atol=1e-8)
            np.testing.assert_allclose(j["reference_volume_interval95_km3"],np.quantile(load_packet(old/"full.npz").volume(),[.025,.975]))
    assert j["exact_site_coverage95"] == {"covered":covered, "exact_sites":exact, "fraction":covered/exact if exact else None}
    with np.load(home/"volume-budget.npz",allow_pickle=False) as b:
        np.testing.assert_allclose(b["sector_km3"].sum(axis=1), b["volume_km3"])
        np.testing.assert_allclose(np.quantile(b["volume_km3"],[.025,.975]),j["conditional_volume_interval95_km3"])
        scenarios=prior_sensitivity(b)["broad_prior_scenarios"]
        for p,s in scenarios.items():
            np.testing.assert_allclose(np.asarray(s["volume_km3"])[[0,-1]],j["prior_sensitivity"]["broad_prior_scenarios"][p]["conditional_volume_interval95_km3"])
    mesh=read(home/"support.json")
    assert j["supported_band_policies"] == [k for k,v in mesh["policies"].items() if v["selected_band"]]
    with gzip.open(home/"local-brackets.csv.gz","rt") as stream:
        for row in csv.DictReader(stream):
            good = int(row["spacing_accepted"]) and int(row["measured_bracket"]) and float(row["posterior_crossing_probability"])>=.9
            assert bool(int(row["display_solid"])) == bool(good)
    return j


def main():
    comp=Path("comparisons/nested-reach")
    with Path("comparisons/nested-isopachs/cases.csv").open(newline="") as stream:
        rows = {r["slug"]:r for r in csv.DictReader(stream) if r["method"] == "Nested corridor"}
    for rec in records():
        j = verify_case(rec)
        if rec["slug"] == "pinatubo-1991":
            continue
        row = rows.pop(rec["slug"])
        assert row["input_sha256"] == j["input_sha256"]
        np.testing.assert_allclose([float(row[k]) for k in ("lower95_km3","upper95_km3")],
                                   j["conditional_volume_interval95_km3"])
        np.testing.assert_allclose(float(row["median_map_km3"]), read(case_home(rec["slug"])/"nested-isopachs/deposit.json")["median_map_volume_km3"])
        np.testing.assert_allclose(float(row["equal_region_log_score"]),j["geographic_score"])
    assert not rows
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
