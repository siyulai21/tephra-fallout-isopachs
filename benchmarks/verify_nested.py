"""Replay the published nested volumes, budgets, event likelihoods and matched row identities."""
import gzip
from hashlib import sha256
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from .nested_summary import ROOT, case_results, csv_rows, matched_challenges, read, sector_evidence
from src.empirical_isopach.core.audit import audited_data
from src.empirical_isopach.core.catalog import datasets
from src.empirical_isopach.models.nested.profile import ASSET
from src.empirical_isopach.models.nested.continuation import PROTOCOL, continuation_digest, radius_for, load_packet as load_draws
from src.empirical_isopach.models.nested.study import digest
from src.empirical_isopach.models.nested.geometry import positions
from src.empirical_isopach.models.nested.diagnostics import edge_sites
from src.empirical_isopach.validation import automatic_buffer


def verify_case(home, data_path):
    data, meta = audited_data(data_path,allow_intervals=True)
    report = read(home/"report.json")
    support = read(home/"support.json")
    assert report["input"]["input_sha256"] == meta["input_sha256"]
    assert report["thinning_source_sha256"] == digest(ASSET)
    geo = read(home/"geographic.json")
    withheld = []
    for fold in [None, *geo["folds"]]:
        name = "full" if fold is None else f"fold-{fold['fold']}"
        receipt = read(home/(name+".json"))
        if fold is not None:
            assert all(receipt[k] == fold[k] for k in receipt)
        file = home/(name+".npz")
        assert digest(file) == receipt["posterior_sha256"]
        expected_engine = (continuation_digest() if report["protocol"]==PROTOCOL else
                           "58ef3d3182fd152595d225df9faba0618f5190b1e06f4f0c87e52deb208344a6")
        assert receipt["identity"]["engine_sha256"] == expected_engine
        assert receipt["identity"]["input_sha256"] == meta["input_sha256"]
        model = load_draws(file)
        last = receipt["numerics"]["attempts"][-1]
        if report["protocol"]==PROTOCOL:
            num = receipt["numerics"]
            assert last["passed"] == num["parent_passed"]
            assert num["passed"] == (num["parent_passed"] and num["continuation"]["passed"])
            assert num["passed"] or last["particles_per_population"]>=num["ceiling"]
        else:
            assert last["passed"] == receipt["numerics"]["passed"]
        assert len(model.z) == last["independent_populations"]*last["particles_per_population"]
        if fold is None:
            assert receipt["identity"]["training_rows"] == data.row_ids.tolist()
            assert report["posterior_sha256"] == digest(file)
            vol = model.volume()
            if report["protocol"] == PROTOCOL:
                verify_edge_study(home, data_path, data, model)
            np.testing.assert_allclose(np.quantile(vol,[.025,.5,.975]),report["conditional_volume_km3"])
            np.testing.assert_allclose(report["prior_sensitivity"]["baseline"]["volume_km3"],report["conditional_volume_km3"])
            with np.load(home/"volume-budget.npz", allow_pickle=False) as budget:
                np.testing.assert_allclose(budget["volume_km3"],vol)
                np.testing.assert_allclose(budget["slab_km3"].sum(axis=0),vol)
                assert np.min(budget["slab_km3"]) >= -1e-8
                for t in budget["level_cm"]:
                    area = np.pi*radius_for(model,t)**2
                    np.testing.assert_allclose(np.quantile(area,[.025,.5,.975]),report["area_km2_at_cm"][str(t)])
        else:
            identity = receipt["identity"]
            assert not (set(identity["training_rows"]) & set(identity["test_rows"]))
            test = data.subset(np.isin(data.row_ids,identity["test_rows"]))
            assert test.row_ids.tolist() == identity["test_rows"]
            withheld.extend(test.row_ids)
            np.testing.assert_allclose(model.score(test),receipt["row_log_scores"],atol=1e-8)
    assert sorted(withheld) == sorted(data.row_ids.tolist())
    assert support["status"] == report["support_status"]
    accepted = [k for k,p in support["policies"].items() if p["selected_band"]]
    assert accepted == report["supported_band_policies"]
    taper = read(home/"distal-taper.json")
    np.testing.assert_allclose(taper["baseline_km3"],report["conditional_volume_km3"])
    previous = np.zeros(3)
    for factor in ("1.0","2.0","4.0"):
        s = taper["scenarios"][factor]
        values = np.array(s["volume_km3"])
        assert np.all(values>=previous) and np.all(values<=np.array(taper["baseline_km3"])*(1+1e-10))
        assert 0 < min(s["retained_volume_fraction"]) <= max(s["retained_volume_fraction"]) <= 1+1e-8
        for trained, reference in zip(s["geographic_folds"],geo["folds"],strict=True):
            for key in ("training_rows","test_rows"):
                assert trained[key] == reference["identity"][key]
        previous = values
    rows = csv_rows(home/"heldout.csv")
    assert sorted(int(r["row_id"]) for r in rows) == sorted(withheld)
    exact = [r for r in rows if r["censored"] == "False"]
    covered = sum(float(r["predictive_lower_log_cm"]) <= float(r["observed_or_bound_log_cm"]) <=
                  float(r["predictive_upper_log_cm"]) for r in exact)
    assert covered == report["exact_site_coverage95"]["covered"]
    assert len(exact) == report["exact_site_coverage95"]["exact_sites"]
    assert report["geographic_folds_passed"] == sum(f["numerics"]["passed"] for f in geo["folds"])
    np.testing.assert_allclose(report["geographic_score"],np.mean([f["negative_log_score"] for f in geo["folds"]]))
    for img in ("isopachs.png","world.png"):
        assert (home/img).read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
    if report["protocol"] == PROTOCOL:
        test = read(home/"bearing-study/report.json")
        assert test["input_sha256"] == meta["input_sha256"]
        np.testing.assert_allclose(test["baseline_km3"],report["conditional_volume_km3"])
        for pair,fold in zip(test["folds"],geo["folds"],strict=True):
            for key in ("training_rows","test_rows"):
                assert pair[key] == fold["identity"][key]
        for key,value in test["paired_scores"].items():
            np.testing.assert_allclose(value,np.mean([f[key] for f in test["folds"]]))
        assert all(0 <= low <= middle <= high <= 1 for low,middle,high in test["sector_fraction"])
        assert len(test["sector_fraction"]) == 8


def verify_edge_study(home, path, data, model):
    import pandas as pd
    import json
    table = edge_sites(model,data,path)
    rows = pd.read_csv(home/"edge-sites.csv")
    np.testing.assert_array_equal(rows.row_id, table.row_id)
    for key in ("edge_probability","edge_probability_in_largest_5pct_volumes"):
        np.testing.assert_allclose(rows[key],table[key],atol=1e-12)
        np.testing.assert_allclose(rows[key].sum(),1.)
    out = home/"edge-deletion"
    trial = json.loads((out/"report.json").read_text())
    assert trial["input_sha256"]==digest(path) and trial["edge_table_sha256"]==digest(home/"edge-sites.csv")
    assert trial["full_posterior_sha256"]==digest(home/"full.npz") and trial["engine_sha256"]==continuation_digest()
    assert trial["dominant_row"]==rows.row_id.iloc[0]
    single = data.row_ids==trial["dominant_row"]
    radius = automatic_buffer(data)[0]
    np.testing.assert_allclose(radius,trial["geographic_buffer_km"])
    cluster = np.linalg.norm(data.xy-data.xy[single][0],axis=1)<=radius
    assert trial["draws_sha256"] == digest(out/"volumes.npz")
    with np.load(out/"volumes.npz",allow_pickle=False) as draws:
        for key,mask in (("single",single),("cluster",cluster)):
            f = trial["fits"][key]
            assert f["test_rows"]==data.row_ids[mask].tolist() and f["training_rows"]==data.row_ids[~mask].tolist()
            np.testing.assert_allclose(np.quantile(draws[f.get("same_rows_as",key)],[.025,.5,.975]),f["volume_km3"])


def verify_distal_holdout(receipt, path):
    data,meta = audited_data(path,allow_intervals=True)
    assert receipt["input_sha256"] == meta["input_sha256"]
    distance = positions(data.xy,meta["vent_lonlat"][1])[0]
    edge = np.quantile(distance,.8)
    withheld = distance>=edge
    spatial_buffer = automatic_buffer(data)[0]
    training = (~withheld)&(cKDTree(data.xy[withheld]).query(data.xy)[0]>=spatial_buffer)
    assert receipt["training_rows"] == data.row_ids[training].tolist()
    assert receipt["test_rows"] == data.row_ids[withheld].tolist()
    for fit in receipt["fits"].values():
        assert len(fit["site_log_scores"]) == sum(withheld)
        np.testing.assert_allclose(fit["equal_site_negative_log_score"],-np.mean(fit["site_log_scores"]))


def verify_preview_peers(preview, source):
    directional = preview/"directional/azimuth"
    spherical = preview/"spherical-mixture"
    key = digest(source)
    d, s = read(directional/"report.json"),read(spherical/"report.json")
    assert d["input"]["input_sha256"] == s["input"]["input_sha256"] == key
    assert set(s["fits"])=={"baseline","diffuse_tail","wide_geography","volume_prior_shift",
                            "one_fan","five_fans","gaussian_errors","diffuse_mixture_weights"}
    assert len(d["members"])==6
    inventories = [*directional.parent.rglob("artifacts.json"),spherical/"artifacts.json"]
    for manifest in inventories:
        assert all(digest(manifest.parent/f)==v for f,v in read(manifest).items())
    with np.load(spherical/"bearing/baseline.npz",allow_pickle=False) as packet:
        q = [s["fits"]["baseline"]["volume"][k] for k in ("lower95","median","upper95")]
        np.testing.assert_allclose(np.quantile(packet["volume_km3"],[.025,.5,.975]),q)
    folds = read(preview/"nested-whole-area/geographic.json")["folds"]
    for n,dr,sr in zip(folds,d["validation"]["geographic"]["folds"],s["validation"]["fits"],strict=True):
        for nk,dk,sk in (("training_rows","train","training_row_ids"),("test_rows","test","withheld_row_ids")):
            assert n["identity"][nk]==dr[dk]==sr[sk]
    for name in ("directional","spherical-mixture"):
        sensitivity=read(preview/name/"sensitivity-study/report.json")
        assert sensitivity["input_sha256"]==key
        assert len(sensitivity["folds"])==len(folds)


def main():
    prior = read(ASSET)
    source = ROOT/"data/sources/nested-isopachs/paladio-table-source.html.gz"
    assert sha256(gzip.decompress(source.read_bytes())).hexdigest() == prior["source_sha256"]
    comp = ROOT/"comparisons/nested-isopachs"
    expected_cases, expected_sensitivities, expected_sectors = [], [], []
    for record in sorted(datasets(),key=lambda r:r["slug"]):
        home = ROOT/"eruptions"/record["slug"]/"results"
        verify_case(home/"nested-whole-area",record["path"])
        verify_case(ROOT/"archive/2026-11-21-nested-initial"/record["slug"],record["path"])
        rows, peer, report = case_results(record)
        expected_cases.extend(rows)
        expected_sensitivities.extend(peer)
        expected_sectors.extend(sector_evidence(record,report))
    with gzip.open(comp/"challenge-receipts.json.gz","rt") as stream:
        import json
        receipts = json.load(stream)
    assert len(receipts["nested"]) == 26
    for record in datasets():
        verify_distal_holdout(receipts["outer"][record["slug"]], record["path"])
    pairs = {(s["scenario"],s["seed"]):s for s in receipts["spherical"]}
    assert len(pairs) == 14
    assert all("candidate_quantiles_km3" in s for s in pairs.values())
    assert all("candidate_quantiles_km3" in s["directional"] for s in receipts["nested"] if s["seed"] in (90,91))
    assert len(receipts["recovery"]) == 4
    for example in receipts["recovery"]:
        assert len(example["row_ids"])==len(example["upper_limits"])==len(example["simulated_log_cm_or_upper_limit"])
        assert example["truth"]["outside_earth_upper_bound_km3"]<1e-7*example["truth"]["bulk_km3"]
    assert len(receipts["initial"]) == 26
    v,a = matched_challenges(receipts["nested"],pairs,receipts["initial"])
    for file, expected in (("whole-area-cases.csv",expected_cases),("peer-sensitivities.csv",expected_sensitivities),("sector-evidence.csv",expected_sectors),
                           ("known-volumes.csv",v),("known-areas.csv",a)):
        actual = csv_rows(comp/file)
        assert len(actual) == len(expected), file
        for row, source_row in zip(actual,expected,strict=True):
            for k,value in source_row.items():
                if isinstance(value,(float,np.floating)):
                    np.testing.assert_allclose(float(row[k]),value,rtol=1e-10,atol=1e-12)
                else:
                    assert row[k] == ("" if value is None else str(value)), (file,k)
    preview = comp/"pinatubo-preview"
    if preview.exists():
        data = preview/"data"
        provenance = read(data/"audit/provenance.json")
        file = data/"Pinatubo_1991_layer_C.csv"
        assert digest(file) == provenance["cases"]["principal"]["sha256"]
        verify_distal_holdout(receipts["outer"]["unregistered"],file)
        assert all(digest(data/"sources"/k)==v for k,v in provenance["source_transcript_sha256"].items())
        for entry in read(data/"originals/sources.json").values():
            path = data/"originals"/entry["stored"]
            stored = path.read_bytes()
            raw = gzip.decompress(stored) if path.suffix == ".gz" else stored
            assert len(raw) == entry["source_bytes"] and sha256(raw).hexdigest() == entry["source_sha256"]
        verify_case(preview/"nested-whole-area",file)
        verify_preview_peers(preview,file)
    print(f"Verified {len(expected_cases)//3} registered cases, the extra published snapshot, "
          f"{len(receipts['nested'])} challenge designs, paired fields, scores, budgets and source identities.")


if __name__ == "__main__":
    main()
