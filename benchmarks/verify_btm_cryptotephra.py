"""Verify the B-Tm core evidence, accepted likelihood, matched refits and volume draws."""
import gzip
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

from benchmarks.btm_cryptotephra import AUDIT, NEW, OLD, ROOT, digest, interpreted_limits, make_input, profile_integrals, rows
from benchmarks.btm_cryptotephra_evidence import ice_budget, kushu_window_check
from src.empirical_isopach.core.audit import audited_data
from src.empirical_isopach.models.directional.ensemble.integrals import interval
from src.empirical_isopach.models.mass_mixture.source_protocol import protocol

HOME = ROOT / "eruptions/b-tm/results"
LABELS = ("lower95", "median", "upper95")


def read(path):
    return json.loads(path.read_text())


def verify_inputs():
    design = read(AUDIT/"cryptotephra-principal.json")
    prior_rows = rows(OLD)
    assert len(prior_rows) == 68
    assert digest(OLD) == design["cases"]["thickness_only"]["sha256"]
    for link in ("original_source_audit", "archive_inventory"):
        assert digest(ROOT/design[link]["file"]) == design[link]["sha256"]
    limits = interpreted_limits()
    assert limits == {r["evidence_id"]:r["interpreted_upper_cm"] for r in design["principal_limits"]}
    assert probe_limits(design) == limits
    settings = {"principal": limits, "one_tenth_response": {k:10*v for k,v in limits.items()},
                "recognition_analogs": {"longgang":1., "quanyang":2., "old-084":1.}}
    with TemporaryDirectory() as scratch:
        for name, bounds in settings.items():
            case = design["cases"][name]
            path = ROOT/case["file"]
            out = Path(scratch)/path.name
            make_input(out, bounds)
            assert digest(path) == digest(out) == case["sha256"]
            selected = rows(path)
            assert selected[:68] == prior_rows and len(selected) == 71
            assert len({(float(r["Lat"]),float(r["Lon"])) for r in selected}) == len(selected)
            for row in selected[68:]:
                assert row["Kind"] == "upper_limit" and row["Thickness"] == ""
                assert float(row["Limit_cm"]) == bounds[row["EvidenceID"]]
    recorded = pd.read_csv(AUDIT/"cryptotephra-count-profiles.csv")
    pd.testing.assert_frame_equal(recorded, pd.DataFrame(profile_integrals()), check_exact=False, rtol=1e-13)
    ice = ice_budget()
    stored = read(AUDIT/"greenland-size-fraction.json")
    assert ice.keys() == stored.keys()
    for key in ("entire_record","946_to_948_CE"):
        for name, value in ice[key].items():
            np.testing.assert_allclose(value, stored[key][name], rtol=1e-12)
    assert ice["sha256"] == stored["sha256"]
    window = read(AUDIT/"kushu-window-sensitivity.json")
    for key, value in kushu_window_check().items():
        if isinstance(value, float):
            np.testing.assert_allclose(window[key],value,rtol=1e-12)
    auxiliary = read(AUDIT/"cryptotephra-profile-sources.json")
    ice_count = auxiliary["neem_recovered_particle_count"]
    area = np.pi*(ice_count["whole_core_diameter_cm"]/2)**2
    np.testing.assert_allclose(area,ice_count["whole_core_area_cm2"])
    for numerator,reported in (("reported_minimum_all_glass_in_band","recovered_all_glass_in_band_per_full_core_cm2_minimum"),
             ("conservative_minimum_B_Tm_identified_fragments_any_size","recovered_matched_fragments_per_full_core_cm2_minimum")):
        np.testing.assert_allclose(ice_count[numerator]/area,ice_count[reported])
    h = auxiliary["sihailongwan_host_mass_check"]
    x0,x1 = h["horizontal_tick_pixels"]
    lo,hi = h["horizontal_ticks_g_cm3"]
    margin = h["digitization_allowance_pixels_per_edge"]
    edge = h["rightmost_density_mark_pixel"] + margin
    upper_density = lo+(hi-lo)*(edge-x0+margin)/(x1-x0-2*margin)
    np.testing.assert_allclose(np.ceil(100*upper_density)/100,h["recent_host_dry_density_ceiling_g_cm3"])
    np.testing.assert_allclose(h["recent_host_dry_density_ceiling_g_cm3"]*h["source_window_cm"]
                               /h["original_bulk_density_floor_g_cm3"],h["full_retention_mass_budget_upper_cm"])
    return design


def probe_limits(design):
    inventory = read(AUDIT/"cryptotephra-probes.json")
    assert inventory["scenarios"] == [] and not inventory.get("validation_scenario")
    assert all(r["unconditional_upper_cm"] is None and r["original_reported_limit_cm"] is None
               for r in design["principal_limits"])
    return {r["id"]:r["principal_effective_upper_cm"] for r in inventory["sites"]
            if r.get("in_principal_as") == "interpreted_upper_limit"}


def verify_release(design, home=HOME):
    study = home/"cryptotephra"
    release = read(study/"release.json")
    for filename, expected in release["files"].items():
        assert digest(ROOT/filename) == expected, filename
    observations, meta = audited_data(NEW)
    ids = set(observations.row_ids)
    bounds = {r["id"]:r["row_id"] for r in meta["interpreted_upper_limits"]}
    regional = read(home/"report.json")
    sphere = read(home/"spherical-mixture/report.json")
    az = read(home/"directional/azimuth/report.json")
    for report in (regional, sphere, az, read(home/"directional/ensemble/report.json"),
                   read(home/"directional/single-structure/report.json")):
        assert report["input"]["input_sha256"] == digest(NEW)
        assert report["input"]["audit"] == meta["audit"]
        assert report["input"]["interpreted_upper_limits"] == meta["interpreted_upper_limits"]
        assert [report["input"][k] for k in ("n_measurements","n_lower_limits","n_upper_limits")] == [64,3,4]
    assert set(az["data_rows"]) == ids
    assert az["archive"]["mapped_scenario"] is None
    scenarios = az["archive"]["scenarios"]
    assert set(scenarios) == {"quantitative"} and not scenarios["quantitative"]["assumed_effective_scales_cm"]
    assert set(bounds) == {sid for sid, s in scenarios["quantitative"]["sites"].items() if s["in_likelihood"]}
    with np.load(home/"directional/azimuth/fields.npz") as fields:
        assert "archive_weights" not in fields
    rv, sv = regional["validation"], sphere["validation"]
    assert set(rv["row_ids"]) == ids
    assert all(rv[k] == sv[k] for k in ("row_ids","folds","buffer_km"))
    for fold, azi in zip(sv["fits"], az["validation"]["geographic"]["folds"], strict=True):
        assert fold["fold"] == azi["fold"]
        for k, short in (("training_row_ids","train"),("withheld_row_ids","test")):
            assert set(fold[k]) == set(azi[short])
            for method in rv["methods"].values():
                assert fold[k] == method["selections"][fold["fold"]][k]
    prior, others = protocol("source-fans")
    assert set(sphere["fits"]) == {p.name for p in (prior,*others)}
    assert len(sv["fits"]) == 4
    assert read(home/"support/report.json")["provenance"]["parent_report_sha256"] == digest(home/"report.json")
    folder = home/"spherical-mixture"
    with gzip.open(folder/"support/b-tm.json.gz","rt") as stream:
        screen = json.load(stream)
    assert screen.pop("surface_samples_sha256") == digest(folder/"support/b-tm.npz")
    assert screen == read(folder/"complete-contour-audit.json")
    table = pd.read_csv(study/"four-method-sensitivities.csv").set_index(["case","method"])
    assert table.index.is_unique and len(table) == 16
    for method in ("stacked","spline"):
        reported = regional["methods"][method]
        row = table.loc[("principal",method)]
        np.testing.assert_allclose(row["median_map_km3"],reported["regional_volume"]["median_surface_km3"])
        quantiles = reported["joint_volume"]["conditional_quantiles_km3"]
        np.testing.assert_allclose(row[["lower95","volume_median_km3","upper95"]].to_numpy(dtype=float),
                                   [quantiles[str(p)] for p in (.025,.5,.975)])
    for method, summary in (("directional",az["volume"]["coherent_volume_km3"]),
                            ("spherical",sphere["fits"]["baseline"]["volume"])):
        np.testing.assert_allclose(table.loc[("principal",method),["lower95","volume_median_km3","upper95"]].to_numpy(dtype=float),
                                   [summary[k] for k in LABELS])
    np.testing.assert_allclose(table.at[("principal","directional"),"median_map_km3"],az["volume"]["median_map_km3"])
    for case in ("one_tenth_response","recognition_analogs"):
        for method in ("spherical","directional"):
            result = read(study/f"{case}-{method}.json")
            packet = study/f"{case}-{method}-volumes.npz"
            assert digest(packet) == result["draws_sha256"]
            assert result["input_sha256"] == design["cases"][case]["sha256"]
            with np.load(packet) as draws:
                values = draws["volume_km3"]
                assert (values > 0).all() and np.isfinite(values).all()
                if method == "directional":
                    assert "b-tm" not in result["weights"]["calibration_cases"]
                    actual = interval(values,draws["weights"])
                    quantiles = [actual[k] for k in LABELS]
                else:
                    assert values.ndim == 2 and values.shape[0] == 4
                    quantiles = np.quantile(values,[.025,.5,.975])
            row = table.loc[(case,method)]
            np.testing.assert_allclose(quantiles,row[["lower95","volume_median_km3","upper95"]].to_numpy(dtype=float),rtol=1e-12)
    held = pd.read_csv(study/"heldout-archive-probabilities.csv")
    assert len(held) == 12 and not held.duplicated(["row_id","method"]).any()
    assert set(held.row_id) == set(bounds.values())
    assert held.heldout_probability_of_bound.dropna().between(0.,1.).all()
    assert held.fitted_repeat_probability_of_bound.between(0.,1.).all()
    print("Verified B-Tm profile integration, three conditional bounds, matched geographic fits and sensitivity draws.")


if __name__ == "__main__":
    verify_release(verify_inputs())
