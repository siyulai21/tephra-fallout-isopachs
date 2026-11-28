"""Verify published display revisions and untouched scientific evidence."""
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from .publication import published_datasets, published_methods, released_datasets, result_home
from src.empirical_isopach.core.catalog import datasets
from .azimuth_summary import case_table
from .publication import read_hashes, verify_files


def digest(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def verify_case_map(record, home):
    from src.empirical_isopach.core.distributions import predictive_distribution
    from src.empirical_isopach.models.directional.azimuth.primary import read_field
    from src.empirical_isopach.models.mass_mixture.geometry import log_field, to_sphere
    from src.empirical_isopach.visualization.comparison import FIGURE, REGIONAL, DIRECTIONAL_LABEL
    from src.empirical_isopach.models.mass_mixture.bearings.field import PROTOCOL
    from src.empirical_isopach.models.mass_mixture.bearings.maps import BearingView
    from src.empirical_isopach.visualization.maps import ASSET, frame_bounds
    from .refresh_figures import recorded_observations, regional_models

    page = home.parent/"README.md"
    figure = home/FIGURE
    assert f"(results/{FIGURE})" in page.read_text()
    with Image.open(figure) as image:
        info = json.loads(image.info["Description"])
    expected = {"accepted_observations.csv", "report.json", "directional/azimuth/report.json",
                "directional/azimuth/median-grids.npz", "directional/azimuth/fields.npz", "spherical-mixture/report.json",
                "spherical-mixture/posterior.npz"}
    assert set(info["sources_sha256"]) == expected
    assert all(digest(home/path) == checksum for path, checksum in info["sources_sha256"].items())
    assert digest(record["path"]) == info["input_sha256"]
    assert digest(ASSET) == info["coastline_sha256"]
    data, meta = recorded_observations(record, home)
    np.testing.assert_allclose(info["common_bounds_km"], frame_bounds(data.xy))
    assert info["panels"] == ["GP ensemble", "Cubic spline", DIRECTIONAL_LABEL, "Spherical mixture"]
    samples = {name: np.asarray(rows) for name, rows in info["check_points_x_km_y_km_log10_cm"].items()}
    assert set(samples) == set(info["panels"]) and all(rows.shape == (5, 3) for rows in samples.values())
    regional, region, _ = regional_models(data, read(home/"report.json"))
    for key, name in REGIONAL.items():
        xy = samples[name][:, :2]
        assert np.all(region.contains(xy))
        median = predictive_distribution(regional[key], xy).quantile(.5)
        np.testing.assert_allclose(median/np.log(10.), samples[name][:, 2], rtol=2e-7, atol=2e-7)
    latitude = meta["vent_lonlat"][1]
    folder = home/"directional/azimuth"
    current = read(folder/"report.json")
    assert info["directional_protocol"] == current["protocol"] == "uncertain-azimuth-v1"
    assert info["directional_median_map_km3"] == current["volume"]["median_map_km3"]
    field = read_field(folder,data,current)
    xy = samples[DIRECTIONAL_LABEL][:, :2]
    median = field.distribution(xy).quantile(.5)
    np.testing.assert_allclose(median/np.log(10.), samples[DIRECTIONAL_LABEL][:, 2], atol=1e-8)
    sphere = read(home/"spherical-mixture/report.json")
    xy = samples["Spherical mixture"][:, :2]
    with np.load(home/"spherical-mixture/posterior.npz") as f:
        if sphere["protocol"]==PROTOCOL:
            assert info["spherical_protocol"]==PROTOCOL
            assert info["spherical_volume_km3"]==sphere["fits"]["baseline"]["volume"]["median"]
            view = BearingView(dict(f),data,latitude,sphere["bearing"]["map_draw_check"]["draws_per_chain"])
            median = np.median(np.concatenate(view.field.components(xy)),axis=0)
        else:
            from src.empirical_isopach.visualization.global_maps import PosteriorView
            view = PosteriorView(dict(f),latitude)
            median = np.median(log_field(view.samples,to_sphere(xy,latitude)),axis=0)
    assert view.n == info["spherical_display_draws"]
    np.testing.assert_allclose(median/np.log(10.), samples["Spherical mixture"][:, 2], atol=1e-8)


def verify_comparison_tables(root):
    """Check that readers of the cross-eruption summaries see the registered case fits."""
    summaries = root/"comparisons"
    directional = pd.read_csv(summaries/"directional/case-comparison.csv")
    pd.testing.assert_frame_equal(directional, case_table(published_root=root/"eruptions"),
                                  check_dtype=False, rtol=1e-12, atol=1e-12)

    regional = pd.read_csv(summaries/"regional/comparison.csv").set_index(["slug", "method"])
    spherical = pd.read_csv(summaries/"spherical-mixture/comparison.csv").set_index("slug")
    sources = read(summaries/"spherical-mixture/report.json")["cases"]
    registered = {r["slug"]: r for r in released_datasets("stacked")}
    sphere_editions = {r["slug"]: r for r in released_datasets("spherical-mixture")}
    spherical_slugs = set(sphere_editions)
    assert spherical.index.is_unique and set(spherical.index) == spherical_slugs
    assert len(sources) == len(spherical_slugs) and {s["slug"] for s in sources} == spherical_slugs
    expected_rows = {(slug,k) for slug,r in registered.items()
                     for k in ("stacked","spline") if k in published_methods(r)}
    assert regional.index.is_unique and set(regional.index) == expected_rows
    for (slug, method), row in regional.iterrows():
        rec = registered[slug]
        reg = read(root/result_home(rec)/"report.json")
        assert row.input_sha256 == digest(rec["path"]) == reg["input"]["input_sha256"]
        metric = reg["validation"]["methods"][method]["metrics"]
        assert (row.n_exact, row.n_lower, row.n_upper) == tuple(
            reg["input"][key] for key in ("n_measurements", "n_lower_limits", "n_upper_limits"))
        assert row.n_intervals == reg["input"].get("n_intervals",0)
        vol = reg["methods"][method]["regional_volume"]
        np.testing.assert_allclose(
            [row.median_surface_km3, row.equal_region_log_score, row.coverage95_observations],
            [vol["median_surface_km3"], metric["equal_region_log_score"], metric["coverage95_observations"]],
            rtol=1e-12)
        assert row.volume_converged == vol["converged"]
    for source in sources:
        slug = source["slug"]
        rec = sphere_editions[slug]
        home = root/result_home(rec)
        assert source["input_sha256"] == digest(rec["path"])
        regional_sha = digest(home/"report.json") if "stacked" in published_methods(rec) else None
        assert source["regional_report_sha256"] == regional_sha
        assert source["global_report_sha256"] == digest(home/"spherical-mixture/report.json")
        sphere = read(home/"spherical-mixture/report.json")
        row = spherical.loc[slug]
        receipt = read(home/"spherical-mixture/deposit.json")
        current = home/"spherical-mixture/extent"
        if current.is_dir():
            val = read(current/"report.json")["validation"]["geographic"]
            priors = read(current/"prior-sensitivity.json")
            envelope = [min(v["conditional_interval"][0] for v in priors.values()),max(v["conditional_interval"][-1] for v in priors.values())]
            score,coverage = val["negative_log_score"],val["coverage95_exact"]
        else:
            envelope = sphere["sensitivity_envelope_km3"]
            val = sphere["validation"]["metrics"]
            score,coverage=val["equal_region_log_score"],val["coverage95_observations"]
        np.testing.assert_allclose(
            [row.median_map_volume_km3, row.thickness_q25_volume_km3, row.thickness_q75_volume_km3, row.sensitivity_lower, row.sensitivity_upper,
             row.global_equal_region_negative_log_score, row.global_observation_coverage95],
            [receipt["median_map_volume_km3"],*np.asarray(receipt["thickness_band"]["volumes_km3"])[[0,2]],*envelope,score,coverage],
            rtol=1e-12)


def main():
    root = Path(__file__).resolve().parents[1]
    verify_comparison_tables(root)
    with Image.open(root/"comparisons/method-scores.png") as figure:
        score_info = json.loads(figure.info["Description"])
        panels = score_info["eruption_panels"]
        assert set(score_info["methods_by_case"]) == set(panels)
        from .principal_volumes import catalogue_rows, LABELS
        current = catalogue_rows()
        assert score_info["methods_by_case"] == {name:[LABELS[r["method"]] for r in current if r["eruption"]==name] for name in panels}
    assert sorted(panels) == sorted(record["name"] for record in datasets())
    hashes = read_hashes()
    verify_files(hashes)
    for record in published_datasets():
        home = root/result_home(record)
        report = read(home/"report.json")
        replay = read(home/"figure-replay.json")
        for path, expected in replay["source_reports"].items():
            assert digest(home/path) == expected
        for check in replay["regional_parity"].values():
            assert check["recorded_parameters_replayed"] and not check["selection_repeated"]
            assert check["expected_volume_relative_difference"] < 2e-7
            assert check["median_surface_volume_relative_difference"] < 2e-7
        regional = read(home/"map-display.json")
        assert regional["entire_observation_hull_in_frame"] and regional["all_accepted_locations_in_frame"]
        assert regional["unchanged_fit_bounds_km"] == report["fit_bounds_km"]
        assert not regional["volume_calculation_changed"]
        ids=regional["accepted_row_ids"]
        assert len(ids)==len(set(ids)) and set(ids)==set(report["validation"]["row_ids"])
        assert set(regional["inner_contours"])==set(report["methods"])
        from src.empirical_isopach.visualization.contour_limits import inner_contour
        for item in regional["inner_contours"].values():
            check=inner_contour({"mapped_log10_cm":item["mapped_site_log10_cm"],"row_ids":ids})
            assert all(check[key]==item[key] for key in check)
            assert item["inside"]>=1 and item["next_level_inside"]==0
            assert all(v<=item["level_cm"] for v in item["drawn_levels_cm"])
        for family in ("directional/single-structure", "spherical-mixture"):
            folder = home/family
            for path, expected in read(folder/"artifacts.json").items():
                assert digest(folder/path) == expected, str(folder/path)
            display = read(folder/"map-display.json")
            assert display["input_sha256"] == report["input"]["input_sha256"]
            assert display["posterior_sha256"] == digest(folder/"posterior.npz")
            assert not display["volume_calculation_changed"]
            assert display["extent_audit"]["edge_refinements"][-1]["bounds_km"] == display["global_bounds_km"]
            assert np.isfinite(display["global_bounds_km"]).all()
    from .verify_deposit_maps import main as verify_deposits
    verify_deposits()
    print(f"Verified {len(hashes)} published files and the matched comparison tables.")


if __name__ == "__main__":
    main()
