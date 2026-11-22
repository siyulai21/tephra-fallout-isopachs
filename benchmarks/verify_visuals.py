"""Verify published display revisions and untouched scientific evidence."""
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

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
    from src.empirical_isopach.visualization.comparison import FIGURE, REGIONAL, DIRECTIONAL_LABEL, comparison_section
    from src.empirical_isopach.models.mass_mixture.bearings.field import PROTOCOL
    from src.empirical_isopach.models.mass_mixture.bearings.maps import BearingView
    from src.empirical_isopach.visualization.maps import ASSET, frame_bounds
    from .refresh_figures import recorded_observations, regional_models

    page = home.parent/"README.md"
    figure = home/FIGURE
    assert comparison_section(home, prefix="results/") in page.read_text()
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
    registered = {r["slug"]: r for r in datasets()}
    assert spherical.index.is_unique and set(spherical.index) == set(registered)
    assert len(sources) == len(registered) and {s["slug"] for s in sources} == set(registered)
    expected_rows = set()
    for source in sources:
        slug = source["slug"]
        home = root/"eruptions"/slug/"results"
        assert source["input_sha256"] == digest(registered[slug]["path"])
        assert source["regional_report_sha256"] == digest(home/"report.json")
        assert source["global_report_sha256"] == digest(home/"spherical-mixture/report.json")
        reg = read(home/"report.json")
        for method, fit in reg["methods"].items():
            expected_rows.add((slug, method))
            row = regional.loc[(slug, method)]
            metric = reg["validation"]["methods"][method]["metrics"]
            assert (row.n_exact, row.n_lower, row.n_upper) == tuple(
                reg["input"][key] for key in ("n_measurements", "n_lower_limits", "n_upper_limits"))
            vol = fit["regional_volume"]
            np.testing.assert_allclose(
                [row.median_surface_km3, row.expected_regional_km3, row.equal_region_log_score, row.coverage95_observations],
                [vol["median_surface_km3"], vol["expected_km3"], metric["equal_region_log_score"], metric["coverage95_observations"]],
                rtol=1e-12)
            assert row.volume_converged == vol["converged"]
        sphere = read(home/"spherical-mixture/report.json")
        row = spherical.loc[slug]
        volume = sphere["fits"]["baseline"]["volume"]
        np.testing.assert_allclose(
            [row["median"], row.lower95, row.upper95, row.sensitivity_lower, row.sensitivity_upper,
             row.global_equal_region_negative_log_score, row.global_observation_coverage95],
            [volume["median"], volume["lower95"], volume["upper95"], *sphere["sensitivity_envelope_km3"],
             sphere["validation"]["metrics"]["equal_region_log_score"], sphere["validation"]["metrics"]["coverage95_observations"]],
            rtol=1e-12)
    assert regional.index.is_unique and set(regional.index) == expected_rows


def main():
    root = Path(__file__).resolve().parents[1]
    verify_comparison_tables(root)
    with Image.open(root/"comparisons/method-scores.png") as figure:
        score_info = json.loads(figure.info["Description"])
        panels = score_info["eruption_panels"]
        assert set(score_info["methods_by_case"]) == set(panels)
        assert all(names == ["GP ensemble", "Cubic spline", "Directional", "Spherical mixture", "Nested corridor"]
                   for names in score_info["methods_by_case"].values())
    assert sorted(panels) == sorted(record["name"] for record in datasets())
    hashes = read_hashes()
    verify_files(hashes)
    for record in datasets():
        home = root/"eruptions"/record["slug"]/"results"
        verify_case_map(record, home)
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
    print(f"Verified {len(hashes)} published files and all {len(datasets())} current four-method maps "
          "and three comparison tables against the saved fits.")


if __name__ == "__main__":
    main()
