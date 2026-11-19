"""Verify published display revisions and untouched scientific evidence."""
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
from PIL import Image

from src.empirical_isopach.core.catalog import datasets
from .publication import read_hashes, verify_files


def digest(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def verify_case_map(record, home):
    from src.empirical_isopach.core.distributions import predictive_distribution
    from src.empirical_isopach.models.directional.ensemble.fields import load_packet
    from src.empirical_isopach.models.mass_mixture.geometry import log_field, to_sphere
    from src.empirical_isopach.visualization.comparison import FIGURE, REGIONAL, comparison_section
    from src.empirical_isopach.visualization.global_maps import PosteriorView
    from src.empirical_isopach.visualization.maps import ASSET, frame_bounds
    from .refresh_figures import recorded_observations, regional_models

    page = home.parent/"README.md"
    figure = home/FIGURE
    assert comparison_section(home, prefix="results/") in page.read_text()
    with Image.open(figure) as image:
        info = json.loads(image.info["Description"])
    expected = {"accepted_observations.csv", "report.json", "directional/ensemble/report.json",
                "directional/ensemble/median-grids.npz", "directional/ensemble/map-fields.npz", "spherical-mixture/report.json",
                "spherical-mixture/posterior.npz"}
    assert set(info["sources_sha256"]) == expected
    assert all(digest(home/path) == checksum for path, checksum in info["sources_sha256"].items())
    assert digest(record["path"]) == info["input_sha256"]
    assert digest(ASSET) == info["coastline_sha256"]
    data, meta = recorded_observations(record, home)
    np.testing.assert_allclose(info["common_bounds_km"], frame_bounds(data.xy))
    assert info["panels"] == ["GP ensemble", "Cubic spline", "Directional ensemble", "Spherical mixture"]
    samples = {name: np.asarray(rows) for name, rows in info["check_points_x_km_y_km_log10_cm"].items()}
    assert set(samples) == set(info["panels"]) and all(rows.shape == (5, 3) for rows in samples.values())
    regional, region, _ = regional_models(data, read(home/"report.json"))
    for key, name in REGIONAL.items():
        xy = samples[name][:, :2]
        assert np.all(region.contains(xy))
        median = predictive_distribution(regional[key], xy).quantile(.5)
        np.testing.assert_allclose(median/np.log(10.), samples[name][:, 2], rtol=2e-7, atol=2e-7)
    latitude = meta["vent_lonlat"][1]
    folder = home/"directional/ensemble"
    ensemble = load_packet(folder, data, latitude, read(folder/"report.json"))
    xy = samples["Directional ensemble"][:, :2]
    median = ensemble.distribution(xy).quantile(.5)
    np.testing.assert_allclose(median/np.log(10.), samples["Directional ensemble"][:, 2], atol=1e-8)
    with np.load(home/"spherical-mixture/posterior.npz") as f:
        view = PosteriorView(dict(f), latitude)
    assert view.n == info["spherical_display_draws"]
    xy = samples["Spherical mixture"][:, :2]
    median = np.median(log_field(view.samples, to_sphere(xy, latitude)), axis=0)
    np.testing.assert_allclose(median/np.log(10.), samples["Spherical mixture"][:, 2], atol=1e-8)


def main():
    root = Path(__file__).resolve().parents[1]
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
        for family in ("directional", "spherical-mixture"):
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
          "against the saved fits.")


if __name__ == "__main__":
    main()
