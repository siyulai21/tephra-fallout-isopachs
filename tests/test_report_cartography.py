"""Display frames must neither hide data nor manufacture contour closure."""
import gzip
from hashlib import sha256
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.empirical_isopach.visualization.maps import (
    ASSET, boundary_points, coastlines, frame_bounds, global_frame, grid_axes,
)
from src.empirical_isopach.visualization.summary import metrics, saved_case_overview


def test_frame_contains_extreme_observations_and_source():
    points = np.array([[600, -20], [40, 900], [1800, 11], [41, -300]])
    b = frame_bounds(points)
    assert np.all(points > b[:2]) and np.all(points < b[2:])
    assert np.all(b[:2] < 0) and np.all(b[2:] > 0)
    x, y = grid_axes(b)
    assert 0 in x and 0 in y
    assert x[0] == b[0] and y[-1] == b[3]


def test_global_frame_finds_a_detached_fan_far_from_the_observations():
    observations = np.array([[0, 0], [50, 50], [-50, 20]])
    def evaluate(xy):
        near = 1 - np.sum(xy**2, axis=1)/100**2
        far = 1 - np.sum((xy-[5000, -2000])**2, axis=1)/600**2
        return np.maximum(near, far)
    b, audit = global_frame(observations, 30., evaluate)
    assert b[2] > 5000+600 and b[1] < -2000-600
    assert audit["lowest_contour_within_chart"]
    assert evaluate(boundary_points(b)).max() < -1
    assert audit["edge_refinements"][-1]["bounds_km"] == b.tolist()


def test_a_positive_whole_earth_field_is_not_declared_closed():
    _, audit = global_frame(np.array([[0, 0], [100, 100]]), 40., lambda xy: np.ones(len(xy)))
    assert not audit["lowest_contour_within_chart"]
    assert audit["physical_rim_maximum_log10_cm"] == 1.


def test_pinned_coastline_is_lossless_and_cannot_change_the_viewport():
    record = json.loads((ASSET.parent/"coastline-source.json").read_text())
    assert sha256(ASSET.read_bytes()).hexdigest() == record["gzip_sha256"]
    assert sha256(gzip.decompress(ASSET.read_bytes())).hexdigest() == record["original_sha256"]
    fig, ax = plt.subplots()
    ax.set(xlim=(-300, 300), ylim=(-300, 300))
    before = ax.axis()
    artist = coastlines(ax, (25.4, 36.4))
    assert ax.axis() == before
    assert len(artist.get_segments()) > 0
    plt.close(fig)


def test_overview_reads_existing_reports_and_preserves_warning(tmp_path):
    root = Path(__file__).resolve().parents[1]/"eruptions/tambora/results"
    paths = ["report.json", "directional/single-structure/report.json", "spherical-mixture/report.json"]
    reports = []
    for p in paths:
        dest = tmp_path/p
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes((root/p).read_bytes())
        reports.append(json.loads(dest.read_text()))
    before = {p: (tmp_path/p).read_bytes() for p in paths}
    rows = metrics(*reports)
    assert rows[2][1] == reports[1]["validation"]["geographic"]["equal_region_log_score"]
    assert not rows[2][3]  # Published directional geographic numerical warning.
    saved_case_overview(tmp_path)
    assert (tmp_path/"overview.png").stat().st_size > 10000
    assert before == {p: (tmp_path/p).read_bytes() for p in paths}


def test_publication_check_rejects_changed_or_missing_evidence(tmp_path):
    import pytest
    from benchmarks.publication import read_hashes, verify_files

    result = tmp_path/"report.json"
    result.write_bytes(b"published")
    receipt = tmp_path/"evidence.sha256"
    receipt.write_text(f"{sha256(result.read_bytes()).hexdigest()}  report.json\n")
    expected = read_hashes(receipt)
    verify_files(expected, tmp_path)
    result.write_bytes(b"changed")
    with pytest.raises(ValueError, match="report.json"):
        verify_files(expected, tmp_path)
    result.unlink()
    with pytest.raises(ValueError, match="report.json"):
        verify_files(expected, tmp_path)


def test_comparison_reads_both_regional_medians_by_coordinate(tmp_path):
    import pandas as pd
    import pytest
    from src.empirical_isopach.visualization.comparison import saved_regional_grid

    table = pd.DataFrame({"x_km": [3, 0, 3, 0], "y_km": [7, 0, 0, 7],
                          "stacked_median_cm": [1000, 1, 10, 100], "spline_median_cm": [1, 1000, 100, 10]})
    path = tmp_path/"predictions.csv"
    table.to_csv(path, index=False)
    x, y, fields = saved_regional_grid(path)
    np.testing.assert_array_equal(x, [0, 3])
    np.testing.assert_array_equal(y, [0, 7])
    np.testing.assert_array_equal(fields["stacked"], [[0, 1], [2, 3]])
    np.testing.assert_array_equal(fields["spline"], [[3, 2], [1, 0]])
    table.iloc[:-1].to_csv(path, index=False)
    with pytest.raises(ValueError, match="incomplete"):
        saved_regional_grid(path)


def test_publication_receipt_rejects_conflicting_entries(tmp_path):
    import pytest
    from benchmarks.publication import read_hashes

    receipt = tmp_path/"evidence.sha256"
    receipt.write_text("first  report.json\nsecond  report.json\n")
    with pytest.raises(ValueError, match="Duplicate publication path"):
        read_hashes(receipt)


def test_old_fit_packets_use_median_maps_and_do_not_substitute_an_unmapped_volume():
    from benchmarks.publication import BTM_THICKNESS_ONLY
    from src.empirical_isopach.visualization.comparison import comparison_section

    home = Path(__file__).resolve().parents[1] / BTM_THICKNESS_ONLY / "results"
    section = comparison_section(home, prefix="results/")
    assert "| Spherical mixture | Map integral pending |" in section
    assert "| Directional | Whole-Earth median map **" in section
    assert section.count("Hull median map") == 2


def test_volume_interval_and_geographic_warnings_are_independent(monkeypatch, tmp_path):
    from src.empirical_isopach.visualization import comparison

    home = Path(__file__).resolve().parents[1] / "eruptions/ilopango-tbj/results"
    import json
    import shutil
    for name in ("directional","spherical-mixture","nested-isopachs"):
        (tmp_path/name).mkdir()
        shutil.copy2(home/name/"deposit.json",tmp_path/name/"deposit.json")
    shutil.copy2(home/"nested-isopachs/report.json",tmp_path/"nested-isopachs/report.json")
    for name in ("report.json","thickness-bands.json","directional/extent/report.json","nested-isopachs/footprints/report.json"):
        (tmp_path/name).parent.mkdir(exist_ok=True)
        shutil.copy2(home/name,tmp_path/name)
    reports = comparison.comparison_reports(home)
    spherical = reports[2]
    monkeypatch.setattr(comparison, "comparison_reports", lambda _: reports)
    for volume_ok, prediction_ok in ((False, True), (True, False)):
        receipt = json.loads((home/"spherical-mixture/deposit.json").read_text())
        receipt["posterior_checks_passed"] = volume_ok
        (tmp_path/"spherical-mixture/deposit.json").write_text(json.dumps(receipt))
        spherical["fits"]["baseline"]["diagnostics"]["passed"] = volume_ok
        spherical["validation"]["all_fits_passed_diagnostics"] = prediction_ok
        path = home/"spherical-mixture/extent/report.json"
        if path.exists():
            current = json.loads(path.read_text())
            current["draw_checks_passed"] = volume_ok
            current["validation"]["geographic"]["all_folds_passed"] = prediction_ok
            target = tmp_path/"spherical-mixture/extent"
            target.mkdir(exist_ok=True)
            (target/"report.json").write_text(json.dumps(current))
        section = comparison.comparison_section(tmp_path)
        row = next(line for line in section.splitlines() if line.startswith("| [Spherical mixture]"))
        _, _, _, interval, score, _, _ = row.split("|")
        assert ("†" in interval) == (not volume_ok)
        assert ("†" in score) == (not prediction_ok)


def test_full_run_opens_with_four_fits_and_retains_regional_detail(tmp_path, monkeypatch):
    from src.empirical_isopach.visualization import comparison

    home = Path(__file__).resolve().parents[1]/"eruptions"/"tambora"/"results"
    for name in ("report.json", "directional/azimuth/report.json", "spherical-mixture/report.json"):
        dest = tmp_path/name
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes((home/name).read_bytes())
    page = tmp_path/"README.md"
    page.write_text("# Regional comparison\n\nOriginal regional details\n")
    monkeypatch.setattr(comparison, "draw_comparison", lambda home, dest: None)
    comparison.add_comparison_to_run(tmp_path)
    first = page.read_text()
    assert first.startswith("# Tambora fallout reconstruction\n\n## Isopach fits")
    assert first.index(comparison.FIGURE) < first.index("Original regional details")
    assert "Whole-Earth median map **95.7**" in first
    assert "| 2.066 † | 84.6% |" in first
    assert "(directional/azimuth/world.png)" in first and "(spherical-mixture/world.png)" in first
    comparison.add_comparison_to_run(tmp_path)
    assert page.read_text() == first
