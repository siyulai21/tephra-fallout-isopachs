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
    paths = ["report.json", "directional/report.json", "spherical-mixture/report.json"]
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


def test_visual_history_never_exempts_a_scientific_record(tmp_path, monkeypatch):
    from benchmarks import visual_history
    path = tmp_path/"report.json"
    path.write_text('{"changed_volume": 123}')
    ledger = tmp_path/"ledger.json"
    ledger.write_text(json.dumps({"replaced_presentation_files": {str(path): {
        "before_sha256": "old", "after_sha256": sha256(path.read_bytes()).hexdigest()}}}))
    monkeypatch.setattr(visual_history, "LEDGER", ledger)
    assert not visual_history.verified_revision(path, "old")
    image = tmp_path/"map.png"
    image.write_bytes(b"display")
    ledger.write_text(json.dumps({"replaced_presentation_files": {str(image): {
        "before_sha256": "old", "after_sha256": sha256(image.read_bytes()).hexdigest()}}}))
    assert visual_history.verified_revision(image, "old")
    assert not visual_history.verified_revision(image, "different-history")
    image.write_bytes(b"unreviewed-change")
    assert not visual_history.verified_revision(image, "old")


def test_navigation_retirement_checks_the_archived_page_and_live_replacement(tmp_path, monkeypatch):
    from zipfile import ZipFile
    from benchmarks import visual_history

    old = tmp_path/"global"/"README.md"
    current = tmp_path/"README.md"
    old.parent.mkdir()
    current.write_text("Full case report")
    archived = b"Superseded index"
    expected = sha256(archived).hexdigest()
    entry = {"prior_sha256": expected, "current_path": "README.md",
             "current_sha256": sha256(current.read_bytes()).hexdigest()}
    ledger = tmp_path/"release.json"
    ledger.write_text(json.dumps({"files": {"global/README.md": entry}, "unchanged_moves": {}}))
    monkeypatch.setattr(visual_history, "ROOT", tmp_path)
    monkeypatch.setattr(visual_history, "NAVIGATION", ledger)
    with ZipFile(ledger.with_suffix(".zip"), "w") as archive:
        archive.writestr("global/README.md", archived)
    assert visual_history.verified_navigation(old, expected)
    assert not visual_history.verified_navigation(old, "different prior")
    with ZipFile(ledger.with_suffix(".zip"), "w") as archive:
        archive.writestr("global/README.md", b"Wrong archive")
    assert not visual_history.verified_navigation(old, expected)
    with ZipFile(ledger.with_suffix(".zip"), "w") as archive:
        archive.writestr("global/README.md", archived)
    current.write_text("Unreviewed replacement")
    assert not visual_history.verified_navigation(old, expected)


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


def test_prior_case_map_release_requires_exact_old_and_current_page(tmp_path, monkeypatch):
    from benchmarks import visual_history

    case = tmp_path/"eruptions"/"example"/"README.md"
    case.parent.mkdir(parents=True)
    case.write_text("Four maps and a result table")
    old = "Earlier case summary"
    old_hash = sha256(old.encode()).hexdigest()
    previous = {"previous_markdown": old, "page_sha256": sha256(case.read_bytes()).hexdigest()}
    ledger = tmp_path/"case-maps.json"
    ledger.write_text(json.dumps({"cases": {"example": previous}}))
    monkeypatch.setattr(visual_history, "ROOT", tmp_path)
    monkeypatch.setattr(visual_history, "CASE_MAPS", ledger)
    assert visual_history.matches_case_publication(case, old_hash)
    assert not visual_history.matches_case_publication(case, "wrong prior")
    case.write_text("Unreviewed case summary")
    assert not visual_history.matches_case_publication(case, old_hash)


def test_full_run_opens_with_four_fits_and_retains_regional_detail(tmp_path, monkeypatch):
    from src.empirical_isopach.visualization import comparison

    home = Path(__file__).resolve().parents[1]/"eruptions"/"tambora"/"results"
    for name in ("report.json", "directional/ensemble/report.json", "spherical-mixture/report.json"):
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
    assert "| 1.964 † | 84.6% |" in first
    assert "(directional/ensemble/world.png)" in first and "(spherical-mixture/world.png)" in first
    comparison.add_comparison_to_run(tmp_path)
    assert page.read_text() == first
