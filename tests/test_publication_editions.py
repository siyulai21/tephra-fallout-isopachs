"""Keep the current eruption catalogue distinct from the fixed comparison studies."""
import csv
import json
import math
from hashlib import sha256

from benchmarks.principal_volumes import catalogue_rows
from benchmarks.publication import matched_edition, published_methods, released_edition, result_home
from src.empirical_isopach.core.audit import audited_data
from src.empirical_isopach.core.catalog import data_directory, datasets, resolve_dataset


def test_current_audits_and_historical_controls_have_distinct_inputs():
    current = {r["slug"]: r for r in datasets()}
    rows = catalogue_rows()
    for slug, count, families, previous_count in (
        ("ilopango-tbj", 168, {"stacked", "spline", "directional", "spherical-mixture", "nested-isopachs"}, 56),
        ("pinatubo-1991-layer-c", 235, {"stacked", "spline", "directional", "spherical-mixture", "nested-isopachs"}, 208),
    ):
        case = current[slug]
        case_rows = [r for r in rows if r["slug"] == slug]
        assert {r["method"] for r in case_rows} == families == set(published_methods(case))
        assert {r["accepted_observations"] for r in case_rows} == {count}
        assert {r["input_sha256"] for r in case_rows} == {sha256(case["path"].read_bytes()).hexdigest()}
        study = matched_edition(case)
        assert len(published_methods(study)) == 5
        assert result_home(study) != result_home(case)
        with study["path"].open() as stream:
            assert len(list(csv.DictReader(stream))) == previous_count


def test_aso_u1429_principal_measurement_and_published_edition():
    record = next(r for r in datasets() if r["slug"] == "aso-4")
    audit = json.loads((data_directory() / record["audit"]).read_text())
    input_path = resolve_dataset("aso-4")
    assert input_path == record["path"]
    observations, meta = audited_data(input_path)
    added = audit["new_observation"]
    row = int((observations.row_ids == added["csv_row"]).nonzero()[0].item())
    assert math.isclose(math.exp(observations.log_value[row]), 27)
    assert not observations.censored[row] and not observations.reported_minimum[row]
    assert meta["input_sha256"] == audit["cases"]["principal"]["fitting_sha256"]
    assert meta["n_reported_measurements"] == 84

    published = released_edition(record)
    if not published_methods(record):
        assert sha256(published["path"].read_bytes()).hexdigest() == audit["preceding_release"]["fitting_sha256"]
        report = json.loads((result_home(published) / "report.json").read_text())
        assert report["input"]["input_sha256"] == audit["preceding_release"]["fitting_sha256"]
