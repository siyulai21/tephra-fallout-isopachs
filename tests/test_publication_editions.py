"""Keep the current eruption catalogue distinct from the fixed comparison studies."""
import csv
from hashlib import sha256

from benchmarks.principal_volumes import catalogue_rows
from benchmarks.publication import matched_edition, published_methods, result_home
from src.empirical_isopach.core.catalog import datasets


def test_current_audits_and_historical_controls_have_distinct_inputs():
    current = {r["slug"]: r for r in datasets()}
    rows = catalogue_rows()
    for slug, count, families, previous_count in (
        ("ilopango-tbj", 168, {"directional", "spherical-mixture", "nested-isopachs"}, 56),
        ("pinatubo-1991-layer-c", 235, {"stacked", "spline", "nested-isopachs"}, 208),
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
