"""Keep the current eruption catalogue distinct from the fixed comparison studies."""
import csv
import json
import math
from decimal import Decimal
from hashlib import sha256

from pyproj import Geod

from benchmarks.principal_volumes import catalogue_rows
from benchmarks.publication import matched_edition, published_methods, released_edition, result_home
from src.empirical_isopach.core.audit import audited_data
from src.empirical_isopach.core.catalog import data_directory, dataset_record, datasets


def test_current_audits_and_historical_controls_have_distinct_inputs():
    current = {r["slug"]: r for r in datasets()}
    rows = catalogue_rows()
    for slug, count, families, previous_count in (
        ("ilopango-tbj", 168, {"stacked", "spline", "directional", "spherical-mixture", "nested-isopachs"}, 56),
        ("pinatubo-1991-layer-c", 235, {"stacked", "spline", "directional", "spherical-mixture", "nested-isopachs"}, 208),
    ):
        case = current[slug]
        published = released_edition(case)
        case_rows = [r for r in rows if r["slug"] == slug]
        assert {r["method"] for r in case_rows} == families == set(published_methods(published))
        assert {r["accepted_observations"] for r in case_rows} == {count}
        assert {r["input_sha256"] for r in case_rows} == {sha256(published["path"].read_bytes()).hexdigest()}
        study = matched_edition(case)
        assert len(published_methods(study)) == 5
        assert result_home(study) != result_home(published)
        with study["path"].open() as stream:
            assert len(list(csv.DictReader(stream))) == previous_count


def test_aso_u1429_principal_measurement_and_published_edition():
    input_path = data_directory() / "eruptions/Aso-4_reviewed_u1429_minimum_fit.csv"
    record = dataset_record(input_path)
    audit = json.loads((data_directory() / record["audit"]).read_text())
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


def test_ulleung_depth_spans_and_core_identifiers():
    root = data_directory()
    with (root / "audits/Aso-4/ulleung/cores.csv").open() as file:
        ledger = {row["evidence_id"]: row for row in csv.DictReader(file)}
    expected = {"95PC-6": "0.8", "95PC-9": "0.6", "95PC-10": "3.0", "96EBP-7": "3.0"}
    assert ledger.keys() == expected.keys()
    assert ledger["96EBP-7"]["longitude_printed"].endswith(" N")
    for suffix in ("", "_minimum_fit"):
        with (root / f"eruptions/Aso-4_reviewed_u1429{suffix}.csv").open() as file:
            previous = list(csv.DictReader(file))
        with (root / f"eruptions/Aso-4_ulleung{suffix}.csv").open() as file:
            current = list(csv.DictReader(file))
        assert len(current) == len(previous) + 4
        for old, new in zip(previous, current):
            assert old == {key: new[key] for key in old}
        for row in current[len(previous):]:
            core = row["EvidenceID"]
            item = ledger[core]
            assert row == current[int(item["input_csv_row"]) - 2]
            span = Decimal(item["aso4_deeper_depth_cm"]) - Decimal(item["aso4_shallower_depth_cm"])
            assert span == Decimal(expected[core]) == Decimal(row["Thickness"])
            assert float(row["Lat"]) == float(item["latitude_deg"])
            assert float(row["Lon"]) == float(item["longitude_deg"]) > 130
            assert row["Kind"] == "measurement" and row["Thickness_basis"] and row["Deposit_evidence"]
    obs, meta = audited_data(root / "eruptions/Aso-4_ulleung_minimum_fit.csv")
    assert (len(obs.xy), meta["n_reported_measurements"], obs.reported_minimum.sum(), obs.censored.sum()) == (92, 88, 3, 1)


def test_petapilla_field_position_moves_one_bed_in_both_tables():
    root = data_directory()
    with (root / "audits/Ilopango-TBJ/petapilla/provenance.json").open() as file:
        site = json.load(file)["corrected_site"]
    for suffix in ("", "_minimum_fit"):
        with (root / f"eruptions/Ilopango_TBJ_total_airfall{suffix}.csv").open() as file:
            previous = list(csv.DictReader(file))
        with (root / f"eruptions/Ilopango_TBJ_petapilla{suffix}.csv").open() as file:
            revised = list(csv.DictReader(file))
        changes = [(i, a, b) for i, (a, b) in enumerate(zip(previous, revised, strict=True), 2) if a != b]
        (number, old, new), = changes
        assert number == site["principal_csv_row"] == 72 and new["EvidenceID"] == "SM1-P118"
        assert {key for key in new if new[key] != old[key]} == {"Lat", "Lon"}
        assert abs(float(new["Lat"]) - (14 + 52.154 / 60)) < 1e-8
        assert abs(float(new["Lon"]) + (89 + 6.979 / 60)) < 1e-8
        az, _, metres = Geod(ellps="WGS84").inv(float(old["Lon"]), float(old["Lat"]), float(new["Lon"]), float(new["Lat"]))
        assert round(metres / 1000, 3) == site["wgs84_shift_km"] == 5.191
        assert round(az % 360, 3) == site["wgs84_initial_bearing_deg"]
        assert Decimal(new["Thickness"]) == Decimal("321.10") - Decimal("320.75")
