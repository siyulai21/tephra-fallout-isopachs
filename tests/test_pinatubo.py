"""Protect published field readings, station joins and visual verification for Pinatubo."""
import csv
import gzip
import json
from collections import Counter
from hashlib import sha256

from benchmarks.pinatubo import AUDIT, INPUT, SOURCE, products


def rows(path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def test_input_replays_from_original_readings_and_units():
    for path, content in products().items():
        assert content == path.read_bytes()
    accepted = rows(INPUT)
    assert Counter(r["Kind"] for r in accepted) == {"measurement": 140, "lower_limit": 26, "upper_limit": 28}
    records = {r["EvidenceID"]: r for r in accepted}
    assert len(records) == len({(r["Lon"], r["Lat"]) for r in accepted}) == 194
    core = records["S99-SO-114-23"]
    assert (core["Lon"], core["Lat"], core["Thickness"]) == ("117.74", "14.07", "1.0")
    assert "S99-SO-114-6" not in records
    assert records["K-03"]["Limit_cm"] == "2.0"
    assert records["H01-14"]["Kind"] == "upper_limit"
    assert float(records["H01-14"]["Limit_cm"]) == 2
    assert records["W04-SO-140-03"]["Thickness"] == "0.7"
    assert {"H01-03", "WW-SCS-C", "P7-056", "P7-101", "P7-148"}.isdisjoint(records)
    ledger = rows(AUDIT / "full-evidence.csv")
    assert len(ledger) == len({r["evidence_id"] for r in ledger}) == 464
    fitted = {r["evidence_id"]: int(r["accepted_csv_row"]) for r in ledger if r["accepted_csv_row"]}
    assert fitted == {r["EvidenceID"]: i for i, r in enumerate(accepted, 2)}
    sheets = rows(AUDIT / "reading-index.csv")
    assert {r["evidence_id"] for r in sheets} >= {key for key in records if key.startswith(("P2-", "K-", "P7-"))}
    assert all((AUDIT / r["sheet"]).is_file() for r in sheets)


def test_preserved_sources_and_marine_station_matches():
    manifest = json.loads((SOURCE / "sources.json").read_text())
    for entry in manifest["sources"]:
        if entry.get("stored") and entry["source_sha256"]:
            path = SOURCE / entry["stored"]
            raw = gzip.decompress(path.read_bytes()) if path.suffix == ".gz" else path.read_bytes()
            assert len(raw) == entry["source_bytes"]
            assert sha256(raw).hexdigest() == entry["source_sha256"]
    for name, identity in manifest["transcripts"].items():
        assert sha256((SOURCE / name).read_bytes()).hexdigest() == identity
    events = {r["event"]: r for r in rows(SOURCE / "pangaea-coring-events.csv")}
    beds = rows(SOURCE / "wiesner-2004-table-1.csv")
    assert len(beds) == 11
    for bed in beds:
        event = events[bed["coordinate_event"]]
        for field in ("longitude", "latitude"):
            assert float(bed[field]) == float(event[field])
        if bed["unit_i_mm"]:
            assert int(bed["unit_i_mm"]) + int(bed["unit_ii_mm"]) == int(bed["bulk_mm"])

    probes = json.loads((AUDIT / "archive-probes.json").read_text())
    assert len(probes["sites"]) == len({site["id"] for site in probes["sites"]}) == 102
    assert not probes["scenarios"]
    assert all(site["reported_fallout_limit_cm"] is None for site in probes["sites"])
    inventory = {row["evidence_id"] for row in rows(AUDIT / "full-evidence.csv")}
    assert {site["id"] for site in probes["sites"]} <= inventory

def test_other_fall_layers_and_cruise_chart_are_separately_traceable():
    from benchmarks.pinatubo_evidence import cruise_registration
    from benchmarks.pinatubo_phases import inventory

    supplemental = inventory()
    index = rows(AUDIT / "other-phases/readings.csv")
    assert len(index) == len(supplemental) == 366
    assert len({r["evidence_id"] for r in index}) == len(index)
    assert {r["evidence_id"] for r in supplemental}.isdisjoint(r["EvidenceID"] for r in rows(INPUT))
    assert all((AUDIT / "other-phases" / r["sheet"]).is_file() for r in index)
    assert all(float(r["single_positive_cm"]) > 0 for r in index if r["single_positive_cm"])
    assert all(r["review"] == "positive point reading" for r in index if r["single_positive_cm"])
    assert len(cruise_registration()) == 15
    assert {r["evidence_id"]: r["reported_or_candidate"] for r in index}["P6-068"] == "23"
