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
    assert Counter(r["Kind"] for r in accepted) == {"measurement": 150, "lower_limit": 28, "upper_limit": 30}
    records = {r["EvidenceID"]: r for r in accepted}
    assert len(records) == len({(r["Lon"], r["Lat"]) for r in accepted}) == 208
    core = records["S99-SO-114-23"]
    assert (core["Lon"], core["Lat"], core["Thickness"]) == ("117.74", "14.07", "1.0")
    assert "S99-SO-114-6" not in records
    assert records["K-03"]["Limit_cm"] == "2.0"
    for key, thickness in {"K-07": "8.5", "K-34": "20.0", "K-52": "4.0"}.items():
        assert (records[key]["Kind"], records[key]["Thickness"]) == ("measurement", thickness)
    assert (records["K-23"]["Kind"], records["K-23"]["Limit_cm"]) == ("lower_limit", "5.8")
    assert records["H01-14"]["Kind"] == "upper_limit"
    assert float(records["H01-14"]["Limit_cm"]) == 2
    assert records["W04-SO-140-03"]["Thickness"] == "0.7"
    assert {"H01-03", "WW-SCS-C", "P7-028", "P7-056", "P7-064", "P7-080", "P7-156", "P7-170"}.isdisjoint(records)
    assert (records["P7-066"]["Kind"], records["P7-066"]["Limit_cm"]) == ("upper_limit", "25.0")
    assert (records["P7-048"]["Kind"], records["P7-048"]["Limit_cm"]) == ("lower_limit", "14.0")
    assert (records["P7-172"]["Kind"], records["P7-172"]["Limit_cm"]) == ("upper_limit", "9.5")
    assert records["P7-101"]["Thickness"] == "15.0" and records["P7-148"]["Thickness"] == "5.1"
    ledger = rows(AUDIT / "full-evidence.csv")
    assert len(ledger) == len({r["evidence_id"] for r in ledger}) == 464
    fitted = {r["evidence_id"]: int(r["accepted_csv_row"]) for r in ledger if r["accepted_csv_row"]}
    assert fitted == {r["EvidenceID"]: i for i, r in enumerate(accepted, 2)}
    from PIL import Image

    with Image.open(AUDIT / "coverage.png") as figure:
        coverage = json.loads(figure.info["Description"])
    assert coverage == {"input_sha256":sha256(INPUT.read_bytes()).hexdigest(), "observations":208,
                        "group_counts":{"Terrestrial surveys":172, "Marine cores":34, "Airfield limits":2}}
    sheets = rows(AUDIT / "reading-index.csv")
    assert {r["evidence_id"] for r in sheets} >= {key for key in records if key.startswith(("P2-", "K-", "P7-"))}
    assert all((AUDIT / r["sheet"]).is_file() for r in sheets)
    for site in sheets:
        if site["evidence_id"] in records:
            accepted_row = records[site["evidence_id"]]
            assert float(site["longitude_e"]) == float(accepted_row["Lon"])
            assert float(site["latitude_n"]) == float(accepted_row["Lat"])
    assert (AUDIT / "visual/coordinate-check.png").is_file()


def test_visual_review_replays_the_field_labels_and_empty_sites():
    from benchmarks.pinatubo import ROOT

    history = ROOT / "archive/2026-11-22-pinatubo-intake"
    before = {r["EvidenceID"]: r for r in rows(history / "koyaguchi-reviewed/input.csv")}
    after = {r["EvidenceID"]: r for r in rows(INPUT)}
    review = rows(AUDIT / "paladio-figure-7-review.csv")
    original = {r["id"]: r for r in rows(history / "pre-visual-review/paladio-figure-7.csv")}
    current = {r["id"]: r for r in rows(SOURCE / "paladio-figure-7.csv")}
    assert len(review) == len({r["evidence_id"] for r in review}) == 32
    assert before.keys() <= after.keys()
    assert {k for k in before if before[k] != after[k]} == {"P7-072", "P7-074"}
    additions = {r["evidence_id"] for r in review if r["model_kind"] and r["evidence_id"] not in before}
    assert len(additions) == 14 and additions == after.keys() - before.keys()
    for r in review:
        key = r["evidence_id"]
        assert r["curator_reading"]
        assert (r["previous_reading_cm"], r["previous_mark"]) == (original[key]["reading_cm"], original[key]["mark"])
        assert (r["reviewed_reading_cm"], r["reviewed_mark"]) == (current[key]["reading_cm"], current[key]["mark"])
        assert (r["source_x"], r["source_y"]) == (current[key]["x"], current[key]["y"])
        if r["model_kind"]:
            assert (r["model_kind"], r["model_value_cm"]) == (after[key]["Kind"], after[key]["Thickness"] or after[key]["Limit_cm"])
        else:
            assert key not in after


def test_numbered_full_maps_locate_every_reading_card_on_the_original_pixels():
    import numpy as np
    from PIL import Image

    guide = (AUDIT / "VISUAL-CHECK.md").read_text()
    cards = rows(AUDIT / "reading-index.csv")
    teal = (0, 85, 103)
    figures = [
        ("koyaguchi-figure-3", "K3", [f"K-{i:02d}" for i in range(1, 53)], ["C1", "C2"]),
        ("paladio-figure-7", "P7", [f"P7-{i:03d}" for i in range(1, 173)], ["Figure 7"]),
    ]
    for stem, tag, ids, names in figures:
        image_path = f"visual/{stem}-numbered.png"
        assert f"]({image_path})" in guide
        index = {r["evidence_id"]: r for r in cards if r["source"] == tag}
        assert set(index) == set(ids)
        with Image.open(AUDIT / image_path) as image, Image.open(SOURCE / (stem + ".gif")) as source:
            meta = json.loads(image.info["Description"])
            assert meta["source"] == stem + ".gif"
            assert meta["source_sha256"] == sha256((SOURCE / meta["source"]).read_bytes()).hexdigest()
            assert meta["source_size"] == list(source.size)
            assert [p["name"] for p in meta["panels"]] == names
            original = np.asarray(source.convert("RGB"))
            scale = meta["scale"]
            previous_stop = 0
            for number, panel in enumerate(meta["panels"], 1):
                start, stop = panel["source_y"]
                assert start == previous_stop
                previous_stop = stop
                left, top, right, bottom = panel["frame"]
                ox, oy = panel["paste_origin"]
                assert 0 <= ox and ox + source.width * scale <= image.width
                assert 0 <= oy and oy + (stop - start) * scale <= image.height
                assert [e["id"] for e in panel["entries"]] == ids
                for entry in panel["entries"]:
                    key = entry["id"]
                    card = index[key]
                    x, y = entry["pixel"]
                    assert (x, y) == (int(card[f"x{number}"]), int(card[f"y{number}"]))
                    assert left <= x < right and top <= y < bottom and start <= y < stop
                    col = (x - left) * meta["columns"] // (right - left)
                    row = (y - top) * meta["rows"] // (bottom - top) + 1
                    assert entry["square"] == chr(65 + col) + str(row)
                    line, = [line for line in guide.splitlines() if line.startswith("| [")
                             and f'({card["sheet"]})' in line and f"{key} **" in line]
                    assert entry["square"] in line.split(f"{key} **", 1)[1].split("**", 1)[0]
                    cx, cy = ox + x*scale + scale//2, oy + (y-start)*scale + scale//2
                    assert image.getpixel((cx, cy)) == tuple(original[y, x])
                    for sign in (-1, 1):
                        assert any(image.getpixel((cx + sign*dx, cy + dy)) == teal
                                   for dx in range(1, 4*scale+1) for dy in (-1, 0, 1)), key
                    l, t, r, b = entry["badge"]
                    assert (original[t:b+1, l:r+1].max(axis=2) >= 185).all(), key
            assert previous_stop == source.height


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


def test_paired_position_refit_changes_only_the_51_sublayer_coordinates(tmp_path):
    from benchmarks.pinatubo_sensitivity import paired_coordinates

    original, alternate = rows(INPUT), rows(paired_coordinates(tmp_path))
    changed = set()
    for before, after in zip(original, alternate, strict=True):
        if any(before[k] != after[k] for k in ("Lon","Lat")):
            changed.add(before["EvidenceID"])
        assert all(before[k] == after[k] for k in before if k not in ("Lon","Lat"))
    pairs = {r["evidence_id"]:r for r in rows(tmp_path/"paired-coordinates.csv")}
    assert len(pairs) == 51 and changed == set(pairs)
    assert pairs["K-03"]["c2_x"] == "302" and pairs["K-03"]["c2_y"] == "458"
    assert pairs["K-03"]["c2_lon"] == "120.580682" and pairs["K-03"]["c2_lat"] == "15.339644"

def test_two_map_reviews_compare_the_same_families_and_only_new_sites(tmp_path):
    from benchmarks.pinatubo import ROOT
    from benchmarks.pinatubo_field_review import ARCHIVE, volume_changes

    home = ROOT / "eruptions/pinatubo-1991-layer-c/results"
    release = home / "source-checks/figure7-review"
    old_rows = {r["EvidenceID"]: r for r in rows(ARCHIVE / "input.csv")}
    new_rows = {r["EvidenceID"]: r for r in rows(INPUT)}
    predictions = rows(release / "previous-field-predictions.csv")
    assert len(predictions) == 14
    assert {r["EvidenceID"] for r in predictions} == new_rows.keys() - old_rows.keys()
    for r in predictions:
        observed = new_rows[r["EvidenceID"]]
        assert r["nearest_earlier_id"] in old_rows
        assert r["Kind"] == observed["Kind"]
        assert float(r["reported_cm"]) == float(observed["Thickness"] or observed["Limit_cm"])
        assert all(float(r[k + "_median_cm"]) > 0 for k in
                   ("stacked", "spline", "directional", "spherical-mixture", "nested-isopachs"))
    replay = tmp_path / "volumes.csv"
    volume_changes(home, replay)
    assert rows(replay) == rows(release / "volume-changes.csv")


def test_reviewed_pinatubo_families_use_identical_geographic_holdouts():
    from benchmarks.pinatubo import ROOT

    home = ROOT / "eruptions/pinatubo-1991-layer-c/results"

    def read(relative):
        return json.loads((home / relative).read_text())

    def membership(folds, train_key, test_key):
        return {f["fold"]: (set(f[train_key]), set(f[test_key])) for f in folds}

    regional = read("report.json")["validation"]["methods"]
    expected = membership(regional["stacked"]["selections"], "training_row_ids", "withheld_row_ids")
    assert len(expected) == 4
    withheld = [i for train, test in expected.values() for i in test]
    assert len(withheld) == len(set(withheld)) == len(rows(INPUT))
    assert all(train.isdisjoint(test) for train, test in expected.values())
    assert membership(regional["spline"]["selections"], "training_row_ids", "withheld_row_ids") == expected
    directional = read("directional/azimuth/report.json")["validation"]["geographic"]
    spherical = read("spherical-mixture/report.json")["validation"]
    assert membership(directional["folds"], "train", "test") == expected
    assert membership(spherical["fits"], "training_row_ids", "withheld_row_ids") == expected
    assert membership(read("nested-isopachs/report.json")["folds"], "training_rows", "test_rows") == expected
