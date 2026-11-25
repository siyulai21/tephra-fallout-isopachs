"""Reconcile the Unit G field review with the preserved source-cell audit."""
import argparse
from collections import Counter
import csv
from hashlib import sha256
from io import StringIO
import json
from pathlib import Path

from .ilopango import AUDIT, BOOK, BOOK_HASH, CORES, CORE_TABLE_HASH, GEOD, ROOT, VENT, original

REV = AUDIT / "reaudit"
INPUT = ROOT / "data/eruptions/Ilopango_TBJ_Unit_G_reviewed.csv"
OUT = ROOT / "archive/2026-11-24-case-editions/ilopango-unit-g-116"
FITTED = {"measurement", "interval", "lower_limit", "upper_limit"}
AGE = "TBJ eruption correlation; Smith et al. 431 ± 2 CE; Dull et al. propose 539/540 CE. No independent local age in this row."


def read(p):
    with p.open(newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def csv_bytes(rows):
    stream = StringIO()
    writer = csv.DictWriter(stream, list(rows[0]), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().encode()


def digest(p):
    return sha256(p.read_bytes()).hexdigest()


def observation(row):
    kind = row["kind"]
    value = row["value_cm"]
    lo, hi = ("", "")
    if kind == "interval":
        lo, hi = value.split("|")
        assert 0 < float(lo) < float(hi)
    else:
        assert float(value) > 0
    return dict(EvidenceID=row["id"], Site=row["site"], Lon=row["longitude"], Lat=row["latitude"],
                Thickness=value if kind == "measurement" else "",
                **{"Vent Lon": VENT[0], "Vent Lat": VENT[1]}, Kind=kind,
                Limit_cm=value if kind in {"lower_limit","upper_limit"} else "", Lower_cm=lo, Upper_cm=hi)


def baseline(x, section):
    setting = x["setting"]
    kind = x["kind"] or "unresolved"
    decision = x["decision"]
    quality = "source-attributed field measurement"
    ref = x["source"]
    identity = "Unit G in column P; localized transition to F and erosion possible."
    if setting != "outcrop":
        identity = "Distal TBJ assigned to co-ignimbrite dispersal; unresolved individual pulses."
    if setting == "marine core":
        ref += "; Kutterolf 2008a Table 1 C1; Dull 2019 Table S2b"
        quality = "compiled local quantity; verified glass correlation"
        decision += " Sample depths in Table S2b identify analysed glass. C1 commonly forms pods; station-specific original bed recovery is uncertain."
    if x["source_row"] and x["unit_g_cm"] == "0":
        quality = "unit entry without original-thickness calibration"
        decision = (f"G=0 in the unit inventory; H={section['total_TBJ']} cm, F={section['F']} cm, "
                    f"post-TBJ={section['post_TBJ']} cm. No demonstrated intact terminal surface and "
                    "no station-specific detection scale. Whole-section arithmetic can mix rounded or overlapping facies.")
    if x["unit_g_cm"] == "?":
        quality = "unresolved unit boundary"
        decision += f" H={section['total_TBJ']} cm and F={section['F']} cm. Original ash retention above F unestablished."
    if x["evidence_id"] == "S20-TUNU2013":
        identity = "Chemically matched glass; phase unresolved."
        quality = "secure occurrence; bulk transfer unavailable"
        decision += " 14–15 identified particles over a 3.3 × 3.3 cm nominal cross-section give 1.29–1.38 recovered particles/cm². The sample spans ~2.5 years. Recovery fraction, full particle volumes, phase share and deposition/preservation transfer are unknown."
    pos = "Workbook C/D on the same source row, degrees and decimal minutes." if x["source_row"] else "Smith 2020 printed coordinates."
    return dict(id=x["evidence_id"], site=x["site_id"], locality=x["site"], longitude=x["longitude"],
                latitude=x["latitude"], setting=setting, position_source=pos,
                age_evidence="Annual ice chronology, 431 ± 2 CE." if setting=="ice core" else AGE,
                identity=identity, reference=ref, raw_G_cm=x["unit_g_cm"], raw_total_cm=x["total_tbj_cm"],
                kind=kind, value_cm=x["fitted_cm"], quality=quality, field_decision=decision)


def verify_workbook():
    import xlrd
    sheet = xlrd.open_workbook(BOOK).sheet_by_name("Hoja1")
    for section in read(REV/"workbook-sections.csv"):
        cells = sheet.row_values(int(section["id"].split("P")[-1]) - 1)
        assert list(section.values())[1:] == list(map(original,[cells[0],cells[1],cells[5],*cells[6:17]]))


def products():
    assert digest(BOOK) == BOOK_HASH and digest(CORES) == CORE_TABLE_HASH
    legacy = read(AUDIT/"selection.csv")
    sections = {x["id"]:x for x in read(REV/"workbook-sections.csv")}
    assert len(legacy) == 110 and len(sections) == 107
    rows = {x["evidence_id"]:baseline(x,sections.get(x["evidence_id"], {})) for x in legacy}
    decisions = read(REV/"field-decisions.csv")
    assert len({r["id"] for r in decisions}) == len(decisions)
    for d in decisions:
        key = d["id"]
        if key not in rows:
            assert all(d[field] for field in ("site", "latitude", "longitude", "reference", "position_source"))
            rows[key] = dict.fromkeys(next(iter(rows.values())), "")
            rows[key].update(id=key, locality=d["site"], age_evidence=AGE,
                identity="TBJ archive correlation; Unit G transfer is described in the field decision.")
        rows[key].update({k:v for k,v in d.items() if v or k=="value_cm"})
    rows["SM1-P116"]["age_evidence"] = "Mehringer Table 2: AMS charcoal at El Cacao, charcoal and water lily seed at Agua Azul; published combined mean 1810 ± 30 radiocarbon yr BP; separate from Smith's ice age."
    rows["SM1-P118"]["age_evidence"] = "Twig against upper contact: 1540 ± 40 radiocarbon yr BP; reported 2σ calibrated 420–620 CE (McNeil et al. 2010)."
    for symbol in read(REV/"kitamura-symbols.csv"):
        if symbol["id"] in rows:
            continue
        key = symbol["id"]
        entry = dict.fromkeys(next(iter(rows.values())), "")
        entry.update(id=key, site=key, locality="Fallout map symbol", longitude=symbol["longitude"],
            latitude=symbol["latitude"], setting="mapped fallout", position_source=symbol["source"],
            age_evidence=AGE, identity="TBJ, individual phase unresolved", reference=symbol["source"],
            raw_total_cm=symbol["raw_reading"], kind=symbol["kind"], quality="map transcription",
            field_decision=symbol["decision"] + " Linked locality: " + (symbol["linked_id"] or "unresolved"))
        rows[key] = entry
    for mark in read(REV/"dull-figure9.csv"):
        if mark["linked_id"]:
            continue
        key = "D19-" + mark["symbol"]
        entry = dict.fromkeys(next(iter(rows.values())), "")
        entry.update(id=key, site=key, locality="Anonymous land symbol", longitude=mark["longitude_from_ticks"],
            latitude=mark["latitude_from_ticks"], setting="mapped TBJ locality", position_source="Dull Fig. 9 ticks",
            age_evidence=AGE, identity="Mapped TBJ; potential overlap with other localities", reference="Dull et al. 2019 Fig. 9",
            raw_total_cm=mark["label"], kind="unresolved", quality="unkeyed figure label",field_decision=mark["decision"])
        rows[key] = entry
    accepted = []
    inventory = []
    for row in rows.values():
        az, _, length = GEOD.inv(*VENT, float(row["longitude"]), float(row["latitude"]))
        row["bearing_deg"], row["distance_km"] = round(az%360,2), round(length/1000,2)
        row["input_csv_row"] = ""
        if row["kind"] in FITTED:
            accepted.append(observation(row))
            row["input_csv_row"] = len(accepted)+1
        if row["kind"] == "occurrence_only" or row["id"] == "D19-CR":
            reading = "attribution_uncertain" if row["id"] == "D19-CR" else "occurrence_only"
            inventory.append(dict(id=row["id"],name=row["site"],archive=row["setting"],
                reading=reading,confidence="uncertain" if reading=="attribution_uncertain" else "secure",
                latitude=float(row["latitude"]),longitude=float(row["longitude"]),
                reported_fallout_limit_cm=None,reference=row["reference"],inspection=row["field_decision"]))
    assert len({(float(r["Lon"]),float(r["Lat"])) for r in accepted}) == len(accepted)
    data = csv_bytes(accepted)
    previous = json.loads((AUDIT/"provenance.json").read_text())
    provenance = dict(sources=previous["sources"], preceding_sha256=previous["cases"]["principal"]["sha256"],
        source_inputs_sha256={str(p.relative_to(ROOT)):digest(p) for p in
            (AUDIT/"selection.csv", REV/"field-decisions.csv", REV/"workbook-sections.csv", REV/"kitamura-symbols.csv", REV/"dull-figure9.csv",
             REV/"san-andres-trenches.csv", REV/"other-field-records.csv", REV/"carmen-wall.csv")},
        cases={"principal":dict(file=str(INPUT.relative_to(ROOT)),sha256=sha256(data).hexdigest(),
            definition=previous["cases"]["principal"]["definition"],counts=dict(Counter(x["Kind"] for x in accepted)))},
        principal_limits=[dict(evidence_id=x["EvidenceID"],interpreted_upper_cm=float(x["Limit_cm"]))
                          for x in accepted if x["Kind"]=="upper_limit"])
    archives=dict(schema="tephra-archive-evidence-v1",target=provenance["cases"]["principal"]["definition"],sites=inventory,scenarios=[])
    def binary(x):
        return (json.dumps(x,indent=2,ensure_ascii=False,allow_nan=False)+"\n").encode()
    return {INPUT:data,REV/"localities.csv":csv_bytes(list(rows.values())),REV/"provenance.json":binary(provenance),
            REV/"archive-probes.json":binary(archives)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare",action="store_true")
    parser.add_argument("--verify-workbook",action="store_true")
    args=parser.parse_args()
    if args.verify_workbook:
        verify_workbook()
    for dest, contents in products().items():
        if args.prepare:
            dest.write_bytes(contents)
        elif dest.read_bytes() != contents:
            raise ValueError(f"Field audit differs at {dest}")
