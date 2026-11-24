"""Build the all-phase primary-airfall edition from the original source ledgers."""

import argparse
from collections import Counter
import json

from .ilopango_field_review import (
    AUDIT,
    REV,
    ROOT,
    VENT,
    GEOD,
    FITTED,
    AGE,
    read,
    csv_bytes,
    digest,
    observation,
)

FIELD = AUDIT / "field-logs"
INPUT = ROOT / "data/eruptions/Ilopango_TBJ_total_airfall.csv"
RESULT = ROOT / "eruptions/ilopango-tbj/results/total-airfall-2026-11-24"
TARGET = "Local bulk thickness in cm of all primary airborne TBJ fallout, including co-ignimbrite fallout."
BASE = ROOT / "data/eruptions/Ilopango_TBJ_Unit_G_reviewed.csv"


def record(kind="", value="", why=""):
    return kind or "unresolved", str(value), why


def value(number):
    return format(number, ".12g")


def workbook(s, previous):
    """A and B are fall, G is co-ignimbrite fall, E is a mixed fall/PDC package."""
    i = int(s["id"].split("P")[-1])
    if i >= 105 or i == 67:
        return record(
            previous["kind"],
            previous["value_cm"],
            "Distal integrated ash bed or independent complete archaeological primary reading. "
            + previous["field_decision"],
        )
    # Some distal localities are explicitly recorded as one bulk bed. Preserve an already bounded bed.
    if s["setting"] == "DISTAL" and s["total_TBJ"] == s["G"] and s["G"] not in {"", "0", "?"}:
        return record(
            previous["kind"],
            previous["value_cm"],
            "Indistinguishable distal primary ash package, with matching total and G readings and no separate fall reported. "
            + previous["field_decision"],
        )
    a = sum(float(s[k]) for k in ("A", "B") if s[k].replace(".", "", 1).isdigit())
    g = s["G"]
    top = float(g) if g.replace(".", "", 1).isdigit() else (50.0 if i in (18, 46) else 0.0)
    minimum = a + top
    detail = f"Measured A+B = {value(a)} cm; identified G minimum = {value(top)} cm. E is interbedded fall/PDC. A0, C, D and F are current deposits."
    if i == 18:
        detail += " G is queried 50/100 and B is queried 100. Count 50 G; B is an unresolved alternative."
    if i == 46:
        detail += " Queried 50–100 cm of G; total airfall can exceed this component. Workbook H equals F."
    if i == 47:
        return record(
            why="H and F both 400 cm, with a conflicting 200? of G and no independently resolved primary component."
        )
    if i in (15, 63):
        return record(
            "interval",
            f"{value(minimum)}|{s['total_TBJ']}",
            detail
            + " Overlying post-TBJ material supports a conditional ceiling at the complete measured package; an erosional hiatus can break it.",
        )
    if i in (26, 66):
        return record(
            "upper_limit",
            s["total_TBJ"],
            detail
            + " Capped primary package; unresolved original retention and zero G are insufficient for an absence claim.",
        )
    if i in (27, 61, 77):
        return record(
            "interval",
            f"{value(minimum)}|{value(minimum + float(s['E']))}",
            detail
            + " Covered terminal G and logged lower units. Excluding or including all E brackets its unknown fallout fraction, conditional on local section completeness.",
        )
    if i == 72:
        return record(
            "interval",
            f"{value(minimum)}|{s['total_TBJ']}",
            detail
            + " Covered distal section, uncertain lower completeness. The whole preserved package gives a deliberately broad conditional upper bound.",
        )
    if minimum > 0:
        return record(
            "lower_limit",
            value(minimum),
            detail
            + " Earlier beds may be absent, unexposed or locally stripped, or the terminal deposit incompletely retained.",
        )
    return record(
        why=detail
        + " No securely quantified primary fall subbed or preservation-calibrated total ceiling. A missing or zero cell is not a measured zero."
    )


def build():
    older = read(REV / "localities.csv")
    rows = {x["id"]: x.copy() for x in older}
    sections = {r["id"]: r for r in read(REV / "workbook-sections.csv")}
    for k, r in rows.items():
        r["previous_G_kind"] = r["kind"]
        r["previous_G_value_cm"] = r["value_cm"]
        if k in sections:
            r["kind"], r["value_cm"], r["field_decision"] = workbook(sections[k], r)
        if r["kind"] in FITTED:
            r["identity"] = (
                "TBJ primary fallout; local sum or component/whole-package transfer specified in field_decision."
            )
        if k.startswith("K09-") and (r["kind"] in FITTED or k in {"K09-08", "K09-27", "K09-28"}):
            m = next(x for x in read(REV / "kitamura-symbols.csv") if x["id"] == k)
            raw = m["raw_reading"]
            r["kind"] = "lower_limit" if "+" in raw else "measurement"
            r["value_cm"] = raw.strip("?+")
            r["field_decision"] = (
                "Kitamura Fig. 2 orange survey point on the primary fallout map. Whole-fall reading "
                + raw
                + " cm. Printed queries and preservation/figure-location uncertainty retained."
            )
            r["quality"] = (
                "field-survey fallout map; qualified reading"
                if "?" in raw
                else "author's mapped fallout reading"
            )
            r["identity"] = "Survey-mapped TBJ fallout, combined phases."
        if k == "I21-SAN":
            r["field_decision"] = (
                "At San Andrés the largest intact excavated primary package reaches 52 cm. Use a conditional local ceiling across airfall phases, given retained contacts. Exclude disturbed sections and metres of construction fill."
            )
        if k == "R10-108":
            r["field_decision"] = (
                "The recovered wetland package is 150 cm under 0.5–1 m of soil. Suspension settling, redistribution or focusing may add material. All-phase airfall is at most 150 cm conditional on retention and negligible compaction/loss."
            )
        if k == "SM1-P124":
            r["field_decision"] += (
                " A possible antecedent is the 60 cm Río Grande excavation. The 1975 laboratory assigned CH-1 to a flow group, and field summaries identify primary TBJ. The precise linkage and airfall share are unresolved. Keep Dull S3 once with this qualification."
            )
            r["quality"] = (
                "published mapped fallout; rounded position; possible competing Río Grande facies evidence"
            )
        if k == "E16-CARMEN":
            r["kind"], r["value_cm"], r["field_decision"] = record(
                "interval",
                "120|190",
                "G and A+B measured separately above and below the current beds. Ten Fig. 1 wall columns, printed 50 cm bar and both published total-depth calibrations give 129.63–185.18 cm. Round outward to 120–190 cm. Thin basal A0 included only in the conservative upper calibration; see carmen-total-contacts.csv.",
            )
    rows["S20-TUNU2013"]["field_decision"] = (
        "The ice-core glass identifies TBJ. Recovery, grain volume and "
        "transfer from counted particles to original bulk fallout remain unresolved."
    )

    def put(key, site, x, y, kind, v, why, source, pos, quality, setting):
        row = rows.get(key, dict.fromkeys(next(iter(rows.values())), ""))
        row.update(
            id=key,
            site=site,
            locality=site,
            longitude=x,
            latitude=y,
            kind=kind,
            value_cm=str(v),
            field_decision=why,
            reference=source,
            position_source=pos,
            quality=quality,
            setting=setting,
            identity="TBJ total airfall, attribution and facies qualifications in field_decision.",
            age_evidence=AGE,
        )
        rows[key] = row

    for code, kind, v in [
        ("L08", "interval", "25|60"),
        ("L10", "lower_limit", 60),
        ("L12", "lower_limit", 65),
        ("M12", "lower_limit", 1),
    ]:
        mark = next(x for x in read(REV / "dull-figure9.csv") if x["symbol"] == code)
        sea = code == "M12"
        key = "D19-CR" if sea else "D19-" + code
        reason = (
            "Author-mapped distal marine fallout. Retain the '?' on >1 as conditional attribution. The core identity and a station-specific chemical match remain unresolved."
            if sea
            else "User-confirmed single white terrestrial field symbol and printed range or inequality, rechecked at its source pixels. Use once as an unpartitioned land fallout reading."
        )
        put(
            key,
            "Osa margin, Dull M12" if sea else "Dull Fig. 9 " + code,
            mark["longitude_from_ticks"],
            mark["latitude_from_ticks"],
            kind,
            v,
            reason,
            "Dull et al. 2019 Fig. 9; supplemental Fig. 6",
            "Printed symbol and integer-degree ticks in dull-figure9.csv; map-scale position.",
            "questioned correlation; strong geographic leverage"
            if sea
            else "mapped terrestrial primary fallout; site ID unspecified",
            "marine core" if sea else "mapped terrestrial fallout",
        )
    for stop in read(FIELD / "hart-steen-mcintyre-stops.csv"):
        n = int(stop["stop"])
        k = stop["id"]
        why = stop["source_review"]
        how = "Appendix 2-A, printed DMS normally rounded to 15 arcsec; original map datum."
        if n == 40:
            how = "Appendix printed UTM zone 16, 273700 E, 1516400 N (Clarke 1866). Printed latitude/longitude duplicate another stop; use road and grid."
        no = not stop["sum_cm"]
        kind = "unresolved" if no else "lower_limit"
        v = stop["sum_cm"]
        if n == 44:
            kind, v = "interval", "45|50"
        if n in (34, 41):
            how = "Road chainage in appendix; no published coordinates. No model observation at a guessed position."
        if n in (49, 50):
            kind = "duplicate"
        if n in (16, 48):
            j = {16: "SM1-P45", 48: "SM1-P89"}[n]
            # Use the better-localized previously registered station and the strongest secure minimum once.
            primary = rows[j]
            primary["kind"] = "lower_limit"
            primary["value_cm"] = value(max(float(primary["value_cm"]), float(v)))
            primary["field_decision"] += (
                " Same named road/site as "
                + k
                + "; two independent descriptions grouped. Use the stronger supported fall minimum once."
            )
            primary["reference"] += "; Hart & Steen-McIntyre 1983 Table 2-1 and Appendix 2-A stop " + str(n)
            kind, v = "duplicate", ""
            why += " Grouped with " + j + "."
        put(
            k,
            stop["place"],
            stop["longitude"],
            stop["latitude"],
            kind,
            v,
            why,
            "Hart & Steen-McIntyre 1983, Table 2-1, Fig. 2-9 and Appendix 2-A Stop " + str(n),
            how,
            "original field log; queried"
            if n in (11, 22, 27, 30)
            else "original field log with preserved-unit and facies audit",
            "excavation or outcrop",
        )
    # Conservative cross-survey grouping at map resolution. Retain each measured value in its source ledger.
    for k, j in {"K09-19": "HS83-28", "K09-17": "HS83-46", "K09-15": "HS83-47"}.items():
        rows[k].update(
            kind="possible_duplicate",
            value_cm="",
            field_decision="Within ~1 km of "
            + j
            + " at the registration scale of the regional map. Keep the original mapped value in kitamura-symbols.csv; use the tabulated section as the spatial representative.",
        )
    put(
        "SM75-SS12",
        "Santiago Texacuangos SS-12t",
        -89.11,
        13.65,
        "lower_limit",
        100,
        "Appendix 1 describes slightly more than a metre of probable airfall. Fig. 24 separates ash from two humic horizons. Petrography supports a qualified fallout interpretation; the authors also discuss a flow alternative. Use a minimum for total fallout.",
        "Sheets 1975 field report, Fig. 2, Table 3, Fig. 24; Steen-McIntyre Appendix 1 p. 73 and Table 1",
        "1975 sampling map Fig. 2 and Appendix 1 Fig. 1, SS-12t. Approximate map coordinates to 0.01 degree.",
        "provisional upper-airfall origin",
        "roadcut",
    )
    put(
        "SM75-CH2",
        "Los Flores (Las Flores) CH-2t",
        -89.04722,
        14.03056,
        "occurrence_only",
        "",
        "Appendix Fig. 1 locates the Las Flores sample. Table 1 assigns CH-2 to airfall on petrography, but the laboratory size fractions are not bed thickness and no intact local thickness or recovered loading is reported.",
        "Steen-McIntyre, 1975 project report, Appendix 1, Fig. 1 and Table 1",
        "Hacienda Los Flores site coordinates 14°01′50″ N, 89°02′50″ W, sheet 2358 II, reported from Fowler 1976 at https://biblioteca.utec.edu.sv/siab/virtual/tesis/55428.pdf pp. 172–173. Site-scale coordinates; mound offset from the site reference uncertain. Appendix Fig. 1 separates CH-1 and CH-2.",
        "primary sample, original source vent qualified by the investigator",
        "field tephra sample",
    )
    for row in rows.values():
        if row["longitude"] and row["latitude"]:
            az, _, d = GEOD.inv(*VENT, float(row["longitude"]), float(row["latitude"]))
            row.update(distance_km=round(d / 1000, 2), bearing_deg=round(az % 360, 2))
        row["input_csv_row"] = ""
    accepted = []
    for row in rows.values():
        if row["kind"] in FITTED:
            accepted.append(observation(row))
            row["input_csv_row"] = len(accepted) + 1
    assert len({(float(r["Lon"]), float(r["Lat"])) for r in accepted}) == len(accepted)
    data = csv_bytes(accepted)
    from hashlib import sha256

    previous = json.loads((REV / "provenance.json").read_text())
    original = next(item for item in previous["sources"] if item.get("sheet") == "Hoja1")
    original["thickness"] = (
        "H is the full package, I–P are A0 through G in centimetres. "
        "J, K and P supply A, B and G, with N (mixed E) used only in "
        "qualified bounds. G and Q document pre- and post-TBJ material."
    )
    provenance = dict(
        sources=previous["sources"],
        external_scans=read(FIELD / "primary-scans.csv"),
        preceding_sha256=digest(BASE),
        target=TARGET,
        source_inputs_sha256={
            str(p.relative_to(ROOT)): digest(p)
            for p in (
                REV / "source-manifest.csv",
                REV / "localities.csv",
                REV / "workbook-sections.csv",
                REV / "kitamura-symbols.csv",
                REV / "dull-figure9.csv",
                FIELD / "table-2-1.csv",
                FIELD / "hart-steen-mcintyre-stops.csv",
                FIELD / "carmen-total-contacts.csv",
                FIELD / "primary-scans.csv",
            )
        },
        cases={
            "principal": dict(
                file=str(INPUT.relative_to(ROOT)),
                sha256=sha256(data).hexdigest(),
                definition=TARGET,
                counts=dict(Counter(r["Kind"] for r in accepted)),
            )
        },
        principal_limits=[
            dict(evidence_id=r["EvidenceID"], interpreted_upper_cm=float(r["Limit_cm"]))
            for r in accepted
            if r["Kind"] == "upper_limit"
        ],
    )
    archives = json.loads((REV / "archive-probes.json").read_text())
    archives.update(target=TARGET, sites=[x for x in archives["sites"] if x["id"] != "D19-CR"])
    polar = next(s for s in archives["sites"] if s["id"] == "S20-TUNU2013")
    polar["inspection"] = (
        "Glass chemistry identifies TBJ. The 14–15 particles over a nominal 3.3 × 3.3 cm "
        "sample section give 1.29–1.38 recovered particles/cm² in about 2.5 years. "
        "Recovery fraction, complete particle volumes and transfer to bulk deposit "
        "thickness are unknown for total TBJ airfall."
    )
    ch = rows["SM75-CH2"]
    archives["sites"].append(
        dict(
            id=ch["id"],
            name=ch["site"],
            archive=ch["setting"],
            reading="occurrence_only",
            confidence="uncertain",
            latitude=ch["latitude"],
            longitude=ch["longitude"],
            reported_fallout_limit_cm=None,
            reference=ch["reference"],
            inspection=ch["field_decision"],
        )
    )

    def as_json(obj):
        return (json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode()

    return {
        INPUT: data,
        FIELD / "localities.csv": csv_bytes(list(rows.values())),
        FIELD / "provenance.json": as_json(provenance),
        FIELD / "archive-probes.json": as_json(archives),
    }


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--prepare", action="store_true")
    a = p.parse_args()
    for path, content in build().items():
        if a.prepare:
            path.write_bytes(content)
        else:
            assert path.read_bytes() == content, path
