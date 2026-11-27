"""Replay Young's reviewed marker positions and their joining to the Mazama field tables."""
import csv
from collections import Counter, defaultdict
from hashlib import sha256
import json
from pathlib import Path

from benchmarks.mazama import write_csv
from benchmarks.mazama_young_maps import displacement, km, offset
from src.empirical_isopach.core.fitting import fitting_bytes

DATA = Path(__file__).resolve().parents[1] / "data"
AUDIT = DATA / "audits/Mazama"
REVIEW = AUDIT / "young-marker-review"
PREVIOUS = DATA / "eruptions/Mazama_bolan_lake.csv"
REPORTED = DATA / "eruptions/Mazama_young_centroids.csv"
FITTED = DATA / "eruptions/Mazama_young_centroids_minimum_fit.csv"


def read_csv(path):
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def reviewed_marks():
    marks = {}
    for clip in json.loads((REVIEW / "clip-registration.json").read_text())["clips"]:
        assert sha256((REVIEW / clip["file"]).read_bytes()).hexdigest() == clip["sha256"]
        cx, cy, scale_x, scale_y = clip["registration"]
        width, height = clip["clip_dimensions"]
        for mark in clip["marks"]:
            x = cx + (mark["clip_x"] - (width - 1) / 2) * scale_x + 0.5
            y = cy + (mark["clip_y"] - (height - 1) / 2) * scale_y + 0.5
            assert abs(x - mark["page_x"]) < 0.001 and abs(y - mark["page_y"]) < 0.001
            assert mark["site"] not in marks
            marks[mark["site"]] = (clip["file"], clip["page"], mark["page_x"], mark["page_y"])
    return marks


def run():
    parts = defaultdict(dict)
    for row in read_csv(AUDIT / "young1990-appendix1.csv"):
        parts[int(row["site"])][row["appendix_table"]] = row
    positions = {int(r["site"]): r for r in read_csv(AUDIT / "young1990-section-positions.csv")}
    assert set(positions) == set(parts)
    prior_decisions = {r["record_id"]: r for r in read_csv(AUDIT / "principal-ranges/locality-decisions.csv")}
    source_rows = read_csv(PREVIOUS)
    accepted = {r["EvidenceID"]: r for r in source_rows}
    origins = json.loads((AUDIT / "young1990-georeferencing.json").read_text())["alternate_origin_lon_lat"]
    section_106 = positions[106]
    anchor_pixel = [float(section_106["pixel_x"]), float(section_106["pixel_y"])]
    assert section_106["map_pdf_page"] == "323"
    marks = reviewed_marks()
    ledger = []

    for site, old in sorted(positions.items()):
        key = f"Y1990-A{site:03}"
        field = parts[site]
        cup = field.get("C", {})
        thickness = cup.get("upper_pumice_th_cm", "")
        prior = prior_decisions[key]
        old_coordinate = [float(old["longitude"]), float(old["latitude"])]
        page = int(old["map_pdf_page"])
        clip = ""
        x, y = old["pixel_x"], old["pixel_y"]
        lon, lat = old_coordinate
        alt_lon, alt_lat = old["upper_kirk_anchor_longitude"], old["upper_kirk_anchor_latitude"]
        spread = old["alternate_registration_max_displacement_km"]
        marker_note = old["marker_note"]
        related = prior["related_record"]
        decision = prior["decision"]
        status = "retained" if key in accepted else "omitted"
        if site in marks:
            clip, checked_page, x, y = marks[site]
            assert checked_page == page
            marker_note = f"Contributor-marked centre in {clip}, transferred to the original Appendix 1d plate."
            alternatives = {}
            for name, origin in origins.items():
                if page == 321:
                    origin = offset(origin, displacement(323, anchor_pixel))
                alternatives[name] = offset(origin, displacement(page, (x, y)))
            lon, lat = alternatives["shoreline_centre"]
            alt_lon, alt_lat = alternatives["upper_kirk_equivalent"]
            spread = round(max(km((lon, lat), pos) for pos in alternatives.values()), 3)
            related = ""
            if thickness:
                minimum = thickness.split()[0]
                decision = f"Young Appendix 1c, PDF p. {cup['pdf_page']}, section {site}, upper pumice thickness {thickness} cm."
                if "+" in thickness:
                    decision += " Young marks the exposed thickness as a minimum."
                decision += " This preserved climactic Plinian component supplies a minimum on total local airborne fallout."
                status = "relocated" if key in accepted else "added"
                # A new section inherits only the accepted table schema. All site-specific fields come from this row.
                row = accepted.get(key, dict.fromkeys(source_rows[0], ""))
                row.update(EvidenceID=key, Site=prior["locality"], Reference=prior["field_reference"],
                           Lon=f"{lon:.6f}", Lat=f"{lat:.6f}", Pseudo="No", Kind="lower_limit",
                           Limit_cm=f"{float(minimum):.1f}", Evidence=decision,
                           Qualification=f"Appendix 1d, PDF p. {page}, contributor-marked centroid in {clip}. An in-place upper pumice bed is required. Compaction, erosion and missing fine ash can lower the preserved thickness. Imported material can increase it.")
                if key not in accepted:
                    source_rows.append(row)
                    accepted[key] = row
            elif cup:
                decision = "Appendix 1c records maximum clast diameters. Bulk upper-pumice thickness is blank. The clasts do not establish a section-averaged thickness."
            else:
                decision = "Appendix 1a gives precursor pumice thickness. No climactic bulk thickness is reported for this section."
        elif site in {15, 100, 133}:
            marker_note = "Separate unmarked dark dot visible beside the contributor-marked section."
            decision = {
                15: "The separate dark dot by 15 remains at its archived position. Appendix 1c gives 194 cm of upper pumice here.",
                100: "The separate dark dot by 100 remains at its archived position. Appendix 1c reports upper-pumice clast sizes only.",
                133: "The separate dark dot by 133 remains at its archived position. Appendix 1c gives 70 cm of upper pumice here.",
            }[site]
        quantity = accepted.get(key, {})
        ledger.append(dict(evidence_id=key, site=site, map_pdf_page=page, annotated_crop=clip, marker_note=marker_note,
                           previous_pixel_x=old["pixel_x"], previous_pixel_y=old["pixel_y"],
                           pixel_x=x, pixel_y=y, previous_longitude=old["longitude"], previous_latitude=old["latitude"],
                           longitude=f"{lon:.6f}", latitude=f"{lat:.6f}",
                           shift_from_previous_km=round(km(old_coordinate, (lon, lat)), 4),
                           alternate_registration_max_displacement_km=spread,
                           upper_kirk_anchor_longitude=f"{float(alt_lon):.6f}", upper_kirk_anchor_latitude=f"{float(alt_lat):.6f}",
                           source_tables_and_pdf_pages=" / ".join(f"1{k.lower()} p. {r['pdf_page']}" for k,r in field.items()),
                           upper_pumice_thickness_cm=thickness, age_basis=prior["local_age"],
                           previous_related_record=prior["related_record"], current_related_record=related,
                           decision=status, principal_kind=quantity.get("Kind", ""), principal_limit_cm=quantity.get("Limit_cm",""),
                           basis=decision))
    assert len(ledger) == 138 and len(marks) == 12
    assert {r["site"] for r in ledger if r["decision"] == "added"} == {95, 134, 150}
    assert {r["site"] for r in ledger if r["decision"] == "relocated"} == {97, 116}
    assert len(accepted) == len(source_rows) == 304
    assert Counter(r["Kind"] for r in source_rows) == dict(measurement=47, interval=29, lower_limit=132, upper_limit=96)
    write_csv(REVIEW / "section-register.csv", ledger)
    write_csv(REPORTED, source_rows)
    FITTED.write_bytes(fitting_bytes(REPORTED))


if __name__ == "__main__":
    run()
