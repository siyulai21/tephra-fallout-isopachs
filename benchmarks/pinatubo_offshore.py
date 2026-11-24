"""Replay the Layer C re-audit from the frozen visual survey and field transcriptions."""
import csv
import gzip
import json
import math
import re
from collections import Counter
from hashlib import sha256
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "data/sources/pinatubo-1991"
AUDIT = ROOT / "data/audits/Pinatubo-1991"
REV = AUDIT / "reaudit"
OLD = ROOT / "data/eruptions/Pinatubo_1991_layer_C.csv"
INPUT = OLD.with_name("Pinatubo_1991_layer_C_reviewed.csv")


def read(p):
    with p.open(newline="") as f:
        return list(csv.DictReader(f))


def write(path, rows):
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0], lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def degrees(dms):
    degree, minute, second = map(float, dms.split())
    return degree + minute / 60 + second / 3600


def farm_position(x, y):
    # Longitude ticks (top and bottom), latitude ticks (left and right).
    lons = 120 + (1.5 + np.arange(8)*5.5)/60
    top = [50,120,190,260,330,400,470,540]
    bottom = [44.5,114.5,184.5,255,325,395.5,466,536]
    lats = 15 + (19.5 - np.arange(5)*5.5)/60
    left, right = [80,155,230,306,381], [87,162,237,312,388]
    lonfit = np.linalg.lstsq([(v,a,1) for vs,a in [(top,33),(bottom,441)] for v in vs],np.tile(lons,2),rcond=None)[0]
    latfit = np.linalg.lstsq([(a,v,1) for vs,a in [(left,23),(right,577)] for v in vs],np.tile(lats,2),rcond=None)[0]
    xy = [float(x), float(y), 1]
    return np.dot(xy,lonfit), np.dot(xy,latfit)


def build():
    assert sha256(OLD.read_bytes()).hexdigest() == "2899fd9e79d88f81065dc87e50da41e0673b525bf370b6788a8f72d4ad14f5ce"
    rows = {r["EvidenceID"]: dict(r,Lower_cm="",Upper_cm="") for r in read(OLD)}
    changes = []
    baseline = list(rows.values())[0]
    def put(key, kind, low, high, cite, rationale, *, name=None, xy=None):
        if key not in rows:
            lon,lat = xy
            rows[key] = dict.fromkeys(baseline,"")
            rows[key].update(EvidenceID=key,Site=name,Lon=f"{lon:.6f}",Lat=f"{lat:.6f}",
                             **{k:baseline[k] for k in ["Vent Lon","Vent Lat"]})
        row=rows[key]
        previous = row["Kind"]+" "+(row["Thickness"] or row["Limit_cm"])
        row.update(Kind=kind,Thickness="",Limit_cm="",Lower_cm="",Upper_cm="")
        if kind=="interval":
            row.update(Lower_cm=low,Upper_cm=high)
        elif kind=="measurement":
            row["Thickness"]=low
        else:
            row["Limit_cm"]=low if kind=="lower_limit" else high
        changes.append(dict(evidence_id=key,site=row["Site"],longitude=row["Lon"],latitude=row["Lat"],
                            prior=previous.strip(),kind=kind,point_or_bound_cm=row["Thickness"] or row["Limit_cm"],
                            lower_cm=row["Lower_cm"],upper_cm=row["Upper_cm"],source=cite,interpretation=rationale))

    hess_log=[]
    for r in read(SRC/"hess-table-1.csv"):
        key = "H01-"+r["site"].zfill(2)
        raw = r["thickness_mm"]
        linked = r["linked"] or key
        if re.fullmatch(r"\d+–\d+",raw) and linked in rows and (
            key in rows or linked.startswith("W04-")):
            lo,hi=(int(v)/10 for v in raw.split("–"))
            if rows[linked]["Thickness"]:
                assert lo <= float(rows[linked]["Thickness"]) <= hi
            put(linked,"interval",lo,hi,"Hess Table 1 "+key+" "+r["stations"],
                "Observed replicate span at this station. Named cruise casts and any direct bulk reading count once.")
            decision="replicate interval"
        elif key in rows or r["linked"]:
            decision="retained direct or phase-qualified reading; see historical selection and marine review"
        else:
            decision="questionable or missing thickness; parentheses give no direction of error"
        if r["site"]=="15":
            decision="table dash; see nearby Wetzel 247 and its direct nondetection evidence"
        hess_log.append(dict(**r,principal_evidence_id=linked if linked in rows else "",decision=decision))
    put("WW-17922-1","interval",1.9,2.0,"SO95 surface Table 1; Wiesner & Wang Table 1",
        "Two independent readings of the same early box-core bed. Exclude pooled, offset Hess revisits.")
    # The two early nearshore cores resolve a coarse basal component even if
    # some of the fine cap belongs to a later phase (SO114 report, sec. 3.4).
    for core, total, maximum in [("17920",6.0,6.5),("17921",6.2,6.2)]:
        put("WW-"+core+"-1","interval",round(total-3.5,1),maximum,
            "Wiesner & Wang Table 1 and Fig. 2; SÜDMEER sec. 3.4; SO95 surface Table 1; Hess groups 1–2",
            "The early cores each have an approximately 1 cm fine cap. Conservatively allow the full 3.5 cm upper-unit maximum reported for the same nearshore latitude band to be non-target. Original basal coarse airfall supplies the lower endpoint. Upper endpoint includes the full early bed and the verified range at 17920; uncertain later reworked 17921 measurements are excluded.")
        hess_log[0 if core=="17920" else 1]["decision"]="same early station; see the phase-qualified principal interval"
    for r in read(AUDIT/"survey-coverage/david-schematic.csv"):
        put("D96-"+r["site"],"lower_limit",float(r["coarse_unit_cm"]),None,"David et al. Table 1 and Fig. 3",
            "Retained coarse June 15 pumice only. Assume in-place fallout without local addition. Approximate survey position; nearby surveys may share local conditions.",
            name=r["site"]+" pumice section",xy=(float(r["approximate_lon"]),float(r["approximate_lat"])))
    pits=read(REV/"quimbo-pits.csv")
    for group in ["P1","P2","P3","P4"]:
        members=[r for r in pits if r["group"]==group]
        min_cm=min(int(r["coarse_base_cm"])-int(r["coarse_top_cm"]) for r in members)
        a=members[0]
        put("Q20-"+group,"lower_limit",min_cm,None,"Quimbo Tables 3.3, A3.1 and A4.1; Fig. 3.3",
            "Smallest retained coarse pumice thickness among nearby pits, conditional on the identified June 15 pumice remaining in place. Fine ash, topsoil and transported material are excluded. Pit a supplies the representative GPS location.",
            name=a["reported_land_use"]+" "+group,xy=(degrees(a["longitude_dms"]), degrees(a["latitude_dms"])))
    for r in read(REV/"quimbo-farms.csv"):
        if r["decision"]=="upper":
            put("Q20-"+r["site"],"upper_limit",None,int(r["interview_cm"].strip("<")),
                "Quimbo Table 6.1 col. 6 and Fig. 6.1",
                "Retrospective all-phase loading on an ash-classified farm. Conditional ceiling assumes recall includes the full local bed without underestimation. Column 8 is a published isopach.",
                name=r["barangay"]+" "+r["site"],xy=farm_position(r["printed_figure_x"],r["printed_figure_y"]))
    put("WW-SCS-C","measurement",0.8,None,"Wiesner & Wang Table 1, trap SCS-C; Wiesner et al. (1995)",
        "Author-reported bulk equivalent. 9 kg/m² divided by a 2.34 g/cm³ grain density and 0.475 solid fraction. Requires representative fallout capture. Simultaneous traps form one station; nearby 0.8 cm box cores corroborate.",
        name="SCS-C sediment-trap mooring",xy=(115+6.4/60,14+36.1/60))
    probes={r["station"]:r for r in read(AUDIT/"survey-coverage/wiesner-2004-station-probes.csv")}
    for station,kind,lo,hi,locator,why in [
        ("24","lower_limit",1.,None,"Fig. 7A","Approximate sharp-based surviving bed, assuming local mixing removed material from the bed without importing additional ash."),
        ("33","measurement",2.,None,"Fig. 5B","Author's reconstructed original thickness. Same-station earlier survey marks a measured bed. Neither the mixed-window depth nor a second independent reading is fitted."),
        ("47","measurement",1.5,None,"Fig. 7C and province III discussion","No visible bed but radiography identifies completely mixed ash originally estimated at 1.5 cm. Use the author's approximate original thickness. The survey nondetection and province-specific conditional 2 cm ceiling are linked checks on this single station.")]:
        r=probes["SO132_"+station]
        put("W09-"+station,kind,lo,hi,"Wetzel (2009) "+locator,why,name="SO132-"+station,
            xy=(float(r["longitude"]),float(r["latitude"])))
    core = hess_log[10]
    assert core["stations"] == "SO-114-4"
    locator = next(r for r in read(AUDIT/"full-evidence.csv") if r["evidence_id"] == "H01-11")
    put("W08-404", "measurement", 5., None, "Wetzel (2008), Figs. 2 and 5B; Hess Table 1 group 11",
        "Author's reconstruction of the original Pinatubo ash completely mixed by large Scolicia. The 0–8 cm X-ray window is not the bed. Hess parenthesizes the disturbed 5.5–6 cm thickness at this station; keep it as a field check. The reconstructed point assumes the original bed is correctly identified.",
        name="SO114-4 (Wetzel 404)", xy=(float(locator["longitude"]), float(locator["latitude"])))
    core.update(principal_evidence_id="W08-404", decision="Wetzel Fig. 5B resolves original thickness at the same station; questioned Hess repeat retained for comparison.")
    write(REV/"hess-decisions.csv",hess_log)
    with gzip.open(SRC/"context/so95-events.html.gz","rt") as f:
        events={}
        for tr in re.findall(r"<tr[^>]*>(.*?)</tr>",f.read(),re.S):
            cells=re.findall(r"<td[^>]*>(.*?)</td>",tr,re.S)
            if len(cells)>7 and cells[3]=="GKG":
                events.setdefault(cells[0][3:8],[]).append(cells)
    for r in read(REV/"so95-surface.csv"):
        if not r["profile_top_y"]:
            continue
        event,=events[r["station"]]
        top,clay,base=(int(r[k]) for k in ["profile_top_y","base_of_mixed_y","column_base_y"])
        # Outward edge allowance of two pixels at each contact on the 105 dpi rendering.
        cap=math.ceil(float(r["core_cm_max"])*(clay+2-(top-2))/((base-2)-(top-2)))
        put("SO95-"+r["station"],"upper_limit",None,cap,"Sarnthein et al. spade-box register, surface Table 1 and section 3.4 Figs. 1–2",
            "Clear overlying water and intact surface above older homogeneous clay. Weak capacity ceiling conditional on the original local ash volume remaining inside the logged surficial package, with no net export or material compaction. Station-specific bracket is independent of visibility.",
            name=event[0],xy=(float(event[7]),float(event[6])))
    # Preserve the confirmed land survey digitizations exactly as supplied.
    for old in read(OLD):
        if old["EvidenceID"].startswith(("P7-", "P2-", "K-")):
            assert all(rows[old["EvidenceID"]][key] == value for key, value in old.items())

    prior = {r["evidence_id"]: r for r in read(AUDIT/"full-evidence.csv")}
    revised = {r["evidence_id"]: r for r in changes}
    ledger = []
    for key, row in rows.items():
        earlier = prior.get(key, {})
        change = revised.get(key, {})
        assert change or earlier.get("accepted_csv_row"), key
        ledger.append(dict(**row, event_date="1991-06-15", unit="climactic Layer C",
                           source=change.get("source") or earlier["source"],
                           earlier_source_record=earlier.get("source_record", ""),
                           earlier_published_reading=earlier.get("published_reading", ""),
                           earlier_units=earlier.get("units",""),
                           field_decision=change.get("interpretation") or earlier["decision"],
                           earlier_position_basis=earlier.get("position_evidence", ""),
                           source_locator="source-notes.md" if change else "../"+earlier["verification"]))
    write(REV/"localities.csv",ledger)
    write(INPUT,list(rows.values()))
    write(REV/"principal-decisions.csv",changes)
    kinds=Counter(r["Kind"] for r in rows.values())
    assert len({(r["Lon"],r["Lat"]) for r in rows.values()})==len(rows)
    audit=dict(cases={"principal":dict(file=INPUT.relative_to(ROOT).as_posix(),sha256=sha256(INPUT.read_bytes()).hexdigest(),
                  definition="June 15 climactic Layer C and its correlated marine fallout, including co-ignimbrite ash, in bulk centimetres.",counts=kinds)},
               source_inputs_sha256={p.relative_to(ROOT).as_posix():sha256(p.read_bytes()).hexdigest()
                    for p in (OLD, AUDIT/"full-evidence.csv", SRC/"hess-table-1.csv", SRC/"context/so95-events.html.gz",
                              AUDIT/"survey-coverage/david-schematic.csv", AUDIT/"survey-coverage/wiesner-2004-station-probes.csv",
                              REV/"quimbo-pits.csv", REV/"quimbo-farms.csv", REV/"so95-surface.csv", REV/"source-manifest.csv")},
               preceding_sha256=sha256(OLD.read_bytes()).hexdigest(),
               principal_limits=[dict(evidence_id=r["evidence_id"],interpreted_upper_cm=r["point_or_bound_cm"]) for r in changes if r["kind"]=="upper_limit"])
    (REV/"provenance.json").write_text(json.dumps(audit,indent=2)+"\n")
    print(len(rows),dict(kinds),audit["cases"]["principal"]["sha256"])


if __name__=="__main__":
    build()
