"""Build the next principal Mazama input from the frozen field audit and cited decisions."""
import csv
from collections import Counter
import json
from pathlib import Path

from pyproj import Transformer

from .mazama import AUDIT, INPUT as PUBLISHED_INPUT, ROOT, sha, write_csv


REVIEW = AUDIT / "principal-ranges"
OUTPUT = ROOT / "data/eruptions/Mazama_primary_ranges.csv"
PUBLISHED_HASH = "1554141b67af566055595624bb7295c40c6da4500f95bc78c912e9d8b07a1b29"
KINDS = {"measurement": "Thickness", "upper_limit": "Limit_cm", "lower_limit": "Limit_cm"}


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def save_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def site_reading(row, spec):
    values = dict(Thickness="", Limit_cm="", Lower_cm="", Upper_cm="")
    if spec["kind"] == "interval":
        lo, hi = spec["range_cm"]
        assert 0 < lo < hi
        values.update(Lower_cm=str(lo), Upper_cm=str(hi))
    else:
        value = next(spec[key] for key in ("thickness_cm", "limit_cm") if key in spec)
        assert value > 0
        values[KINDS[spec["kind"]]] = str(value)
    return {
        "Lon": row["longitude"], "Lat": row["latitude"], **values, "Pseudo": "No", "Kind": spec["kind"],
        "LogSD": "", "Vent Lon": "", "Vent Lat": "", "EvidenceID": row["record_id"],
        "Site": row["locality"], "Reference": row["primary_reference"],
        "Evidence": spec["reason"], "Qualification": spec["qualification"],
    }


def cypress_checks(specs):
    grouped = {}
    for row in read_csv(REVIEW/"cypress-sections.csv"):
        thickness = int(row["base_cm"]) - int(row["top_cm"])
        assert thickness == int(row["thickness_cm"])
        if row["Robertson_context"] == "Primary":
            grouped.setdefault(row["model_record"], []).append(thickness)
        else:
            assert not row["model_record"]
    for site, values in grouped.items():
        spec = specs[site]
        assert spec.get("range_cm", [spec.get("thickness_cm")]*2) == [min(values), max(values)]


def build():
    assert sha(PUBLISHED_INPUT) == PUBLISHED_HASH
    old = {r["EvidenceID"]: r for r in read_csv(PUBLISHED_INPUT)}
    ledger = {r["record_id"]: r for r in read_csv(AUDIT/"revised-locality-ledger.csv")}
    s1 = {f"B2020-{int(r['buckland_id']):03}": r for r in read_csv(AUDIT/"buckland2020-293-localities.csv")}
    prior_review = {r["record_id"]: r for r in read_csv(AUDIT/"distal-review/locality-facies-crosswalk.csv")}
    decisions = json.loads((REVIEW/"decisions.json").read_text())
    specs = decisions["changes"]
    cypress_checks(specs)
    assert len(s1) == 293 and len(ledger) == 867 and old.keys() <= ledger.keys()
    assert not ledger.keys() & decisions["additional_records"].keys()
    for rid, record in decisions["additional_records"].items():
        ledger[rid] = {"record_id": rid, **record}
    assert specs.keys() <= ledger.keys()
    utm = Transformer.from_crs(26912, 4326, always_xy=True)
    accepted, crosswalk, range_review, primary_review = [], [], [], []
    for rid, row in ledger.items():
        spec = specs.get(rid, {})
        source = s1.get(rid, {})
        previous = old.get(rid)
        compilation_class = source.get("remobilisation_raw", "")
        field_ref = row.get("primary_reference", "")
        row["primary_reference"] = spec.get("reference", field_ref)
        if spec and source and not spec.get("reference"):
            row["primary_reference"] += f"; Buckland et al. (2020), S1 ID {int(source['buckland_id'])}, row {source['source_excel_row']}."
        if "utm12" in spec:
            lon, lat = utm.transform(*spec["utm12"])
            row.update(longitude=f"{lon:.6f}", latitude=f"{lat:.6f}", locality=spec["site"])
        observed_lo, observed_hi = row.get("observed_min_cm", ""), row.get("observed_max_cm", "")
        if spec.get("kind") == "interval":
            observed_lo, observed_hi = map(str, spec["range_cm"])
        scope = decisions["range_scope"].get(rid, row.get("interval_meaning", ""))
        if rid.startswith("E1997-"):
            scope = ("Graphical uncertainty in one core package. The four lake surveys show strong within-basin "
                     "variation in ash accumulation. Only the selected central core per basin supplies a conditional "
                     "ceiling from its complete published log. The two graphic readings use different figures.")
        if rid.startswith("SHP2023-") and observed_lo:
            scope = "Multi-bed lithofacies envelope. A site-specific minimum for the primary bed is unresolved."
        if spec:
            active = site_reading(row, spec)
            reason = spec["reason"]
        elif previous:
            active = dict(previous, Lower_cm="", Upper_cm="")
            reason = previous["Evidence"]
        else:
            active = None
            reason = decisions["excluded_primary_reasons"].get(rid) or prior_review[rid]["review_reason"] or row["caveats"]
        kind = active["Kind"] if active else "omitted"
        if compilation_class == "P" and not active:
            assert rid in decisions["excluded_primary_reasons"], rid
        if active:
            active["ReportedLower_cm"], active["ReportedUpper_cm"] = observed_lo, observed_hi
            accepted.append(active)
        if spec.get("kind") == "interval":
            scope = spec["qualification"]
        action = ("added" if not previous else "updated" if spec else "retained") if active else "omitted"
        if rid in ("B2020-039", "B2020-121"):
            action = "represented by another record"
        related = {"B2020-039":"B2023-GOLD", "B2020-121":"B2020-120",
                   "J2019-UA2825":"B2020-184", "J2019-S1R62":"B2020-227"}.get(rid, row.get("parent_record_id", ""))
        if rid in ("J2019-UA2825", "J2019-S1R62"):
            reason = f"Same archive as {related}. The field reading is documented in the selected locality."
        info = dict(record_id=rid, locality=row["locality"], latitude=row["latitude"], longitude=row["longitude"],
            coordinate_basis=spec.get("coordinate_basis", row["coordinate_basis"]),
            local_age=spec.get("age", row["local_age"]), identity=spec.get("identity",row["identity"]), field_verification=spec.get("verification",row["verification"]),
            field_reference=row["primary_reference"], evidence_source=spec.get("url",row["evidence_source"]),
            evidence_locator=spec.get("locator",row["evidence_locator"]), S1_excel_row=source.get("source_excel_row",""),
            S1_deposit_code=compilation_class, S1_value_cm=source.get("thickness_cm_raw",""),
            S1_value_qualifier=source.get("thickness_metadata_raw",""), related_record=related,
            reported_lower_cm=observed_lo, reported_upper_cm=observed_hi, former_kind=previous["Kind"] if previous else "omitted",
            action=action, principal_kind=kind, point_cm=active["Thickness"] if active else "",
            limit_cm=active["Limit_cm"] if active else "", lower_cm=active["Lower_cm"] if active else "",
            upper_cm=active["Upper_cm"] if active else "", decision=reason,
            qualification=active["Qualification"] if active else row.get("caveats",""), range_scope=scope)
        crosswalk.append(info)
        if compilation_class == "P":
            primary_review.append(info)
        if (observed_lo and observed_hi or rid in decisions["range_scope"] or
                info["S1_value_qualifier"].startswith("±")):
            range_review.append(info)
    assert len(primary_review) == 71
    assert len(accepted) == len({r["EvidenceID"] for r in accepted})
    assert len(accepted) == len({(float(r["Lon"]),float(r["Lat"])) for r in accepted})
    assert set(old) <= {r["EvidenceID"] for r in accepted}
    fields = ["Lon","Lat","Thickness","Pseudo","Kind","Limit_cm","Lower_cm","Upper_cm","LogSD","Vent Lon","Vent Lat",
              "EvidenceID","Site","Reference","Evidence","Qualification","ReportedLower_cm","ReportedUpper_cm"]
    write_csv(OUTPUT, [{k:r[k] for k in fields} for r in accepted])
    for name, rows in [("locality-decisions", crosswalk), ("buckland-primary", primary_review), ("range-review", range_review)]:
        write_csv(REVIEW/(name+".csv"), rows)
    archive = json.loads((AUDIT/"archive-sites.json").read_text())
    prior_limits = {r["evidence_id"]:r for r in json.loads((AUDIT/"provenance.json").read_text())["principal_limits"]}
    by_site = {r["id"]:r for r in archive["sites"]}
    limits = []
    for r in accepted:
        rid = r["EvidenceID"]
        if rid in by_site and r["Kind"] != "upper_limit":
            site = by_site[rid]
            site.pop("principal_effective_upper_cm", None)
            site["in_principal_as"] = r["Kind"]
            site["control"], site["preservation"] = r["Evidence"], r["Qualification"]
        if r["Kind"] == "upper_limit":
            value = float(r["Limit_cm"])
            limits.append(prior_limits.get(rid, dict(evidence_id=rid, interpreted_upper_cm=value,
                           derivation=r["Evidence"], qualifications=r["Qualification"])))
            assert limits[-1]["interpreted_upper_cm"] == value
            if rid not in by_site:
                site = dict(id=rid, name=r["Site"], longitude=float(r["Lon"]), latitude=float(r["Lat"]),
                            archive=ledger[rid].get("setting","field section"), reading="occurrence_only",
                            confidence="qualified", inspection=specs[rid]["reason"], reference=r["Reference"],
                            url=specs[rid].get("url","https://doi.org/10.1007/s00445-020-1362-1"),
                            control=r["Evidence"], preservation=r["Qualification"], reported_fallout_limit_cm=None,
                            in_principal_as="interpreted_upper_limit", principal_effective_upper_cm=value)
                archive["sites"].append(site)
    save_json(REVIEW/"archive-sites.json", archive)
    counts = Counter(r["Kind"] for r in accepted)
    case = dict(file=str(OUTPUT.relative_to(ROOT)), sha256=sha(OUTPUT),
                definition="Climactic Mazama bulk airborne fallout, including co-ignimbrite ash. Both endpoints of an eligible complete-bed range enter as one interval event. Primary-coded Buckland observations with a local quantity enter by default, subject to documented site exceptions.",
                n_measurements=counts["measurement"], n_lower_limits=counts["lower_limit"],
                n_upper_limits=counts["upper_limit"], n_intervals=counts["interval"])
    sources = [PUBLISHED_INPUT, AUDIT/"revised-locality-ledger.csv", AUDIT/"buckland2020-293-localities.csv",
               AUDIT/"distal-review/locality-facies-crosswalk.csv", AUDIT/"provenance.json", AUDIT/"archive-sites.json",
               ROOT/"data/sources/mazama/buckland2020-esm1.xlsx", REVIEW/"decisions.json", REVIEW/"cypress-sections.csv",
               REVIEW/"source-notes.md"]
    save_json(REVIEW/"provenance.json", dict(cases={"principal":case}, principal_limits=limits,
              previous_principal_sha256=PUBLISHED_HASH, sources={str(p.relative_to(ROOT)):sha(p) for p in sources},
              decision_sha256={p.name:sha(p) for p in [REVIEW/"buckland-primary.csv",REVIEW/"range-review.csv",
                              REVIEW/"locality-decisions.csv",REVIEW/"archive-sites.json"]},
              total_evidence_records=len(ledger), Buckland_primary_classes=dict(Counter(r["action"] for r in primary_review))))
    print(f"{len(accepted)} principal localities: {dict(counts)}; {len(primary_review)} primary-coded S1 rows; {len(range_review)} range reviews.")


if __name__ == "__main__":
    build()
