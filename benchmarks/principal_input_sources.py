"""Join figure rows to the registered source evidence without changing fitting inputs."""
import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
AUDITS = ROOT / "data/audits"

# These are the row-level ledgers for the registered editions. Source IDs already in
# the fitting table take precedence. The two older manifests expose their CSV row.
LEDGERS = {
    "aso-4": "Aso-4/record-review.csv",
    "b-tm": "B-Tm/two-phase/sites.csv",
    "ilopango-tbj": "Ilopango-TBJ/field-logs/localities.csv",
    "minoan": "Minoan/selection_manifest.csv",
    "pinatubo-1991-layer-c": "Pinatubo-1991/reaudit/localities.csv",
    "sakurajima-satsuma": "Sakurajima-Satsuma/selection.csv",
    "ytt": "YTT/selection_manifest.csv",
}
PINATUBO_SOURCES = {
    "P2": "Paladio-Melosantos et al., Table 2",
    "P7": "Paladio-Melosantos et al., Fig. 7",
    "K3": "Koyaguchi & Tokuno, Fig. 3",
    "WW1": "Wiesner & Wang, Table 1",
    "W04": "Wiesner et al. (2004), Table 1",
    "H01": "Hess et al. (2001), Table 1",
    "AIR": "Casadevall et al., airfield observations",
    "S99": "SÜDMEER (1999), section 3.4",
}


def lookup(rows, field, *, accepted=None):
    """Use original evidence identity or an explicitly recorded accepted row."""
    found = {}
    for row in rows:
        if not row[field] or (accepted is not None and not row[accepted]):
            continue
        key = row[field]
        if key in found:
            raise ValueError(f"Repeated source-ledger key {key}")
        found[key] = row
    return found


def check_position(raw, audit, lat, lon):
    if any(abs(float(raw[column]) - float(audit[field])) > 1e-7
           for column, field in (("Lat", lat), ("Lon", lon))):
        raise ValueError(f"Source row position differs from fitting table: {audit}")


def provenance_reader(slug):
    """Return the ledger and a callable providing the ID, full source and optional short label."""
    ledger = AUDITS / LEDGERS[slug] if slug in LEDGERS else None
    if ledger:
        with ledger.open(newline="", encoding="utf-8-sig") as stream:
            rows = list(csv.DictReader(stream))
    if slug == "aso-4":
        keyed = lookup(rows, "csv_row")
    elif slug == "minoan":
        keyed = lookup(rows, "combined_csv_row")
    elif slug == "ytt":
        keyed = lookup(rows, "principal_csv_row")
    elif slug == "sakurajima-satsuma":
        keyed = lookup(rows, "evidence_id", accepted="accepted_csv_row")
    elif slug in {"b-tm", "ilopango-tbj", "pinatubo-1991-layer-c"}:
        keyed = lookup(rows, "EvidenceID" if slug == "pinatubo-1991-layer-c" else "id")

    def get(raw, number):
        identity = raw.get("EvidenceID", "")
        label = ""
        source = raw.get("Reference") or raw.get("Refs") or raw.get("Source", "")
        if slug == "aso-4":
            row = keyed.get(str(number))
            if row:
                check_position(raw, row, "reviewed_latitude", "reviewed_longitude")
                if row["s2_number"]:
                    identity = f'S2 #{float(row["s2_number"]):g}'
        elif slug == "minoan":
            row = keyed[str(number)]
            check_position(raw, row, "latitude", "longitude")
            identity = identity or row["core_id"] or f'Tabelle1 row {row["xlsx_row"]}'
            source = source or f'Karstens et al. (2023), Supplementary Data 2, row {row["xlsx_row"]}'
            if row["reference"]:
                source += f' (workbook cites {row["reference"]})'
        elif slug == "ytt":
            row = keyed[str(number)]
            check_position(raw, row, "effective_latitude", "effective_longitude")
            identity = identity or row["record_id"]
            source = source or row["reference"] or row["origin"]
        elif slug == "b-tm":
            source = source or keyed[identity]["source"]
        elif slug == "ilopango-tbj":
            source = source or keyed[identity]["reference"]
        elif slug == "pinatubo-1991-layer-c":
            row = keyed[identity]
            citation = row["source"]
            source = source or PINATUBO_SOURCES.get(citation, citation)
        elif slug == "sakurajima-satsuma":
            locator = keyed[identity]["source_cell_or_text"]
            family = identity.split("-")[0]
            citation = {"S2": "Nishihara et al. (2022)", "GSJ": "GSJ Open-File Report 737",
                        "M11": "Moriwaki et al. (2011)"}[family]
            if not locator.startswith(citation):
                source = source or f"{citation}, {locator}"
            else:
                source = source or locator
            if family == "GSJ":
                label = f'{citation}, site {identity.removeprefix("GSJ-")}'
        elif slug == "tambora":
            source = source or "Supplied Tambora CSV (row citation unavailable)"
        return identity, source or "Row citation unavailable", label

    return ledger, get
