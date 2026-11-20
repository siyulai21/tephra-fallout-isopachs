"""Verify or rebuild Minoan combined-fallout inputs from the original core table."""
import argparse
import csv
from hashlib import sha256
import io
import json
from pathlib import Path
import posixpath
import xml.etree.ElementTree as ET
import zipfile

import numpy as np
from src.empirical_isopach.core.observations import read_observations

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT/'data/sources/minoan/supplementary_data_2.xlsx'
AUDIT = ROOT/'data/audits/Minoan'
SOURCE_SHA = 'a4046d43e4b28afebc7ce1bd8f73284089ea607679cd9b54c1f348bec8a9eee6'
VENT = (25.396, 36.404)
FIELDS = ['map_id', 'core_id', 'latitude', 'longitude', 'total_cm', 'plinian_cm',
          'turbidite_cm', 'co_ignimbrite_cm', 'location', 'reference', 'comment']
CSV_FIELDS = ['Lon', 'Lat', 'Thickness', 'Vent Lon', 'Vent Lat', 'Kind', 'Limit_cm']
CASES = {
    'combined': ('data/eruptions/Minoan_tephra.csv', 'Combined fallout: consistent reported totals minus identified turbidites; incomplete cores are lower bounds; conflicting rows excluded; identical colocated records counted once.'),
    'exact_only': ('data/audits/Minoan/exact_only.csv', 'Combined main selection excluding incomplete-core lower bounds.'),
    'component_conflicts': ('data/audits/Minoan/component_conflicts.csv', 'Combined main selection plus map IDs 83 and 85 using their listed Plinian plus co-ignimbrite sum, 10 and 5 cm.'),
    'total_conflicts': ('data/audits/Minoan/total_conflicts.csv', 'Combined main selection plus map IDs 83 and 85 using their reported total, 12 and 15 cm.'),
    'without_pos513_41': ('data/audits/Minoan/without_pos513_41.csv', 'Combined main selection minus map ID 24, whose workbook and Supplementary Note 1 describe different total sequences.'),
}


def source_rows(path=SOURCE):
    """Read unchanged static XLSX values, preserving inequalities and blanks."""
    ns = {'s': 'http://schemas.openxmlformats.org/spreadsheetml/2006/main'}
    with zipfile.ZipFile(path) as z:
        strings = [''.join(t.text or '' for t in item.findall('.//s:t', ns))
                   for item in ET.fromstring(z.read('xl/sharedStrings.xml'))]
        sheet = next(s for s in ET.fromstring(z.read('xl/workbook.xml')).findall('s:sheets/s:sheet', ns)
                     if s.attrib['name'] == 'Tabelle1')
        rid = sheet.attrib['{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id']
        target = next(r.attrib['Target'] for r in ET.fromstring(z.read('xl/_rels/workbook.xml.rels')) if r.attrib['Id'] == rid)
        target = target.lstrip('/') if target.startswith('/') else posixpath.normpath(posixpath.join('xl', target))
        rows = []
        for row in ET.fromstring(z.read(target)).findall('s:sheetData/s:row', ns):
            if int(row.attrib['r']) < 3:
                continue
            values = [None]*len(FIELDS)
            for cell in row.findall('s:c', ns):
                if cell.find('s:f', ns) is not None:
                    raise ValueError('Review source formulas before relying on cached values.')
                column = 0
                for letter in cell.attrib['r']:
                    if not letter.isalpha(): break
                    column = 26*column+ord(letter.upper())-64
                value = cell.find('s:v', ns)
                if value is None: continue
                kind = cell.attrib.get('t')
                if kind == 's': value = strings[int(value.text)]
                elif kind == 'str': value = value.text
                else:
                    value = float(value.text)
                    if value.is_integer(): value = int(value)
                values[column-1] = value
            if values[0] is not None:
                rows.append({'xlsx_row': int(row.attrib['r']), **dict(zip(FIELDS, values))})
    return rows


def numeric(value):
    return isinstance(value, (int, float)) and np.isfinite(value)


def classify(row, case='combined'):
    """Explicit deposit arithmetic, with no phase attribution for unsplit totals."""
    total = row['total_cm']
    parts = [row[k] for k in ('plinian_cm', 'co_ignimbrite_cm')]
    component_sum = sum(v for v in parts if numeric(v))
    flow = row['turbidite_cm'] if numeric(row['turbidite_cm']) else 0.
    if row['map_id'] == 89:
        return None, None, 'Identical coordinates and total to map ID 88; preserve both source labels but use one location constraint.'
    if case == 'without_pos513_41' and row['map_id'] == 24:
        return None, None, 'Sensitivity: omit workbook/Note 1 sequence discrepancy at POS513-41.'
    if total == '?':
        return None, None, 'Minoan deposit not reached beneath Kolumbo; no thickness or detection limit.'
    if isinstance(total, str) and total.startswith('>'):
        if case == 'exact_only':
            return None, None, 'Sensitivity: omit incomplete-core lower bound.'
        if component_sum <= 0:
            raise ValueError('Incomplete core lacks a documented positive fallout minimum.')
        return 'lower_limit', float(component_sum), 'Incomplete core: sum of recovered fallout components is a lower bound; identified flow excluded.'
    if not numeric(total) or total <= flow:
        raise ValueError(f'Invalid source total at {row["map_id"]}')
    # Both numeric components do not reconcile with the reported total at these sites.
    conflict = all(numeric(v) for v in parts) and not np.isclose(total-flow, component_sum)
    if conflict:
        if case == 'component_conflicts':
            return 'measurement', float(component_sum), 'Sensitivity only: resolve inconsistency using sum of listed fallout components.'
        if case == 'total_conflicts':
            return 'measurement', float(total-flow), 'Sensitivity only: resolve inconsistency using reported total minus identified flow.'
        return None, None, 'Reported total and listed components disagree; excluded pending source clarification.'
    return 'measurement', float(total-flow), 'Reported total minus identified turbidite; no separate fan fit or invented phase allocation.'


def prepare_inputs(write=False):
    if sha256(SOURCE.read_bytes()).hexdigest() != SOURCE_SHA:
        raise ValueError('Source workbook changed; review and version the audit.')
    rows = source_rows()
    if len(rows) != 105 or len({r['map_id'] for r in rows}) != 105:
        raise ValueError('Expected 105 distinct source records.')
    manifest = [dict(row) for row in rows]
    prepared, cases = {}, {}
    for name, (file, description) in CASES.items():
        stream = io.StringIO(newline='')
        writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS, lineterminator='\n'); writer.writeheader()
        selected, counts = [], {'measurement': 0, 'lower_limit': 0}
        for row, record in zip(rows, manifest):
            kind, value, reason = classify(row, name)
            record.update({name+'_kind': kind, name+'_value_cm': value,
                           name+'_csv_row': len(selected)+2 if kind else None, name+'_decision': reason})
            if not kind: continue
            writer.writerow(dict(zip(CSV_FIELDS, [row['longitude'], row['latitude'],
                value if kind == 'measurement' else '', *VENT, kind, value if kind == 'lower_limit' else ''])))
            selected.append(row['map_id']); counts[kind] += 1
        content = stream.getvalue().encode()
        prepared[file] = content
        cases[name] = {'file': file, 'definition': description, 'map_ids': selected,
                       'n_measurements': counts['measurement'], 'n_lower_limits': counts['lower_limit'],
                       'sha256': sha256(content).hexdigest()}
    stream = io.StringIO(newline='')
    writer = csv.DictWriter(stream, fieldnames=list(manifest[0]), lineterminator='\n')
    writer.writeheader(); writer.writerows(manifest)
    prepared['data/audits/Minoan/selection_manifest.csv'] = stream.getvalue().encode()
    for file, content in prepared.items():
        path = ROOT/file
        if write:
            path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes(content)
        elif path.read_bytes() != content:
            raise ValueError(f'Audited input does not reproduce exactly: {file}')
    data, _ = read_observations(ROOT/CASES['combined'][0])
    # Same rectangle for every sensitivity: primary-data bounding box + 25%, rounded outward to 25 km.
    sites = np.vstack((data.xy, [0., 0.]))
    lo, hi = sites.min(axis=0), sites.max(axis=0)
    span = hi - lo
    padding = .25 * np.maximum(span, max(span.max(), 1.) * .05)
    bounds = np.r_[lo - padding, hi + padding]
    bounds = np.r_[np.floor(bounds[:2]/25)*25, np.ceil(bounds[2:]/25)*25].tolist()
    provenance = {'audit_date': '2026-11-03', 'source': {'file': str(SOURCE.relative_to(ROOT)),
        'sha256': SOURCE_SHA, 'sheet': 'Tabelle1', 'rows': 105,
        'article': 'https://doi.org/10.1038/s41467-023-38176-3',
        'download': 'https://static-content.springer.com/esm/art%3A10.1038%2Fs41467-023-38176-3/MediaObjects/41467_2023_38176_MOESM5_ESM.xlsx'},
        'cases': cases, 'bounds_km': bounds, 'bounds_rule': 'Primary observation/source rectangle padded by 25%, rounded outward to 25 km; fixed across sensitivity cases.',
        'vent': {'longitude': VENT[0], 'latitude': VENT[1], 'source': 'https://volcano.si.edu/volcano.cfm?vn=212040',
                 'meaning': 'GVP Santorini reference, not a uniquely known Minoan vent.'},
        'selection_manifest_sha256': sha256(prepared['data/audits/Minoan/selection_manifest.csv']).hexdigest(),
        'preserved_semantics': ['Fit a single total-fallout field over all fans; no individual fan volumes are fitted or added.',
            'Subtract numeric identified turbidite thickness; a blank flow cell means none identified in this compilation, not independently proved absence.',
            'Incomplete cores contribute lower bounds from recovered Plinian plus co-ignimbrite material; limits are not exact observations.',
            'Unpartitioned numeric totals remain totals; blank component cells are not assigned fabricated phase thicknesses.',
            'Map IDs 83 and 85 conflict internally; keep separate component-sum and total-column sensitivity cases.',
            'Map IDs 88 and 89 share exact coordinates/thickness; one location constraint, with both source records retained.',
            'Map ID 24 (POS513-41) is 32 cm in the workbook, while Supplementary Note 1 describes a 66 cm sequence; preserve the workbook, test omission.',
            'Black Sea 0.1 cm entries are used as published nominal thicknesses; original publications were not independently re-audited.',
            '2021 PANGAEA catalogue is a cross-check, not extra observations or an override of the 2023 revised data.']}
    if write:
        (AUDIT/'provenance.json').write_text(json.dumps(provenance, indent=2, allow_nan=False)+'\n')
    elif json.loads((AUDIT/'provenance.json').read_text()) != provenance:
        raise ValueError('Provenance changed; rebuild and review the audit.')
    return provenance


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare-only", action="store_true", help="Rebuild the reviewed input tables and provenance.")
    args = parser.parse_args(argv)
    provenance = prepare_inputs(write=args.prepare_only)
    print(f"Verified {provenance['source']['rows']} Minoan source rows and {len(provenance['cases'])} reviewed selections.")


if __name__ == "__main__":
    main()
