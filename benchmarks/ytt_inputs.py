"""Deterministically prepare reviewed YTT inputs from the unchanged source audit."""
import csv
from hashlib import sha256
import io
import json
from pathlib import Path
import posixpath
import xml.etree.ElementTree as ET
import zipfile

ROOT = Path(__file__).resolve().parents[1]
NS = {'s': 'http://schemas.openxmlformats.org/spreadsheetml/2006/main'}
FIELDS = ['Lon', 'Lat', 'Thickness', 'Vent Lon', 'Vent Lat', 'Kind']


def audit_rows(path):
    """Read the source's static occurrence values without modifying the XLSX."""
    with zipfile.ZipFile(path) as z:
        strings = []
        if 'xl/sharedStrings.xml' in z.namelist():
            strings = [''.join(t.text or '' for t in item.findall('.//s:t', NS))
                       for item in ET.fromstring(z.read('xl/sharedStrings.xml'))]
        sheets = ET.fromstring(z.read('xl/workbook.xml')).findall('s:sheets/s:sheet', NS)
        sheet = next(s for s in sheets if s.attrib['name'] == 'Occurrence audit')
        rid = sheet.attrib['{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id']
        target = next(r.attrib['Target'] for r in ET.fromstring(z.read('xl/_rels/workbook.xml.rels')) if r.attrib['Id'] == rid)
        target = target.lstrip('/') if target.startswith('/') else posixpath.normpath(posixpath.join('xl', target))
        values = []
        for row in ET.fromstring(z.read(target)).findall('s:sheetData/s:row', NS):
            cells = {}
            for cell in row.findall('s:c', NS):
                if cell.find('s:f', NS) is not None:
                    raise ValueError('Audit contains a formula; review before reading cached values.')
                index = 0
                for char in cell.attrib['r']:
                    if not char.isalpha():
                        break
                    index = index*26+ord(char.upper())-64
                value = cell.find('s:v', NS)
                kind = cell.attrib.get('t')
                if kind == 'inlineStr':
                    value = ''.join(t.text or '' for t in cell.findall('.//s:t', NS))
                elif value is None or value.text is None:
                    value = None
                elif kind == 's':
                    value = strings[int(value.text)]
                elif kind in ('str', 'e'):
                    value = value.text
                else:
                    number = float(value.text)
                    value = int(number) if number.is_integer() else number
                cells[index-1] = value
            values.append(cells)
    headers = [values[0].get(i) for i in range(max(values[0])+1)]
    return [{key: r.get(i) for i, key in enumerate(headers)} for r in values[1:] if r.get(0)]


def csv_bytes(rows, fields):
    stream = io.StringIO(newline='')
    writer = csv.DictWriter(stream, fieldnames=fields, lineterminator='\n')
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().encode()


def prepared_files(root=ROOT):
    """Return expected generated files; never infer precision or interpolate ranges."""
    audit = root/'data/audits/YTT'
    rules_path = audit/'selection_rules.json'
    rules = json.loads(rules_path.read_text())
    workbook = audit/'YTT_thickness_audit.xlsx'
    if sha256(workbook.read_bytes()).hexdigest() != rules['source_sha256']:
        raise ValueError('Audit workbook hash changed; a new audit version is required.')
    rows = audit_rows(workbook)
    by_id = {r['record_id']: r for r in rows}
    if len(rows) != 88 or len(by_id) != 88:
        raise ValueError('Expected 88 distinct audited records.')
    base = {r['record_id'] for r in rows if r['fit_cm'] is not None}
    overrides = rules['overrides']
    for record_id, override in overrides.items():
        source = by_id[record_id]
        if not (source['reported_low_cm'] == source['reported_high_cm'] == override['thickness_cm']):
            raise ValueError(f'Override is not a reported point thickness: {record_id}')
    principal = base | {k for k, v in overrides.items() if v['principal']}
    review_path = audit/'record_review.csv'
    review = list(csv.DictReader(io.StringIO(review_path.read_text())))
    if [r['record_id'] for r in review] != list(by_id):
        raise ValueError('Review must cover all 88 records in source order.')
    for r in review:
        if (r['principal'] == '1') != (r['record_id'] in principal):
            raise ValueError(f'Review/selection disagreement: {r["record_id"]}')
    manifest = [dict(r) for r in rows]
    for r in manifest:
        override = overrides.get(r['record_id'], {})
        r['effective_longitude'] = override.get('longitude', r['longitude'])
        r['effective_latitude'] = override.get('latitude', r['latitude'])
        r['review_decision'] = next(x['decision'] for x in review if x['record_id'] == r['record_id'])
    files, cases = {}, {}
    for name, definition in rules['cases'].items():
        selected = (principal - set(definition['exclude'])) | set(definition['add'])
        if not selected <= set(by_id):
            raise ValueError(f'Unknown record in {name}')
        model_rows, ids = [], []
        for r in manifest:
            record_id = r['record_id']
            included = record_id in selected
            r[f'included_{name}'] = int(included)
            r[f'{name}_csv_row'] = len(model_rows)+2 if included else ''
            value = overrides[record_id]['thickness_cm'] if record_id in overrides else r['fit_cm']
            r[f'{name}_thickness_cm'] = value if included else ''
            if included:
                if value is None or r['effective_longitude'] is None or r['effective_latitude'] is None:
                    raise ValueError(f'No measured thickness or location for {record_id}')
                ids.append(record_id)
                model_rows.append(dict(zip(FIELDS, [r['effective_longitude'], r['effective_latitude'], value, 98.8417, 2.6083, 'measurement'])))
        content = csv_bytes(model_rows, FIELDS)
        files[definition['file']] = content
        cases[name] = {'file': definition['file'], 'sha256': sha256(content).hexdigest(),
                       'n_measurements': len(ids), 'definition': definition['definition'], 'record_ids': ids,
                       'overrides': {k: v for k, v in overrides.items() if k in selected}}
    manifest_file = 'data/audits/YTT/selection_manifest.csv'
    files[manifest_file] = csv_bytes(manifest, list(manifest[0]))
    original = json.loads((audit/'history/2026-11-02/provenance.json').read_text())
    provenance = {k: original[k] for k in ('audit_date', 'source', 'vent', 'input_contract', 'range_handling', 'spatial_dependence', 'import_verification')}
    provenance.update({
        'review_date': rules['review_date'], 'principal_case': rules['principal_case'],
        'selection_rules': {'file': str(rules_path.relative_to(root)), 'sha256': sha256(rules_path.read_bytes()).hexdigest()},
        'record_review': {'file': str(review_path.relative_to(root)), 'sha256': sha256(review_path.read_bytes()).hexdigest(), 'n_rows': 88},
        'preparation_script': {'file': 'benchmarks/ytt_inputs.py', 'sha256': sha256(Path(__file__).read_bytes()).hexdigest()},
        'selection_manifest': {'file': manifest_file, 'sha256': sha256(files[manifest_file]).hexdigest(), 'n_rows': 88,
                               'csv_row_definition': '1-based physical CSV row including header; source order retained.'},
        'cases': cases,
        'preserved_semantics': [
            'The original workbook and all 17 source columns remain unchanged.',
            'Principal overrides: N001/N002 approximately 23 cm; N005 8 cm with verified hole coordinates.',
            'N006 20 cm is an inclusion sensitivity only; no distribution is inferred from conflicting contacts.',
            'Chala is gross preserved-deposit thickness, not solid-glass equivalent.',
            'RC14-37 uses observed 15 cm, not reconstructed approximately 20 cm.',
            'Malawi assumed thickness, Huguangyan hypothetical thickness and other held records are not constraints.',
            'Approximate coordinates and preservation uncertainty are not propagated into model intervals.'
        ],
        'evidence_scope': 'All 88 audit records screened. Targeted new primary-source checks: Kutterolf et al. (2023) Table 1 pp.18,20 and IODP U1445/U1448 reports. Other source assessments are inherited unless marked in record_review.csv.',
        'history': {'directory': 'data/audits/YTT/history', 'note': 'Historical provenance files retain paths used at the time; see history/README.md for relocated files.'}
    })
    files['data/audits/YTT/provenance.json'] = (json.dumps(provenance, indent=2, allow_nan=False)+'\n').encode()
    return files, provenance


def prepare_inputs(root=ROOT):
    files, provenance = prepared_files(root)
    for name, content in files.items():
        path = root/name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    return provenance


def verify_inputs(root=ROOT):
    files, provenance = prepared_files(root)
    for name, content in files.items():
        if (root/name).read_bytes() != content:
            raise ValueError(f'Prepared input does not reproduce the reviewed audit exactly: {name}')
    # Historical fixtures stay independently tied to the original recorded checksums.
    audit = root/'data/audits/YTT'
    old = json.loads((audit/'history/2026-11-02/provenance.json').read_text())
    historical = {'main43': 'without_maninjau_43.csv', 'no_md16138': 'no_md161_38.csv', 'with_maninjau45': 'with_maninjau_45.csv'}
    for case, name in historical.items():
        if sha256((audit/'history/2026-11-02'/name).read_bytes()).hexdigest() != old['cases'][case]['sha256']:
            raise ValueError(f'Historical input changed: {case}')
    return {'workbook_sha256': provenance['source']['sha256'], 'audit_records': 88,
            'cases': {name: {'file': case['file'], 'measurements': case['n_measurements'], 'sha256': case['sha256']}
                      for name, case in provenance['cases'].items()}}
