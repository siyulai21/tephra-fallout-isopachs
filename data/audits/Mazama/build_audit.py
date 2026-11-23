"""Rebuild the Mazama evidence ledger; never changes registered model inputs.
Run from any directory with Python 3 (standard library only).
Scientific judgments live in review-decisions.json and additional-records.json.
"""
import csv
import hashlib
import json
import math
import shutil
import statistics
import xml.etree.ElementTree as ET
import zipfile
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
SOURCES = ROOT / 'data/sources/mazama'
NS = {'m': 'http://schemas.openxmlformats.org/spreadsheetml/2006/main'}


def workbook_rows(path, sheet='sheet1.xml'):
    with zipfile.ZipFile(path) as z:
        shared = []
        if 'xl/sharedStrings.xml' in z.namelist():
            shared = [''.join(e.itertext()) for e in ET.fromstring(z.read('xl/sharedStrings.xml')).findall('m:si', NS)]
        rows = {}
        for r in ET.fromstring(z.read('xl/worksheets/' + sheet)).findall('.//m:row', NS):
            values = {}
            for c in r.findall('m:c', NS):
                col = ''.join(x for x in c.attrib['r'] if x.isalpha())
                v = c.find('m:v', NS)
                val = v.text if v is not None else None
                if c.get('t') == 's' and val is not None:
                    val = shared[int(val)]
                elif c.get('t') == 'inlineStr':
                    val = ''.join(c.find('m:is', NS).itertext())
                elif val is not None:
                    try:
                        val = float(val)
                        if val.is_integer():
                            val = int(val)
                    except ValueError:
                        pass
                values[col] = val
            rows[int(r.attrib['r'])] = values
        return rows


def write_csv(name, rows):
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with (HERE / name).open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def distance(lat1, lon1, lat2, lon2):
    p1, p2 = map(math.radians, (lat1, lat2))
    a = math.sin((p2-p1)/2)**2 + math.cos(p1)*math.cos(p2)*math.sin(math.radians(lon2-lon1)/2)**2
    return 6371.0088 * 2 * math.asin(min(1, math.sqrt(a)))


def write_register(records):
    def cell(value):
        return str(value or '').replace('|', '\\|').replace('\n', ' ').replace('<', '&lt;')

    lines = ['# Mazama locality register', '',
             '[Assessment](README.md) · [Full evidence ledger](revised-locality-ledger.csv) · [Field definitions](DATA_DICTIONARY.md)', '',
             'This index includes every ledger record. Values describe the stated quantity, not automatically original fallout. The [principal selection](model-selection.csv) identifies the conditional fitting observations; all other records remain in the evidence inventory. The CSV retains ages, identity, native locations, full caveats and original compilation notes. Unknown thickness is not zero. Coordinates are inherited or source-qualified, not uniformly surveyed.', '']
    for start in range(0, len(records), 50):
        group = records[start:start+50]
        lines += [f'## Records {start+1}–{start+len(group)}', '',
                  '| ID / locality | Latitude, longitude | Quantity (cm) | Verification / decision | Reference / evidence locator |',
                  '|---|---|---|---|---|']
        for r in group:
            quantities = []
            if r.get('observed_point_cm', '') != '': quantities.append(str(r['observed_point_cm']))
            if r.get('observed_min_cm', '') != '' or r.get('observed_max_cm', '') != '':
                quantities.append(f"{r.get('observed_min_cm', '')}–{r.get('observed_max_cm', '')}")
            if r.get('bound_cm', '') != '': quantities.append(f"{r['bound_relation']}{r['bound_cm']}")
            q = '; '.join(quantities) or 'not quantified'
            xy = f"{r['latitude']}, {r['longitude']}" if r.get('latitude', '') != '' else 'not recovered; see native location in CSV'
            values = [f"{r['record_id']} — {r['locality']}", xy,
                      f"{q}; {r['quantity_kind'].replace('_', ' ')}",
                      r['verification'].replace('_', ' ') + '; ' + r['fallout_model_status'].replace('_', ' ') + ('; PRINCIPAL '+r['principal_kind'].replace('_',' ') if r['accepted_for_model'] else '; omitted'),
                      str(r.get('primary_reference', '')) + '; ' + str(r.get('evidence_locator', ''))]
            lines.append('| ' + ' | '.join(cell(v) for v in values) + ' |')
        lines.append('')
    (HERE/'LOCALITIES.md').write_text('\n'.join(lines).rstrip()+'\n')


def main():
    cols = ['buckland_id','locality','references_raw','latitude_raw','longitude_raw','state_raw','thickness_cm_raw','thickness_metadata_raw','grain_size_um_raw','grain_statistic_raw','grain_notes_raw','sampling_raw','discipline_raw','remobilisation_raw','comments_raw']
    cells = workbook_rows(SOURCES/'buckland2020-esm1.xlsx')
    raw = [dict(zip(cols, [cells[n].get(chr(65+i)) for i in range(15)]), source_excel_row=n) for n in range(13,306)]
    assert [r['buckland_id'] for r in raw] == list(range(1,294))
    write_csv('buckland2020-293-localities.csv', raw)
    (HERE/'buckland2020-cells.json').write_text(json.dumps(raw, ensure_ascii=False, indent=2)+'\n')
    decisions = json.loads((HERE/'review-decisions.json').read_text())
    records = []
    for r in raw:
        i = r['buckland_id']
        numeric = isinstance(r['thickness_cm_raw'], (int,float))
        trace = 'trace' in str(r['thickness_metadata_raw']).lower()
        code = r['remobilisation_raw']
        status = 'hold_primary_verification' if numeric else 'occurrence_only_unquantified'
        if code in ('S','B'):
            status = 'exclude_bulk_fallout_secondary' if code == 'S' else 'hold_primary_component_separation'
        if trace:
            status = 'occurrence_only_no_detection_limit'
        record = {
            'record_id':f'B2020-{i:03d}', 'parent_record_id':'', 'locality':r['locality'],
            'source_excel_row':r['source_excel_row'], 'latitude':r['latitude_raw'], 'longitude':r['longitude_raw'],
            'coordinate_basis':'Buckland2020 S1; datum and site precision not independently established',
            'location_description':'', 'primary_reference':r['references_raw'],
            'evidence_source':'buckland2020-esm1.xlsx', 'evidence_locator':f'Sheet1!A{i+12}:O{i+12}',
            'verification':'compilation_checked_primary_pending',
            'identity':'Mazama attribution inherited from compilation; independent check pending',
            'local_age':'not independently recovered', 'age_basis':'correlation only; no local radiometric age assigned',
            'deposit_class':{'P':'compiler_primary','S':'compiler_secondary','B':'compiler_both'}.get(code,'unknown'),
            'setting':r['sampling_raw'], 'compiler_thickness_cm':r['thickness_cm_raw'],
            'quantity_kind':'trace_no_numeric_limit' if trace else ('compilation_point' if numeric else 'unquantified'),
            'observed_point_cm':r['thickness_cm_raw'] if numeric and not trace else '',
            'observed_min_cm':'', 'observed_max_cm':'', 'bound_relation':'', 'bound_cm':'',
            'bound_target':'', 'interval_meaning':'', 'core_depth_or_sample_interval':'',
            'fallout_model_status':status, 'accepted_for_model':False,
            'identity_quality':'not independently assessed', 'coordinate_quality':'compilation only',
            'thickness_quality':'qualitative; no detection limit' if trace else 'primary evidence pending',
            'caveats':'Compiler P/S/B is not an independent verification. No preservation correction or measurement uncertainty invented.',
            'source_thickness_notes':r['thickness_metadata_raw'], 'source_description':r['comments_raw'],
        }
        if 'pers comm' in str(r['references_raw']).lower():
            record['caveats'] += ' Personal communication cited; original field record not available in this audit.'
        record.update(decisions.get(str(i), {}))
        records.append(record)
    records.extend(json.loads((HERE/'additional-records.json').read_text()))
    assert len({r['record_id'] for r in records}) == len(records)
    for r in records:
        assert r.get('accepted_for_model') is False
        if r.get('quantity_kind') in ('cryptotephra','trace_no_numeric_limit','tentative_signal'):
            assert all(r.get(x,'') == '' for x in ['observed_point_cm','observed_min_cm','observed_max_cm'])
        if r.get('bound_cm','') != '':
            assert r.get('bound_target') and r.get('bound_relation') in ('<','<=','>','>=')
        for key in ('latitude','longitude'):
            if r.get(key,'') != '': assert isinstance(r[key],(float,int))
    selection = {r['record_id']: r for r in json.loads((HERE/'model-selection.json').read_text())}
    assert set(selection) <= {r['record_id'] for r in records}
    for r in records:
        item = selection.get(r['record_id'], {})
        r['accepted_for_model'] = bool(item)
        r['principal_kind'] = item.get('kind', 'omitted')
        r['principal_thickness_cm'] = item.get('thickness_cm', '')
        r['principal_limit_cm'] = item.get('limit_cm', '')
        r['principal_decision'] = item.get('decision_basis', '')
        r['principal_assumption'] = item.get('preservation_assumption', '')
    write_csv('revised-locality-ledger.csv', records)
    write_register(records)
    write_csv('additional-localities.csv', records[293:])
    write_csv('bounds-and-ranges.csv', [r for r in records if r.get('observed_min_cm','') != '' or r.get('bound_cm','') != ''])
    write_csv('occurrences-and-traces.csv', [r for r in records if r.get('quantity_kind') in ('unquantified','cryptotephra','trace_no_numeric_limit','tentative_signal')])
    original = ROOT/'data/eruptions/Mazama_tephra.csv'
    snapshot = HERE/'Mazama_supplied_52rows.csv'
    if snapshot.exists(): assert snapshot.read_bytes() == original.read_bytes(), 'Supplied input changed; review explicitly'
    else: shutil.copyfile(original,snapshot)
    supplied = list(csv.DictReader(original.open()))
    matches = [291,42,229,210,160,134,274,174,126,52,165,23,121,107,119,104,99,72,169,149,227,205,264,148,54,177,59,122,120,63,94,73,76,49,77,53,113,47,58,46,38,39,33,26,28]
    it = iter(matches); crosswalk=[]
    for n,p in enumerate(supplied,1):
        row={'project_data_row':n,'project_csv_line':n+1,'project_latitude':p['Lat'],'project_longitude':p['Lon'],'project_thickness_cm':p['Thickness'],'pseudo':p['Pseudo']}
        if p['Pseudo'].lower() == 'yes':
            row.update(record_id='',match_basis='supplied invented constraint; excluded',review_status='exclude_pseudo',coordinate_displacement_km='')
        else:
            i=next(it); s=raw[i-1]
            row.update(record_id=f'B2020-{i:03d}',locality=s['locality'],source_excel_row=s['source_excel_row'],source_latitude=s['latitude_raw'],source_longitude=s['longitude_raw'],source_thickness_cm=s['thickness_cm_raw'],thickness_delta_cm=float(p['Thickness'])-s['thickness_cm_raw'],coordinate_displacement_km=round(distance(float(p['Lat']),float(p['Lon']),s['latitude_raw'],s['longitude_raw']),6),match_basis='candidate correspondence from spatial proximity, thickness and compilation primary selection; not original digitisation provenance',review_status=records[i-1]['fallout_model_status'])
            if i==169:row['match_basis']+='; ID168 shares coordinates but lacks thickness; ID169 supplies 8 cm'
            if i==205:row['match_basis']+='; nearer ID206 lacks thickness; ID205 supplies 10 cm'
        crosswalk.append(row)
    assert len(crosswalk)==52 and len([x for x in crosswalk if x['record_id']])==45
    write_csv('supplied-row-crosswalk.csv',crosswalk)
    groups = {}
    legacy = {r['record_id'] for r in crosswalk if r['record_id']}
    for r in records[:293]:
        if r['verification'] == 'compilation_checked_primary_pending':
            groups.setdefault(r['primary_reference'], []).append(r['record_id'])
    write_csv('primary-source-work-queue.csv', [
        dict(source_citation=citation, record_ids=';'.join(ids),
             legacy_input_record_ids=';'.join(i for i in ids if i in legacy),
             principal_input_record_ids=';'.join(i for i in ids if i in selection),
             status='not independently verified; not necessarily inaccessible',
             priority='principal input' if any(i in selection for i in ids) else ('legacy input' if any(i in legacy for i in ids) else 'catalogue'),
             record_count=len(ids)) for citation, ids in sorted(groups.items())])
    later=workbook_rows(SOURCES/'buckland2022-esm3.xlsx','sheet2.xml')
    comparison=[]
    for n,d in later.items():
        if isinstance(d.get('A'),int):
            i=d['A']; comparison.append({'buckland_id':i,'source_2022_excel_row':n,'thickness_2020_cm':raw[i-1]['thickness_cm_raw'],'thickness_2022_cm':d.get('F'),'note':'Later model input, not an independent field measurement'})
    write_csv('buckland2022-input-comparison.csv',comparison)
    with zipfile.ZipFile(SOURCES/'buckland2020-esm2.docx') as z:
        doc=ET.fromstring(z.read('word/document.xml'))
        ns={'w':'http://schemas.openxmlformats.org/wordprocessingml/2006/main'}
        paras=[''.join(t.itertext()) for t in doc.findall('.//w:p',ns)]
        refs=paras[paras.index('References')+1:]
    write_csv('compilation-bibliography.csv',[{'reference_number':n,'citation':p,'verification':'citation transcription only; see locality ledger'} for n,p in enumerate([p for p in refs if p.strip()],1)])
    shifts=[r['coordinate_displacement_km'] for r in crosswalk if r['record_id']]
    summary={'audit_date':'2026-11-23','source_localities':293,'additional_records':len(records)-293,'total_records':len(records),'supplied_rows':52,'supplied_nonpseudo':45,'supplied_pseudo':7,'crosswalk_status':'45 candidate matches; original extraction provenance unavailable','coordinate_displacement_km':{'min':min(shifts),'median':statistics.median(shifts),'max':max(shifts)},'verification_counts':dict(Counter(r['verification'] for r in records)),'decision_counts':dict(Counter(r['fallout_model_status'] for r in records)),'quantity_counts':dict(Counter(r['quantity_kind'] for r in records)),'builder_modifies_registered_input':False,'builder_runs_models':False,'primary_audit_complete':False,'limitations':'All 293 compilation records reconciled; independent primary verification remains incomplete. Bounds concern preserved deposits unless explicitly qualified otherwise.'}
    summary.update(registered_input_modified=True, principal_observations=len(selection),
                   principal_kinds=dict(Counter(r['kind'] for r in selection.values())),
                   fitting_input='data/eruptions/Mazama_reviewed.csv',
                   new_model_run='See case results and validation record; this builder only prepares evidence')
    (HERE/'audit-summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2))

if __name__=='__main__': main()
