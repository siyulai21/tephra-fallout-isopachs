"""Independently check Mazama source cells, decisions, inputs and retained identities."""
import argparse
import csv
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import openpyxl

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
SOURCES = ROOT / 'data/sources/mazama'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def historical_checks(file, revision):
    records = json.loads((HERE / file).read_text())
    for path, expected in records.items():
        content = subprocess.check_output(['git', 'show', f'{revision}:{path}'], cwd=ROOT)
        assert hashlib.sha256(content).hexdigest() == expected, (revision, path)
    return len(records)


def main(published=False):
    manifest = json.loads((SOURCES / 'source-manifest.json').read_text())
    assert len({s['file'] for s in manifest}) == len(manifest)
    checked_payloads, linked_only_missing = [], []
    for s in manifest:
        path = SOURCES / s['file']
        if not path.exists() and not s.get('bundled', True):
            assert s['url'] and len(s['sha256']) == 64 and s['bytes'] > 0
            linked_only_missing.append(s['file'])
            continue
        assert path.stat().st_size == s['bytes'], s['file']
        assert digest(path) == s['sha256'], s['file']
        checked_payloads.append(s['file'])
    historic = historical_checks('historical-input-result-hashes.json', '41b8f2f')
    previous = historical_checks('pre-revision-hashes.json', '0a53d02')
    raw = json.loads((HERE / 'buckland2020-cells.json').read_text())
    wb = openpyxl.load_workbook(SOURCES / 'buckland2020-esm1.xlsx', data_only=True, read_only=True)
    fields = list(raw[0])[:15]
    count = 0
    for record, row in zip(raw, wb.worksheets[0].iter_rows(min_row=13, max_row=305, max_col=15, values_only=True), strict=True):
        for key, value in zip(fields, row, strict=True):
            assert record[key] == value, (record['buckland_id'], key)
            count += 1
    assert count == 4395
    wb.close()
    rows = list(csv.DictReader((HERE / 'revised-locality-ledger.csv').open()))
    assert len(rows) == len({r['record_id'] for r in rows}) == 713
    lookup = {r['record_id']: r for r in rows}
    assert sum(r['record_id'].startswith('B2020-') for r in rows) == 293
    chosen = json.loads((HERE / 'model-selection.json').read_text())
    accepted = list(csv.DictReader((ROOT / 'data/eruptions/Mazama_reviewed.csv').open()))
    assert len(chosen) == len(accepted) == 236
    assert sum(r['Kind'] == 'measurement' for r in accepted) == 57
    assert sum(r['Kind'] == 'lower_limit' for r in accepted) == 152
    assert sum(r['Kind'] == 'upper_limit' for r in accepted) == 27
    assert float(lookup['S2023-DOG-OR']['bound_cm']) == 920 - 908
    assert float(lookup['S2023-WHITEPINE']['bound_cm']) == 79 - 54
    assert float(lookup['S2023-CONNLEY2']['bound_cm']) == 180 - 152
    assert float(lookup['MW1986-SKULL']['bound_cm']) == 5
    assert abs(float(lookup['W2016-LYMAN']['latitude']) - (48 + 11/60 + 30/3600)) < 1e-12
    assert sum(r['accepted_for_model'] == 'True' for r in rows) == 236
    for a, s in zip(accepted, chosen, strict=True):
        assert a['EvidenceID'] == s['record_id']
        assert float(a['Lat']) == float(s['latitude'])
        assert float(a['Lon']) == float(s['longitude'])
        # Moss's catalogue row describes both central and fringe cores. The
        # selection explicitly uses the thesis's central-core coordinate.
        if s['record_id'] == 'B2020-119':
            assert float(a['Lat']) == 47.69325
            assert abs(float(a['Lon']) - (-121-50.81/60)) < 1e-12
        else:
            assert float(a['Lat']) == float(lookup[s['record_id']]['latitude'])
            assert float(a['Lon']) == float(lookup[s['record_id']]['longitude'])
        for key, dest in [('thickness_cm', 'Thickness'), ('limit_cm', 'Limit_cm')]:
            assert str(s[key]) == a[dest]
        assert a['Kind'] == s['kind'] and a['Pseudo'] == 'No' and not a['LogSD']
    sections = list(csv.DictReader((HERE/'young1990-appendix1.csv').open()))
    assert len(sections)==295 and len({r['site'] for r in sections})==138
    c = {int(r['site']): r['upper_pumice_th_cm'] for r in sections if r['appendix_table']=='C'}
    yaccepted=[r for r in accepted if r['EvidenceID'].startswith('Y1990-A')]
    assert len(yaccepted)==103
    for r in yaccepted:
        raw=c[int(r['EvidenceID'].split('-A')[1])]
        assert raw and 'J' not in raw and float(raw.replace('+','').strip())==float(r['Limit_cm'])
        assert r['Kind']=='lower_limit'
    assert 'Y1990-A150' not in {r['EvidenceID'] for r in accepted}
    assert len(list(csv.DictReader((HERE/'young1990-distal.csv').open())))==25
    for sid,x in [('W2015-RM-V2',290-210),('W2015-RM-V3',330-130),('Y1990-C-NEW',3),('C2014-WIL',87),('J2025-NTL',224-143-5),('B2020-229',1),('R1967-BB326-36',4)]:
        assert float(lookup[sid]['bound_cm'])==x,sid
    by_id = {row['EvidenceID']:row for row in accepted}
    for sid, value in {'M2018-OVERLAND':8,'M2018-SOLDIER':4,'J2025-GSL4':1,'W2016-LYMAN':4}.items():
        assert float(by_id[sid]['Thickness']) == value
    mapped = json.loads((HERE/'evidence-map.json').read_text())
    assert mapped['ledger_sha256'] == digest(HERE/'revised-locality-ledger.csv')
    assert mapped['input_sha256'] == digest(ROOT/'data/eruptions/Mazama_reviewed.csv')

    for sid, contacts in {'CH2008-THU':(170,202),'CH2008-WIN':(287,373),'HER2014-DIS':(593,734),
            'HER2014-HOR':(178,389),'HER2014-RRL':(278,439),'ROS2004-FRO':(155,163),
            'ROS2004-EAG':(81,96),'CH2008-RED':(409,520),'SP2007-THU':(320.5,440),
            'SP2008-RID':(166,277),'OS2012-POC':(55,93)}.items():
        row = by_id[sid]
        assert float(row['Thickness'] or row['Limit_cm']) == contacts[1]-contacts[0],sid
    assert not {'SC2020-RS10','SC2020-RS12','SC2020-RS25','OS2012-EAS','SHP2023-RDF2A','SHP2023-RDF3A'} & by_id.keys()
    for sid, mass in [('B2020-251',5),('B2020-203',3.5)]:
        assert float(by_id[sid]['Thickness'])==mass
    for sid, limit in [('B2020-129',456-315),('B2020-141',605-500),('B2020-147',1080),('R1982-KOOT-776',1.5+.5)]:
        assert float(by_id[sid]['Limit_cm'])==limit
    ages=list(csv.DictReader((HERE/'egan-age-crosswalk.csv').open()))
    assert len(ages)==83 and len({r['lab_id'].upper() for r in ages[:81]})==81
    assert all(set(filter(None,r['candidate_ledger_ids'].split('|')))<=lookup.keys() for r in ages)
    assert all(r['age_system']=='ice-core layer-counted BP' for r in ages[-2:])
    assert not {'TX1980-SIMP','R1982-BCV5','B2020-142'} & by_id.keys()
    assert float(by_id['MAR2009-S43']['Thickness'])==3
    assert lookup['B2020-274']['accepted_for_model'] == 'False'
    assert lookup['B2020-028']['accepted_for_model'] == 'False'
    assert lookup['B2020-107']['accepted_for_model'] == 'False'
    assert abs(float(lookup['B2020-257']['latitude']) - (50 + 8/60 + 59/3600)) < 1e-12
    assert float(lookup['B2020-134']['observed_min_cm']) == 1.27
    assert float(lookup['B2020-134']['observed_max_cm']) == 1.905
    for r in rows:
        if r['quantity_kind'] in ('cryptotephra', 'trace_no_numeric_limit', 'tentative_signal'):
            assert not any(r[x] for x in ('observed_point_cm', 'observed_min_cm', 'observed_max_cm'))
            if r['accepted_for_model'] == 'True':
                assert r['bound_target'] and r['principal_kind'] in ('lower_limit','upper_limit')
        if r['latitude']:
            assert -90 <= float(r['latitude']) <= 90 and -180 <= float(r['longitude']) <= 180
        if r['observed_min_cm'] and r['observed_max_cm']:
            assert float(r['observed_min_cm']) <= float(r['observed_max_cm'])
        if r['bound_cm']:
            assert r['bound_target'] and r['bound_relation'] in ('<', '<=', '>', '>=')
    crosswalk = list(csv.DictReader((HERE / 'supplied-row-crosswalk.csv').open()))
    assert len(crosswalk) == 52 and sum(bool(r['record_id']) for r in crosswalk) == 45
    assert sum(r['pseudo'].lower() == 'yes' for r in crosswalk) == 7
    assert all(int(r['project_csv_line']) == int(r['project_data_row']) + 1 for r in crosswalk)
    assert (HERE / 'Mazama_supplied_52rows.csv').read_bytes() == (ROOT / 'data/eruptions/Mazama_tephra.csv').read_bytes()
    provenance = json.loads((HERE / 'provenance.json').read_text())
    expected = provenance['cases']['principal']['sha256']
    assert digest(ROOT / 'data/eruptions/Mazama_reviewed.csv') == expected
    for f, sha in provenance['audit_artifacts'].items():
        assert digest(HERE / f) == sha, f
    registry = json.loads((ROOT / 'data/datasets.json').read_text())['datasets']
    assert next(r for r in registry if r['slug'] == 'mazama')['file'] == 'eruptions/Mazama_reviewed.csv'
    outputs = ['revised-locality-ledger.csv', 'buckland2020-cells.json', 'buckland2020-293-localities.csv', 'additional-localities.csv', 'bounds-and-ranges.csv', 'occurrences-and-traces.csv', 'supplied-row-crosswalk.csv', 'buckland2022-input-comparison.csv', 'compilation-bibliography.csv', 'audit-summary.json', 'LOCALITIES.md', 'primary-source-work-queue.csv']
    before = {name: digest(HERE / name) for name in outputs}
    subprocess.run([sys.executable, str(HERE / 'build_audit.py')], cwd=ROOT, check=True, capture_output=True)
    assert before == {name: digest(HERE / name) for name in outputs}
    queue = list(csv.DictReader((HERE / 'primary-source-work-queue.csv').open()))
    queued = {i for r in queue for i in r['record_ids'].split(';')}
    assert queued == {r['record_id'] for r in rows[:293] if r['verification'] in ('compilation_checked_primary_pending','young_compiled_table_read_original_observer_pending')}
    assert sum(int(r['record_count']) for r in queue) == len(queued)
    models = []
    if published:
        for family in ('', 'directional/azimuth/', 'spherical-mixture/', 'nested-isopachs/', 'nested-whole-area/'):
            p = ROOT / 'eruptions/mazama/results' / family / 'report.json'
            report = json.loads(p.read_text())
            assert report.get('input', report)['input_sha256'] == expected, family
            models.append(str(p.relative_to(ROOT)))
    result = dict(audit_date='2026-11-23', source_payloads_hash_verified=len(checked_payloads),
                  source_payloads_bundled=sum(s.get('bundled',True) for s in manifest),
                  linked_only_sources_not_locally_reverified=linked_only_missing,
                  historical_revision='41b8f2f', historical_files_hash_verified=historic,
                  previous_github_revision='0a53d02', previous_release_files_hash_verified=previous,
                  independently_compared_source_cells=count, unique_source_records=293,
                  unique_revised_records=len(rows), additional_records=len(rows)-293, supplied_rows=52,
                  candidate_matches=45, excluded_pseudo_rows=7, principal_observations=len(accepted),
                  principal_point_values=sum(r["Kind"]=="measurement" for r in accepted), principal_lower_bounds=sum(r["Kind"]=="lower_limit" for r in accepted), principal_upper_bounds=sum(r["Kind"]=="upper_limit" for r in accepted), input_sha256=expected,
                  missing_decimal_coordinates=sum(not r['latitude'] for r in rows),
                  original_observer_pending_in_queue=len(queued), compilation_only_original_records=sum(r['verification']=='compilation_checked_primary_pending' for r in rows[:293]), deterministic_rebuild=True,
                  derived_output_sha256=before, published_model_input_checks=models,
                  primary_audit_complete=False,
                  test_and_navigation_results='See case results/source-audit/publication-checks.json; this checker does not invent test results.',
                  visual_source_checks=['Lemke1975 Table4 scan', 'Crandell1967 Table1 p.8', 'Buckley1970 p.101 fractions', 'Yu2006 Fig.2 qualitative only', 'Morgan2023 Tables S1-S2 core identities and coordinates', 'Munroe2018 Table1 lake positions', 'LacCore GLAD GSL00 hole4A position sheet', 'Walsh2017 Table1 Rainier lake coordinates', 'Wan1995 printed p.60 core TT063-13', 'Dallimore2008 p.1348 core MD02-2494 coordinates', 'Saban2023 pp.37,66,76,95,99,103: bed boundaries, site positions and conflicting summaries'])
    (HERE / 'validation.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: v for k, v in result.items() if k != 'derived_output_sha256'}, indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--published', action='store_true')
    main(**vars(p.parse_args()))
