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
    assert len(rows) == len({r['record_id'] for r in rows}) == 453
    lookup = {r['record_id']: r for r in rows}
    assert sum(r['record_id'].startswith('B2020-') for r in rows) == 293
    chosen = json.loads((HERE / 'model-selection.json').read_text())
    accepted = list(csv.DictReader((ROOT / 'data/eruptions/Mazama_reviewed.csv').open()))
    assert len(chosen) == len(accepted) == 62
    assert sum(r['Kind'] == 'measurement' for r in accepted) == 42
    assert sum(r['Kind'] == 'lower_limit' for r in accepted) == 20
    assert sum(r['accepted_for_model'] == 'True' for r in rows) == 62
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
    assert lookup['B2020-274']['accepted_for_model'] == 'False'
    assert lookup['B2020-028']['accepted_for_model'] == 'False'
    assert lookup['B2020-107']['accepted_for_model'] == 'False'
    assert abs(float(lookup['B2020-257']['latitude']) - (50 + 8/60 + 59/3600)) < 1e-12
    assert float(lookup['B2020-134']['observed_min_cm']) == 1.27
    assert float(lookup['B2020-134']['observed_max_cm']) == 1.905
    for r in rows:
        if r['quantity_kind'] in ('cryptotephra', 'trace_no_numeric_limit', 'tentative_signal'):
            assert not any(r[x] for x in ('observed_point_cm', 'observed_min_cm', 'observed_max_cm', 'bound_cm'))
            assert r['accepted_for_model'] == 'False'
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
    assert sum(int(r['record_count']) for r in queue) == 259
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
                  unique_revised_records=len(rows), additional_records=160, supplied_rows=52,
                  candidate_matches=45, excluded_pseudo_rows=7, principal_observations=62,
                  principal_point_values=42, principal_lower_bounds=20, input_sha256=expected,
                  missing_decimal_coordinates=sum(not r['latitude'] for r in rows),
                  compilation_only_original_records=259, deterministic_rebuild=True,
                  derived_output_sha256=before, published_model_input_checks=models,
                  primary_audit_complete=False,
                  test_and_navigation_results='See case results/source-audit/publication-checks.json; this checker does not invent test results.',
                  visual_source_checks=['Lemke1975 Table4 scan', 'Crandell1967 Table1 p.8', 'Buckley1970 p.101 fractions', 'Yu2006 Fig.2 qualitative only'])
    (HERE / 'validation.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: v for k, v in result.items() if k != 'derived_output_sha256'}, indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--published', action='store_true')
    main(**vars(p.parse_args()))
