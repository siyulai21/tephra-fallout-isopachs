"""Verify the second 946 CE field audit and its source-to-fit translation without fitting models."""
from collections import Counter
import csv
from hashlib import sha256
import json
from pathlib import Path

from .minimum_fit_inputs import verify as verify_source_translation
from .publication import result_home
from src.empirical_isopach.core.audit import audited_data
from src.empirical_isopach.core.catalog import dataset_record
from src.empirical_isopach.models.directional.archive_readings import load_inventory

ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT / 'data/audits/B-Tm/two-phase'


def rows(path):
    with path.open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def verify():
    manifest = json.loads((AUDIT/'provenance.json').read_text())
    record = dataset_record(ROOT / manifest['fitting_table']['file'])
    case = manifest['cases']['principal']
    raw_path = ROOT / case['file']
    assert case['sha256'] == digest(raw_path)
    assert record['path'] == ROOT / manifest['fitting_table']['file']
    assert digest(record['path']) == manifest['fitting_table']['sha256']
    evidence = rows(AUDIT/'sites.csv')
    sites = {s['id']: s for s in evidence}
    raw = rows(raw_path)
    selected = {r['EvidenceID']: r for r in raw}
    assert len(sites) == len(evidence) == manifest['evidence_counts']['field_ledger']
    assert all(s['source'] and s['locator'] and s['decision'] for s in evidence)
    assert all(not s['principal_id'] or s['principal_id'] in selected for s in evidence)
    assert len(raw) == len(selected) == len({(float(r['Lat']), float(r['Lon'])) for r in raw})
    assert {s['id'] for s in evidence if s['kind']} == set(selected)
    for sid, row in selected.items():
        s = sites[sid]
        assert (row['Lat'], row['Lon'], row['Kind'], row['Thickness']) == (
            s['latitude'], s['longitude'], s['kind'], s['thickness_cm'])
        if s['kind'] == 'interval':
            assert (row['Lower_cm'], row['Upper_cm']) == (s['lower_cm'], s['upper_cm'])
            assert 0 < float(row['Lower_cm']) < float(row['Upper_cm'])
        elif s['kind'].endswith('_limit'):
            assert row['Limit_cm'] == s['lower_cm' if s['kind'] == 'lower_limit' else 'upper_cm']
    assert Counter(r['Kind'] for r in raw) == dict(measurement=75, lower_limit=34, upper_limit=13, interval=28)
    assert verify_source_translation(record) == 34
    data, meta = audited_data(record['path'])
    assert len(data.row_ids) == 150 and len(meta['interpreted_upper_limits']) == 12
    probes, probe_path = load_inventory(record['path'])
    assert probe_path == AUDIT / 'archive-evidence.json'
    assert {s['id']:s['principal_effective_upper_cm'] for s in probes['sites'] if s.get('in_principal_as')} == {
        s['id']:s['limit_cm'] for s in meta['interpreted_upper_limits']}
    old = rows(ROOT/'data/eruptions/B-Tm_cryptotephra.csv')
    assert {r['EvidenceID'] for r in old} <= selected.keys()
    old_sites = rows(AUDIT.parent/'evidence.csv') + rows(AUDIT.parent/'additional-evidence.csv')
    assert {s['id'] for s in old_sites} <= sites.keys()
    for table, key, ix in [('costa-crosswalk','costa_id',{f'P{i:02d}' for i in range(1,84)}),
                           ('yang-crosswalk','source_sheet_row',{str(i) for i in range(2,78)}),
                           ('nakanishi-survey','site_number',{str(i) for i in range(1,43)})]:
        cross = rows(AUDIT/(table+'.csv'))
        assert len(cross) == len(ix) and {r[key] for r in cross} == ix
        assert all(r.get('second_review_id', r.get('evidence_id')) in sites for r in cross)
        assert all(not r['principal_id'] or r['principal_id'] in selected for r in cross)
    dates = rows(AUDIT/'sun-dates.csv')
    assert len(dates) == 10 and len({r['lab_or_sample'] for r in dates}) == 10
    assert all(r['sun_original_assignment'] for r in rows(AUDIT/'sun-units.csv'))
    assert not sites['sun-1702-document']['principal_id']
    checks = rows(AUDIT/'nakanishi-survey.csv')
    assert all(not r['principal_id'] for r in checks if not r['btm_cell'])
    assert len(rows(AUDIT/'tmu-specimens.csv')) == 39
    for sid, n in [('miy-guan',1500),('miy-west',1200),('sun-NTC',18),('sun-YC',95),('sun-HSG',20),('machida-11',130)]:
        assert selected[sid]['Kind']=='lower_limit' and float(selected[sid]['Limit_cm'])==n
    for sid, n in [('old-002',150),('old-008',26),('tashiro-core',4),('machida-12',135)]:
        assert float(selected[sid]['Thickness']) == n
    for sid, ends in [('miy-X',(80,90)),('miy-E',(385,475)),('sun-DYT',(55,65)),('sun-SMF',(40,50)),('sun-SMFD',(185,205)),('old-061',(1,1.5)),('ru-kit',(0.1,1))]:
        assert tuple(float(selected[sid][k]) for k in ('Lower_cm','Upper_cm')) == ends
    assert sites['pan16-70m']['lower_cm']=='7000' and not sites['pan16-70m']['latitude']
    assert sites['kase-takadai']['principal_id']=='old-047'
    assert selected['costa-P79']['Lon'] != sites['miy-B08']['longitude']
    assert sites['raz-9515']['lower_cm'] == '1' and sites['old-054']['upper_cm'] == '2'
    assert sites['old-070']['upper_cm'] == '1'
    assert selected['costa-P01']['Thickness'] == '9' and selected['costa-P07']['Thickness'] == '10'
    released = dataset_record(ROOT / 'data' / manifest['previous_release']['file'])
    assert digest(released['path']) == manifest['previous_release']['sha256']
    receipt = json.loads((ROOT/result_home(released)/released['release_receipt']).read_text())
    assert released['path'].name in json.dumps(receipt) or digest(released['path']) in json.dumps(receipt)
    for f, expected in manifest['evidence_files'].items():
        assert digest(AUDIT/f) == expected, f
    plates = json.loads((AUDIT/'figures/registration.json').read_text())
    picks = rows(AUDIT/'figure-readings.csv')
    assert len({(r['plate'],r['mark']) for r in picks}) == len(picks)
    for panel, info in plates.items():
        part = [r for r in picks if r['plate']==panel]
        assert len(part)==info['readings'] and all(not r['evidence_id'] or r['evidence_id'] in sites for r in part)
        for pic, check in {panel+'-source.png':info['source_crop_sha256'],panel+'-numbered.jpg':info['annotated_sha256'],**info['details']}.items():
            assert digest(AUDIT/'figures'/pic)==check, pic
    print(f"Verified {len(raw)} combined-fall sites, {len(evidence)} evidence entries and {len(picks)} figure selections.")


if __name__ == '__main__':
    verify()
