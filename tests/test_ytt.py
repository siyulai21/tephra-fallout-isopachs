"""Protect the audited input chain and dataset-specific analysis defaults."""
import json
from hashlib import sha256
from pathlib import Path
import shutil

import pytest

from benchmarks.ytt import verify_inputs
from src.probabilistic_isopach.catalog import dataset_record, resolve_dataset


ROOT = Path(__file__).resolve().parents[1]


def test_audited_selections_reproduce_exactly_and_detect_modified_input(tmp_path):
    result = verify_inputs()
    assert result['audit_records'] == 88
    assert {name: case['measurements'] for name, case in result['cases'].items()} == {
        'principal': 46, 'without_maninjau': 44, 'without_u1445b': 45, 'without_md161': 41, 'with_u1448b': 47}
    shutil.copytree(ROOT/'data', tmp_path/'data')
    path = tmp_path/'data/eruptions/YTT_tephra.csv'
    path.write_text(path.read_text().replace(',measurement', ',measurement ', 1))
    with pytest.raises(ValueError, match='does not reproduce'):
        verify_inputs(tmp_path)


def test_catalog_profile_is_not_applied_by_filename(tmp_path):
    source = resolve_dataset('YTT')
    external = tmp_path/source.name
    shutil.copyfile(source, external)
    assert dataset_record(source)['audit'] == 'audits/YTT/provenance.json'
    assert dataset_record(external) is None


def test_reviewed_principal_uses_one_u1445_hole_and_preserves_source_values():
    import csv
    from benchmarks.ytt_inputs import audit_rows, prepared_files
    audit = ROOT/'data/audits/YTT'
    provenance = json.loads((audit/'provenance.json').read_text())
    cases = provenance['cases']
    principal = set(cases['principal']['record_ids'])
    assert {'N001', 'N002', 'N005', 'N021', 'N022', 'N023', 'N024', 'N025'} <= principal
    assert not {'T022', 'N006', 'T063', 'N020'} & principal
    assert set(cases['with_u1448b']['record_ids']) == principal | {'N006'}
    assert set(cases['without_u1445b']['record_ids']) == principal - {'N005'}
    assert set(cases['without_maninjau']['record_ids']) == principal - {'N001', 'N002'}
    original = {r['record_id']: r for r in audit_rows(audit/'YTT_thickness_audit.xlsx')}
    manifest = {r['record_id']: r for r in csv.DictReader((audit/'selection_manifest.csv').open())}
    assert original['N005']['fit_cm'] is None and manifest['N005']['fit_cm'] == ''
    assert manifest['N005']['principal_thickness_cm'] == '8'
    assert float(manifest['N005']['effective_latitude']) == pytest.approx(17 + 44.7098/60)
    assert float(manifest['N005']['effective_longitude']) == pytest.approx(84 + 47.2498/60)
    assert manifest['N005']['latitude'] == str(original['N005']['latitude'])
    files, _ = prepared_files()
    for name, content in files.items():
        assert (ROOT/name).read_bytes() == content
    # Recorded main45 SHA-256 from the intermediate Maninjau audit linked in history/README.md.
    assert sha256((audit/'sensitivity/without_u1445b.csv').read_bytes()).hexdigest() == (
        "28fecaaf595bf1824c85bd8c70064cea4926108fa7a33b9ff35676727c3f61fb"
    )


def test_review_membership_cannot_drift_from_selection_rules(tmp_path):
    import csv
    shutil.copytree(ROOT/'data', tmp_path/'data')
    path = tmp_path/'data/audits/YTT/record_review.csv'
    with path.open() as f:
        rows = list(csv.DictReader(f))
    next(r for r in rows if r['record_id'] == 'N006')['principal'] = '1'
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator='\n')
        writer.writeheader(); writer.writerows(rows)
    with pytest.raises(ValueError, match='Review/selection disagreement'):
        verify_inputs(tmp_path)
