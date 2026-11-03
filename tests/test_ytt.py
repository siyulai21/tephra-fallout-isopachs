"""Protect the audited input chain and dataset-specific analysis defaults."""
import json
from pathlib import Path
import shutil

import pytest

from benchmarks.ytt import verify_inputs
from src.probabilistic_isopach.catalog import dataset_record, resolve_dataset
from src.probabilistic_isopach.cli import main


ROOT = Path(__file__).resolve().parents[1]


def test_audited_selections_reproduce_exactly_and_detect_modified_input(tmp_path):
    result = verify_inputs()
    assert result['audit_records'] == 88
    assert [case['measurements'] for case in result['cases'].values()] == [43, 38, 45]
    shutil.copytree(ROOT/'data', tmp_path/'data')
    path = tmp_path/'data/eruptions/YTT_tephra.csv'
    path.write_text(path.read_text().replace(',measurement', ',measurement ', 1))
    with pytest.raises(ValueError, match='does not reproduce'):
        verify_inputs(tmp_path)


def test_catalog_profile_is_not_applied_by_filename(tmp_path):
    source = resolve_dataset('YTT')
    external = tmp_path/source.name
    shutil.copyfile(source, external)
    assert dataset_record(source)['analysis_defaults']['buffer_km'] == 50
    assert dataset_record(external) is None


@pytest.mark.parametrize('overrides,buffer,bounds,extent', [
    ([], 50, [-6900, -1400, 2400, 2900], 8000),
    (['--buffer-km', '0', '--bounds', '-7000', '-1500', '2500', '3000',
      '--max-extent-km', '7500', '--levels', '.1', '1', '10'],
     0, [-7000, -1500, 2500, 3000], 7500),
])
def test_ytt_cli_records_profile_and_honors_explicit_overrides(tmp_path, overrides, buffer, bounds, extent):
    main(['YTT', '--output', str(tmp_path/'run'), '--no-publication',
          '--grid', '8', '--volume-grid', '8', '--draws', '20', *overrides])
    report = json.loads((tmp_path/'run/report.json').read_text())
    settings = report['settings']
    assert settings['method'] == 'selected'
    assert settings['buffer_km'] == buffer
    assert settings['bounds'] == bounds
    assert settings['max_extent_km'] == extent
    assert report['input']['n_measurements'] == 43
    assert report['input']['audit']['source_workbook_sha256'] == verify_inputs()['workbook_sha256']
    assert report['volume']['bounds_km'] == bounds
