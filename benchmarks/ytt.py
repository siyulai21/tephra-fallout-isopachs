"""Reproduce the audited YTT inputs, matched method comparison and reports.

python -m benchmarks.ytt --output results/ytt
No new thickness interpretation or pseudo-observation is introduced here.
"""
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
import pandas as pd
from src.probabilistic_isopach.cli import main as fit
from src.probabilistic_isopach.publication import write_short_report

ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT/'data/audits/YTT'
NS = {'s': 'http://schemas.openxmlformats.org/spreadsheetml/2006/main'}
MD161 = {f'N0{i}' for i in range(21, 26)}
MANINJAU = {'N001', 'N002'}
LEVELS = [.01, .03, .1, .3, 1, 3, 10, 30, 100, 300]
BOUNDS = [-6900, -1400, 2400, 2900]


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


def verify_inputs(root=ROOT):
    """Rebuild all three CSVs in memory and check source/manifest hash chains."""
    audit = root/'data/audits/YTT'
    provenance = json.loads((audit/'provenance.json').read_text())
    workbook = root/provenance['source']['file']
    if sha256(workbook.read_bytes()).hexdigest() != provenance['source']['sha256']:
        raise ValueError('Audit workbook hash changed; a new audit version is required.')
    rows = audit_rows(workbook)
    if len(rows) != 88 or len({r['record_id'] for r in rows}) != 88:
        raise ValueError('Expected 88 distinct audited records.')
    manifest_path = root/provenance['selection_manifest']['file']
    if sha256(manifest_path.read_bytes()).hexdigest() != provenance['selection_manifest']['sha256']:
        raise ValueError('Selection manifest hash changed.')
    manifest = list(csv.DictReader(io.StringIO(manifest_path.read_text())))
    if len(manifest) != len(rows):
        raise ValueError('Manifest and workbook record counts differ.')
    for source, recorded in zip(rows, manifest):
        for key, value in source.items():
            same = (recorded[key] == '' if value is None else
                    float(recorded[key]) == value if isinstance(value, (int, float)) else recorded[key] == value)
            if not same:
                raise ValueError(f'Audit/manifest mismatch at {source["record_id"]}/{key}')
    fields = ['Lon', 'Lat', 'Thickness', 'Vent Lon', 'Vent Lat', 'Kind']
    verified = {}
    for name, case in provenance['cases'].items():
        selected = [r for r in rows if ((r['fit_cm'] is not None and not (name == 'no_md16138' and r['record_id'] in MD161))
                    or (name == 'with_maninjau45' and r['record_id'] in MANINJAU))]
        if [r['record_id'] for r in selected] != case['record_ids']:
            raise ValueError(f'Changed case membership: {name}')
        stream = io.StringIO(newline='')
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator='\n'); writer.writeheader()
        for index, row in enumerate(selected, 2):
            value = row['fit_cm']
            if value is None:
                if name != 'with_maninjau45' or row['record_id'] not in MANINJAU or row['reported_low_cm'] != 23 or row['reported_high_cm'] != 23:
                    raise ValueError('Unapproved thickness substitution')
                value = 23
            writer.writerow(dict(zip(fields, [row['longitude'], row['latitude'], value, 98.8417, 2.6083, 'measurement'])))
            recorded = next(r for r in manifest if r['record_id'] == row['record_id'])
            if int(recorded[f'{name}_csv_row']) != index or float(recorded[f'{name}_thickness_cm']) != value:
                raise ValueError('Manifest/model row mismatch')
        rebuilt = stream.getvalue().encode()
        if sha256(rebuilt).hexdigest() != case['sha256'] or (root/case['file']).read_bytes() != rebuilt:
            raise ValueError(f'Prepared input does not reproduce the audit exactly: {name}')
        verified[name] = {'file': case['file'], 'measurements': len(selected), 'sha256': case['sha256']}
    return {'workbook_sha256': provenance['source']['sha256'], 'audit_records': len(rows), 'cases': verified}


def summarize(output):
    definitions = [('main43_selected', 'main43', 'selected'), ('main43_stacked', 'main43', 'stacked'),
                   ('no_md16138_stacked', 'no_md16138', 'stacked'), ('with_maninjau45_stacked', 'with_maninjau45', 'stacked')]
    reports = {key: json.loads((output/key/'report.json').read_text()) for key, _, _ in definitions}
    manifest = pd.read_csv(AUDIT/'selection_manifest.csv')
    comparisons, sites, folds = [], [], []
    for key, case, method in definitions:
        r = reports[key]; v = r['nested_validation']; regional = r['volume_analysis']['regional']
        comparisons.append({'run': key, 'measurements': r['input']['n_measurements'], 'method': method,
                            **{metric: v[metric] for metric in ('mean_log_score', 'equal_region_log_score', 'rmse_log', 'coverage95_observations', 'mean_interval_width_log')},
                            'expected_regional_km3': regional['mean_km3'], 'median_surface_km3': regional['median_surface_km3'],
                            'adjusted_log_score': v['discrepancy_adjusted']['mean_log_score'],
                            'adjusted_coverage95': v['discrepancy_adjusted']['coverage95_observations']})
        lookup = manifest[manifest[f'included_{case}'] == 1].set_index(f'{case}_csv_row')
        run_sites = []
        for i, rowid in enumerate(v['row_ids']):
            source = lookup.loc[rowid]; observed = float(source[f'{case}_thickness_cm'])
            lo, median, hi = np.exp([v['lower95_log'][i], v['median_log'][i], v['upper95_log'][i]])
            run_sites.append({'run': key, 'record_id': source.record_id, 'site': source.site, 'source_row': rowid,
                              'fold': v['folds'][i], 'observed_cm': observed, 'predicted_median_cm': median,
                              'lower95_cm': lo, 'upper95_cm': hi, 'covered': bool(lo <= observed <= hi),
                              'negative_log_score': -v['point_log_scores'][i]})
        sites.extend(run_sites)
        for label in sorted(set(v['folds'])):
            region = [x for x in run_sites if x['fold'] == label]
            folds.append({'run': key, 'fold': label, 'observations': len(region),
                          'covered': sum(x['covered'] for x in region),
                          'negative_log_score': np.mean([x['negative_log_score'] for x in region])})
    pd.DataFrame(comparisons).to_csv(output/'comparison.csv', index=False)
    pd.DataFrame(sites).to_csv(output/'site_holdouts.csv', index=False)
    pd.DataFrame(folds).to_csv(output/'fold_comparison.csv', index=False)
    a, b = reports['main43_selected'], reports['main43_stacked']
    assert a['nested_validation']['row_ids'] == b['nested_validation']['row_ids']
    assert a['nested_validation']['folds'] == b['nested_validation']['folds']
    same_fit = len(b['model']['components']) == 1 and b['model']['components'][0]['model'] == a['model']
    same_grids = np.allclose(pd.read_csv(output/'main43_selected/predictions.csv')['conditional_mean_cm'],
                            pd.read_csv(output/'main43_stacked/predictions.csv')['conditional_mean_cm'], rtol=1e-12, atol=1e-12)
    av, bv = a['nested_validation'], b['nested_validation']
    if not (bv['mean_log_score'] > av['mean_log_score'] and
            bv['coverage95_observations'] == av['coverage95_observations'] and
            bv['rmse_log'] < av['rmse_log']):
        raise ValueError('YTT comparison changed; reassess the method decision before publishing.')
    decision = {'default': 'selected', 'promote_stacked': False,
                'reason': 'On matched audited YTT holdouts the stacked negative log score is worse, coverage is unchanged, and only log RMSE improves. No clear predictive superiority supports replacement.',
                'main43_same_final_component': same_fit, 'main43_same_expected_thickness_grid': bool(same_grids),
                'negative_log_score_difference_stacked_minus_selected': b['nested_validation']['mean_log_score']-a['nested_validation']['mean_log_score'],
                'criterion': 'Proper predictive score is primary; also inspect coverage, width and point error. A small point-error gain alone does not establish superiority.',
                'input_verification': verify_inputs(),
                'reproduce': 'python -m benchmarks.ytt --output results/ytt'}
    (output/'method_decision.json').write_text(json.dumps(decision, indent=2, allow_nan=False)+'\n')
    return comparisons, decision


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists() and any(args.output.iterdir()):
        parser.error('Choose a new output directory to preserve earlier runs.')
    verify_inputs()
    provenance = json.loads((AUDIT/'provenance.json').read_text())
    cases = provenance['cases']
    for name, case, method in [('main43_selected','main43','selected'), ('main43_stacked','main43','stacked'),
                              ('no_md16138_stacked','no_md16138','stacked'), ('with_maninjau45_stacked','with_maninjau45','stacked')]:
        titles = {'main43_selected': 'YTT · selected', 'main43_stacked': 'YTT · ensemble',
                  'no_md16138_stacked': 'YTT · no MD161', 'with_maninjau45_stacked': 'YTT · with Maninjau'}
        command = [str(ROOT/cases[case]['file']), '--output', str(args.output/name), '--name', titles[name],
                   '--method', method, '--nested-cv', '--folds', '4', '--buffer-km', '50',
                   '--bounds', *map(str, BOUNDS), '--grid', '128', '--map-grid', '481',
                   '--volume-grid', '48', '--draws', '1500', '--seed', '1729', '--max-extent-km', '8000',
                   '--levels', *map(str, LEVELS)]
        fit(command)
        report_path = args.output/name/'report.json'
        report = json.loads(report_path.read_text())
        report['input']['audit'] = {
            'provenance_file': 'data/audits/YTT/provenance.json',
            'provenance_sha256': sha256((AUDIT/'provenance.json').read_bytes()).hexdigest(),
            'source_workbook_sha256': provenance['source']['sha256'],
            'selection': cases[case]['definition'], 'preserved_semantics': provenance['preserved_semantics']}
        report['reproducibility']['benchmark_sha256'] = sha256(Path(__file__).read_bytes()).hexdigest()
        report_path.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
        write_short_report(args.output/name, report['dataset'], report)
    summarize(args.output)


if __name__ == '__main__':
    main()
