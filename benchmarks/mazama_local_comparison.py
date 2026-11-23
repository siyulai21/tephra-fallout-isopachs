"""Summarize saved Mazama source-refit predictions without refitting models."""
import argparse
import csv
import gzip
import io
import json
from pathlib import Path

from benchmarks.mazama import AUDIT, sha, write_csv


def read(path):
    return json.loads(path.read_text())


def collect(root):
    """Read the same source probes and adjacent bearings in every scenario."""
    records, sources = {}, {}
    for folder in sorted(root.iterdir()):
        if not folder.is_dir():
            continue
        regional = folder / 'regional-details.json'
        if regional.exists():
            sources[str(regional.relative_to(root))] = sha(regional)
            for family, record in read(regional)['methods'].items():
                records[folder.name, family] = record
        for family in ('spherical-mixture', 'directional', 'nested-isopachs'):
            path = folder / family / ('local-checks.json' if folder.name == 'principal' else 'report.json')
            if path.exists():
                sources[str(path.relative_to(root))] = sha(path)
                records[folder.name, family] = read(path)['local_checks']
    return records, sources


def summarize(root, output):
    records, sources = collect(root)
    probe_ids = {r['EvidenceID'] for r in csv.DictReader((AUDIT/'model-probes.csv').open())}
    probe_count = len(probe_ids)
    scenarios = {name for name, _ in records}
    assert len(records) == 5*len(scenarios), sorted(records)
    assert 'principal' in scenarios
    site_rows, reach_rows = [], []
    summaries = []
    for (scenario, family), record in sorted(records.items()):
        reference = records['principal', family]
        baseline = {s['evidence_id']: s for s in reference['sites']}
        base_crossings = {(c['evidence_id'], c['bearing_offset_deg'], c['level_cm']): c
                          for c in reference['crossings']}
        assert {s['evidence_id'] for s in record['sites']} == probe_ids
        assert len(record['sites']) == probe_count and len(record['crossings']) == probe_count*12
        changes = []
        for site in record['sites']:
            sid = site['evidence_id']
            q = site['latent_q025_q50_q975_cm']
            bq = baseline[sid]['latent_q025_q50_q975_cm']
            ratio = q[1] / bq[1] if q is not None and bq is not None else None
            row = dict(scenario=scenario, method=family, evidence_id=sid,
                       inside_scenario_domain=site.get('inside_scenario_hull', True),
                       latent_lower95_cm=q[0] if q else '',
                       latent_median_cm=q[1] if q else '',
                       latent_upper95_cm=q[2] if q else '',
                       principal_median_cm=bq[1] if bq else '',
                       median_ratio_to_principal=ratio if ratio is not None else '',
                       median_change_cm=q[1]-bq[1] if ratio is not None else '')
            site_rows.append(row)
            if ratio is not None:
                changes.append(row)
        single_changes = []
        for crossing in record['crossings']:
            key = crossing['evidence_id'], crossing['bearing_offset_deg'], crossing['level_cm']
            base = base_crossings[key]
            current = crossing['crossings_authalic_km']
            previous = base['crossings_authalic_km']
            # Compare a single transition only when both fields have exactly
            # one sampled crossing. Preserve every transition in all cases.
            delta = current[0]-previous[0] if len(current) == len(previous) == 1 else None
            row = dict(scenario=scenario, method=family, evidence_id=key[0],
                       bearing_offset_deg=key[1], level_cm=key[2],
                       crossings_authalic_km=json.dumps(current),
                       principal_crossings_authalic_km=json.dumps(previous),
                       crossing_grid_brackets_authalic_km=json.dumps(crossing['crossing_grid_brackets_authalic_km']),
                       principal_crossing_grid_brackets_authalic_km=json.dumps(base['crossing_grid_brackets_authalic_km']),
                       single_crossing_change_km=delta if delta is not None else '',
                       source_above=crossing.get('source_above', ''),
                       antipode_above=crossing.get('antipode_above', ''),
                       scope='Whole-Earth conditional field' if 'source_above' in crossing else 'Only crossings inside this scenario observation hull; others unavailable')
            reach_rows.append(row)
            if delta is not None:
                single_changes.append(row)
        extreme_site = max(changes, key=lambda r: abs(r['median_ratio_to_principal']-1))
        extreme_reach = max(single_changes, key=lambda r: abs(r['single_crossing_change_km']), default=None)
        summaries.append(dict(scenario=scenario, method=family,
            largest_site_relative_change=extreme_site,
            largest_comparable_single_crossing_change=extreme_reach,
            comparable_sites=len(changes), comparable_single_crossings=len(single_changes)))
    output.mkdir(parents=True, exist_ok=True)
    assert len(site_rows) == len(records)*probe_count and len(reach_rows) == len(records)*probe_count*12
    write_csv(output/'local-thickness-comparison.csv', site_rows)
    with (output/'local-crossing-comparison.csv.gz').open('wb') as raw:
        with gzip.GzipFile(fileobj=raw,mode='wb',mtime=0) as compressed:
            with io.TextIOWrapper(compressed,encoding='utf-8',newline='') as stream:
                writer=csv.DictWriter(stream,fieldnames=list(reach_rows[0]))
                writer.writeheader()
                writer.writerows(reach_rows)
    (output/'local-comparison.json').write_text(json.dumps(dict(
        script_sha256=sha(Path(__file__)), source_files=sources,
        site_rows=len(site_rows), crossing_rows=len(reach_rows), summaries=summaries,
        interpretation=f'Conditional latent-field intervals; no new observations or measured support. Source-refit comparisons are not held-out validation. Largest changes are descriptive maxima across {probe_count} probes and {probe_count*3} rays, not uncertainty intervals. Only two single sampled transitions are differenced. Multiple transitions remain explicit. Regional out-of-domain predictions are unavailable. Numerical resolution is given by both crossing brackets.'
    ), indent=2)+'\n')
    print(f'Summarized {len(site_rows)} site predictions and {len(reach_rows)} contour comparisons')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scenarios', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    summarize(args.scenarios, args.output)
