"""Update the current catalogue while retaining the original matched development experiment."""
import csv
import gzip
import json

import pandas as pd
from pathlib import Path
from tempfile import TemporaryDirectory

from benchmarks.extent_challenges import run as check_sectors
from benchmarks.extent_spherical_check import run as check_kernels
from benchmarks.mazama import sha, write_csv
from benchmarks.footprint_study import controls
from src.empirical_isopach.core.audit import audited_data
from src.empirical_isopach.core.catalog import dataset_record, resolve_dataset
from src.empirical_isopach.visualization.deposit import FAMILIES, checked_field, read_json, write_json
from src.empirical_isopach.visualization.deposit_field import parent_view
from src.empirical_isopach.visualization.deposit_integrals import within_cells


PAIRED = Path('comparisons/global-extent')


def paired_estimates(home, scratch):
    """Compare both continuations on this version's observations, fields and geographic folds."""
    path = resolve_dataset("Mazama")
    record = dataset_record(path)
    data, meta = audited_data(path)
    checks = {}
    with (PAIRED / 'paired-results.csv').open(newline='') as stream:
        rows = list(csv.DictReader(stream))
    matches = {row['method']: row for row in rows if row['slug'] == 'mazama'}
    if set(matches) != set(FAMILIES):
        raise ValueError('Expected three paired global fits for Mazama')
    for family, row in matches.items():
        folder = controls("mazama",record)[0] if family=="nested-isopachs" else home/family
        extent = read_json(folder / 'extent/report.json')
        deposit = read_json(folder / 'deposit.json')
        if extent['input_sha256'] != meta['input_sha256'] or deposit['input_sha256'] != meta['input_sha256']:
            raise ValueError(f'Mazama input differs for {family}')
        view, _, sources = parent_view(folder, family, data, meta)
        field, quadrature = checked_field(view, data, meta['vent_lonlat'][1])
        parent_volume, _ = within_cells(field)
        checks[family] = dict(input_sha256=meta['input_sha256'], parent_sources=sources,
                              parent_median_map_km3=parent_volume, grid_shape=list(field.log10_cm.shape),
                              independent_parent_integrals=quadrature,
                              finite_spread_report_sha256=sha(folder / 'extent/report.json'))
        geo = extent['validation']['geographic']
        with gzip.open(folder / 'extent/support.json.gz', 'rt') as stream:
            screen = json.load(stream)
        bands = [name for name, policy in screen.get('policies', {}).items() if policy['selected_band']]
        row.update(input_sha256=meta['input_sha256'], accepted=len(data.xy), earlier_km3=parent_volume,
                   earlier_lower95=extent['parent_conditional_interval95_km3'][0],
                   earlier_upper95=extent['parent_conditional_interval95_km3'][1],
                   principal_km3=deposit['median_map_volume_km3'],
                   lower95=extent['conditional_interval95_km3'][0], upper95=extent['conditional_interval95_km3'][1],
                   parent_matched_negative_log_score=geo['parent_matched_score'],
                   principal_negative_log_score=geo['negative_log_score'],
                   parent_coverage95=geo['matched_parent_coverage95'], principal_coverage95=geo['coverage95_exact'],
                   passing_folds=geo['passed_folds'], complete_draw_checks=extent['draw_checks_passed'],
                   map_integral_passed=deposit['integration_passed'], qualified_contour_policies=','.join(bands) or 'none',
                   maximum_training_log_cm_change=max(f['maximum_training_log_cm_change'] for f in extent['folds']
                                                       if f['status'] == 'estimated'))
    sectors_path = PAIRED/'sector-areas.csv.gz'
    sectors = pd.read_csv(sectors_path,dtype=str)
    other = sectors[sectors.slug != "mazama"].copy()
    identities = {(r["slug"],r["method"]):r["input_sha256"] for r in rows}
    other["input_sha256"] = [identities[k] for k in zip(other.slug,other.method)]
    parts = []
    for family in matches:
        check_sectors(("mazama",family,0,scratch,None,None))
        part = pd.read_csv(scratch/f"mazama-{family}-0/sectors.csv")
        assert len(part) == 32 and part.scramble.nunique() == 2 and part.sector.nunique() == 8
        parts.append(part.assign(slug="mazama",method=family,input_sha256=meta["input_sha256"]))
    pd.concat([other,*parts],ignore_index=True).to_csv(sectors_path,index=False,compression={"method":"gzip","mtime":0})
    write_csv(PAIRED / 'paired-results.csv', rows)
    write_json(PAIRED / 'mazama-parent-checks.json', checks)
    with TemporaryDirectory(dir=scratch) as temp:
        check_kernels((record, Path(temp), None))
        current = read_json(Path(temp) / 'mazama.json')
    current.update(input_sha256=sha(path), posterior_sha256=sha(home / 'spherical-mixture/posterior.npz'))
    kernels = read_json(PAIRED / 'kernel-integrals.json')
    kernels['mazama'] = current
    write_json(PAIRED / 'kernel-integrals.json', kernels)
    return matches, current


def update_pages(rows, kernel):
    page = PAIRED/'README.md'
    table = ['# Global fallout models and geographic validation','',
        '## Finite-spread validation on 300 Mazama localities','',
        'The directional, spherical and corridor pairs each share the same data, geographic holdouts and parent fields. '
        'The nested corridor is a comparator for this development experiment. The current nested fit uses dispersal footprints. '
        '[Five current maps and integrated thickness bands](../../eruptions/mazama/README.md) · [Paired predictions](paired-results.csv) · [Parent map integrals](mazama-parent-checks.json)','',
        '| Family | Parent median map (km³) | With finite spread (km³) | Parent geographic score ↓ | With finite spread ↓ | Passing folds |',
        '|---|---:|---:|---:|---:|---:|']
    for k,name in (('directional','Directional ensemble'),('spherical-mixture','Spherical mixture'),('nested-isopachs','Nested corridor control')):
        row = rows[k]
        table.append(f"| {name} | {row['earlier_km3']:,.1f} | {row['principal_km3']:,.1f} | {row['parent_matched_negative_log_score']:.3f} | {row['principal_negative_log_score']:.3f} | {row['passing_folds']}/4 |")
    status = 'passed' if kernel['passed'] else 'retained a numerical warning'
    table.extend(['',f'The independent spherical-kernel integral {status}. [Kernel checks](kernel-integrals.json)','',
        '## Across the nine eruptions','',
        'The paired study measures how limiting lateral spread away from observations changes isopach area, integrated volume and prediction in a withheld region. Each eruption contributes one geographic score to the average. Lower scores indicate better predictions. [All 27 pairs](paired-results.csv) · [Sector areas and volumes](sector-areas.csv.gz)','',
        '| Family | Mean parent score | Mean finite-spread score | Improved cases |','|---|---:|---:|---:|'])
    pairs = pd.read_csv(PAIRED/'paired-results.csv')
    for key,label in (('directional','Directional ensemble'),('spherical-mixture','Spherical mixture'),('nested-isopachs','Nested corridor control')):
        part = pairs[pairs.method == key]
        a,b = part.parent_matched_negative_log_score,part.principal_negative_log_score
        table.append(f'| {label} | {a.mean():.3f} | {b.mean():.3f} | {int((b<a-1e-10).sum())}/{len(part)} |')
    table.extend(['','![Parent and finite-spread contours on matched axes for five eruptions](affected-contours.png)','',
        'Each pair uses the same observations, axis limits and posterior subset. Sites are colored by thickness bracket. The Mazama corridor pair uses all 300 current localities. [Angular variation and input hashes](angular-variation.csv)','',
        'The independent survey tests compare predictions at field sites withheld from training. The simulated deposits measure both excess mapped area and missed area against known contours. [External field surveys](external-surveys.csv) · [Known-deposit trials](known-deposits.csv.gz) · [Finite-spread method](../../docs/methods/finite-spread.md)',''])
    page.write_text('\n'.join(table))


if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('--scratch',required=True,type=Path)
    args = parser.parse_args()
    args.scratch.mkdir(exist_ok=True,parents=True)
    update_pages(*paired_estimates(Path('eruptions/mazama/results'),args.scratch))
