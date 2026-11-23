"""Update the current catalogue while retaining the original matched development experiment."""
import csv
import gzip
import json
from pathlib import Path
from tempfile import TemporaryDirectory

from benchmarks.extent_spherical_check import run as check_kernels
from benchmarks.mazama import INPUT, sha, write_csv
from src.empirical_isopach.core.audit import audited_data
from src.empirical_isopach.core.catalog import dataset_record
from src.empirical_isopach.visualization.deposit import FAMILIES, checked_field, read_json, write_json
from src.empirical_isopach.visualization.deposit_field import parent_view
from src.empirical_isopach.visualization.deposit_integrals import within_cells


HISTORICAL = 'https://github.com/siyulai21/tephra-fallout-isopachs/blob/0a53d0211f2ac9cfc9fbeaff95217fc0d319a18b/comparisons/global-extent/'
PAIRED = Path('comparisons/global-extent')


def paired_estimates(home, scratch):
    """Compare both continuations on this version's observations, fields and geographic folds."""
    data, meta = audited_data(INPUT)
    checks = {}
    with (PAIRED / 'paired-results.csv').open(newline='') as stream:
        rows = list(csv.DictReader(stream))
    matches = {row['method']: row for row in rows if row['slug'] == 'mazama'}
    if set(matches) != set(FAMILIES):
        raise ValueError('Expected three paired global fits for Mazama')
    for family, row in matches.items():
        folder = home / family
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
    write_csv(PAIRED / 'paired-results.csv', rows)
    write_json(home / 'source-audit/paired-continuation-checks.json', checks)
    record = dataset_record(INPUT)
    with TemporaryDirectory(dir=scratch) as temp:
        check_kernels((record, Path(temp), None))
        current = read_json(Path(temp) / 'mazama.json')
    current.update(input_sha256=sha(INPUT), posterior_sha256=sha(home / 'spherical-mixture/posterior.npz'))
    kernels = read_json(PAIRED / 'kernel-integrals.json')
    kernels['mazama'] = current
    write_json(PAIRED / 'kernel-integrals.json', kernels)
    return matches, current


def update_pages(rows, kernel):
    page = PAIRED / 'README.md'
    heading = '## Original nine-eruption experiment'
    content = page.read_text()
    original = (content.split('<!-- original study -->', 1)[-1] if '<!-- original study -->' in content
                else content.split('\n', 1)[1])
    original = original.strip()
    for filename in ('paired-results.csv', 'kernel-integrals.json'):
        original = original.replace(f']({filename})', f']({HISTORICAL}{filename})')
    original = original.replace('](../principal-volumes.csv)', f']({HISTORICAL}../principal-volumes.csv)')
    table = [
        '# Global fallout models and geographic validation', '',
        '## Mazama after the source audit', '',
        f"The fits use {rows['nested-isopachs']['accepted']} distinct locations. Each pair shares the same observations, geography and training folds. The two map integrals measure the effect of the assumed continuation into unsampled areas. Volumes are in km³. [Current case](../../eruptions/mazama/README.md) · [Matched outcomes](paired-results.csv) · [Parent map calculations](../../eruptions/mazama/results/source-audit/paired-continuation-checks.json)", '',
        '| Family | Parent map | Revised map | Revised conditional 95% | Geographic score, parent → revised | Withheld 95% coverage |',
        '|---|---:|---:|---:|---:|---:|']
    for family, label in FAMILIES.items():
        r = rows[family]
        flag = '' if r['complete_draw_checks'] else ' · provisional'
        table.append(f"| {label} | {r['earlier_km3']:,.1f} | {r['principal_km3']:,.1f} | {r['lower95']:,.1f}–{r['upper95']:,.1f}{flag} | {r['parent_matched_negative_log_score']:.3f} → {r['principal_negative_log_score']:.3f} | {r['principal_coverage95']:.1%} |")
    status = 'passed' if kernel.get('passed') or kernel.get('exact_parent_identity') else 'retained a numerical warning'
    table += ['', f'The independent spherical-kernel integral {status}. The case pages retain the sampling, spatial-integration and contour-support diagnostics for each family. [Kernel diagnostics](kernel-integrals.json)', '',
              heading, '', 'The following experiment used the earlier 45-observation Mazama dataset. Its maps, sector areas and study-wide averages retain that input. [Original publication](' + HISTORICAL + 'README.md)', '', '<!-- original study -->', '', original, '']
    page.write_text('\n'.join(table))
    support = Path('comparisons/support/README.md')
    lines = support.read_text().splitlines()
    for i, line in enumerate(lines):
        if line.startswith('| Mazama |'):
            lines[i] = ('| Mazama | [GP and cubic spline](../../eruptions/mazama/results/support/README.md) | '
                        '[Directional ensemble](../../eruptions/mazama/results/directional/extent/support.json.gz) · '
                        '[Spherical mixture](../../eruptions/mazama/results/spherical-mixture/extent/support.json.gz) · '
                        '[Nested corridors](../../eruptions/mazama/results/nested-isopachs/extent/support.json.gz) |')
    support.write_text('\n'.join(lines) + '\n')


def source_summary(folder):
    """Put the largest source-interpretation changes next to the comparable principal fit."""
    with (folder / 'source-sensitivity-volumes.csv').open(newline='') as stream:
        rows = list(csv.DictReader(stream))
    families = {'stacked': 'GP ensemble', 'spline': 'Cubic spline', **FAMILIES}
    lines = [
        '## Sensitivity to the geological interpretation', '',
        'Each endpoint below is a separate refit in km³. The GP and spline use the fixed footprint of the 26 historical measurements for this comparison. The other three models integrate over Earth. The full table gives each fit’s own conditional interval and numerical checks. [All source refits](source-sensitivity-volumes.csv)', '',
        '| Family | Comparable principal map | Smallest source-scenario map | Largest source-scenario map | Scenarios with a numerical warning |',
        '|---|---:|---:|---:|---:|']
    for family, name in families.items():
        subset = [row for row in rows if row['method'] == family]
        original, = [row for row in subset if row['scenario'] == 'principal']
        refits = [row for row in subset if row['scenario'] != 'principal']
        ranked = sorted(refits, key=lambda r: float(r['median_map_volume_km3']))
        extremes = []
        for row in (ranked[0], ranked[-1]):
            scenario = row['scenario']
            extremes.append(f"{float(row['median_map_volume_km3']):,.1f} ([{scenario}](sensitivities/{scenario}/))")
        flags = ('fit_passed','draw_checks_passed','map_integration_passed')
        warned = sum(any(row[key] == 'False' for key in flags) for row in refits)
        lines.append(f"| {name} | {float(original['median_map_volume_km3']):,.1f} | {extremes[0]} | {extremes[1]} | {warned}/{len(refits)} |")
    lines += ['', 'The endpoint range describes the tested source choices. Sampling intervals and geographic prediction have separate assessments. [Withheld site predictions](heldout-source-observations.csv) · [Probabilities assigned to withheld upper and lower bounds](heldout-bound-probabilities.csv) · [Site-by-site changes](local-thickness-comparison.csv) · [All contour crossings, compressed CSV](local-crossing-comparison.csv.gz)', '']
    page = folder / 'README.md'
    text = page.read_text()
    first, rest = text.split('\n\n', 1)
    page.write_text(first + '\n\n' + '\n'.join(lines) + '\n' + rest)
