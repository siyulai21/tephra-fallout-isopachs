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


def distal_facies_assessment(folder):
    """Report the effect of the actual source decisions on matched volumes and site fields."""
    with (folder / 'source-sensitivity-volumes.csv').open(newline='') as stream:
        volumes = {(r['method'], r['scenario']): r for r in csv.DictReader(stream)}
    families = {'stacked': 'GP ensemble', 'spline': 'Cubic spline', **FAMILIES}
    cases = {
        'principal': 'Revised evidence',
        'prior-244': 'Earlier 244 rows',
        'alternate-field-bounds': 'Alternate field bounds',
        'restore-thick-archive-readings': 'Restore thick readings',
        'omit-reinterpreted-ceilings': 'Omit revised ceilings',
        'half-capture-ceilings': 'Half capture',
        'omit-all-archive-ceilings': 'Omit all ceilings',
    }
    site_ids = {
        'CH2008-WIN': 'Windy Lake, BC', 'CH2008-THU': 'Thunder Lake, BC',
        'B2020-250': 'Johnson Lake', 'B2020-106': 'Sheep Mountain Bog',
        'S2009-FOY-A': 'Foy Lake core A', 'W2016-LYMAN': 'Lyman Lake',
        'M2023-YL16-5A': 'Yellowstone Lake YL16-5A',
    }
    with (folder / 'local-thickness-comparison.csv').open(newline='') as stream:
        sites = {(r['evidence_id'], r['method']): r for r in csv.DictReader(stream)
                 if r['scenario'] == 'prior-244' and r['evidence_id'] in site_ids}

    lines = [
        '# Model response to the distal field evidence', '',
        'The comparisons below use median-map volumes and local median thicknesses from the saved fits. '
        'The GP and cubic spline share the fixed hull of 26 historical measurements in the source comparison. '
        'Their case-home maps cover the larger current observation hull. The other three families integrate '
        'a conditional continuation over Earth. Each volume interval comes from the corresponding family’s draws. '
        '[Field evidence and original references](../../../../data/audits/Mazama/distal-review/FIELD_EVIDENCE.md) · '
        '[Principal maps, current-hull volumes and intervals](../../README.md)', '',
        '## Volumes under the tested readings', '',
        'All entries are in km³. Each row uses one method and one integration domain. A dagger marks a retained '
        'fit, draw or integration warning. The earlier 244 rows are refitted with the source-comparison procedure. '
        'The archived standalone 244-position calculation has its own geographic tuning and observation hull. '
        '[Original 244-position results](../../../../archive/2026-11-24-mazama-distal-facies/README.md)', '',
        '| Method | Revised map (conditional 95%) | ' + ' | '.join(list(cases.values())[1:]) + ' |',
        '|---|' + '---:|' * len(cases),
    ]
    for family, label in families.items():
        items = []
        for case in cases:
            r = volumes[family, case]
            value = f"{float(r['median_map_volume_km3']):,.3g}"
            if case == 'principal':
                value += f" ({float(r['lower95_km3']):,.3g}–{float(r['upper95_km3']):,.3g})"
            if any(r[k] == 'False' for k in ('fit_passed', 'draw_checks_passed', 'map_integration_passed')):
                value += ' †'
            items.append(value)
        lines.append('| ' + label + ' | ' + ' | '.join(items) + ' |')
    lines += [
        '', 'Restoring the thick readings reinstates the previous interpretations at nine archives, including '
        'Windy and Thunder lakes in British Columbia and Round Lake in Washington. '
        'The field alternative varies Johnson, Kilpoola, Foy and '
        'three Greater Victoria core brackets together. Half capture doubles the revised package ceilings as a '
        'stress test. Each scenario changes a group of evidence decisions, so the differences cannot apportion '
        'causality to one site. [All unrounded volumes, conditional intervals and flags](source-sensitivity-volumes.csv) · '
        '[Scenario definitions and input hashes](sensitivities/scenarios.json)', '',
        '## Modelled thickness at the reviewed locations', '',
        'Each cell gives the **earlier 244-row refit → revised principal** median thickness in cm. These are '
        'fitted latent thicknesses. Laboratory identities, enclosing-section bounds and geological observations '
        'are recorded separately. A dash indicates an unavailable prediction on the regional domain. '
        '[Site intervals and all comparisons](local-thickness-comparison.csv) · '
        '[Observed beds, bounds and withheld readings](../../../../data/audits/Mazama/distal-review/locality-facies-crosswalk.csv)', '',
        '| Site | ' + ' | '.join(families.values()) + ' |', '|---|' + '---:|' * len(families),
    ]
    def thickness(value):
        return f'{float(value):.3g}' if value else '—'

    for sid, label in site_ids.items():
        entries = []
        for family in families:
            r = sites[sid, family]
            entries.append(f"{thickness(r['latent_median_cm'])} → {thickness(r['principal_median_cm'])}")
        lines.append('| ' + label + ' | ' + ' | '.join(entries) + ' |')
    with (folder / 'heldout-source-observations.csv').open(newline='') as stream:
        withheld = {(r['evidence_id'], r['method']): r for r in csv.DictReader(stream)
                    if r['evidence_id'] in site_ids and r['method'] in {'stacked', 'spline'}}
    lines += [
        '', '## When the surrounding geographic region is withheld', '',
        'The two regional methods predict each observation using other geographic folds. Values below are '
        'predictive medians in cm. For a ceiling, the probability column gives the fitted probability of an '
        'observation at or below that ceiling; the measurement rows have no one-sided event. '
        'A predictive median above a conditional field ceiling shows a conflict under the local-retention assumption. '
        'The next columns give the probability assigned to the observed bound. '
        '[Withheld intervals, folds and scores for every observation](heldout-source-observations.csv)', '',
        '| Site | Observed point or bound (cm) | GP median | GP bound probability | Spline median | Spline bound probability |',
        '|---|---:|---:|---:|---:|---:|',
    ]
    for sid, label in site_ids.items():
        gp, spline = (withheld[sid, method] for method in ('stacked', 'spline'))
        reading = {'measurement': '', 'upper_limit': '≤', 'lower_limit': '≥'}[gp['kind']] + thickness(gp['reported_value_cm'])
        pair = []
        for model in (gp, spline):
            value = model['bound_probability']
            pair.extend([thickness(model['predictive_median_cm']), f'{float(value):.1%}' if value else '—'])
        lines.append('| ' + label + ' | ' + reading + ' | ' + ' | '.join(pair) + ' |')
    lines += [
        '', 'The geographic folds assess prediction away from training data. The contour screen tests which bands '
        'have observational support, including unsupported interior holes. The complete-Earth estimates condition '
        'on the family, continuation, retained evidence and numerical status. [Geographically withheld sites](heldout-source-observations.csv) · '
        '[Probabilities at held-out bounds](heldout-bound-probabilities.csv) · [Regional contour support](../support/README.md) · '
        '[Global support decisions by family](../../../../comparisons/support/README.md) · [Numerical checks](publication-checks.json)', '',
    ]
    (folder / 'distal-facies-results.md').write_text('\n'.join(lines))
