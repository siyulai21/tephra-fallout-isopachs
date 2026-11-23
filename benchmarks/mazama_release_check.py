"""Verify the published Mazama source refits and their unchanged evidence identities.

Numerical warnings are reported, not converted into successful fits. Geological
interpretations remain conditional even when this consistency check passes.
"""
import argparse
import csv
import json
from pathlib import Path

from benchmarks.mazama import AUDIT, INPUT, sha
from benchmarks.mazama_sensitivity import CASES


def read(path):
    return json.loads(path.read_text())


def rows(path):
    with path.open() as source:
        return list(csv.DictReader(source))


def check(home):
    source = home/'source-audit'
    cases = source/'sensitivities'
    reference = sha(INPUT)
    accepted = rows(INPUT)
    probes = rows(AUDIT/'model-probes.csv')
    ids = {r['EvidenceID'] for r in probes}
    old_ids = {r['EvidenceID'] for r in rows(AUDIT/'history/2026-11-23-before-distal-review/input.csv')}
    assert len(probes) == len(ids)
    assert ids == old_ids | {r['EvidenceID'] for r in accepted} | {'M2023-YL16-6A'}
    scenarios = set(CASES)
    assert {p.name for p in cases.iterdir() if p.is_dir()} == scenarios | {'principal'}
    expected = read(cases/'scenarios.json')
    assert expected['principal_input_sha256'] == reference
    assert set(expected['cases']) == set(CASES)
    script = source/'reproduction/mazama_sensitivity.py'
    assert sha(script) == sha(Path('benchmarks/mazama_sensitivity.py'))
    principal = read(home/'report.json')
    assert principal['input']['input_sha256'] == reference
    partitions = principal['validation']['methods']['stacked']['selections']
    assert partitions == principal['validation']['methods']['spline']['selections'] or all(
        a['training_row_ids'] == b['training_row_ids'] and
        a['withheld_row_ids'] == b['withheld_row_ids']
        for a, b in zip(partitions, principal['validation']['methods']['spline']['selections'], strict=True))
    family_names = ('directional', 'spherical-mixture', 'nested-isopachs')
    warnings, principal_checks, artifact_count = [], [], 0

    def local(record):
        assert len(record['sites']) == len(ids)
        assert {r['evidence_id'] for r in record['sites']} == ids
        keys = {(r['evidence_id'], r['bearing_offset_deg'], r['level_cm']) for r in record['crossings']}
        wanted = {(sid, offset, cm) for sid in ids for offset in (-11.25, 0., 11.25) for cm in (.5, 1., 5., 10.)}
        assert len(record['crossings']) == len(wanted) and keys == wanted

    for family in family_names:
        folder = home/family
        extent = read(folder/'extent/report.json')
        deposit = read(folder/'deposit.json')
        assert extent['input_sha256'] == deposit['input_sha256'] == reference
        assert sha(folder/'extent/heldout.csv.gz') == extent['heldout_sha256']
        assert sha(folder/'extent/volumes.npz') == extent['volume_draws_sha256']
        for fold in extent['folds']:
            if fold['design'] != 'geographic' or fold['status'] != 'estimated':
                continue
            base = partitions[fold['fold']]
            assert fold['train'] == base['training_row_ids']
            assert fold['test'] == base['withheld_row_ids']
            assert not set(fold['train']) & set(fold['test'])
        local(read(cases/'principal'/family/'local-checks.json')['local_checks'])
        principal_checks.append(dict(family=family,
            draw_checks_passed=extent['draw_checks_passed'],
            geographic=extent['validation']['geographic']))
    priors = read(home/'spherical-mixture/extent/prior-sensitivity.json')
    assert len(priors) == 8 and set(priors) == set(read(home/'spherical-mixture/report.json')['fits'])

    for case in sorted(scenarios | {'principal'}):
        folder = cases/case
        input_hash = reference if case == 'principal' else sha(folder/'input.csv')
        if case in CASES:
            assert input_hash == expected['cases'][case]['sha256']
        regional = read(folder/'regional-details.json')
        assert regional['input_sha256'] == input_hash
        assert regional['principal_input_sha256'] == reference
        assert regional['script_sha256'] == sha(source/'reproduction/mazama_release.py')
        assert set(regional['methods']) == {'stacked', 'spline'}
        for family, details in regional['methods'].items():
            local(details)
            assert details['profiles_sha256'] == sha(folder/(family+'-profiles.npz'))
        if case == 'principal':
            continue
        for family in ('regional', *family_names):
            packet = folder/family
            report = read(packet/'report.json')
            assert report['input_sha256'] == input_hash
            assert report['principal_input_sha256'] == reference
            assert report['script_sha256'] == sha(script)
            for name, expected_sha in report['artifacts'].items():
                assert sha(packet/name) == expected_sha, (case, family, name)
                artifact_count += 1
            if family != 'regional':
                assert 'mazama' not in report['calibration']['donors']
                local(report['local_checks'])
                flags = {key: report[key] for key in ('fit_passed', 'draw_checks_passed', 'map_integration_passed')}
                if not all(flags.values()):
                    warnings.append(dict(scenario=case, family=family, **flags))
    local_report = read(source/'local-comparison.json')
    family_cases = 5 * (len(scenarios) + 1)
    assert local_report['site_rows'] == family_cases * len(ids)
    assert local_report['crossing_rows'] == family_cases * len(ids) * 12
    volumes = rows(source/'source-sensitivity-volumes.csv')
    assert len(volumes) == family_cases
    assert {(r['scenario'], r['method']) for r in volumes} == {
        (case, method) for case in scenarios | {'principal'}
        for method in ('stacked', 'spline', *family_names)}
    return dict(input_sha256=reference, principal_observations=len(accepted),
        source_scenarios=len(scenarios), family_scenario_reports=4*len(scenarios),
        artifact_hashes_verified=artifact_count, principal=principal_checks,
        source_numerical_warnings=warnings, source_prior_sensitivities=8,
        local_site_predictions=local_report['site_rows'],
        local_contour_comparisons=local_report['crossing_rows'],
        interpretation='Identity and completeness checks passed; numerical warnings and conditional geological interpretations remain explicit.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--home', type=Path, default=Path('eruptions/mazama/results'))
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    result = check(args.home)
    if args.output:
        args.output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))
