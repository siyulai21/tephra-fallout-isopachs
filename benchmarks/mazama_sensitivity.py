"""Source-interpretation tests with independent all-data refits of the five families.

These scenarios alter source selection and conditional interpretations. They are assumptions,
not added geological observations or alternative estimates of measurement error.
Geographic validation and fixed-prior sweeps belong to the principal comparison.
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np

from benchmarks.mazama import INPUT, AUDIT, sha, write_csv
from src.empirical_isopach.core.audit import audited_data
from src.empirical_isopach.core.distributions import predictive_distribution
from src.empirical_isopach.models.extent_integrals import integrate
from src.empirical_isopach.models.extent_release import draw_check
from src.empirical_isopach.models.extent_review import ExtentFields, ExtentView
from src.empirical_isopach.models.extent_weights import calibration
from src.empirical_isopach.models.nested.study import write_json
from src.empirical_isopach.visualization.deposit_integrals import independent_integral
from src.probabilistic_isopach.global_volume import EarthDomain

CASES = {
    'before-field-243': 'Exact archived review before the return to original Alberta, Montana and Vancouver Island field logs. This isolates the added field records from the earlier facies reclassification.',
    'alternate-field-bounds': 'Use 14 cm from Kilpoola core 1 at the common location, the 108 cm main Johnson package alone, retain only Foy core A, and widen the three Greater Victoria host ceilings to the part of physical core left after subtracting the sampled early and late sections.',
    'prior-244': 'Exact archived 244-position input. Regional settings come from the revised geographic selection; compare all families on their stated domains.',
    'omit-reinterpreted-ceilings': 'Omit every new or changed ceiling in the deposit-nature review. This removes the complete-local-retention assumption for those packages.',
    'half-capture-ceilings': 'Double each new or revised package ceiling as a sensitivity to a retained thickness fraction of 0.5. The factor is a stress assumption.',
    'omit-all-archive-ceilings': 'Remove all conditional upper limits, including the earlier cryptotephra and enclosing-section upper envelopes.',
    'omit-qualified-identities': 'Drop low-confidence sources including the one-shard On-off Lake correlation, probable Stowell ash, distal ice/cryptotephra assignments and distorted Hasse horizon.',
    'jenny-one': 'Retain the directly registered JEN13-2E photo as the sole Jenny Lake likelihood.',
    'jenny-omit': 'Omit all five Jenny Lake stations, including their conditional enclosing-section bounds.',
    'cathedral-one': 'Use only the Lake of the Woods A12 central core among the four adjacent Cathedral basins. This tests closely correlated archive behavior and shared source attribution.',
    'yellowstone-one': 'Retain central core YL16-5A as the single Yellowstone Lake likelihood. All core positions remain in the evidence inventory.',
    'restore-williams': 'Restore the former 46 cm Williams Lake Fen point alone. The unresolved two-horizon interpretation is a historical stress case.',
    'restore-thick-archive-readings': 'Restore the earlier readings at Windy, Thunder, Round WA, Tumalo, Round OR, Sunrise, Shadow, Dog OR and White Pine Marsh as a joint historical archive-transfer scenario.',
    'omit-basin-beds': 'Remove selected measured lake beds and conditional upper limits from the Ruby Mountains, Lyman, Roe, Great Salt Lake, Rainier and Yellowstone. This tests shared limnological preservation.',
    'omit-young-proximal': 'Omit Young’s individually mapped upper pumice minima as a group to test leverage of spatially clustered partial-component constraints.',
}
LEGACY = AUDIT/'history/2026-11-23-before-yellowstone/Mazama_reviewed_40.csv'
OLD244 = AUDIT/'history/2026-11-23-before-distal-review/input.csv'


def common_region():
    """Fixed original26-point footprint, independently of expanded scenario hulls."""
    from src.empirical_isopach.region import SurveyRegion
    data, _ = audited_data(LEGACY)
    assert int(np.sum(~data.censored)) == 26
    return SurveyRegion(data.xy[~data.censored])


def probe_data():
    path = AUDIT/'model-probes.csv'
    data, _ = audited_data(path)
    return data, list(csv.DictReader(path.open()))


def local_checks(view, folder):
    """Same actual bound sites and adjacent rays for every source interpretation.

    Rays are on the authalic sphere used by the published global integration.
    Retain every crossing and the sampled profiles; do not hide disconnected
    exceedances by reporting only an outer radius.
    """
    from src.empirical_isopach.models.directional.ensemble.fields import inverse_cdf
    full, records = probe_data()
    ids = np.arange(len(full.xy))
    domain = view.domain
    xy = full.xy[ids]
    values = np.concatenate(view.fields.variants(xy))
    quantiles = np.exp(inverse_cdf(values.T, view.weights, [.025, .5, .975])).T
    positions = xy * [1/domain.axis_scale, domain.axis_scale]
    bearings = np.arctan2(positions[:, 0], positions[:, 1])
    radii = np.r_[0., np.geomspace(.02, domain.rho_max*(1-1e-10), 768)]
    rays, crossings, sites = [], [], []
    for i, index in enumerate(ids):
        # Observation canonicalization reorders rows spatially. Recover the
        # original CSV row identity rather than using that array position.
        record = records[int(full.row_ids[index])-2]
        sites.append(dict(evidence_id=record['EvidenceID'], latitude=float(record['Lat']),
            longitude=float(record['Lon']), latent_q025_q50_q975_cm=quantiles[i].tolist(),
            authalic_bearing_deg=float(np.rad2deg(bearings[i]) % 360)))
        for offset in (-11.25, 0., 11.25):
            angle = bearings[i] + np.deg2rad(offset)
            ray = radii[:, None] * [np.sin(angle)*domain.axis_scale, np.cos(angle)/domain.axis_scale]
            z = view.central(ray)
            rays.append(z)
            for level in (.5, 1., 5., 10.):
                threshold = np.log10(level)
                inside = z >= threshold
                changes = np.flatnonzero(inside[:-1] != inside[1:])
                r = [radii[j] + (radii[j+1]-radii[j])*(threshold-z[j])/(z[j+1]-z[j]) for j in changes]
                crossings.append(dict(evidence_id=record['EvidenceID'], bearing_offset_deg=offset,
                    level_cm=level, source_above=bool(inside[0]), antipode_above=bool(inside[-1]),
                    crossings_authalic_km=domain.arc_km(r).tolist(),
                    crossing_grid_brackets_authalic_km=[[float(domain.arc_km(radii[j])),float(domain.arc_km(radii[j+1]))] for j in changes]))
    np.savez_compressed(folder/'local-profiles.npz', radii_km=radii, log10_cm=np.asarray(rays))
    return dict(sites=sites, crossings=crossings,
        profile_order='Canonical order of model-probes.csv, then bearing offsets -11.25, 0, +11.25 degrees',
        scope='Conditional latent thickness and all sampled median-field crossings; no claim of measured contour support. Crossing brackets show radial sampling resolution; sub-grid structure is unresolved.')


def prepare(root):
    import shutil
    with INPUT.open() as source:
        principal = list(csv.DictReader(source))
    with OLD244.open() as source:
        former = {r['EvidenceID']:r for r in csv.DictReader(source)}
    interpretations = json.loads((AUDIT/'distal-review/interpretations.json').read_text())
    changed_caps = {key for key,r in interpretations.items() if r['use']=='upper_limit'}
    weak = {'M2012-ONOFF02','L2016-STOWELL','B2020-285','B2020-003','P2018-MORT12','B2020-229','B2020-291','FLV93-T206','H2026-FROG','H2026-SWANSON','H2026-WORLEY','B2019-BEGBIE','H1976-ELK','H1999-KILPOOLA-2'}
    basin = {'M2018-OVERLAND','M2018-SOLDIER','W2016-LYMAN','L2012-ROE','J2025-GSL4','W2017-SHADOW','W2017-SUNRISE'}
    result = {}
    for name, definition in CASES.items():
        folder = root/name
        folder.mkdir(parents=True, exist_ok=True)
        path = folder/'input.csv'
        if name in ('prior-244','before-field-243'):
            frozen=OLD244 if name=='prior-244' else AUDIT/'history/2026-11-24-before-field-logs/input.csv'
            shutil.copyfile(frozen,path)
            with frozen.open() as source:
                rows=list(csv.DictReader(source))
        else:
            rows = [r.copy() for r in principal]
            if name == 'alternate-field-bounds':
                alternate={'H1999-KILPOOLA-2':14,'B2020-250':108,'H2026-FROG':562-180-(495-405),
                           'H2026-SWANSON':654-112-(549-380),'H2026-WORLEY':474-60-(360-250)}
                rows=[r for r in rows if r['EvidenceID']!='S2009-FOY-C']
                for r in rows:
                    if r['EvidenceID'] in alternate:
                        r.update(Limit_cm=alternate[r['EvidenceID']],Qualification=definition)
                        if r['EvidenceID']=='H1999-KILPOOLA-2':
                            r.update(EvidenceID='H1999-KILPOOLA-1',Site='Kilpoola Lake, 1993 core 1',Evidence='1993 core at the common lake coordinate, contacts 290–304 cm.')
            elif name == 'omit-reinterpreted-ceilings':
                rows = [r for r in rows if r['EvidenceID'] not in changed_caps]
            elif name == 'half-capture-ceilings':
                for r in rows:
                    if r['EvidenceID'] in changed_caps:
                        r.update(Limit_cm=2*float(r['Limit_cm']),Qualification=definition)
            elif name == 'omit-all-archive-ceilings':
                rows = [r for r in rows if r['Kind']!='upper_limit']
            elif name == 'omit-qualified-identities':
                rows = [r for r in rows if r['EvidenceID'] not in weak]
            elif name in ('jenny-one','jenny-omit'):
                keep='L2024-JEN13-2E' if name=='jenny-one' else None
                rows=[r for r in rows if r['EvidenceID']==keep or (r['EvidenceID']!='B2020-042' and not r['EvidenceID'].startswith('L2024-JEN'))]
            elif name == 'cathedral-one':
                rows = [r for r in rows if not r['EvidenceID'].startswith('E1997-') or r['EvidenceID']=='E1997-A12']
            elif name == 'yellowstone-one':
                rows = [r for r in rows if not r['EvidenceID'].startswith('M2023-YL') or r['EvidenceID']=='M2023-YL16-5A']
            elif name == 'restore-williams':
                rows.append(former['B2020-113'].copy())
            elif name == 'restore-thick-archive-readings':
                ids={'CH2008-THU','CH2008-WIN','M2024-ROUNDWA','B2020-047','B2020-054','W2017-SHADOW','W2017-SUNRISE','S2023-DOG-OR','S2023-WHITEPINE'}
                rows=[former[r['EvidenceID']].copy() if r['EvidenceID'] in ids else r for r in rows]
            elif name == 'omit-basin-beds':
                rows=[r for r in rows if r['EvidenceID'] not in basin and not r['EvidenceID'].startswith('M2023-YL')]
            elif name=='omit-young-proximal':
                rows=[r for r in rows if not r['EvidenceID'].startswith('Y1990-A')]
            else:
                raise ValueError(name)
            for i,row in enumerate(rows):
                for field in ('Vent Lon','Vent Lat'):
                    row[field]=principal[0][field] if i==0 else ''
            write_csv(path,rows)
        assert sha(path)!=sha(INPUT),name
        result[name]=dict(file=str(path),sha256=sha(path),definition=definition,
                       measurements=sum(r['Kind']=='measurement' for r in rows),
                       lower_limits=sum(r['Kind']=='lower_limit' for r in rows),
                       upper_limits=sum(r['Kind']=='upper_limit' for r in rows))
    write_json(root/'scenarios.json',dict(principal_input_sha256=sha(INPUT),cases=result))



def regional(data, full, report):
    from src.empirical_isopach.models.gaussian_process.model import SpatialModel, Candidate
    from src.empirical_isopach.models.gaussian_process.ensemble import StackedModel
    from src.empirical_isopach.models.cubic_spline.model import TrendRegularizedSpline, SplineCandidate
    from src.empirical_isopach.region import SurveyRegion, integrate_region
    region = common_region()
    gp = report['methods']['stacked']['model']
    components = [SpatialModel(Candidate(**c['model']['candidate'])).fit(data) for c in gp['components']]
    models = {'stacked': StackedModel(components, [c['weight'] for c in gp['components']],
                                     gp['observation_discrepancy_variance_log'])}
    s = report['methods']['spline']['model']
    models['spline'] = TrendRegularizedSpline(SplineCandidate(**s['candidate']),
        quadrature_order=s['quadrature_order']).fit(data, bounds=report['fit_bounds_km'])
    return dict(common_region=region.describe(), tuning='Fixed principal hyperparameters and ensemble weights; coefficients refitted to each scenario. No new model selection or held-out skill claim.',
                methods={k:dict(model=m.describe(), volume=integrate_region(m, region),
                    bound_site_median_cm=np.exp(predictive_distribution(m, full.xy[full.censored]).quantile(.5)).tolist()) for k,m in models.items()})


def nested(data, lat, folder):
    from src.empirical_isopach.models.nested.inference import fit
    from src.empirical_isopach.models.nested.continuation import check_and_refine
    from src.empirical_isopach.models.nested.reach import condition
    raw, internal = fit(data, lat, initial=768, ceiling=6144)
    raw, parent, checks = check_and_refine(raw, data, internal, ceiling=12288)
    np.savez_compressed(folder / 'whole-area.npz', **vars(parent))
    model, importance = condition(parent, data, seed=71183)
    attempts = [importance]
    passed = checks['passed']
    details = dict(initial=internal, parent=checks)
    if not importance['passed']:
        positive = data.subset(~data.censored | data.lower_censored)
        raw, info = fit(positive, lat, initial=768, ceiling=12288, seed=71183)
        raw, proposal, check = check_and_refine(raw, positive, info, ceiling=12288, seed=71183)
        np.savez_compressed(folder / 'positive-proposal.npz', **vars(proposal))
        model, importance = condition(proposal, data, positive_proposal=True, seed=71183)
        attempts.append(importance)
        passed = check['passed']
        details['positive_proposal'] = check
    model.save(folder / 'corridor.npz')
    return model, dict(numerics=details, importance_attempts=attempts,
                       passed=bool(passed and importance['passed']))


def global_fit(family, data, meta, folder, work):
    lat = meta['vent_lonlat'][1]
    cal = calibration(family, meta['input_sha256'], exclude_slug='mazama')
    assert 'mazama' not in cal['donors']
    if family == 'spherical-mixture':
        from src.empirical_isopach.models.mass_mixture.study import fit_checked
        from src.empirical_isopach.models.mass_mixture.prior import BASELINE
        from src.empirical_isopach.models.mass_mixture.bearings.field import BearingField, BearingSupport, selected, phases
        posterior, fit = fit_checked(data, lat, BASELINE, work)
        np.savez_compressed(folder / 'posterior.npz', **posterior)
        def parent(n):
            draws, _, shape = selected(posterior, n)
            return BearingField(draws, BearingSupport(data, lat), phases(shape))
        fit_passed = fit['diagnostics']['passed']
        counts = (128, 256, 512, 1024)
    elif family == 'directional':
        from src.empirical_isopach.models.directional.ensemble.members import run_members
        from src.empirical_isopach.models.directional.azimuth.calibration import calibrate
        from src.empirical_isopach.models.directional.azimuth.report import model_for
        run_members(data, lat, work, kinds=('full',))
        weights = calibrate(meta, 'mazama')
        assert 'mazama' not in weights['calibration_cases']
        def parent(n):
            return model_for(data, lat, work, weights, n)
        check = parent(64)
        fit = dict(weights=weights, members={m.geometry.policy.name:m.records for m in check.members})
        fit_passed = all(m.records[-1]['diagnostics']['passed'] for m in check.members)
        counts = (64, 128, 256, 512)
    else:
        model, fit = nested(data, lat, folder)
        def parent(n):
            return model.reduced(n)
        fit_passed = fit['passed']
        counts = (2048, 4096)
    refinements = []
    for count in counts:
        fields = ExtentFields(family, parent(count), data, lat)
        vol, history = integrate(fields, lat)
        q, gap = draw_check(vol, fields.weights, cal)
        refinements.append(dict(draw_parameter=count, split_relative_difference=gap, integration=history))
        print(f'{folder}: integrated {count}, split gap {gap:.3g}, integration {history[-1]["passed"]}', flush=True)
        if gap < .15 and history[-1]['passed']:
            break
    domain = EarthDomain.wgs84(lat)
    indices = [fields.states.index(s) for s in cal['states']]
    mixture = [cal['weights'][indices.index(i)] if i in indices else 0. for i in range(len(fields.states))]
    view = ExtentView(fields, domain, mixture)
    maps = independent_integral(view, domain)
    np.savez_compressed(folder / 'extent-volumes.npz', volume_km3=vol, weights=fields.weights)
    return dict(calibration=cal, fit=fit, median_map_volume_km3=float(np.mean(maps[-1]['km3'])),
                map_integrals=maps, conditional_q025_q50_q975_km3=q.tolist(), refinements=refinements,
                local_checks=local_checks(view, folder),
                fit_passed=bool(fit_passed), draw_checks_passed=bool(fit_passed and gap < .15 and history[-1]['passed']),
                map_integration_passed=bool(maps[-1]['relative_change'] < .01))


def run(root, case, family, principal):
    path = root / case / 'input.csv'
    data, meta = audited_data(path)
    folder = root / case / family
    folder.mkdir(parents=True, exist_ok=True)
    receipt = folder / 'report.json'
    if receipt.exists():
        old = json.loads(receipt.read_text())
        if old['input_sha256'] != meta['input_sha256'] or old['script_sha256'] != sha(Path(__file__)):
            raise ValueError('Sensitivity identity changed; use a fresh output directory')
        print('Verified completed sensitivity:', case, family, flush=True)
        return
    result = dict(case=case, family=family, definition=CASES[case], input_sha256=meta['input_sha256'],
                  principal_input_sha256=sha(INPUT), script_sha256=sha(Path(__file__)),
                  scope='Independent full-data source-interpretation refit. This is not an independent geographic validation or a replacement for the principal fixed-prior study.')
    if family == 'regional':
        full, _ = audited_data(INPUT)
        report = json.loads((principal / 'report.json').read_text())
        assert report['input']['input_sha256'] == sha(INPUT)
        result.update(regional(data, full, report))
        if not (root / 'principal-common-hull.json').exists():
            write_json(root / 'principal-common-hull.json', regional(full, full, report))
    else:
        result.update(global_fit(family, data, meta, folder, folder / 'work'))
    result['artifacts'] = {str(f.relative_to(folder)):sha(f) for f in folder.rglob('*') if f.is_file() and f != receipt}
    write_json(receipt, result)
    print('Completed sensitivity:', case, family, flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--principal', type=Path)
    p.add_argument('--case', choices=CASES)
    p.add_argument('--family', choices=('regional', 'spherical-mixture', 'directional', 'nested-isopachs'))
    a = p.parse_args()
    prepare(a.output)
    if a.family:
        if a.family == 'regional' and a.principal is None:
            p.error('--principal is required for matched regional settings')
        for case in ([a.case] if a.case else CASES):
            run(a.output, case, a.family, a.principal)
