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
from benchmarks.mazama_great_basin import BASIN_IDS
from src.empirical_isopach.core.audit import audited_data
from src.empirical_isopach.core.distributions import predictive_distribution
from src.empirical_isopach.models.extent_integrals import integrate
from src.empirical_isopach.models.extent_release import draw_check
from src.empirical_isopach.models.extent_review import ExtentFields, ExtentView
from src.empirical_isopach.models.extent_weights import calibration
from src.empirical_isopach.models.nested.study import write_json
from src.empirical_isopach.visualization.deposit_integrals import independent_integral
from src.probabilistic_isopach.global_volume import EarthDomain

DATED_IDS={'B2020-129','B2020-141','B2020-147','B2020-203','B2020-251','R1982-KOOT-776'}
CASES = {
    'great-basin-omit': 'Omit the eight additional quantitative observations in the Great Basin and Carp Lake source review. The predecessor had 236 observations.',
    'great-basin-alternate': 'Read the Leidy trench at its upper endpoint, omit the less clean first peat hole, double the second cap to test net loss and move the Furnace auger to a corner of its mapped frame. Treat Patterson and Fish as lake ceilings, omit the assumed Lily bed and use the 26 cm Table 1 Dead Horse endpoint. Carp uses the 2 cm compiler alternative.',
    'dated-sections-omit': 'Omit the two corrected UBC Research Forest beds, both dated host brackets, the complete Huff archive and the Kootenai pit. The other reviewed regions are retained.',
    'dated-capture-alternate': 'Allow net focusing in both UBC lake beds by treating them as maxima. Allow half capture at Waits and Huff. Replace the Hager mire with the central core host ceiling. Use only basal BCV1 and the western Kootenai section centre.',
    'kootenai-position-east': 'Move the single Kootenai pit likelihood to the eastern section centre without altering its component minimum.',
    'lake-field-omit': 'Omit the final lake-core and Mount Baker/Three Sisters interpretation group. Retain the independent Young and Round Meadow sections.',
    'lake-capture-alternate': 'For British Columbia measured beds, use the whole preserved package as a ceiling. Loosen original-to-preserved capture to at least one half in the three Idaho lake packages, two undated-width Washington beds, Redmountain, Pocket Lake and Pettit. Keep one Redfish core, RDF-4A. The facies cap is doubled for Redfish under the same response assumption.',
    'field-range-alternate': 'Use the opposite endpoints for mapped Mount Baker (RS–5, 17, 21) and Three Sisters (5, 15, 41) ranges, conditional on capture and no erosion. Omit RS–18 at its ambiguous coordinate.',
    'young-sections-omit': 'Omit the Young Appendix 1 upper-pumice sections together. The distinct modern measured locations retain their original treatment.',
    'young-section-positions': 'Translate the Young Appendix 1 observations coherently to the alternative Kirk Hill registration tying Young section 47 to the modern equivalent Upper Kirk site. All measured values are unchanged.',
    'young-distal-omit': 'Omit the three field-checked bog estimates, the five Young ceilings, and the Lofty Lake alternative-bearing record together. This tests the shared upland conversion and local preservation.',
    'distal-transfer-alternates': 'Jesmond uses the original observer’s approximately 0.5 cm corer estimate under negligible smearing. Lofty uses Young’s 0.2 cm ceiling. The three field-checked peat estimates retain the principal endpoints.',
    'round-meadow-one': 'Retain only vibracore 3 from the Round Meadow 2015 group. Its unambiguous 330–130 cm primary section tests the contribution of adjacent borings and less certain contacts.',
    'offshore-omit': 'Omit the conditional Willapa Canyon ceiling. Marine reworking, compaction or export may decouple the 4 cm bed from local primary fallout.',
    'western-field-omit': 'Omit the new Stillwater, North Twin Lake, Indian Prairie, Gold Lake Bog, Newberry and Walker Rim field studies together. Other Oregon sections and Young’s table remain in the likelihood.',
    'omit-bounds': 'Omit every conditional upper and lower bound. The remaining likelihood consists of measured beds.',
    'half-transfer': 'Halve the lower limits and double the upper limits together. This tests weaker preservation and original-to-recovered transfer across sites. A factor of two is a stress assumption.',
    'portage-upper': 'Replace the Portage lower endpoint of 1.27 cm with the upper endpoint of 1.905 cm. This one-sided approximation adds negligible loss and compaction assumptions.',
    'range-upper': 'Replace the lower endpoints for Independence, Cleveland and Newman with upper endpoints of 6, 6 and 30 cm. Each horizon enters once. The upper endpoints additionally assume negligible loss and compaction.',
    'omit-new-records': 'Retain the frozen 40-row selection to assess the combined effect of subsequent primary-source discoveries.',
    'new-beds-lower': 'Treat point beds added since the 40-row trial as lower limits at their observed thickness, conditional on primary attribution and negligible import. This tests incomplete preservation.',
    'weak-identity-omit': 'Omit the listed beds whose local climactic correlation or stratigraphic assignment is particularly uncertain. See WEAK_IDS in the reproduction script.',
    'yellowstone-one': 'Retain YL16-5A as the central Yellowstone Lake representative and omit six other lake sites. Goose Lake remains a separate observation.',
    'yellowstone-2c-5mm': 'Use Schiller 2020 YL16-2C 0.5 cm in place of Morgan 2023 0.4 cm at the same core.',
    'yellowstone-slump': 'Admit YL16-6A 0.5 cm under the assumption that the deformed slump-toe bed retains the original local thickness.',
    'omit-gold': 'Omit the 74 cm Gold Lake coarse-component minimum to test uncertain precursor contribution and lake focusing.',
    'late-source-omit': 'Omit Dog Lake Oregon, White Pine Marsh, Connley Cave 2, Skull Creek and Lyman Lake to isolate their joint influence.',
    'skull-omit': 'Omit the Skull Creek author-estimated upper limit to test its uncertain decimal typography and transfer from a mixed interval.',
    'field-source-omit': 'Remove the nine additions from the final primary-source review. This isolates the change from the 67-observation reference.',
    'bracket-omit': 'Remove the four new conditional upper limits at Deep Creek, Medicine Lake, Mahoney Lake and Lodgepole Quarry. The Skull Creek estimate remains.',
    'lake-core-alternates': 'Use the separate Little Lake core 91 bed of 1 cm and the companion Mahoney Lake 80 cm package. Each lake enters once under the principal preservation assumptions.',
    'bachelor-west-alternate': 'Replace the Mount Bachelor 44 cm upper-pumice minimum with the historical nearby West Village two-unit package as a 38 cm ceiling. Exclude the overlying 70 cm reworked material. This tests an alternative section, phase attribution and preservation together.',
}

LEGACY = AUDIT/'history/2026-11-23-before-yellowstone/Mazama_reviewed_40.csv'
BRACKET_IDS = {'B2020-206','S2003-MEDICINE','H1997-MAHONEY','SG1989-LODGEPOLE'}
FIELD_IDS = BRACKET_IDS | {'B2020-095','B2020-052','L2007-LOST-OR','J1985-FLODELLE','P1999-BEAVER-OR'}
WEAK_IDS = {'W2017-SHADOW','W2017-SUNRISE','W2017-LSUNRISE','D2025-LOST',
            'M2024-ROUNDWA','R2021-LONG','MK2012-PATJENS','M2014-CLELAND-F',
            'S2023-DOG-OR','S2023-WHITEPINE','S2023-CONNLEY2','B2020-095',
            'B2020-206','L2007-LOST-OR','P1999-BEAVER-OR'}
LATE_IDS = {'S2023-DOG-OR','S2023-WHITEPINE','S2023-CONNLEY2','MW1986-SKULL','W2016-LYMAN'}


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
    source = list(csv.DictReader(INPUT.open()))
    legacy_ids = {r['EvidenceID'] for r in csv.DictReader(LEGACY.open())}
    result = {}
    for name, definition in CASES.items():
        folder = root / name
        folder.mkdir(parents=True, exist_ok=True)
        rows = [dict(r) for r in source]
        if name == 'great-basin-omit':
            rows=[r for r in rows if r['EvidenceID'] not in BASIN_IDS]
        elif name == 'great-basin-alternate':
            rows=[r for r in rows if r['EvidenceID'] not in {'FLV93-C1','MWB07-LIL'}]
            for r in rows:
                sid=r['EvidenceID']
                if sid=='FLV93-T206':
                    r.update(Kind='upper_limit',Limit_cm=1,Qualification='Upper end of the preserved range under complete capture with no appreciable thinning.')
                elif sid in {'MWB07-PAT','B2020-023'}:
                    r.update(Kind='upper_limit',Limit_cm=r['Thickness'],Thickness='',Qualification='Whole lake bed bounds the original fall if retained despite focusing or rapid sinking.')
                elif sid in {'FLV93-C7','MWB07-DH','B2020-079'}:
                    alt={'FLV93-C7':5,'MWB07-DH':26,'B2020-079':2}
                    r.update(Limit_cm=alt[sid],Qualification='Alternative retained value or enclosure described in the interpretation tests.')
                elif sid=='FLV94-FC-AUG':
                    geo=json.loads((AUDIT/'great-basin-registration.json').read_text())
                    lat,lon=geo['mapped_positions'][sid]['approximate_area_corners_lat_lon'][2]
                    r.update(Lat=lat,Lon=lon,Qualification='Position stress at the eastern corner of the mapped study-area frame; it is a location assumption.')
        elif name == 'dated-sections-omit':
            rows=[r for r in rows if r['EvidenceID'] not in DATED_IDS]
        elif name in ('dated-capture-alternate','kootenai-position-east'):
            registered=json.loads((AUDIT/'kootenai-section-positions.json').read_text())['sections']
            for r in rows:
                sid=r['EvidenceID']
                if sid=='R1982-KOOT-776':
                    section=registered[0 if name=='dated-capture-alternate' else 1]
                    r.update(Lon=section['longitude'],Lat=section['latitude'])
                    if name=='dated-capture-alternate':
                        r.update(Limit_cm=1.5,Qualification='Only the lower reported thickness of the first secure component is transferred.')
                elif name=='dated-capture-alternate':
                    if sid in {'B2020-203','B2020-251'}:
                        r.update(Kind='upper_limit',Limit_cm=r['Thickness'],Thickness='',Qualification='Whole discrete bed caps local deposition if focusing can add material and net loss is negligible.')
                    elif sid in {'B2020-129','B2020-147'}:
                        r.update(Limit_cm=2*float(r['Limit_cm']),Qualification='Hypothetical half-capture floor for the dated or whole-host interval.')
                    elif sid=='B2020-141':
                        r.update(Limit_cm=530,Lat=48.596931,Lon=-116.971806,Site='Hager Lake central coring area',
                            Reference='Moseley, Bursik and Mehringer (1992), Hager locality 2.',
                            Evidence='Linked replacement of the 1973 mire bracket by the single 1992 central-basin package. 17.4 minus 12.1 m.',
                            Qualification='Complete capture in the combined four overlapping drives, despite observed mixing and sinking.')
        elif name == 'lake-field-omit':
            rows=[r for r in rows if r['EvidenceID']!='B2020-045' and not r['EvidenceID'].startswith(('SP2007-', 'SP2008-', 'CH2008-', 'HER2014-', 'ROS2004-', 'SC2020-', 'OS2012-', 'MAR2009-', 'SHP2023-'))]
        elif name == 'lake-capture-alternate':
            rows=[r for r in rows if r['EvidenceID']!='SHP2023-RDF1A']
            for r in rows:
                sid=r['EvidenceID']
                if sid.startswith(('ROS2004-','CH2008-')) and r['Kind']=='measurement':
                    r.update(Kind='upper_limit',Limit_cm=r['Thickness'],Thickness='',Qualification='Complete recovered lake package permits lateral addition. Requires no net thinning relative to original local deposition.')
                elif sid.startswith(('HER2014-','SP2007-','SP2008-','SHP2023-')) or sid in {'CH2008-RED','OS2012-POC','B2020-045'}:
                    r['Limit_cm']=2*float(r['Limit_cm'])
                    r['Qualification']='A recovered thickness at least half the original local deposit; a conditional stress scenario without a measured retention fraction.'
        elif name == 'field-range-alternate':
            rows=[r for r in rows if r['EvidenceID']!='SC2020-RS18']
            endpoints={'SC2020-RS05':5,'SC2020-RS17':4,'SC2020-RS21':4,'MAR2009-S05':9,'MAR2009-S15':66,'MAR2009-S41':8}
            for r in rows:
                if r['EvidenceID'] in endpoints:
                    r.update(Kind='upper_limit',Limit_cm=endpoints[r['EvidenceID']],Qualification='Alternative recovered upper endpoint assumes complete local capture and no net thinning.')
        elif name == 'young-sections-omit':
            rows = [r for r in rows if not r['EvidenceID'].startswith('Y1990-A')]
        elif name == 'young-section-positions':
            points = {int(v['site']):v for v in csv.DictReader((AUDIT/'young1990-section-positions.csv').open())}
            for r in rows:
                if r['EvidenceID'].startswith('Y1990-A'):
                    v = points[int(r['EvidenceID'].split('-A')[1])]
                    r.update(Lon=v['upper_kirk_anchor_longitude'], Lat=v['upper_kirk_anchor_latitude'],
                             Qualification='Alternative coherent registration using the author-identified equivalent section 47.')
        elif name == 'young-distal-omit':
            ids = {f'B2020-{n:03}' for n in (4,63,68,76,229,232,264,290,291)}
            rows = [r for r in rows if r['EvidenceID'] not in ids]
        elif name == 'distal-transfer-alternates':
            for r in rows:
                if r['EvidenceID']=='B2020-229':
                    r.update(Kind='measurement', Thickness=.5, Limit_cm='', Evidence='Mathews approximate 5 mm with smearing reported. Tests the assumption that the apparent thickness represents the complete bed.')
                elif r['EvidenceID']=='B2020-291':
                    r.update(Kind='upper_limit', Limit_cm=.2, Evidence='Young Table 1.1 gives ≤0.2 cm at the upland reference density.')
        elif name == 'round-meadow-one':
            rows = [r for r in rows if not r['EvidenceID'].startswith('W2015-RM-') or r['EvidenceID']=='W2015-RM-V3']
        elif name == 'offshore-omit':
            rows = [r for r in rows if r['EvidenceID']!='R1967-BB326-36']
        elif name == 'western-field-omit':
            rows = [r for r in rows if not r['EvidenceID'].startswith(('W2015-RM-','C2014-','SW1995-','J2025-NTL','BC2010-8','Y1990-C-'))]
        elif name == 'omit-bounds':
            rows = [r for r in rows if r['Kind'] == 'measurement']
        elif name == 'half-transfer':
            for r in rows:
                if r['Kind'] == 'lower_limit':
                    r['Limit_cm'] = float(r['Limit_cm']) / 2
                elif r['Kind'] == 'upper_limit':
                    r['Limit_cm'] = float(r['Limit_cm']) * 2
        elif name == 'portage-upper':
            r = next(r for r in rows if r['EvidenceID'] == 'B2020-134')
            r['Kind'], r['Limit_cm'] = 'upper_limit', 1.905
        elif name == 'range-upper':
            limits = {'B2020-017':6, 'B2020-019':6, 'B2020-122':30}
            for r in rows:
                if r['EvidenceID'] in limits:
                    r['Kind'], r['Limit_cm'] = 'upper_limit', limits[r['EvidenceID']]
        elif name == 'omit-new-records':
            rows = [r for r in rows if r['EvidenceID'] in legacy_ids]
            assert len(rows) == 40
        elif name == 'new-beds-lower':
            for r in rows:
                if r['EvidenceID'] not in legacy_ids and r['Kind'] == 'measurement':
                    r['Kind'], r['Limit_cm'], r['Thickness'] = 'lower_limit', r['Thickness'], ''
        elif name == 'weak-identity-omit':
            rows = [r for r in rows if r['EvidenceID'] not in WEAK_IDS]
        elif name == 'yellowstone-one':
            rows = [r for r in rows if not r['EvidenceID'].startswith('M2023-YL') or r['EvidenceID']=='M2023-YL16-5A']
        elif name == 'yellowstone-2c-5mm':
            next(r for r in rows if r['EvidenceID']=='M2023-YL16-2C')['Thickness'] = .5
        elif name == 'yellowstone-slump':
            r = next(r for r in csv.DictReader((AUDIT/'model-probes.csv').open()) if r['EvidenceID']=='M2023-YL16-6A')
            rows.append(r)
        elif name == 'omit-gold':
            rows = [r for r in rows if r['EvidenceID']!='B2023-GOLD']
        elif name == 'late-source-omit':
            rows = [r for r in rows if r['EvidenceID'] not in LATE_IDS]
            assert len(rows) == len(source) - len(LATE_IDS)
        elif name == 'skull-omit':
            rows = [r for r in rows if r['EvidenceID'] != 'MW1986-SKULL']
        elif name in ('field-source-omit', 'bracket-omit'):
            excluded = FIELD_IDS if name == 'field-source-omit' else BRACKET_IDS
            rows = [r for r in rows if r['EvidenceID'] not in excluded]
            assert len(rows) == len(source) - len(excluded)
        elif name == 'lake-core-alternates':
            r = next(r for r in rows if r['EvidenceID'] == 'B2020-052')
            r.update(EvidenceID='L1998-LITTLE91', Site='Little Lake, core 91', Thickness=1,
                     Evidence='Long et al. 1998 Table 2, core 91, 7.74–7.75 m, lake-scale coordinates.')
            r = next(r for r in rows if r['EvidenceID'] == 'H1997-MAHONEY')
            r.update(Limit_cm=80, Evidence='Alternate retained package from Overmann et al. 1993 Fig. 3 and text. Same primary-capture assumption as the principal upper bound.')
        elif name == 'bachelor-west-alternate':
            with (AUDIT/'revised-locality-ledger.csv').open() as stream:
                west = next(r for r in csv.DictReader(stream) if r['record_id'] == 'SG1989-WEST')
            r = next(r for r in rows if r['EvidenceID'] == 'B2020-046')
            r.update(EvidenceID=west['record_id'], Site=west['locality'], Lat=west['latitude'],
                     Lon=west['longitude'], Kind='upper_limit', Limit_cm=west['bound_cm'],
                     Reference=west['primary_reference'], Evidence=west['evidence_locator'],
                     Qualification=west['caveats'])
        path = folder / 'input.csv'
        write_csv(path, rows)
        result[name] = dict(file=str(path), sha256=sha(path), definition=definition,
                            measurements=sum(r['Kind']=='measurement' for r in rows),
                            lower_limits=sum(r['Kind']=='lower_limit' for r in rows),
                            upper_limits=sum(r['Kind']=='upper_limit' for r in rows))
    write_json(root / 'scenarios.json', dict(principal_input_sha256=sha(INPUT), cases=result))


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
