"""Prepare source-sensitivity diagnostics and assemble the reviewed Mazama release.

This module uses saved choices and posterior packets; it changes no method or
observation. Publication remains a separate reviewed Git operation.
"""
import argparse
import csv
import json
from pathlib import Path
import shutil

import numpy as np

from benchmarks.mazama import INPUT, AUDIT, sha, write_csv
from benchmarks.mazama_sensitivity import local_checks
from src.empirical_isopach.core.audit import audited_data
from src.empirical_isopach.core.distributions import predictive_distribution
from src.empirical_isopach.models.extent_review import ExtentFields, ExtentView
from src.empirical_isopach.region import SurveyRegion, integrate_region, regional_draws
from src.empirical_isopach.visualization.deposit_field import open_field, FAMILIES
from src.empirical_isopach.visualization.principal import case_section, volume_rows


def read(path):
    return json.loads(Path(path).read_text())


def save(path, obj):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, indent=2, allow_nan=False)+'\n')


def models(data, descriptions, bounds):
    from src.empirical_isopach.models.gaussian_process.model import SpatialModel, Candidate
    from src.empirical_isopach.models.gaussian_process.ensemble import StackedModel
    from src.empirical_isopach.models.cubic_spline.model import TrendRegularizedSpline, SplineCandidate
    gp = descriptions['stacked']['model']
    fitted = [SpatialModel(Candidate(**c['model']['candidate'])).fit(data) for c in gp['components']]
    s = descriptions['spline']['model']
    return {'stacked':StackedModel(fitted, [c['weight'] for c in gp['components']],
                                 gp['observation_discrepancy_variance_log']),
            'spline':TrendRegularizedSpline(SplineCandidate(**s['candidate']),
                        quadrature_order=s['quadrature_order']).fit(data, bounds=bounds)}


def regional_details(principal, scenarios):
    """Fixed chosen settings, common footprint and coherent conditional draws."""
    from src.probabilistic_isopach.global_volume import EarthDomain
    full, meta = audited_data(INPUT)
    records = list(csv.DictReader(INPUT.open()))
    bound_idx = np.flatnonzero(full.censored)
    ids = [records[int(full.row_ids[i])-2]['EvidenceID'] for i in bound_idx]
    region = SurveyRegion(full.xy[~full.censored])
    reference = read(principal/'report.json')
    domain = EarthDomain.wgs84(meta['vent_lonlat'][1])
    angles = np.arctan2(full.xy[bound_idx,0]/domain.axis_scale,
                        full.xy[bound_idx,1]*domain.axis_scale)
    radii = np.r_[0., np.geomspace(.02, domain.rho_max*(1-1e-10), 768)]
    jobs = [('principal', INPUT, reference['methods'])]
    jobs += [(p.name, p/'input.csv', read(p/'regional/report.json')['methods'])
             for p in sorted(scenarios.iterdir()) if (p/'regional/report.json').is_file()]
    for name, path, descriptions in jobs:
        target = scenarios/name/'regional-details.json'
        if target.exists():
            assert read(target)['input_sha256'] == sha(path)
            continue
        data, _ = audited_data(path)
        footprint = SurveyRegion(data.xy)
        fitted = models(data, descriptions, reference['fit_bounds_km'])
        output = dict(input_sha256=sha(path), principal_input_sha256=sha(INPUT),
            common_hull=region.describe(), model_domain=footprint.describe(),
            script_sha256=sha(Path(__file__)), methods={},
            scope='Conditional on principal selected settings and weights; no retuning. All volume comparisons use the unchanged26-measurement hull. Predictions outside a scenario observation hull are unavailable, not zero. Latent site intervals omit parameter/selection and source-transfer uncertainty.')
        for key, model in fitted.items():
            volume = integrate_region(model, region)
            joint = regional_draws(model, region, volume['expected_km3'])
            inside = footprint.contains(full.xy[bound_idx])
            sites = []
            for sid, idx, valid in zip(ids, bound_idx, inside, strict=True):
                q = (np.exp([predictive_distribution(model, full.xy[idx:idx+1]).quantile(p)[0]
                             for p in (.025,.5,.975)]).tolist() if valid else None)
                sites.append(dict(evidence_id=sid, inside_scenario_hull=bool(valid),
                                  latent_q025_q50_q975_cm=q))
            rays, crossings = [], []
            for sid, angle in zip(ids, angles, strict=True):
                for offset in (-11.25,0.,11.25):
                    a = angle+np.deg2rad(offset)
                    xy = radii[:,None]*[np.sin(a)*domain.axis_scale,np.cos(a)/domain.axis_scale]
                    mask = footprint.contains(xy)
                    z = np.full(len(radii),np.nan)
                    z[mask] = predictive_distribution(model,xy[mask]).quantile(.5)/np.log(10.)
                    rays.append(z)
                    for cm in (.5,1.,5.,10.):
                        threshold = np.log10(cm)
                        idx = np.flatnonzero(mask[:-1]&mask[1:]&((z[:-1]>=threshold)!=(z[1:]>=threshold)))
                        rr = [radii[j]+(radii[j+1]-radii[j])*(threshold-z[j])/(z[j+1]-z[j]) for j in idx]
                        crossings.append(dict(evidence_id=sid,bearing_offset_deg=offset,level_cm=cm,
                            crossings_authalic_km=domain.arc_km(rr).tolist(),
                            crossing_grid_brackets_authalic_km=[[float(domain.arc_km(radii[j])),float(domain.arc_km(radii[j+1]))] for j in idx],
                            scope='Only crossings bracketed entirely inside this scenario hull; outside-domain reach is unknown.'))
            target.parent.mkdir(parents=True,exist_ok=True)
            packet=target.parent/(key+'-profiles.npz')
            np.savez_compressed(packet,radii_km=radii,log10_cm=np.asarray(rays))
            output['methods'][key] = dict(volume=volume,joint_volume=joint,sites=sites,crossings=crossings,
                profiles_sha256=sha(packet),profile_order='Canonical principal bound order, offsets -11.25,0,+11.25 degrees')
        save(target,output)
        print('Regional uncertainty and local checks:',name,flush=True)


def principal_details(principal, target):
    """Use the exact current continuation, including section-location diagnostics."""
    full, meta = audited_data(INPUT)
    georef = read(AUDIT/'camp-creek-georeferencing.json')
    from src.empirical_isopach.models.directional.ensemble.fields import inverse_cdf
    from src.Laea import make_laea_transform
    forward, _, _ = make_laea_transform(*meta['vent_lonlat'])
    for family in FAMILIES:
        folder = target/'principal'/family
        folder.mkdir(parents=True,exist_ok=True)
        view, _, sources = open_field(principal/family,family,full,meta)
        if not isinstance(view,ExtentView):
            parent = view.field if family=='spherical-mixture' else view.model
            view = ExtentView(ExtentFields(family,parent,full,meta['vent_lonlat'][1],('parent',)),view.domain,[1.])
        result = dict(input_sha256=sha(INPUT),family=family,source_sha256=sources,
                      local_checks=local_checks(view,folder),section_checks=[])
        # Project polygon vertices directly. They carry no observation values
        # and never enter a likelihood.
        for group in georef:
            vertices = group['polygon']['rings'][0][:-1]
            coords = np.asarray(vertices)
            xy = np.column_stack(forward.transform(coords[:,0],coords[:,1]))
            values = np.concatenate(view.fields.variants(xy))
            q = np.exp(inverse_cdf(values.T,view.weights,[.025,.5,.975])).T
            result['section_checks'].append(dict(representative_id=group['representative_id'],
                blm_section_id=group['blm_section_id'],vertices_lonlat=vertices,
                vertex_latent_q025_q50_q975_cm=q.tolist(),
                minimum_vertex_median_cm=float(q[:,1].min()),maximum_vertex_median_cm=float(q[:,1].max()),
                scope='Fixed-fit predictions at section vertices, no new likelihood and no measurement. This checks positional scale under the fitted surface, not the correctness of the primary interpretation.'))
        save(folder/'local-checks.json',result)
        print('Principal local checks:',family,flush=True)


def assessment(home):
    source = home/'source-audit'
    scenarios = source/'sensitivities'
    rows = []
    for folder in sorted(scenarios.iterdir()):
        scenario=folder.name
        if (folder/'regional-details.json').exists():
            report=read(folder/'regional-details.json')
            for family,m in report['methods'].items():
                q=m['joint_volume']['conditional_quantiles_km3']
                rows.append(dict(scenario=scenario,method=family,domain='common26-measurement hull',
                    median_map_volume_km3=m['volume']['median_surface_km3'],lower95_km3=q['0.025'],upper95_km3=q['0.975'],
                    fit_passed='',draw_checks_passed=m['joint_volume']['expected_grid_converged'],
                    map_integration_passed=m['volume']['converged']))
        for family in FAMILIES:
            path=folder/family/'report.json'
            if path.exists():
                r=read(path);q=r['conditional_q025_q50_q975_km3']
                rows.append(dict(scenario=scenario,method=family,domain='whole Earth',median_map_volume_km3=r['median_map_volume_km3'],
                    lower95_km3=q[0],upper95_km3=q[2],fit_passed=r['fit_passed'],draw_checks_passed=r['draw_checks_passed'],map_integration_passed=r['map_integration_passed']))
    for r in volume_rows(home):
        if r['method'] in FAMILIES:
            rows.append(dict(scenario='principal',method=r['method'],domain='whole Earth',median_map_volume_km3=r['median_map_volume_km3'],
                lower95_km3=r['conditional_volume_lower95_km3'],upper95_km3=r['conditional_volume_upper95_km3'],fit_passed='',
                draw_checks_passed=r['complete_draw_checks_passed'],map_integration_passed=r['map_integration_passed']))
    rows.sort(key=lambda r:(r['method'],r['scenario']))
    write_csv(source/'source-sensitivity-volumes.csv',rows)
    lines=['# Mazama source interpretation tests','',
        '[Case home](../../README.md) · [Evidence and calculations](../../../../data/audits/Mazama/BOUND_REVIEW.md) · [Numerical table](source-sensitivity-volumes.csv)','',
        'All scenarios retain the same26 reported point values except the separately labelled Williams omission. Bounds are counted once. Global scenarios are independent full-data refits, with continuation calibration excluding Mazama. Regional scenarios refit coefficients with the principal chosen settings and weights and integrate the original26-measurement hull. These refits test evidence assumptions; they do not repeat the principal geographic validation or all eight spherical prior fits.','',
        '| Method | Scenario | Median-map volume (km³) | Conditional95% interval (km³) | Draw checks | Map integration |',
        '|---|---|---:|---:|---|---|']
    for r in rows:
        lines.append(f"| {r['method']} | {r['scenario']} | {r['median_map_volume_km3']:.3g} | {r['lower95_km3']:.3g}–{r['upper95_km3']:.3g} | {r['draw_checks_passed']} | {r['map_integration_passed']} |")
    lines += ['','The principal case home uses its full accepted hull for regional volumes; the two principal regional rows above deliberately use the smaller common footprint. The median-map integral and the posterior distribution of complete-deposit integrals are different quantities. False numerical flags remain provisional; they are not hidden by a successful map integral.','',
        'Each scenario folder preserves its input, global posterior packets, attempt histories, local point intervals and all sampled0.5,1,5,10 cm crossings on each bound bearing and adjacent±11.25° bearings. Regional crossings are limited to the scenario observation hull; a missing crossing outside it is unavailable. Crossing brackets give the radial grid resolution. These are conditional predictions, not independent measured brackets. Principal local checks also retain Camp Creek section-vertex predictions.','',
        'The full comparison uses identical input hashes and geographic folds. The reports retain geographic and quadrant warnings, supported-contour rejection reasons and the eight spherical prior intervals. Numerical convergence does not establish observational support over Earth.','',
        '[Omit bounds](sensitivities/omit-bounds/) · [Half transfer](sensitivities/half-transfer/) · [Portage upper endpoint](sensitivities/portage-upper/) · [Other range endpoints](sensitivities/range-upper/) · [Williams omission](sensitivities/omit-williams/) · [Principal local checks](sensitivities/principal/)','',
        '[Publication checks](publication-checks.json) · [Release identities](release.json) · [Earlier published results](../../../../archive/2026-11-23-mazama-source-audit/README.md)']
    (source/'README.md').write_text('\n'.join(lines)+'\n')
    intro=['# Mazama climactic fallout','',
        '**40 source-qualified observations:26 reported thicknesses and14 conditional lower bounds.** The audit reconciles293 compilation entries plus60 additional records or leads. It corrects source identity, phase, location and thickness problems and retains traces and cryptotephra without inventing a thickness.260 original records still await independent primary verification; the accepted set remains conditional.','',
        '[Accepted input](../../data/eruptions/Mazama_reviewed.csv) · [Audit assessment](../../data/audits/Mazama/README.md) · [Every locality](../../data/audits/Mazama/LOCALITIES.md) · [Bound derivations](../../data/audits/Mazama/BOUND_REVIEW.md) · [Sources and hashes](../../data/sources/mazama/README.md) · [Source sensitivities](results/source-audit/README.md)','',
        'The target is primary airborne bulk fallout from the climactic eruption, including co-ignimbrite ash. Identified precursor beds, flow deposits and reworked totals are excluded. Preserved primary components and range minima enter once as lower bounds, conditional on their identity and preservation. Green and Tepee Lake positions are corrected; the850 cm Muir Creek facies-ambiguous value is quarantined. The Williams46 cm phase ambiguity is retained explicitly and tested by omission.','',
        'These global volumes are conditional estimates. Sparse bearings and unsampled regions remain uncertain even where a whole-Earth integral converges. Each family keeps its own interval, prior sensitivity and numerical status. [Shared support decisions](results/support/README.md) and each global extent report preserve rejected contours and unsupported gaps.','',
        case_section(home,prefix='results/'),'',
        '## Evidence, reproduction and history','',
        '![Mazama evidence coverage](../../data/audits/Mazama/evidence-map.png)','',
        'Run `tephra-isopachs compare Mazama --output results/new-mazama-run` for the registered comparison. The source-refit commands and publication verification are recorded with the [release](results/source-audit/README.md). Earlier published results and superseded audit trials retain their inputs and hashes in [dated archives](../../archive/2026-11-23-mazama-source-audit/README.md).']
    (home.parent/'README.md').write_text('\n'.join(intro)+'\n')


def assemble(principal, scenarios, phase):
    home=Path('eruptions/mazama/results')
    prior=Path('archive/2026-11-23-mazama-source-audit/previous-release.tar.gz')
    assert prior.is_file() and prior.stat().st_size>0
    assert read(principal/'report.json')['input']['input_sha256']==sha(INPUT)
    for family in FAMILIES:
        assert read(principal/family/'deposit.json')['input_sha256']==sha(INPUT)
        assert (principal/family/'extent/heldout.csv.gz').is_file()
    if home.exists():
        # The original release is already frozen in a verified dated archive.
        shutil.rmtree(home)
    shutil.copytree(principal,home)
    dest=home/'source-audit'
    shutil.copytree(scenarios,dest/'sensitivities')
    shutil.copytree(phase/'omit-williams',dest/'sensitivities/omit-williams')
    reproduction=dest/'reproduction';reproduction.mkdir(parents=True,exist_ok=True)
    for name in ('mazama.py','mazama_sensitivity.py','mazama_phase_sensitivity.py','mazama_release.py'):
        shutil.copyfile(Path('benchmarks')/name,reproduction/name)
    assessment(home)
    from src.empirical_isopach.core.catalog import datasets
    from benchmarks.directional_comparison import case_extremes
    record=next(r for r in datasets() if r['slug']=='mazama')
    extremes=case_extremes(record,read(home/'directional/single-structure/report.json'),by_octant=True)
    write_csv(dest/'octant-extremes.csv',extremes)
    files=[INPUT,home/'directional/single-structure/report.json',home/'directional/ensemble/report.json',dest/'octant-extremes.csv']
    save(dest/'release.json',dict(input_sha256=sha(INPUT),prior_revision='0a53d0211f2ac9cfc9fbeaff95217fc0d319a18b',
        primary_audit_complete=False,files={str(p.resolve().relative_to(Path.cwd())):sha(p) for p in files}))
    print('Assembled current Mazama case',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=('regional','principal','assemble'))
    p.add_argument('--principal',type=Path,required=True)
    p.add_argument('--scenarios',type=Path,required=True)
    p.add_argument('--phase',type=Path)
    a=p.parse_args()
    if a.action=='regional':regional_details(a.principal,a.scenarios)
    elif a.action=='principal':principal_details(a.principal,a.scenarios)
    else:assemble(a.principal,a.scenarios,a.phase)
