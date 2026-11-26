"""Create a case release from five completed fits and their matched geographic checks."""
import argparse
from collections import Counter
import json
import os
from pathlib import Path

import pandas as pd

from benchmarks.global_case_validation import field_validation, verify_sampling
from benchmarks.minimum_fit_results import assess
from benchmarks.publication import reported_edition, result_home, published_methods
from src.empirical_isopach.core.audit import audited_data
from src.empirical_isopach.core.catalog import data_directory, datasets
from src.empirical_isopach.core.fitting import POLICY
from src.empirical_isopach.visualization.deposit_field import digest
from src.empirical_isopach.visualization.principal import case_section, volume_rows


def link(target, directory):
    return Path(os.path.relpath(target, directory)).as_posix()


def publish(record):
    slug = record["slug"]
    previous = reported_edition(record)
    _, meta = audited_data(record["path"])
    home = result_home(record)
    if not (home/'sampling/attempts.csv').is_file() or not (home/'directional/ensemble/report.json').is_file():
        return publish_current(record)
    verify_sampling(home)
    field_validation(home, record["path"], slug, verify=False)
    field_validation(home, record["path"], slug)
    assess(record, write=True)
    sites = pd.read_csv(home / "reported-and-fitted-sites.csv").drop_duplicates("row_id")
    kinds = Counter(sites.reported_kind)
    methods = volume_rows(home, input_sha256=meta["input_sha256"])
    controls = json.loads((home / "directional/ensemble/report.json").read_text())
    reports = [home / "report.json", home / "geographic-site-validation.csv",
               home / "reported-and-fitted-sites.csv", home / "minimum-fit-comparison.csv",
               home / "sampling/attempts.csv", home / "map-display.json", home / "isopachs.png"]
    for family in ("directional", "spherical-mixture", "nested-isopachs"):
        reports.extend(home / family / f for f in ("deposit.json", "deposit.png", "median-grid.npz"))
    reports += [home / "directional/ensemble/report.json", home / "directional/single-structure/report.json",
                record["path"], previous["path"], data_directory() / record["audit"]]
    computation = home / "spherical-mixture/workflow-provenance.json"
    if computation.is_file():
        reports.append(computation)
    receipt = dict(input_sha256=meta["input_sha256"], reported_source_sha256=meta["reported_source_sha256"],
                   previous_revision=previous.get("revision"), observation_policy=POLICY,
                   observations=len(sites), reported_site_types=dict(kinds), fitted_site_types=dict(Counter(sites.fit_kind)),
                   matched_geographic_folds=int(sites.geographic_fold.nunique()), spherical_prior_fits=8,
                   fixed_direction_control_km3=controls["median_surface"]["volume_km3"],
                   directional_median_map_km3=next(r["median_map_volume_km3"] for r in methods if r["method"]=="directional"),
                   methods=methods, files={link(f, Path.cwd()):digest(f) for f in reports})
    (home / "release.json").write_text(json.dumps(receipt, indent=2, allow_nan=False)+"\n")
    write_case_page(record, sites)


def publish_current(record):
    """Authenticate current principal packets without claiming absent older controls.

    Several registered minimum-fit editions contain the newer directional
    partitions and spherical workflow provenance, but no unified legacy
    sampling ledger or fixed-direction control. Preserve that distinction.
    """
    home=result_home(record)
    data,meta=audited_data(record['path'])
    previous=reported_edition(record)
    field_validation(home,record['path'],record['slug'],verify=False)
    assess(record,write=True)
    sites=pd.read_csv(home/'reported-and-fitted-sites.csv').drop_duplicates('row_id')
    methods=volume_rows(home,input_sha256=meta['input_sha256'])
    assert {r['method'] for r in methods}==set(published_methods(record))
    files=[record['path'],previous['path'],data_directory()/record['audit']]
    for name in ('report.json','thickness-bands.json','accepted_observations.csv','heldout_predictions.csv',
                 'geographic-site-validation.csv','reported-and-fitted-sites.csv','minimum-fit-comparison.csv',
                 'map-display.json','isopachs.png','principal-volumes.csv'):
        if (home/name).is_file():files.append(home/name)
    for family in ('directional','spherical-mixture','nested-isopachs'):
        if family not in published_methods(record):continue
        files.extend(p for p in (home/family).rglob('*') if p.is_file() and p.suffix in ('.json','.npz','.gz','.csv','.png'))
    sphere=home/'spherical-mixture/report.json'
    missing=[name for name in ('sampling/attempts.csv','directional/ensemble/report.json','directional/single-structure/report.json')
             if not (home/name).is_file()]
    receipt=dict(protocol='current-principal-release-v2',input_sha256=meta['input_sha256'],
                 reported_source_sha256=meta['reported_source_sha256'],previous_revision=previous.get('revision'),
                 observation_policy=POLICY,observations=len(data.xy),published_methods=list(published_methods(record)),
                 reported_site_types=dict(Counter(sites.reported_kind)),fitted_site_types=dict(Counter(sites.fit_kind)),
                 matched_geographic_folds=int(sites.geographic_fold.nunique()),
                 spherical_prior_fits=len(json.loads(sphere.read_text())['fits']) if sphere.is_file() else 0,
                 methods=methods,unavailable_legacy_controls=missing,
                 retention_note='Current principal packets and their saved attempts are authenticated here. Missing older controls are not represented as new fits; existing external computing provenance is retained.',
                 files={link(f,Path.cwd()):digest(f) for f in sorted(set(files))})
    (home/'release.json').write_text(json.dumps(receipt,indent=2,allow_nan=False)+'\n')
    write_case_page(record,sites)


def write_case_page(record, sites):
    """Link the checked principal deposits and their available numerical evidence."""
    home = result_home(record)
    case = Path("eruptions") / record["slug"]
    previous = reported_edition(record)
    kinds = Counter(sites.reported_kind)
    history = case / "previous-minimum-treatment.md"
    old_page = case / "README.md"
    if not history.exists() and previous.get("revision") and old_page.is_file():
        history.write_bytes(old_page.read_bytes())
    prefix = link(home, case) + "/"
    audit = data_directory() / Path(record["audit"]).parent / "README.md"
    lower = kinds["lower_limit"]
    distal = int((sites.reported_kind.eq("lower_limit") & sites.projected_distance_km.gt(25)).sum())
    evidence = (("Matched predictions by locality", "geographic-site-validation.csv"),
                ("Sampling attempts and diagnostics", "sampling/attempts.csv"),
                ("Numerical checks and file identities", "release.json"),
                ("Spherical fits and computing revisions", "spherical-mixture/workflow-provenance.json"),
                ("Regional support", "support/README.md"), ("Whole-area sensitivity", "nested-whole-area/README.md"))
    checks = " · ".join(f"[{label}]({prefix}{path})" for label, path in evidence if (home / path).is_file())
    nested_review = Path('comparisons/nested-isopachs/adaptive-growth/visual-review.md')
    assessment = []
    if nested_review.is_file():
        headings={'aso-4':'Aso-4','b-tm':'B-Tm','mazama':'Mazama','ytt':'YTT',
                  'minoan':'Minoan','ilopango-tbj':'Ilopango TBJ','sakurajima-satsuma':'Sakurajima Satsuma',
                  'pinatubo-1991-layer-c':'Pinatubo Layer C','tambora':'Tambora'}
        heading=headings[record['slug']]
        body=nested_review.read_text()
        marker='\n## '+heading+'\n'
        if marker in body:
            text=body.split(marker,1)[1].split('\n## ',1)[0].strip().split('\n\n')
            # Keep the case-specific geological and predictive qualifications
            # beside the current maps, with the full three-scale review linked.
            assessment=['## Nested geometry and field agreement','',*[line for p in text[1:] for line in (p,'')],
                        f"[Three-scale comparison, site identities and contour diagnostics]({link(nested_review,case)}#{heading.lower().replace(' ','-')})",'']
    count=len(published_methods(record))
    number={4:'four',5:'five'}.get(count,str(count))
    lines = [f"# {record['name']}", "",
             f"The {number} reconstructions use the same {len(sites)} accepted fallout localities. Field reports give "
             f"{kinds['measurement']} measured thicknesses, {lower} lower limits, {kinds['upper_limit']} upper limits "
             f"and {kinds['interval']} two-sided ranges. Each lower limit enters the fit as a point at its reported "
             f"value, including the {distal} beyond 25 km. Upper limits and ranges retain their respective one-sided "
             f"and two-sided probabilities. [Source audit]({link(audit,case)}) · [Reported table]({link(previous['path'],case)}) "
             f"· [Fitting table]({link(record['path'],case)})", "",
             "The site table pairs each source classification with its fitting value, geographic fold, mapped thickness "
             "and held-out score. It includes the predictive median or cumulative probability at the site value where the method saves it. "
             "A fitted minimum can underestimate total fallout at an incomplete section. "
             f"[Reported and fitted sites]({prefix}reported-and-fitted-sites.csv) · "
             f"[Changes by method and coverage by source type]({prefix}minimum-fit-comparison.csv) · "
             "[Fitting convention](../../docs/methods/reported-minimum-fitting.md)", "",
             case_section(home, prefix=prefix, name=record["name"]).strip(), "", *assessment,
             "## Evidence and checks", "",
             checks, "",
             ("[Previous maps and geological assessment](previous-minimum-treatment.md) · " if history.is_file() else "")
             + "[Change at all distances across eruptions](../../comparisons/reported-minima/README.md)", ""]
    old_page.write_text("\n".join(lines))


def verify_release(record):
    """Check the published files, fitted site counts, prior fits and actual holdouts."""
    home = result_home(record)
    release = json.loads((home / record["release_receipt"]).read_text())
    data, meta = audited_data(record["path"])
    assert release["input_sha256"] == meta["input_sha256"]
    assert release["reported_source_sha256"] == meta["reported_source_sha256"]
    assert release["observation_policy"] == POLICY
    assert release["observations"] == len(data.xy)
    assert release["reported_site_types"]["lower_limit"] == int(data.reported_minimum.sum())
    assert release["fitted_site_types"]["measurement"] == int((~data.censored).sum())
    assert not release["fitted_site_types"].get("lower_limit", 0)
    assert release["methods"] == volume_rows(home, input_sha256=meta["input_sha256"])
    for filename, expected in release["files"].items():
        assert digest(Path(filename)) == expected, filename
    if 'spherical-mixture' in published_methods(record):
        sphere = json.loads((home / "spherical-mixture/report.json").read_text())
        assert len(sphere["fits"]) == release["spherical_prior_fits"] == 8
        assert len(sphere["validation"]["fits"]) == release["matched_geographic_folds"] == 4
    field_validation(home, record["path"], record["slug"])
    if release.get('protocol')=='current-principal-release-v2':
        assert release['published_methods']==list(published_methods(record))
        assert all(not (home/p).exists() for p in release['unavailable_legacy_controls'])
        from src.empirical_isopach.models.nested.footprints.continuous import engine,PROTOCOL
        folder=home/'nested-isopachs/footprints'
        nested=json.loads((folder/'report.json').read_text())
        assert nested['protocol']==PROTOCOL and nested['engine_sha256']==engine()
        assert nested['input_sha256']==meta['input_sha256']
        assert nested['fits']['full']['train']==data.row_ids.tolist()
        assert all(digest(folder/p)==h for p,h in nested['posterior_sha256'].items())
        for fit in nested['fits'].values():
            assert all(digest(folder/c['packet'])==c['posterior_sha256'] for c in fit['checkpoints'])
    else:
        verify_sampling(home)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("case", nargs="?", help="Registered eruption slug")
    parser.add_argument("--verify", action="store_true", help="Check a completed case or all registered minimum-fit releases")
    args = parser.parse_args()
    if not args.verify and not args.case:
        parser.error("Specify a case to publish, or --verify to check published cases")
    cases = [r for r in datasets() if r.get("reported_file") and (not args.verify or r.get("release_receipt"))
             and (not args.case or r["slug"] == args.case)]
    if not cases:
        parser.error("No matching registered source and fitting tables")
    for record in cases:
        if args.verify:
            verify_release(record)
            print(record["name"], "release files, sampling attempts and geographic holdouts verified")
        else:
            publish(record)
