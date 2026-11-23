"""Assemble independently calculated Mazama reports and refresh the catalogue from the fitted packets."""
import csv
import json
from pathlib import Path
import shutil

from benchmarks.mazama import AUDIT, INPUT, sha
from benchmarks.mazama_catalogue import paired_estimates, source_summary, update_pages
from benchmarks.extent_catalogue import summary as finite_catalogue
from benchmarks.mazama_recompute import PRINCIPAL, ROOT, STUDIES, PHASE, execute
from benchmarks.mazama_release import assemble, principal_details, regional_details, save
from benchmarks.mazama_sensitivity import prepare
from benchmarks.nested_edge_deletion import run as nested_edge_deletion
from src.empirical_isopach.core.catalog import datasets
from src.empirical_isopach.visualization.comparison import add_comparison_to_run
from src.empirical_isopach.visualization.summary import saved_case_overview, all_case_scores, principal_report
from src.empirical_isopach.visualization.principal import volume_rows

HOME = Path('eruptions/mazama/results')


def read(path):
    return json.loads(path.read_text())


def refresh_summaries():
    commands=[
        ('regional','benchmarks.regional',['--summarize-only','--published']),
        ('spherical-mixture','benchmarks.global_posterior',['--summarize-only','--published']),
        ('directional','benchmarks.azimuth_summary',[]),
        ('nested-isopachs','benchmarks.nested_summary',[])]
    for family,command,extra in commands:
        folder=ROOT/'summaries'/family
        execute(command,*extra,'--output',folder)
        published=Path('comparisons')/family
        for path in sorted(folder.rglob('*')):
            target = published / path.relative_to(folder)
            if path.is_file() and target.is_file():
                shutil.copy2(path,target)
    # Parent sensitivities above remain accessible alongside the current finite-spread estimates.
    destination = ROOT / 'summaries/finite-spread'
    finite_catalogue(destination)
    for path in sorted(destination.rglob('*')):
        published = path.relative_to(destination)
        if path.is_file() and published.is_file():
            shutil.copy2(path, published)
    paired, kernel = paired_estimates(HOME, ROOT)
    update_pages(paired, kernel)
    reviews=[]
    homes={}
    for record in datasets():
        folder=Path('eruptions')/record['slug']/'results'
        homes[record['name']]=folder
        reviews.append((record['name'],read(folder/'report.json'),read(folder/'directional/azimuth/report.json'),
                        read(folder/'spherical-mixture/report.json'),principal_report(folder)))
    all_case_scores(Path('comparisons'),reviews,homes=homes)
    cases=Path('eruptions/README.md')
    values={r['method']:r['median_map_volume_km3'] for r in volume_rows(HOME)}
    counts=read(AUDIT/'audit-summary.json')['principal_kinds']
    new='| [Mazama](mazama/README.md) | '+' | '.join(f'{values[k]:,.3g}' for k in ('directional','spherical-mixture','nested-isopachs'))
    new+=f" | {counts['measurement']} measurements, {counts['lower_limit']} lower and {counts['upper_limit']} upper bounds | Strong source and archive clustering, wide unsampled sectors and provisional numerical checks. |"
    text=cases.read_text().splitlines()
    assert sum(line.startswith('| [Mazama]') for line in text)==1
    cases.write_text('\n'.join(new if line.startswith('| [Mazama]') else line for line in text)+'\n')


def update_inventory():
    from benchmarks.publication import INVENTORY,read_hashes
    previous=read_hashes()
    areas=('eruptions/mazama/results/','comparisons/regional/','comparisons/nested-isopachs/',
           'comparisons/directional/','comparisons/spherical-mixture/','comparisons/global-extent/','comparisons/support/')
    specific={'data/datasets.json','eruptions/mazama/README.md','eruptions/README.md','comparisons/principal-volumes.csv','comparisons/method-scores.png'}
    registered = next(r for r in read(Path('data/datasets.json'))['datasets'] if r['slug'] == 'mazama')
    provenance = AUDIT / 'provenance.json'
    assert (Path('data') / registered['file']).resolve() == INPUT.resolve()
    assert (Path('data') / registered['audit']).resolve() == provenance.resolve()
    assert read(provenance)['cases']['principal']['sha256'] == sha(INPUT)
    for item in (INPUT, provenance):
        previous[item.resolve().relative_to(Path.cwd()).as_posix()] = sha(item)
    for name in list(previous):
        if name.startswith(areas) or name in specific:
            path=Path(name)
            if path.is_file():
                previous[name]=sha(path)
            elif name.startswith('eruptions/mazama/results/'):
                del previous[name]
            else:
                raise FileNotFoundError(name)
    for p in HOME.rglob('*'):
        if p.is_file() and not p.is_relative_to(HOME/'source-audit'):
            previous[str(p)]=sha(p)
    INVENTORY.write_text(''.join(f'{h}  {name}\n' for name,h in sorted(previous.items())))


def main():
    parent = PRINCIPAL / 'nested-whole-area'
    nested_edge_deletion((INPUT, parent, parent / 'edge-deletion'))
    saved_case_overview(PRINCIPAL)
    add_comparison_to_run(PRINCIPAL)
    prepare(STUDIES)
    regional_details(PRINCIPAL,STUDIES,only='principal')
    principal_details(PRINCIPAL,STUDIES)
    assemble(PRINCIPAL,STUDIES,PHASE)
    extra=HOME/'source-audit'
    receipt = read(extra/'release.json')
    receipt['directional_median_map_km3'] = read(HOME/'directional/ensemble/report.json')['median_surface']['volume_km3']
    save(extra/'release.json', receipt)
    case_page = HOME.parent / 'README.md'
    overview = '![Mazama evidence coverage](../../data/audits/Mazama/evidence-map.png)'
    lake_detail = ('Ten Yellowstone Lake cores contain 4–5 mm of Mazama ash at eight positions. '
                   'Seven positions enter the principal fit. Two core pairs share positions, and YL16-6A lies at a slump toe. '
                   'The enlarged map resolves the cores that overlap at the continental scale. '
                   '[Core identities and bed measurements](../../data/audits/Mazama/YELLOWSTONE.md)\n\n'
                   '![Yellowstone Lake core locations and ash thicknesses](../../data/audits/Mazama/yellowstone-cores.png)\n\n'
                   'The fit also includes 8 cm at Overland Lake, 4 cm at Soldier Lake, 1 cm in Great Salt Lake core GSL00-4A-2H-2, '
                   'and the approximately 4 cm airfall bed in the Lyman Lake composite. '
                   '[Primary core reports](../../data/audits/Mazama/FOLLOWUP_SEARCH.md) · '
                   '[Lyman position and stratigraphy](../../data/audits/Mazama/LATE_SOURCE_REVIEW.md)')
    page_text = case_page.read_text()
    assert page_text.count(overview) == 1
    edge_note = ('The nested-model edge test refits the deposit after withholding the most influential outer site '
                 'and again after withholding its geographic cluster. '
                 '[Withheld rows, volumes and numerical checks](results/nested-whole-area/edge-deletion/README.md)')
    case_page.write_text(page_text.replace(overview, overview + '\n\n' + lake_detail + '\n\n' + edge_note))
    execute('benchmarks.mazama_local_comparison','--scenarios',extra/'sensitivities','--output',extra)
    execute('benchmarks.mazama_heldout','--home',HOME,'--output',extra)
    execute('benchmarks.mazama_release_check','--home',HOME,'--output',extra/'publication-checks.json')
    source_summary(extra)
    from benchmarks.refresh_figures import refresh_case
    record=next(r for r in datasets() if r['slug']=='mazama')
    replay=ROOT/'visual-replay'
    refresh_case(record,replay,scopes=('comparison',))
    shutil.copyfile(replay/'mazama/figure-replay.json',HOME/'figure-replay.json')
    refresh_summaries()
    update_inventory()
    origin = AUDIT / 'publication-request.json'
    provenance = dict(input_sha256=sha(INPUT),observations=len(list(csv.DictReader(INPUT.open()))),
        principal_sources={name:sha(HOME/name) for name in ('report.json','nested-isopachs/deposit.json',
            'spherical-mixture/deposit.json','directional/deposit.json')},prior_evidence_revision='929f26864cce78ff71a9af52492492094784f583')
    if origin.is_file() and read(origin)['input_sha256']==sha(INPUT):
        computation = read(origin)
        assert computation['input_sha256'] == provenance['input_sha256']
        provenance['calculation'] = computation
    provenance['nested_edge_deletion_sha256'] = sha(HOME / 'nested-whole-area/edge-deletion/report.json')
    save(extra/'publication-input.json', provenance)
    checkpoint=HOME.parent/'audit-run'
    if checkpoint.exists() and read(checkpoint/'nested-map.json')['input_sha256'] in {sha(INPUT), 'caeb106f5e3e9e835c8569da245cefdbbfa58de7c26ce88636f59fa51221d2db', '8ff013f2f0026d5eef7c0ca76e0d12584b4fbf11861160e972c9d0a81f847bfe'}:
        shutil.rmtree(checkpoint)
    print('Prepared Mazama reports and cross-eruption summaries for verification.',flush=True)


if __name__=='__main__':
    main()
