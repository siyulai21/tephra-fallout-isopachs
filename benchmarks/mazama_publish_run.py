"""Assemble independently calculated Mazama reports and refresh the catalogue from the fitted packets."""
import csv
import json
from pathlib import Path
import shutil

from benchmarks.mazama import AUDIT, INPUT, sha
from benchmarks.mazama_recompute import PRINCIPAL, ROOT, STUDIES, PHASE, execute
from benchmarks.mazama_release import assemble, principal_details, regional_details, save
from benchmarks.mazama_sensitivity import prepare
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
        for path in sorted(folder.iterdir()):
            if path.is_file() and (published/path.name).is_file():
                shutil.copy2(path,published/path.name)
    execute('benchmarks.principal_volumes','--output','comparisons/principal-volumes.csv')
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
           'comparisons/directional/','comparisons/spherical-mixture/')
    specific={'eruptions/mazama/README.md','eruptions/README.md','comparisons/principal-volumes.csv','comparisons/method-scores.png'}
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
    saved_case_overview(PRINCIPAL)
    add_comparison_to_run(PRINCIPAL)
    prepare(STUDIES)
    regional_details(PRINCIPAL,STUDIES,only='principal')
    principal_details(PRINCIPAL,STUDIES)
    assemble(PRINCIPAL,STUDIES,PHASE)
    extra=HOME/'source-audit'
    execute('benchmarks.mazama_local_comparison','--scenarios',extra/'sensitivities','--output',extra)
    execute('benchmarks.mazama_heldout','--home',HOME,'--output',extra)
    execute('benchmarks.mazama_release_check','--home',HOME,'--output',extra/'publication-checks.json')
    from benchmarks.refresh_figures import refresh_case
    record=next(r for r in datasets() if r['slug']=='mazama')
    replay=ROOT/'visual-replay'
    refresh_case(record,replay,scopes=('comparison',))
    shutil.copyfile(replay/'mazama/figure-replay.json',HOME/'figure-replay.json')
    refresh_summaries()
    update_inventory()
    save(extra/'publication-input.json',dict(input_sha256=sha(INPUT),observations=len(list(csv.DictReader(INPUT.open()))),
        principal_sources={name:sha(HOME/name) for name in ('report.json','nested-isopachs/deposit.json',
            'spherical-mixture/deposit.json','directional/deposit.json')},prior_evidence_revision='5fa6cae5a4e17636dd878b2776b9f71a837601e4'))
    checkpoint=HOME.parent/'audit-run'
    if checkpoint.exists() and read(checkpoint/'nested-map.json')['input_sha256']==sha(INPUT):
        shutil.rmtree(checkpoint)
    print('Prepared Mazama reports and cross-eruption summaries for verification.',flush=True)


if __name__=='__main__':
    main()
