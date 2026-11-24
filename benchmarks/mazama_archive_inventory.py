"""Build the Mazama site inventory for location-specific posterior checks."""
import csv
import json
from pathlib import Path

from benchmarks.mazama import AUDIT

def build():
    upper = json.loads((AUDIT/'interpreted-archives.json').read_text())
    manifest=json.loads((AUDIT.parents[1]/'sources/mazama/source-manifest.json').read_text())
    urls=json.loads((AUDIT/'source-url-catalogue.json').read_text())
    urls.update({r['file']:r['url'] for r in manifest})
    def public_url(record):
        for source in record['evidence_source'].split('; '):
            if source.startswith(('http://','https://')):
                return source
            if source.strip() in urls:
                return urls[source.strip()]
        return 'https://doi.org/10.1007/s00445-020-1362-1' if record['record_id'].startswith('B2020-') else ''
    sites = list(upper['sites'])
    groups = {'cryptotephra':'cryptotephra_only','tentative_signal':'attribution_uncertain','trace_no_numeric_limit':'occurrence_only'}
    with (AUDIT/'revised-locality-ledger.csv').open() as stream:
        for r in csv.DictReader(stream):
            unmeasured = r['quantity_kind']=='unquantified' and r['verification']!='compilation_checked_primary_pending' and r['deposit_class'] not in ('precursor_comparison','mixed_or_redistributed_wetland_pumice') and r['fallout_model_status'] not in ('duplicate_field_log','hold_phase_or_bulk_transfer')
            if (r['quantity_kind'] not in groups and not unmeasured) or not r['latitude'] or r['accepted_for_model']=='True':
                continue
            reading = groups.get(r['quantity_kind'],'occurrence_only')
            if r['record_id'] in {'B2020-003','P2018-MORT12'}:
                reading = 'attribution_uncertain'
            sites.append(dict(id=r['record_id'],name=r['locality'],latitude=float(r['latitude']),
                longitude=float(r['longitude']),archive=r['setting'],reading=reading,confidence='qualified',
                horizon=r['core_depth_or_sample_interval'] or r['identity'],inspection=r['evidence_locator'],
                preservation=r['caveats'],control=r['thickness_quality'],reported_fallout_limit_cm=None,
                reference=r['primary_reference'],url=public_url(r),in_principal_as=None))
    target = dict(schema=upper['schema'],target=upper['target'],sites=sites,scenarios=[],
        qualification='Quantitative measurements and interpreted bounds enter the principal likelihood once. The remaining sites receive map and local predictions only. They supply no separately calibrated recognition likelihood.')
    (AUDIT/'archive-sites.json').write_text(json.dumps(target,indent=2,ensure_ascii=False)+'\n')
    print(len(sites),'registered archive locations')

if __name__=='__main__':
    build()
