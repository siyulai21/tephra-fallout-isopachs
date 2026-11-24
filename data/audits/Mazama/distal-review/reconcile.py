"""Apply a source-led deposit interpretation overlay to the frozen locality evidence."""
import csv
import json
from pathlib import Path
from urllib.parse import quote

HERE = Path(__file__).resolve().parent
AUDIT = HERE.parent
COMPLETE_RETENTION = ('Conditional on complete local retention within this package and sufficient thickness after compaction. '
                      'Extra catchment or lake sediment can enlarge the package. Erosion, compaction, or noncapture can invalidate the ceiling.')
PRESERVED_BED = ('The primary bed approximates the original local bulk fall at this position. No numerical correction for focusing, loss or compaction has been measured.')
TARGET = 'original local climactic Mazama airborne bulk thickness, conditional on the stated preservation assumptions'


def read(path):
    return json.loads(path.read_text())


def write(path, data):
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + '\n')


def qualification(item):
    if item.get('assumption'):
        return item['assumption']
    if item['scope'] == 'historical author upper equivalent':
        return 'Conditional on Young’s published 700 kg/m³ bulk equivalent, correlation and validity of the author’s recovery and density transfer at this bog.'
    if item['use'] == 'lower_limit':
        return 'The primary layer was in place on the drainage divide and its surviving thickness can be reduced by erosion or soil development. No depositional thickening is assumed.'
    return COMPLETE_RETENTION if item['use'] == 'upper_limit' else PRESERVED_BED


def apply(records):
    records.extend(read(HERE/'new-localities.json'))
    lookup = {r['record_id']: r for r in records}
    for key, changes in read(HERE/'ledger-updates.json').items():
        lookup[key].update(changes)
    for key, item in read(HERE/'interpretations.json').items():
        r = lookup[key]
        kind = item['use']
        r['fallout_model_status'] = {'omitted':'hold_original_fallout_transfer', 'measurement':'use_primary_bed',
                                    'upper_limit':'conditional_local_fallout_ceiling', 'lower_limit':'primary_component_minimum'}[kind]
        r['thickness_quality'] = item['evidence']
        r['caveats'] = ' '.join(dict.fromkeys(filter(None,[r['caveats'],item['evidence'],qualification(item) if kind!='omitted' else ''])))
        if kind in ('upper_limit','lower_limit'):
            r.update(bound_relation='<=' if kind=='upper_limit' else '>=',bound_cm=item['cm'],bound_target=TARGET)
        elif kind=='omitted' and 'original local' in str(r.get('bound_target','')).lower():
            r.update(bound_relation='',bound_cm='',bound_target='')
    return records


def registered_selection(records):
    lookup = {r['record_id']:r for r in records}
    old = AUDIT/'history/2026-11-23-before-distal-review/model-selection.csv'
    decisions = read(HERE/'interpretations.json')
    with old.open() as f:
        former = list(csv.DictReader(f))
    def number(x):
        return float(x) if x!='' else ''
    rows = []
    for legacy in former:
        key=legacy['record_id']
        if key in decisions:
            continue
        rows.append(dict(record_id=key,kind=legacy['kind'],thickness_cm=number(legacy['thickness_cm']),
                    limit_cm=number(legacy['limit_cm']),latitude=float(legacy['latitude']),longitude=float(legacy['longitude']),
                    decision_basis=legacy['decision_basis'],preservation_assumption=legacy['qualification']))
    for key,item in decisions.items():
        kind=item['use']
        if kind=='omitted':
            continue
        r=lookup[key]
        rows.append(dict(record_id=key,kind=kind,thickness_cm=item['cm'] if kind=='measurement' else '',
                         limit_cm=item['cm'] if kind!='measurement' else '',latitude=float(r['latitude']),longitude=float(r['longitude']),
                         decision_basis=item['evidence'],preservation_assumption=qualification(item)))
    order={x['record_id']:i for i,x in enumerate(former)}
    rows.sort(key=lambda x:(order.get(x['record_id'],len(order)),x['record_id']))
    write(AUDIT/'model-selection.json',rows)
    archives=read(HERE/'baseline-archive-definitions.json')
    previous={r['id']:r for r in archives['sites']}
    sources=read(AUDIT/'source-url-catalogue.json')
    result=[]
    for row in rows:
        if row['kind']!='upper_limit':
            continue
        key=row['record_id']
        if key not in decisions:
            result.append(previous[key])
            continue
        r=lookup[key]
        candidates=[s.strip() for s in r['evidence_source'].split(';')]
        url=next((s if s.startswith('http') else sources[s] for s in candidates if s.startswith('http') or s in sources),None)
        if not url:
            url='https://github.com/siyulai21/tephra-fallout-isopachs/blob/main/data/audits/Mazama/distal-review/README.md'
        result.append(dict(id=key,name=r['locality'],longitude=row['longitude'],latitude=row['latitude'],archive=r['setting'] or 'See field setting in original source',
                 reading='attribution_uncertain' if key in ('M2012-ONOFF02','L2016-STOWELL') else 'occurrence_only',confidence='qualified',horizon=r['identity'],
                 inspection=r['evidence_locator'],preservation=row['preservation_assumption'],control=row['decision_basis'],
                 reported_fallout_limit_cm=None,reference=r['primary_reference'],url=url,in_principal_as='interpreted_upper_limit',
                 principal_effective_upper_cm=row['limit_cm'],source_quantity=decisions[key]['scope']))
    archives['sites']=result
    write(AUDIT/'interpreted-archives.json',archives)

