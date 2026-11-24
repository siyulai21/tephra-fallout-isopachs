"""Publish the row-by-row facies crosswalk and distal evidence plot."""
import csv
import json
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from benchmarks.mazama import AUDIT, write_csv

OUT = AUDIT/'distal-review'


def keyed(path, field):
    with path.open() as source:
        return {r[field]:r for r in csv.DictReader(source)}


def build():
    ledger=keyed(AUDIT/'revised-locality-ledger.csv','record_id')
    geom=keyed(AUDIT/'site-evidence-decisions.csv','record_id')
    old=keyed(AUDIT/'history/2026-11-23-before-distal-review/model-selection.csv','record_id')
    raw=keyed(AUDIT/'buckland2020-293-localities.csv','buckland_id')
    changed=json.loads((OUT/'interpretations.json').read_text())
    separated={'B2020-073','J1985-FLODELLE','R2001-KSNEA','W2016-LYMAN','M2018-OVERLAND','M2018-SOLDIER'}
    rows=[]
    for key,r in ledger.items():
        compiler=raw[str(int(key.split('-')[1]))] if key.startswith('B2020-') else {}
        prior=old.get(key,{})
        kind=r['principal_kind']
        facies=r['deposit_class']
        if key in separated:
            evidence='Primary component distinguished in original field or core description'
        elif kind=='measurement':
            evidence='Retained primary-bed proxy; see original field or core interpretation'
        elif kind=='upper_limit':
            evidence='Conditional ceiling; recovered archive or author-equivalent bound'
        elif kind=='lower_limit':
            evidence='Conditional in-place primary-component minimum'
        elif key in changed:
            evidence=changed[key]['scope']
        elif any(w in (facies+' '+r['fallout_model_status']).lower() for w in ('rework','redepos','secondary','flow','lahar')):
            evidence='Redistributed, mixed or flow deposit; see component decision'
        elif r['quantity_kind'] in ('trace_no_numeric_limit','cryptotephra','tentative_signal'):
            evidence='Trace or shard signal; original-bulk transfer unresolved'
        elif 'pending' in r['verification']:
            evidence='Original observer or depositional context unresolved'
        else:
            evidence='Evidence or coverage record without an assigned local fallout likelihood'
        rows.append(dict(record_id=key,locality=r['locality'],latitude=r['latitude'],longitude=r['longitude'],
                      coordinate_basis=r['coordinate_basis'],distance_from_vent_km=geom[key]['distance_km'],
                      local_age=r['local_age'],event_identity=r['identity'],original_S1_row=compiler.get('source_excel_row',''),
                      original_S1_deposit_code=compiler.get('remobilisation_raw',''),related_ledger_record=r['parent_record_id'],
                      S1_reported_cm=compiler.get('thickness_cm_raw',''),reported_point_cm=r['observed_point_cm'],
                      reported_min_cm=r['observed_min_cm'],reported_max_cm=r['observed_max_cm'],quantity_kind=r['quantity_kind'],
                      deposit_class=facies,facies_review=evidence,prior_use=prior.get('kind','omitted'),
                      prior_value_cm=prior.get('thickness_cm') or prior.get('limit_cm',''),revised_use=kind,
                      revised_value_cm=r['principal_thickness_cm'] or r['principal_limit_cm'],
                      original_evidence=geom[key]['verification'],review_reason=r['principal_decision'] or r['thickness_quality'] or r['fallout_model_status'],
                      qualification=r['principal_assumption'] or r['caveats'],primary_reference=r['primary_reference'],
                      evidence_source=r['evidence_source'],evidence_locator=r['evidence_locator']))
    write_csv(OUT/'locality-facies-crosswalk.csv',rows)
    thick=[]
    for r in rows:
        if not r['distance_from_vent_km'] or float(r['distance_from_vent_km'])<300:
            continue
        readings=[('S1 numeric column',r['S1_reported_cm']),('recorded point',r['reported_point_cm']),
                  ('reported lower end',r['reported_min_cm']),('reported upper end',r['reported_max_cm'])]
        if r['prior_use'] in ('measurement','lower_limit'):
            readings.append(('previous point or minimum',r['prior_value_cm']))
        if r['revised_use'] in ('measurement','lower_limit'):
            readings.append(('current point or minimum',r['revised_value_cm']))
        flagged=[name for name,value in readings if value not in ('','-') and float(value)>=30]
        if flagged:
            thick.append(dict(screen_fields=', '.join(flagged),**r))
    thick.sort(key=lambda r:float(r['distance_from_vent_km']))
    write_csv(OUT/'thick-distal-records.csv',thick)
    dist=[r for r in rows if r['distance_from_vent_km'] and float(r['distance_from_vent_km'])>100]
    accepted=[r for r in rows if r['revised_use']!='omitted']
    summary=dict(ledger_rows=len(rows),numeric_positions=sum(bool(r['latitude']) for r in rows),
                 distal_rows=len(dist),distal_in_principal=dict(Counter(r['revised_use'] for r in dist)),principal=dict(Counter(r['revised_use'] for r in accepted)),
                 exact_original_S1_codes=dict(Counter(r['original_S1_deposit_code'] for r in rows if r['original_S1_row'])),
                 additions=sum(r['prior_use']=='omitted' and r['revised_use']!='omitted' for r in rows),
                 removals=sum(r['prior_use']!='omitted' and r['revised_use']=='omitted' for r in rows),
                 changed_kind=sum(r['prior_use']!=r['revised_use'] and 'omitted' not in (r['prior_use'],r['revised_use']) for r in rows),
                 missing_coordinates=sum(not r['latitude'] for r in rows),source_status=dict(Counter(r['original_evidence'] for r in rows)),
                 S1_verification_labels_containing_pending=sum('pending' in r['original_evidence'] for r in rows if r['original_S1_row']),
                 original_observer_queue_S1=sum(r['original_evidence'] in ('compilation_checked_primary_pending','young_compiled_table_read_original_observer_pending') for r in rows if r['original_S1_row']),
                 thick_distal_screen_rows=len(thick),thick_distal_S1_numeric=sum('S1 numeric column' in r['screen_fields'] for r in thick))
    (OUT/'counts.json').write_text(json.dumps(summary,indent=2)+'\n')
    plot(dist)
    return summary


def plot(dist):
    fig,ax=plt.subplots(figsize=(10,5.8),layout='constrained')
    kinds=[('measurement','o','#00695e','Primary-bed proxy'),('upper_limit','v','#4765aa','Conditional ceiling (≤)'),('lower_limit','^','#b95324','Conditional minimum (≥)')]
    for kind,marker,color,label in kinds:
        group=[r for r in dist if r['revised_use']==kind]
        ax.scatter([float(r['distance_from_vent_km']) for r in group],[float(r['revised_value_cm']) for r in group],
                   s=38,marker=marker,color=color,edgecolor='white',linewidth=.45,alpha=.84,label=label,zorder=3)
    removed=[r for r in dist if r['prior_use']!='omitted' and r['revised_use']=='omitted']
    ax.scatter([float(r['distance_from_vent_km']) for r in removed],[float(r['prior_value_cm']) for r in removed],
               color='#6e7073',marker='x',s=32,linewidth=1,label='Previous input, now withheld')
    labels={'CH2008-WIN':('Windy',(-4,13)),'CH2008-THU':('Thunder, BC',(4,-13)),'M2024-ROUNDWA':('Round, WA',(3,10)),
            'B2020-073':('Andies Prairie',(-42,12)),'J1985-FLODELLE':('Flodelle',(12,-13)),
            'B2020-250':('Johnson',(-42,16)),'S2009-FOY-A':('Foy A',(4,-12)),
            'B2020-113':('Williams Fen',(-70,4)),'M2013-NEL03':('NEL03',(15,7))}
    for r in dist:
        if r['record_id'] in labels:
            name,offset=labels[r['record_id']]
            ax.annotate(name,(float(r['distance_from_vent_km']),float(r['revised_value_cm'] or r['prior_value_cm'])),xytext=offset,textcoords='offset points',fontsize=8,
                    arrowprops=dict(arrowstyle='-',color='#818981',lw=.6))
    ax.set(xscale='log',yscale='log',xlabel='Geodesic distance from Crater Lake (km)',ylabel='Reported thickness or one-sided endpoint (cm)',
           title='Distal Mazama evidence beyond 100 km')
    ticks=[100,200,400,700,1000,1500]
    ax.set_xticks(ticks,labels=[f'{t:,}' for t in ticks]);ax.grid(which='major',alpha=.16)
    ax.legend(loc='lower left',fontsize=8,ncol=2)
    fig.savefig(OUT/'distal-thickness.png',dpi=155);plt.close(fig)


if __name__=='__main__':
    print(json.dumps(build(),indent=2))

