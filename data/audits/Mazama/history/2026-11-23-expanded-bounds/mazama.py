"""Rebuild the conditional Mazama input and site-level evidence decisions."""
import csv
import hashlib
import json
from pathlib import Path

from pyproj import Geod

ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT/'data/audits/Mazama'
INPUT = ROOT/'data/eruptions/Mazama_reviewed.csv'
GEOD = Geod(ellps='WGS84')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_csv(path, rows):
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator='\n')
        writer.writeheader(); writer.writerows(rows)


def prepare():
    ledger = list(csv.DictReader((AUDIT/'revised-locality-ledger.csv').open()))
    lookup = {r['record_id']:r for r in ledger}
    selection = json.loads((AUDIT/'model-selection.json').read_text())
    supplied = list(csv.DictReader((ROOT/'data/eruptions/Mazama_tephra.csv').open()))
    vent = next((float(r['Vent Lon']),float(r['Vent Lat'])) for r in supplied if r['Vent Lon'] and r['Vent Lat'])
    rows, chosen = [], {}
    for item in selection:
        record = lookup[item['record_id']]
        row = dict(Lon=item['longitude'],Lat=item['latitude'],Thickness=item['thickness_cm'],
            Pseudo='No',Kind=item['kind'],Limit_cm=item['limit_cm'],LogSD='',
            **{'Vent Lon':vent[0] if not rows else '', 'Vent Lat':vent[1] if not rows else ''},
            EvidenceID=item['record_id'],Site=record['locality'],Reference=record['primary_reference'],
            Evidence=item['decision_basis'],Qualification=item['preservation_assumption'])
        rows.append(row); chosen[item['record_id']]={**item,'accepted_csv_row':len(rows)+1}
    assert len(rows)==29 and sum(r['Kind']=='measurement' for r in rows)==26
    assert len(chosen)==29
    write_csv(INPUT,rows)
    decisions=[]
    for r in ledger:
        rid=r['record_id']; included=rid in chosen; item=chosen.get(rid,{})
        if included:
            reason=item['decision_basis']; missing='No invented additional measurement error. '+item['preservation_assumption']
        else:
            reason=r['fallout_model_status']+': '+r['caveats']
            if r['quantity_kind'] in ('cryptotephra','tentative_signal'):
                missing='Original local loading requires profile-integrated target mass (or compatible count index), a site-specific capture/focusing response, and bulk-density or calibrated count response. No positive response floor or upper focusing factor was established. Sample depth/width is not bulk thickness.'
            elif r['quantity_kind']=='trace_no_numeric_limit':
                missing='Positive trace has no established bulk-thickness recognition scale or minimum preserved response. The 0.1 cm convention is not a measured value or a detection threshold.'
            elif r['deposit_class'] in ('compiler_secondary','compiler_both','secondary_or_mixed_sedimentary_context') or 'rework' in r['fallout_model_status']:
                missing='Original local fallout cannot be bounded from total sediment thickness without primary fraction, redistribution/focusing and loss constraints. Preserved-bed endpoints remain in the evidence ledger.'
            elif not r['latitude']:
                missing='Precise site or core georeferencing remains missing; any component thickness is preserved but cannot yet constrain a regional field at a defensible position.'
            else:
                missing='A source-supported local primary thickness or valid transfer to it remains unresolved. See the specific source conflict, coordinate, phase or field-log gap. No low-influence exemption is claimed.'
        lat=float(item.get('latitude',r['latitude'])) if item.get('latitude',r['latitude'])!='' else None
        lon=float(item.get('longitude',r['longitude'])) if item.get('longitude',r['longitude'])!='' else None
        if lat is not None:
            az,_,dist=GEOD.inv(*vent,lon,lat)
            near=sorted((GEOD.inv(lon,lat,x['Lon'],x['Lat'])[2]/1000,x['EvidenceID']) for x in rows if x['EvidenceID']!=rid)
            distance=round(dist/1000,3); bearing=round(az%360,3); neighbours='; '.join(f'{n}:{d:.2f} km' for d,n in near[:3])
        else:distance=bearing=neighbours=''
        decisions.append(dict(record_id=rid,locality=r['locality'],accepted_csv_row=item.get('accepted_csv_row',''),
            principal_kind=item.get('kind','omitted'),principal_thickness_cm=item.get('thickness_cm',''),principal_limit_cm=item.get('limit_cm',''),
            latitude=lat if lat is not None else '',longitude=lon if lon is not None else '',distance_km=distance,bearing_deg=bearing,
            nearby_principal_sites=neighbours,verification=r['verification'],evidence_source=r['evidence_source'],evidence_locator=r['evidence_locator'],
            observed_quantity=r['quantity_kind'],decision=reason,missing_quantity_or_assumption=missing,
            dependence=r['parent_record_id'] or 'Check source/core identity; no new independence assumption from proximity',
            influence_scope='No low-influence claim. Source support, not modeled volume, determines inclusion.'))
    write_csv(AUDIT/'model-selection.csv',[dict(record_id=r['EvidenceID'],accepted_csv_row=i,kind=r['Kind'],thickness_cm=r['Thickness'],limit_cm=r['Limit_cm'],latitude=r['Lat'],longitude=r['Lon'],decision_basis=r['Evidence'],qualification=r['Qualification']) for i,r in enumerate(rows,2)])
    write_csv(AUDIT/'site-evidence-decisions.csv',decisions)
    definition='Source-qualified climactic Mazama airborne fallout reference: 26 reported point values and three lower bounds. Compilation-supported values are not claimed independently primary-verified. Known corrections, unresolved conflicts, explicit precursor totals, reworked totals and unquantified traces/proxies are omitted. Moss central core uses4 cm; Mount Bachelor UPU>=44 cm, Upper Kirk Road UPU>=35 cm and Portage>=1.27 cm (one-sided interval approximation). Twobit and Lavington use primary coordinates; Green Lake1 cm recovered from primary text. No invented uncertainties, zeros or trace thresholds.'
    provenance=dict(audit_date='2026-11-22',scope='Complete293-entry compilation reconciliation plus60 additional records/leads. Independent primary verification remains incomplete. This is a conditional reference selection, not certification of original deposition at every included site.',
      original_input=dict(file='data/eruptions/Mazama_tephra.csv',sha256=sha(ROOT/'data/eruptions/Mazama_tephra.csv')),
      source=dict(file='data/sources/mazama/buckland2020-esm1.xlsx',sha256=sha(ROOT/'data/sources/mazama/buckland2020-esm1.xlsx'),doi='https://doi.org/10.1007/s00445-020-1362-1'),
      cases={'principal':dict(file='data/eruptions/Mazama_reviewed.csv',sha256=sha(INPUT),n_measurements=26,n_lower_limits=3,n_upper_limits=0,definition=definition)},
      target='Climactic Mazama primary airborne bulk deposit, including co-ignimbrite ash; exclude identified older precursor units and flow/reworked material. Legacy phase correlations remain conditional where primary logs are unavailable.',
      thickness_convention='Preserved bulk centimetres as a proxy for original local deposition. No universal decompaction correction. Component lower bounds assume in-place primary pumice without depositional thickening.',
      coordinate_datum='Published decimal or DMS positions used without an undocumented datum conversion; precision does not establish survey accuracy.',
      primary_audit_complete=False,prior_release=json.loads((AUDIT/'prior-release.json').read_text()),
      preparation_script={'file':'benchmarks/mazama.py','sha256':sha(Path(__file__))},
      audit_artifacts={f:sha(AUDIT/f) for f in ['model-selection.json','model-selection.csv','site-evidence-decisions.csv','revised-locality-ledger.csv','supplied-row-crosswalk.csv']})
    (AUDIT/'provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
    print(json.dumps({'input':str(INPUT.relative_to(ROOT)),**provenance['cases']['principal']},indent=2))


if __name__=='__main__':prepare()
