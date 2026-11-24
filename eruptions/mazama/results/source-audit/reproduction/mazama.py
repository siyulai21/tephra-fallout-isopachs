"""Rebuild the conditional Mazama input and site-level evidence decisions."""
import csv
import hashlib
import io
from collections import Counter
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
    stream = io.StringIO(newline='')
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator='\n')
    writer.writeheader()
    writer.writerows(rows)
    data = stream.getvalue().encode('utf-8')
    if not path.is_file() or path.read_bytes() != data:
        path.write_bytes(data)


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
    counts = Counter(r['Kind'] for r in rows)
    assert len(chosen) == len(rows) == len(selection)
    assert set(counts) == {'measurement', 'lower_limit', 'upper_limit'}
    write_csv(INPUT,rows)
    slump = lookup['M2023-YL16-6A']
    probe = {k:'' for k in rows[0]}
    probe.update(Lon=slump['longitude'], Lat=slump['latitude'], Thickness=.5,
                 Pseudo='No', Kind='measurement', EvidenceID=slump['record_id'],
                 Site=slump['locality'], Reference=slump['primary_reference'],
                 Evidence='Morgan2023 reported5 mm; excluded from principal because of slump-toe deformation.',
                 Qualification='Prediction only. Slump-toe deformation excludes this observation from the current likelihood.')
    former = list(csv.DictReader((AUDIT/'history/2026-11-23-before-distal-review/input.csv').open()))
    probes = [dict(r, **{'Vent Lon':'','Vent Lat':''}, Qualification='Prediction location only. Frozen pre-review selection; omitted from the revised input.') for r in former if r['EvidenceID'] not in chosen]
    write_csv(AUDIT/'model-probes.csv',rows+[probe]+probes)
    decisions=[]
    for r in ledger:
        rid=r['record_id']; included=rid in chosen; item=chosen.get(rid,{})
        if included:
            reason=item['decision_basis']; missing='No invented additional measurement error. '+item['preservation_assumption']
        else:
            reason=r['fallout_model_status']+': '+r['caveats']
            if r['quantity_kind'] in ('cryptotephra','tentative_signal'):
                missing='Recovered glass, mineral or concentration data and site-specific alternatives are recorded in THICKNESSLESS_SITES.md and cryptotephra-profile-calculations.json. A source-supported original-bulk transfer or recognition threshold is unresolved at this location. '+r['caveats']+''
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
    definition = (f"Climactic Mazama airborne bulk fallout. {counts['measurement']} point values, {counts['lower_limit']} lower limits and {counts['upper_limit']} upper limits are shared by all five families. "
                  'Core duplicates and inferred phase totals carry explicit qualifications. The current input incorporates the published Yellowstone core data, '
                  'Lyman Lake and the Ruby Mountain, Great Salt Lake, Mazama source and thicknessless-site reviews, Young’s separately mapped upper pumice sections, the Round Meadow borings, North Cascades and British Columbia lake cores, northern Idaho lake packages, Mount Baker field sections and the distal deposit-nature revision. Bounds and sensitivity alternatives are in the locality ledger.')
    interpreted = {r['id']: r for r in json.loads((AUDIT/'interpreted-archives.json').read_text())['sites']}
    upper = [r for r in selection if r['kind'] == 'upper_limit']
    assert set(interpreted) == {r['record_id'] for r in upper}
    provenance=dict(audit_date='2026-11-23',scope=f'All 293 original catalogue rows and {len(ledger)-293} additional records, leads or controls. Verification level is recorded at each site.',
      original_input=dict(file='data/eruptions/Mazama_tephra.csv',sha256=sha(ROOT/'data/eruptions/Mazama_tephra.csv')),
      source=dict(file='data/sources/mazama/buckland2020-esm1.xlsx',sha256=sha(ROOT/'data/sources/mazama/buckland2020-esm1.xlsx'),doi='https://doi.org/10.1007/s00445-020-1362-1'),
      cases={'principal':dict(file='data/eruptions/Mazama_reviewed.csv',sha256=sha(INPUT),n_measurements=counts['measurement'],n_lower_limits=counts['lower_limit'],n_upper_limits=counts['upper_limit'],definition=definition)},
      principal_limits=[dict(evidence_id=r['record_id'],interpreted_upper_cm=r['limit_cm'],derivation=r['decision_basis'],qualifications=r['preservation_assumption']) for r in upper],
      target='Climactic Mazama primary airborne bulk deposit, including co-ignimbrite ash; exclude identified older precursor units and flow/reworked material. Legacy phase correlations remain conditional where primary logs are unavailable.',
      thickness_convention='Preserved bulk centimetres as a proxy for original local deposition. No universal decompaction correction. Component lower bounds assume in-place primary pumice without depositional thickening.',
      coordinate_datum='Published decimal or DMS positions used without an undocumented datum conversion; precision does not establish survey accuracy.',
      primary_audit_complete=False,prior_release=json.loads((AUDIT/'prior-release.json').read_text()),
      preparation_script={'file':'benchmarks/mazama.py','sha256':sha(Path(__file__))},
      audit_artifacts={f:sha(AUDIT/f) for f in ['model-selection.json','model-selection.csv','site-evidence-decisions.csv','revised-locality-ledger.csv','supplied-row-crosswalk.csv']})
    (AUDIT/'provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
    print(json.dumps({'input':str(INPUT.relative_to(ROOT)),**provenance['cases']['principal']},indent=2))


if __name__=='__main__':prepare()
