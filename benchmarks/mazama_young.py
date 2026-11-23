"""Reconcile Young's separately numbered distal and field-section data with the frozen audit."""
import csv
import json
import re
from collections import defaultdict

from benchmarks.mazama import AUDIT
from benchmarks.mazama_young_maps import generate

HISTORY = AUDIT / 'history/2026-11-23-before-young'
THESIS = 'Young, S.R. 1990. Physical volcanology of Holocene airfall deposits from Mt Mazama, Crater Lake, Oregon. Lancaster University PhD thesis.'
URL = 'https://ethos.bl.uk/OrderDetails.do?uin=uk.bl.ethos.315541'
LOWER = 'Requires in-place climactic fallout without net thickening by slope wash, lake focusing or imported clasts. Erosion, compaction and loss of unpreserved fine ash can reduce the recovered component.'
UPPER = 'Use Young’s published upper equivalent if the original target mass is represented by the site estimate and original bulk density is at least 700 kg/m³. Lower density or net export can violate this ceiling. Other additive phases or imported material can inflate the ceiling.'
AGE = 'Climactic event correlation, about 7.6 ka cal BP. Young cites 6845 ± 50 conventional radiocarbon yr BP for the eruption.'
FIELDS = list(json.loads((AUDIT/'additional-records.json').read_text())[0])
def dump(path,value):
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')


def observation(record_id,name,lat,lon,**evidence):
    row=dict.fromkeys(FIELDS,'')
    row.update(record_id=record_id,locality=name,latitude=lat,longitude=lon,accepted_for_model=False,
        local_age=AGE,age_basis='Event correlation unless the locality field reports independent stratigraphic age control.',
        primary_reference=THESIS,evidence_source='young1990.pdf',verification='primary_text_table_and_map_checked',
        identity_quality='Field stratigraphy or published correlation detailed at the stated source locator.',
        coordinate_quality='Figure-derived locality, not a surveyed section coordinate.',thickness_quality='Primary source quantity identified by column and units.')
    row.update(evidence)
    return row


def choose(row,kind,x,basis,assumption):
    return dict(record_id=row['record_id'],kind=kind,thickness_cm=x if kind=='measurement' else '',
        limit_cm='' if kind=='measurement' else x,latitude=row['latitude'],longitude=row['longitude'],
        decision_basis=basis,preservation_assumption=assumption)


def archive(row,limit,interpretation):
    return dict(id=row['record_id'],name=row['locality'],longitude=row['longitude'],latitude=row['latitude'],
        archive=row['setting'],reading='dispersed_no_continuous_bed' if row['quantity_kind']=='trace_no_numeric_limit' else 'occurrence_only',
        confidence='qualified',horizon=row['identity'],inspection=row['evidence_locator'],preservation=interpretation,
        control='Source estimate and original-to-preserved transfer. No separate scenario likelihood at this site.',
        reported_fallout_limit_cm=None,reference=row['primary_reference'],url=URL,
        in_principal_as='interpreted_upper_limit',principal_effective_upper_cm=limit)


def append_sections(points,selected):
    groups=defaultdict(dict)
    for r in csv.DictReader((AUDIT/'young1990-appendix1.csv').open()):
        groups[int(r['site'])][r['appendix_table']]=r
    assert set(groups)==set(points)
    # Some labels cannot be split. One likelihood per mapped group. Known approximate
    # reoccupations use the later measured position or authoritative section.
    linked={47:('B2024-UKR','Buckland 2024 Fig. 1D explicitly uses equivalent Young site 47. Its auger supplies a 35–40 cm UPU bracket.'),
            103:('B2020-046','Buckland 2020 identifies the Mount Bachelor pit as approximately coincident with a Young site. Sites 103 and 128 are in this group.'),
            128:('B2020-046','Source map does not identify which of two close Bachelor sites the later 44 cm UPU pit reoccupied. Keep one group likelihood.'),
            95:('Y1990-A015','The 95/15 marker assignment is unresolved. The shared footprint uses the lower of the two section minima, 194 cm.'),
            134:('Y1990-A133','The 133/134 marker assignment is unresolved. Use the weaker 70 cm minimum for the shared footprint.'),
            150:('Y1990-A116','The north-wall site labels overlap and the accessible map does not resolve the two azimuths confidently. Use one footprint for the borehole-free field sections.')}
    rows=[]
    for i,parts in sorted(groups.items()):
        pos=points[i]
        c=parts.get('C',{})
        raw=c.get('upper_pumice_th_cm','')
        pages=', '.join(r['pdf_page'] for r in parts.values())
        locator=f"Appendix 1, source section {i}, table(s) {', '.join(parts)}, PDF pp. {pages}. Position on Appendix 1d PDF p. {pos['map_pdf_page']}. Legend p. 299. Field method p. 50 and fallout preservation pp. 93–95."
        row=observation(f'Y1990-A{i:03}',f'Young Appendix 1 section {i}, Oregon',pos['latitude'],pos['longitude'],
            evidence_locator=locator,coordinate_basis=f"Appendix 1d map digitization. Source pixel {pos['pixel_x']}, {pos['pixel_y']}. See young1990-section-positions.csv and alternative registrations. "+pos['marker_note'],
            identity='UPU is the climactic Plinian Upper Pumice Unit. Buckland et al. 2024 separates the lower pumice and older ashes from it.' if c else 'This section supplies precursor-unit measurements only in Appendix 1.',
            deposit_class='stratigraphically_identified_airfall_component' if c else 'precursor_comparison',
            setting='Terrestrial exposure or pit, southern Oregon. Individual logging environments are not uniformly specified.',
            source_thickness_notes=f"Upper-pumice thickness cell: {raw or 'blank or no C entry'}. Other phase thicknesses and maximum clast dimensions remain in young1990-appendix1.csv.",
            quantity_kind='unquantified',fallout_model_status='hold_phase_or_bulk_transfer',
            caveats='Clast measurements are mean maximum diameters of the three largest pumice or lithic clasts. No bulk thickness follows from these diameters without areal coverage and packing. Precursor fall and flow deposits are excluded.')
        if raw and raw!='0':
            x=float(re.search(r'[0-9.]+',raw).group())
            if x.is_integer():x=int(x)
            if raw.startswith('J'):
                row.update(quantity_kind='undivided_pumice_package',deposit_class='mixed_eruption_phase_package',
                    caveats='J combines lower and upper pumice. Their division is unknown. Co-ignimbrite ash can be lost above the pumice, so even a complete package cannot cap the original all-phase deposit.')
                if '+' in raw:
                    row.update(bound_relation='>=',bound_cm=x,bound_target='Combined preserved lower and upper pumice only.')
                else:
                    row.update(observed_point_cm=x,bound_relation='<=',bound_cm=x,bound_target='UPU share of undivided preserved pumice only.')
            else:
                incomplete='+' in raw
                basis=f"Young Appendix 1c section {i} reports {raw} cm of upper pumice. This primary Plinian component supplies a minimum on combined climactic airborne fallout."
                if incomplete:basis+=' The author marks a minimum because exposure or preservation was incomplete.'
                row.update(quantity_kind='incomplete_primary_component' if incomplete else 'reported_primary_component',
                    observed_point_cm='' if incomplete else x,bound_relation='>=',bound_cm=x,
                    bound_target='Original local combined climactic bulk fallout, conditional on in-place UPU.',caveats=LOWER,
                    fallout_model_status='conditional_principal_lower_bound')
                if i in linked:
                    rid,explain=linked[i]
                    row.update(parent_record_id=rid,fallout_model_status='source_duplicate_or_unresolved_marker_group',caveats=LOWER+' '+explain)
                else:
                    if i==116:
                        # Use the weaker measured north-wall minimum while preserving the scarcer 20 m original.
                        basis+=' The nearby 116/150 positions cannot be separated confidently. The group uses the weaker 1200 cm bound from section 116.'
                    selected[row['record_id']]=choose(row,'lower_limit',x,basis,LOWER+' The tabulated map origins are tested coherently.')
        elif raw=='0':
            row.update(quantity_kind='component_zero',observed_point_cm=0,
                caveats='Source zero concerns only UPU at section 81. It gives no bound on all-phase ash, including missing fine fallout.')
        rows.append(row)
    return rows


def review():
    points,distal=generate()
    selected={r['record_id']:r for r in json.loads((HISTORY/'model-selection.json').read_text())}
    archives=json.loads((HISTORY/'interpreted-archives.json').read_text())
    originals={
        f"B2020-{int(r['buckland_id']):03}":r for r in csv.DictReader((AUDIT/'buckland2020-293-localities.csv').open())}
    updates={}
    rows=append_sections(points,selected)
    # One event/borehole, retaining the source's normalized reading and a separate original-record crosswalk.
    for r in distal:
        i=r['table_site']; mid=r['candidate_catalogue_id']; orig=originals[mid]; value=r['upland_equivalent_cm']
        lat,lon=r['map_latitude'],r['map_longitude']
        status='map_or_cross_reference_only'
        evidence=observation(f'Y1990-D{i:02}',f'Young Table 1.1 distal site {i}',lat,lon,
            parent_record_id='' if i==9 else mid,evidence_locator=f'Fig. 1.14 and Table 1.1, pp. 28–29, site {i}. '+r['source_reference'],
            verification='young_field_check_and_table_read' if r['field_checked_by_young'] else 'young_compiled_table_read',
            coordinate_basis=f"Small-scale Figure 1.14. Candidate {mid} differs by {r['distance_to_candidate_km']} km. "+r['match_note'],
            identity='Distal Mazama ash as correlated by Young and the cited observers. '+r['match_note'],
            deposit_class='distal_ash_attribution',setting=orig['sampling_raw'] or 'distal site',
            quantity_kind='trace_no_numeric_limit' if value=='T' else 'upland_equivalent',
            source_thickness_notes=f'Table 1.1 gives {value} on the 700 kg/m³ upland-equivalent cm scale.',
            caveats='Site numbers refer solely to the distal map. The source normalizes all numeric values to upland bulk density. '+r['match_note'],
            fallout_model_status=status)
        if value=='T':evidence['caveats']+=' T records presence with no measured thickness or demonstrated recognition threshold.'
        elif value[0] in '<≤':
            evidence.update(bound_relation='<' if value[0]=='<' else '<=',bound_cm=float(value[1:]),bound_target='Young’s upland-equivalent ash thickness at 700 kg/m³.')
        elif '–' in value:
            evidence.update(observed_min_cm=5,observed_max_cm=10,interval_meaning='Range on the normalized upland-equivalent scale.')
        else:evidence['observed_point_cm']=float(value)
        rows.append(evidence)
        if i==9:continue
        mid_num=str(int(mid.split('-')[1]))
        old_checks=json.loads((AUDIT/'review-decisions.json').read_text()).get(mid_num,{})
        addition=dict(evidence_source=old_checks.get('evidence_source','buckland2020-esm1.xlsx')+'; young1990.pdf',
            evidence_locator=old_checks.get('evidence_locator',f"Sheet1 row {orig['source_excel_row']}")+f'; Young Table 1.1 row {i} and Fig. 1.14',
            caveats=old_checks.get('caveats','Published catalogue and Table 1.1 are not independent field observations.')+f" Young reports {value} on the upland-equivalent scale. "+r['match_note'])
        if old_checks.get('verification','compilation_checked_primary_pending')=='compilation_checked_primary_pending':
            addition['verification']='young_field_check_and_table_read' if r['field_checked_by_young'] else 'young_compiled_table_read_original_observer_pending'
        updates[mid_num]=addition

    def revise(n,kind,x,basis,assume,**other):
        rid=f'B2020-{n:03}'
        orig=originals[rid]
        extra=updates.setdefault(str(n),{})
        lat=other.pop('latitude',float(orig['latitude_raw']));lon=other.pop('longitude',float(orig['longitude_raw']))
        relation='>=' if kind=='lower_limit' else '<='
        extra.update(latitude=lat,longitude=lon,quantity_kind='conditional_primary_component' if kind=='lower_limit' else 'interpreted_original_upper',
            bound_relation=relation,bound_cm=x,bound_target='Original local climactic airborne bulk fallout, conditional on the specified transfer.',
            fallout_model_status='conditional_principal_'+kind,caveats=basis+' '+assume,**other)
        item=observation(rid,orig['locality'],lat,lon,setting=orig['sampling_raw'] or 'terrestrial exposure or core',
            primary_reference=extra.get('primary_reference',orig['references_raw']),identity=extra.get('identity','Mazama correlation of the cited investigators.'),
            evidence_locator=extra['evidence_locator'],quantity_kind=extra['quantity_kind'])
        selected[rid]=choose(item,kind,x,basis,assume)
        if kind=='upper_limit':archives['sites'].append(archive(item,x,assume))

    peat='Young field-checked this area and tabulated an equivalent on a 700 kg/m³ scale. The conservative lower endpoint allows the historical twofold bog-to-upland conversion (1400 versus 700 kg/m³), a mainly climactic bed and no local focusing. Older phases, stronger compaction or imported ash can invalidate the transfer.'
    for n,x,description in [(76,17.5,'Distal site 3, 35/2 = 17.5 cm. Dick Springs and Meadowwood are plausible Tollgate counterparts. Enter the mapped area once.'),
            (68,12.5,'Distal site 4, 40/2 = 20 cm. Use the smaller published historical bog reading 12.5 cm at Hurricane Creek.'),
            (63,15,'Distal site 5, 30/2 = 15 cm. The catalogue separately reports an 18 cm bog reading at Anthony Lake Meadow.')]:
        revise(n,'lower_limit',x,description,peat,observed_point_cm='',interval_meaning='Historical values describe different local sections or density bases.',verification='young_field_check_and_table_read')
    for n,x,description in [(4,5,'Young explicitly gives <5 cm for distal site 24. Buckland lists 5 cm. Preserve the strict source reading on the equivalent scale.'),
            (264,25,'Young gives <25 cm at site 13. The catalogue reports a 12 cm local value and a 42–47 inch depth interval at Saskatchewan Crossing.'),
            (290,5,'Young gives <5 cm at site 14. The Beverly terrace description includes one inch of ash below about three feet of soil. The catalogue carries 1.5 cm.'),
            (232,5,'Young gives <5 cm for site 16. At the described eastern bluff a sharp basal contact and mixed gradational top make the total a conditional ceiling. Individual nearby river exposures remain distinguished.')]:
        revise(n,'upper_limit',x,description,UPPER,observed_point_cm='',interval_meaning='Young reports a strict < on his equivalent scale. The continuous one-sided likelihood uses its numeric ceiling.')
    revise(229,'upper_limit',1,'Young Table 1.1 site 12 gives ≤1 cm. Mathews reported approximately 5 mm at 433 cm depth. Williams and Goles say he did not directly measure it and warned the Hiller borer smeared ash. Smearing invalidates the former 0.5 cm minimum.',
        UPPER+' The upper endpoint additionally assumes no material lost by coring. Try the author’s original 0.5 cm estimate separately.',
        latitude=51+5/60,longitude=-(121+59/60),observed_point_cm='',verification='original_observer_account_and_borehole_checked',
        primary_reference='Mathews in Williams and Goles 1968 pp. 37–38, Lowdon and Blake 1968 p. 227, Young 1990 Table 1.1 site 12',
        evidence_source='williams-goles1968.pdf; https://emrlibrary.gov.yk.ca/gsc/papers/68-2b.pdf; young1990.pdf',
        evidence_locator='Jesmond Borehole #2, 500 ft from north edge, 433 cm. GSC-511 and GSC-529 p. 227. Williams and Goles pp. 37–38. Young p. 29.',
        coordinate_basis='Lowdon and Blake 1968 p. 227 location of Mathews et al. Borehole #2: 51°05′N, 121°59′W.',
        coordinate_quality='Original collector borehole location, coordinates published to the minute.',
        local_age='Basal peat GSC-511 9210 ± 150 14C BP at 630–640 cm, below possible Mazama. GSC-529 2440 ± 140 14C BP at 160–170 cm below much younger Bridge River ash.',
        identity='Mathews reports Mazama glass in the peat section. Lowdon and Blake give a tentative Mazama identification for this same depth.',
        core_depth_or_sample_interval='433 cm; distinct Bridge River ash at 142–160 cm.')
    orig=selected['B2020-291']
    orig.update(latitude=54+44/60,longitude=-(112+29/60),decision_basis='The Lofty Lake record has a reported 1 mm bed at 398 cm. Buckland 2022 treats 0.1 cm as a minimum. Young’s ≤0.2 cm is the competing upland-equivalent endpoint.')
    updates['291'].update(latitude=orig['latitude'],longitude=orig['longitude'],coordinate_basis='Lichti-Federovich 1972, GSC Paper 72-1B, p. 113: 54°44′N, 112°29′W.',
        evidence_source='buckland2020-esm1.xlsx; young1990.pdf; https://publications.gc.ca/collections/collection_2025/rncan-nrcan/M41-1-4-72-1B-eng.pdf',
        coordinate_quality='Original author gives lake coordinates to one arc-minute. Specific borehole offset unresolved.',
        quantity_kind='reported_preserved_component',observed_point_cm=.1,
        interval_meaning='Conditional original lower 0.1 cm from the preserved layer. Young Table 1.1 gives ≤0.2 cm on the upland-equivalent scale. Preserve both, fit the reported lower endpoint once.')
    updates['292']=dict(verification='original_paper_and_reference_glass_checked',evidence_source='zdanowicz1999.pdf; gisp2-inventory.html',
        evidence_locator='Zdanowicz et al. 1999 pp. 621–623, Figs. 1–3 and Table 1. GISP2D borehole inventory.',
        identity='SEM-EDX glass comparison to proximal Mazama rhyodacitic pumice and silicic ignimbrite. Four shard-bearing filters.',
        local_age='7627 ± 150 cal yr BP on the GISP2 ice chronology, as reported in 1999.',
        core_depth_or_sample_interval='1325.11–1326.15 m on four filters. Ten >10 µm shards selected from the 1325.89–1326.15 m filter for chemistry.',
        caveats='Grains of about ≤1 to 20 µm are reported. The ten selected grains are a geochemical subset. Total attributable mass, sampled horizontal area and recovered fraction are unresolved. The sulfate record describes a different constituent.')
    from benchmarks.mazama_field_followup import field_records
    new,decisions,more_archives=field_records(selected,points)
    rows.extend(new);updates.update(decisions);archives['sites'].extend(more_archives)
    dump(AUDIT/'young1990-review.json',dict(revisions=updates,additional=rows))
    dump(AUDIT/'model-selection.json',list(selected.values()))
    dump(AUDIT/'interpreted-archives.json',archives)


if __name__=='__main__':
    review()

