"""Reconcile individually logged Great Basin cores, cave facies and the Carp Lake depth datum."""
import json

from benchmarks.mazama import AUDIT
from benchmarks.mazama_young import observation, choose, archive

FLV = 'Throckmorton and Reheis (1993), USGS OFR 93-620, doi:10.3133/ofr93620.'
FURNACE = 'Reheis (1994), USGS Miscellaneous Field Studies Map MF-2266, doi:10.3133/mf2266.'
MWB = 'Minckley, Whitlock and Bartlein (2007), Quaternary Science Reviews 26, 2167–2184, doi:10.1016/j.quascirev.2007.04.009.'
WHIT = 'Whitlock and Bartlein (1997), Nature 388, Table 1; Whitlock et al. (2000), Tables 1–2, doi:10.1016/S0031-0182(99)00092-9.'
JEN = 'Jenkins et al. (2012), Science 337 supplement, pp. 9–10 and Tables S13–S15, doi:10.1126/science.1218443; Jenkins et al. (2013), Geochronology, Archaeological Context, and DNA at the Paisley Caves, ch. 28.'
CAP = 'Conditional local ceiling assumes the original fall was fully retained in this interval and compaction, erosion or noncapture did not make the preserved package thinner than the original local fall. Focusing or added sediment can enlarge it.'
POINT = 'Discrete primary bed approximates local fallout. Basin focusing, erosion, compaction and collection disturbance are unresolved.'
SHELTER = 'The original sedimentological study describes immediate alluvial redistribution and concentration inside the caves. Roof shielding, inflow and wind have opposed effects on thickness. An outside-fall response or pure in-place basal fraction is unavailable.'
BASIN_IDS = {'FLV93-C1','FLV93-C7','FLV93-T206','FLV94-FC-AUG','MWB07-PAT','MWB07-LIL','MWB07-DH','B2020-079'}


def records(selected):
    registration = json.loads((AUDIT/'great-basin-registration.json').read_text())
    mapped = registration['mapped_positions']
    rows, updates, archives = [], {}, []

    def add(rid, name, xy, coordinate, reference, files, where, identity, age, setting, quantity,
            detail, assumption, kind=None, limit=None, url='', **other):
        row = observation(rid, name, *xy, coordinate_basis=coordinate, coordinate_quality=coordinate,
            primary_reference=reference, evidence_source=files, evidence_locator=where, identity=identity,
            local_age=age, age_basis='Local dates and imported event ties are distinguished in the age field.',
            deposit_class='occurrence_or_depositional_control', setting=setting, quantity_kind=quantity,
            source_thickness_notes=detail, caveats=assumption, identity_quality=identity,
            thickness_quality=detail, verification='primary_description_or_table_checked',
            fallout_model_status='retain_occurrence_without_original_fallout_bound')
        row.update(other)
        if kind:
            if kind=='measurement':
                row['observed_point_cm']=limit
            else:
                row.update(bound_relation='>=' if kind=='lower_limit' else '<=', bound_cm=limit,
                    bound_target='Original climactic bulk fallout conditional on the stated preservation transfer.')
            row.update(fallout_model_status='include_qualified_'+kind)
            selected[rid]=choose(row,kind,limit,detail,assumption)
            if kind=='upper_limit':
                item=archive(row,limit,assumption)
                item.update(url=url, reading='occurrence_only', source_quantity=quantity, control=detail)
                archives.append(item)
        if rid.startswith('B2020-'):
            updates[str(int(rid.split('-')[1]))]={k:v for k,v in row.items() if v!='' and k not in
                ('record_id','locality','accepted_for_model','parent_record_id')}
        else:
            rows.append(row)
        return row

    xyz='USGS OFR 93-620 Fig. 2, numbered hole or trench symbol. Map ticks and scale are recorded in great-basin-registration.json. Datum not stated.'
    geo='Glass compositions matched to Mazama by A. M. Sarna-Wojcicki. The Mono Craters and older reworked glasses in the same survey have separate assignments.'
    for rid, core, page, read, cap, deposit, date in [
        ('FLV93-C1',1,33,'core1',2,'3.14 m','WW79, organic-rich clay at 3.18–3.20 m below ash. Table 2 gives 6730 ±30 conventional BP; the core graphic reads ±70.'),
        ('FLV93-C7',7,67,'core7',2.5,'2.80 m','WW92 at 2.35–2.37 m, 5780 ±80 BP; WW93 at 2.90–2.92 m, 7110 ±100 BP, above and below respectively.')]:
        contact=registration['core_log_digitization'][read]
        lo,hi=contact['graphical_enclosure_cm']
        layer=f"Graphic lithology band about {contact['nominal_graphical_width_cm']:.1f} cm; outward reading {lo:g}–{hi:g} cm. Use the upper endpoint. "
        layer+=('Reheis et al. (1996) also list moderately clean Mazama at 3.14–3.15 m. The interval sampled need not be the full bed.' if core==1 else 'The bed lies between laminated clay with rootlets and organic-rich silty clay. Its plotted contacts enclose about 2 cm.')
        add(rid,f'Fish Lake Valley, Leidy Creek peat core {core}',mapped[rid]['latitude_longitude'],xyz,FLV,
            'leidy1993.pdf; https://doi.org/10.1046/j.1365-2117.1996.00205.x',f'OFR pp. 4–5, 19 and {page}, Table 2, Fig. 2, Appendix 1.',
            geo,date,'Peat/marsh at toe of Leidy Creek alluvial fan. Whole campaign recovery 96%.','graphical_preserved_bed_range',
            layer,CAP,'upper_limit',cap,url='https://doi.org/10.3133/ofr93620',observed_min_cm=lo,observed_max_cm=hi,
            core_depth_or_sample_interval=deposit,deposit_class='discrete_tephra_in_organic_sediment')
    rid='FLV93-T206'
    add(rid,'Fish Lake Valley, Leidy Creek trench 206-LC',mapped[rid]['latitude_longitude'],xyz,FLV,'leidy1993.pdf',
        'OFR Fig. 2 and Fig. 5 continuation, unit 13, printed pp. 19, 21–22.',geo,
        '6850 radiocarbon BP is the eruption tie used by the authors. No local date on the layer.',
        'Top of debris-flow unit 13. The ash forms a separate almost continuous horizon.',
        'preserved_terrestrial_range','The ash layer varies from 1 mm to 1 cm. Use the 0.1 cm minimum on the authors’ possible primary-airfall interpretation.',
        'Requires in-place fall without local import at the measured range minimum. Adjacent debris-flow gravel is excluded.',
        'lower_limit',.1,observed_min_cm=.1,observed_max_cm=1,core_depth_or_sample_interval='Approximately 4.5 m in trench.',
        deposit_class='possible_primary_airfall_over_debris_flow')
    rid='FLV94-FC-AUG'
    fc=mapped[rid]['latitude_longitude']
    fp=mapped[rid]['locator']
    add(rid,'Fish Lake Valley, Furnace Creek hand auger',fc,fp,FURNACE,'flv1994.pdf',
        'Tectonic setting p. 3, Figs. 1 and 2b, Table 2 footnote 2.',
        'Field report cites the original 1990 Mazama identification by Sarna-Wojcicki, separately from tentative trench samples.',
        'Eruption tie 6.85 ka radiocarbon in the report. The nearby trench charcoal dates belong to its own stratigraphy.',
        'Ponded sediment behind shutter ridge. Auger depth 6.5 m.','reported_strict_preserved_ceiling',
        'A discrete layer less than 1 cm thick at about 2.25 m is represented as an upper-censored 1 cm observation. The nearest trench is 50 m north.',
        CAP,'upper_limit',1,url='https://doi.org/10.3133/mf2266',core_depth_or_sample_interval='At about 225 cm in hand auger.',
        interval_meaning='Strict reported <1 cm in the auger; non-strict threshold in continuous likelihood.')
    for sample,depth,quantity,field in [
        ('224',234,'mixed_unidentified_pod','Heavily reworked material containing less than 10% shards. No analysis; compared to 226 on stratigraphic grounds.'),
        ('225',255,'tentative_lens_with_measured_thickness','Clean silt-size lens, about 2 mm thick and 20 cm long. Tentative Mazama(?) in Table 2.'),
        ('226',233,'mixed_unidentified_pod','Reworked pod with about 50% shards in silt. Mazama or Mono Craters(?).')]:
        add('FLV94-FC-'+sample,'Furnace Creek trench 93-2a, FLV-'+sample+'-FC',fc,fp,FURNACE,'flv1994.pdf',
            'Table 2; trench 93-2a unit S. '+('Section 15S3.' if sample=='225' else 'Upper part of unit S.'),
            'Tentative identification only; see individual sample description.',
            'Four combined charcoal samples from upper S, CAMS-8358, 6320 ±60 radiocarbon BP. Separate from tephra analysis.',
            'Ponded terrace/fault trench.',quantity,field,
            'Nearby auger and trench preserve different materials. A clean lens does not independently confirm the event.',
            parent_record_id=rid,core_depth_or_sample_interval=f'{depth} cm below surface',observed_point_cm=.2 if sample=='225' else '')
    add('SL96-FC71','Furnace Creek, FLV-71-FC / T-195-5',('', ''),'Native Furnace Creek drainage. Table 1 alone does not locate this sample relative to trench 93-2a.',
        'Slate, USGS Circular 1119 (1996), geochronologic controls on Fish Lake Valley fan deposits.','flv-circ1996.pdf',
        'Geochronologic controls Table 1, unit Qfce, FLV-71-FC / T-195-5.',
        'Electron-microprobe correlated Mazama ash; original author identified it as the southernmost known ash locality.',
        '6845 ±50 radiocarbon BP is the imported event age.','Alluvial fan sample.','burial_depth_only',
        '490 cm is depth below surface. The table gives no ash-bed thickness.',
        'Potential overlap with other Furnace Creek work is unresolved. A burial depth alone supplies no local fallout bound.',
        core_depth_or_sample_interval='490 cm')
    for rid,name,coords,core,depth,value,kind,detail,age in [
        ('MWB07-PAT','Patterson Lake',(41.3867,-120.2236),'2KC','122–123',1,'measurement',
         'Discrete 1 cm Mazama ash. A separate 1 cm Llao bed lies at 127–128 cm.',
         'CURL-5741 pine needle above 6030 ±40 conventional BP; CURL-5742 twig beneath both ashes 7190 ±60 BP. Imported Mazama tie 7627 cal BP.'),
        ('MWB07-LIL','Lily Lake',(41.9758,-120.2097),'99B','729–774',45,'upper_limit',
         '729–774 cm gives 45 cm of assumed Mazama. Inflow and closed-basin capture can thicken it. A separate susceptibility sentence describes the basal clay at 9.95–9.00 m; Fig. 2, Table 1 and the lithologic description identify the 7.29–7.74 m ash.',
         'AA-51467 wood above, 5073 ±61 radiocarbon BP; CURL-5746 needles below, 7920 ±45 BP. Other dates around a younger clay unit are out of order. Imported Mazama tie 7627 cal BP.'),
        ('MWB07-DH','Dead Horse Lake',(42.5606,-120.7781),'99B','113–139 (Table 1) or 113–140 (text)',27,'upper_limit',
         'Table 1 places the bed at 113–139 cm, giving 26 cm. The authors describe 113–140 cm in the lithology text. Use the wider 27 cm package.',
         'AA-55855 lake sediment above 4593 ±46 radiocarbon BP; AA-55856 lake sediment below 8566 ±72 BP. A deeper charcoal date is inverted. Imported Mazama tie 7627 cal BP.')]:
        add(rid,name,coords,'Reported site coordinates, Minckley et al. pp. 2168–2169. Individual hole position and datum unspecified.',MWB,
            'minckley2007.pdf',f'Core {core}, section 3.1, Fig. 2 and Table 1, pp. 2171–2173.',
            'Patterson identification credited to Sarna-Wojcicki; Lily is assumed and Dead Horse attributed to Mazama in the lithology text. Table 1 footnote e credits Sarna-Wojcicki.',
            age,'High-elevation lake, 5 cm modified Livingstone core.','measured_bed' if kind=='measurement' else 'bounded_lacustrine_package',
            detail,POINT if kind=='measurement' else CAP+' The glass assignment at this site is conditional.',kind,value,
            url='https://doi.org/10.1016/j.quascirev.2007.04.009',core_depth_or_sample_interval=depth+' cm below mud surface.',
            deposit_class='primary_ash_in_lacustrine_record',observed_min_cm=26 if rid=='MWB07-DH' else '',observed_max_cm=27 if rid=='MWB07-DH' else '')
    add('MWB07-PAT-LLAO','Patterson Lake, older Llao horizon',(41.3867,-120.2236),'Same core 2KC as MWB07-PAT.',
        MWB,'minckley2007.pdf','Table 1, core 2KC and lithology pp. 2171–2172.',
        'Older Llao Rock tephra, distinguished from climactic Mazama.','Imported age 7798 cal BP.',
        'Lake gyttja.','other_eruption_measured_bed','Separate 1 cm bed at 127–128 cm.',
        'Different eruptive phase. Exclude its thickness from the climactic bed.',parent_record_id='MWB07-PAT',
        observed_point_cm=1,core_depth_or_sample_interval='127–128 cm',deposit_class='precursor')
    add('B2020-079','Carp Lake',(45.918393,-120.882501),
        'Compiler lake coordinate matches Whitlock and Bartlein’s 45°55′N, 120°53′W to arcminute precision. Hole-specific position unresolved.',
        WHIT,'carp1997.pdf; https://doi.org/10.1016/S0031-0182(99)00092-9',
        '1997 Table 1, 1985 core; 2000 pp. 11–13 and Tables 1–2, core 90. Depths referenced to the 1990 water surface.',
        'Carp Ash-3 matched to climactic Mazama by electron-microprobe glass composition.',
        '1997 table uses 6800 ±200 conventional BP, 7581 cal BP. 2000 imports 6730 ±40 BP, 7540 cal BP. Local organic dates bracket the ash.',
        'Intermittently dry crater lake and composite core chronology.','core_contact_ceiling',
        'Both original studies place Carp Ash-3 at 364–368 cm. The 4 cm thickness is a preserved-bed reading. S1 lists 2 cm. The fit uses a 4 cm ceiling and one likelihood across the source core labels.',
        CAP+' The early-Holocene drying history makes loss possible.', 'upper_limit',4,url='https://doi.org/10.1016/S0031-0182(99)00092-9',
        core_depth_or_sample_interval='3.64–3.68 m below the reference water surface.',observed_point_cm=4)
    updates['23']=dict(verification='primary_measurement_and_geochemistry_checked',
        evidence_source='https://doi.org/10.1029/JB091iB03p03609; fish-pollen.json',
        evidence_locator='Verosub et al. 1986 pp. 3610–3614, Figs. 2–3 and 8; Neotoma site 10263, dataset 15879.',
        coordinate_basis='S1 lake position lies inside Neotoma’s investigator-deposited site polygon. Five coring holes at the centre occupied less than 50 m².',
        identity='Tephra V was distinguished from earlier tephra VI by mineralogy and major-element chemistry.',
        local_age='6790 ±70 conventional radiocarbon BP from their composite age model.',
        quantity_kind='measured_preserved_bed',observed_point_cm=5,thickness_quality='Verosub et al. explicitly measure 5 cm.',
        caveats='Magnetic direction in the ash is anomalous. The authors discuss rapid settling and possible subaqueous flow. The five nearby holes do not provide five independent depositional observations.')
    updates['59']=dict(verification='primary_investigator_archive_and_horizon_checked_thickness_pending',
        evidence_source='breit-geochron.json; https://doi.org/10.1016/j.yqres.2015.12.010',
        evidence_locator='Neotoma site 14314, collection BREITEN 16241, dataset 22406, chronological control 47445.',
        coordinate_basis='Published S1 position within the Neotoma lake polygon. Neotoma locates the hole in the deepest part, 7.6 m water.',
        identity='Minckley and Long identifies a recovered Mazama horizon. Neotoma control 47445 identifies tephra on the age model.',
        thickness_quality='Neotoma’s 1 cm control thickness at 230 cm describes the chronological sample. It does not independently measure the compiled 17 cm bed at 249 cm.',
        caveats='Depth origins and the compiler’s 17 cm still require the original 2014 log.')
    updates['49']=dict(verification='primary_site_archive_only_compilation_thickness_pending',
        evidence_source='neotoma-three.json; buckland2020-esm1.xlsx',
        evidence_locator='Neotoma Three Creek Lake site 27608, collection MIN71, dataset 49580 is a modern surface sample.',
        coordinate_basis='Compiled coordinate falls within the Neotoma lake footprint. The modern sampling station is not the coring position.',
        thickness_quality='37 cm remains a compilation-only bed measurement. The modern surface sample cannot verify its age or thickness.')
    updates['22']=dict(verification='primary_excavation_facies_checked_legacy_sample_uncertain',
        evidence_source='paisley-jenkins.pdf; paisley-botany.pdf; https://doi.org/10.1126/science.1218443',
        evidence_locator='Jenkins et al. chapter 28 pp. 487–498 and 2012 supporting materials pp. 9–10, Tables S13–S15. Original 1951 sample not matched to an excavation unit.',
        identity='Mazama Set O in numbered Paisley Caves. Related modern profiles are registered separately.',
        local_age='Y-109, 7610 ±120 conventional BP, rat dung immediately beneath a pumice layer in the legacy date series. The published excavation columns have their own dates.',
        thickness_quality='1951 entry is an occurrence. Preserve modern 35 cm, 30 cm and 1–3 cm records in their own excavation contexts.',
        caveats=SHELTER)
    group='35LK3400, Five Mile Point on the eastern shore of pluvial Summer Lake. Named caves and excavation grid maps in Jenkins, ch. 28. The 1951 compilation coordinate is not a surveyed position for any numbered excavation.'
    cave_files='paisley-jenkins.pdf; paisley-botany.pdf; https://doi.org/10.1126/science.1218443'
    cave_age='Deposits at about 7.6 ka cal BP. Each field age dates material in its own archaeological context.'
    for suffix,name,loc,detail,low,high,age in [
        ('C1','Cave 1, unit 1/2A','Ch. 28 Table 28.1, unit 1/2A.',
         'One coprolite at 1366.33 m elevation is labelled Mazama and measured twice. No bed contacts at that unit.',None,None,
         'OxA-16496, 6608 ±35 BP and Beta-213428, 6640 ±40 BP on the same specimen.'),
        ('C2','Cave 2, mouth','Ch. 28 pp. 494–495, Figs. 28.3–4.',
         'The mouth exposure beneath later roof fall contains 35 cm of undisturbed laminated tephra above compact bat guano. Undisturbed bedding describes subsequent preservation.',
         35,35,'Guano at the basal contact, UCIAMS-68046, 6790 ±15 conventional BP.'),
        ('C5','Cave 5, south block mouth','Ch. 28 pp. 497–498, south block, Cave 5.',
         '30 cm of white laminated ash beneath disturbed surface sediment. The layer thins rapidly toward the rear and north.',
         30,30,'Nearby pre-Mazama hearth Beta-191539, 7640 ±50 conventional BP.'),
        ('C5-N','Cave 5, northern profiles I and II','2012 supplement Table S13, units 5/12A and 5/12C.',
         'LU4 is 1–3 cm thick here and pinches out northward. In profiles III/IV and Cave 2, 1–3 cm refers to individual beds inside the deposit (Tables S14–S15).',
         1,3,cave_age)]:
        fields={}
        if low is not None:
            fields=dict(observed_point_cm=low) if low==high else dict(observed_min_cm=low,observed_max_cm=high)
        add('P13-'+suffix,'Paisley '+name,('',''),group,JEN,cave_files,loc,'Field-identified laminated Mazama fallout-derived sediment in a protected, partly alluvial archive.',
            age,'Rock shelter and dripline excavation.','preserved_cave_quantity' if fields else 'archaeological_horizon_only',
            detail,SHELTER,parent_record_id='B2020-022',deposit_class='fallout_material_redistributed_in_shelter',**fields)
    add('K18-LSP1','LSP-1 Rockshelter (35HA3735)',('',''),
        'Native archaeological site 35HA3735, northern Warner Valley, Oregon, excavation section and Fig. 4.13. The publication does not locate each pocket in geographic coordinates.',
        'Kennedy (2018), A Paleoethnobotanical Approach to 14,000 Years of Great Basin Prehistory, University of Oregon.',
        'paisley-botany.pdf','pp. 124–125, Stratum V; Fig. 4.13.',
        'Author describes small pockets of Mazama tephra within Stratum V.',cave_age,
        'Poorly sorted fan gravel, eolian sand and silt in a rockshelter.','unquantified_patch',
        'Individual pockets interrupt the host stratum; no pocket thickness or capture efficiency is given.',
        'Lateral transport, shielding and discontinuity prevent transfer to open-surface fallout.')
    add('KB96-WASHOE','Washoe Lake, 1992 drilling survey',(39.235756,-119.773844),
        'USGS lake station 10348700 is a waterbody reference near its southern margin. The climate reports do not tie the Mazama horizon to a named borehole. It is not a plotted sample position.',
        'Karlin, Trexler and Petersen (1996), USGS Circular 1119, pp. 37–38; final DOE report, doi:10.2172/2571.',
        'washoe-doe.pdf; flv-circ1996.pdf; https://waterdata.usgs.gov/monitoring-location/10348700/',
        'Drilling program and upper-core chronology, final report pp. 1–2, USGS Circular pp. 37–38.',
        'Reported Mount Mazama horizon used as an age tie. No published glass data or bed contacts in these two reports.',
        '6850 conventional BP imported as the age tie. Three shell dates in the upper metre. The final report gives 23 and 11 cm/kyr for adjacent intervals.',
        'Shallow, periodically dry playa lake. Four drill sites, two to about 25 m, plus measured trenches. Overall 85% recovery for the deeper coring.',
        'archive_age_tie_with_unresolved_fallout','The chronology implies approximately 1.07 m to the tie from 2.6 × 23 + 4.25 × 11 cm. This is an age-model depth only.',
        'Repeated subaerial exposure, erosional lags, sheet flow and unknown hole assignment preclude a reliable thickness transfer, even though the upper 3 m are described.',
        core_depth_or_sample_interval='Original bed contacts unreported. 3 m and 12 m refer to larger stratigraphic packages.')
    add('BGH05-WADS','Wadsworth Amphitheater, Nixon terrace',('',''),
        'Southern edge of the Wadsworth Amphitheater, Nevada. NBMG M153, Qtn in the Wadsworth 7.5-minute quadrangle, 39°37′30″–39°45′N, 119°15′–119°22′30″W. No numbered ash station on plate.',
        'Bell, Garside and House (2005), Nevada Bureau of Mines and Geology Map 153; Davis (1978) cited on plate.',
        'wadsworth-map.pdf; flv-circ1996.pdf','M153 unit Qtn, original plate wording; USGS Circular 1119, Table 1 of the regional tephrochronology.',
        'The map equates the Davis Tsoyawata bed to 6.85 ka Mazama. USGS lists Tsoyowata separately as the older Llao Rock fall. The underlying glass comparison is unresolved.',
        'Imported historical 6.85 ka conventional tie on the map; unit attribution is contested.',
        'Nixon terrace constructional and strath gravels.','ambiguous_phase_and_unmeasured_bed',
        'A 5–10 m exposure at the quarry is terrace fill. Soil-horizon dimensions are not tephra thickness.',
        'Phase ambiguity, possible fluvial transport, and absence of bed contacts at the ash locality prevent a primary thickness estimate.')
    add('FM16-DIAM','Diamond Pond (Malheur Maar), Steens project',(43+6/60+15/3600,-(118+49/60+1/3600)),
        'Lake-level USGS station 430615118490100. Coordinates decoded from its original surface-water station name, not an investigator core.',
        'Verosub et al. (1986); Foit and Mehringer (2016), Holocene tephra stratigraphy in four lakes; Wigand (1987), Diamond Pond.',
        'https://doi.org/10.1029/JB091iB03p03609; https://www.cambridge.org/core/journals/quaternary-research/article/abs/holocene-tephra-stratigraphy-in-four-lakes-in-southeastern-oregon-and-northwestern-nevada-usa/2CF58B4E6E4837F89276B5F0E3BA6062',
        'Verosub et al. pp. 3609–3611; Foit and Mehringer study-site summary; Wigand’s late-Holocene pollen sequence.',
        'Fish/Wildhorse are explicitly identified for the Llao and Mazama pair. Overlap of other tephras across all four lakes does not establish the same climactic layer in this core.',
        'Accessible Diamond Pond paleoecological record spans approximately the last six radiocarbon millennia; depth coverage differs among studies.',
        'Maar pond archive.','cross_reference_without_target_confirmation',
        'The five-centimetre Fish Lake reading is not transferable to Diamond Pond. No climactic bed is measured here in the verified passages.',
        'Source cross-reference only. Neither deposition nor absence is established for this core.')
    return rows,updates,archives
