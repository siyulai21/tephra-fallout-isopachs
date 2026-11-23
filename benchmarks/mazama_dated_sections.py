"""Primary core and field records located through the radiocarbon citation network."""
import json

from benchmarks.mazama import AUDIT
from benchmarks.mazama_young import archive, choose, observation

MAT = 'Mathewes (1973), Canadian Journal of Botany 51, 2085–2103, doi:10.1139/b73-271.'
MAT_URL = 'https://doi.org/10.1139/b73-271'
TEX = 'Valastro, Davis and Varela (1980), University of Texas at Austin radiocarbon dates XIV, doi:10.1017/S0033822200011590.'
TEX_URL = 'https://doi.org/10.1017/S0033822200011590'
HUF = 'Moseley, Bursik and Mehringer (1992), Paleoecology of peatlands at Huff and Hager Lakes, Idaho Department of Fish and Game.'
HUF_URL = 'https://idfg.idaho.gov/ifwis/idnhp/cdc_pdf/moser92j.pdf'
KOO = 'Roll (ed.), 1982, Kootenai Canyon Archaeology, 1979 LAURD final mitigation report, Chapter 2, DTIC ADA151998.'
KOO_URL = 'https://archive.org/details/DTIC_ADA151998'
LAKE = 'Assumes the recovered discrete bed records local fall. Differential compaction, strong inflow, basin focusing or erosion could change the thickness.'
CAP = 'Conditional whole-horizon ceiling. Local deposition must be captured between the observed stratigraphic limits, with no net loss sufficient to exceed the endpoint. Additional sediment or imported ash can enlarge the package.'
MIN = 'Lower bound requires the two identified basal units to be in place, with no material addition or local thickening. Allows loss, bioturbation and unresolved additional airborne ash.'


def records(selected, decisions):
    rows, updates, archives = [], {}, []
    def add(rid, name, xy, basis, ref, files, locator, identity, age, setting, detail,
            kind=None, value=None, assumption='', url='', **fields):
        row = observation(rid,name,*xy,primary_reference=ref,evidence_source=files,evidence_locator=locator,
            coordinate_basis=basis,coordinate_quality=basis,identity=identity,local_age=age,
            age_basis='Age wording distinguishes local dated material and imported eruption correlation.',
            setting=setting,quantity_kind='unquantified',verification='primary_site_description_checked',
            identity_quality=identity,deposit_class='occurrence_or_ambiguous_context',
            fallout_model_status='hold_original_fallout_quantity',source_thickness_notes=detail,
            caveats=assumption or detail,thickness_quality='See dated original observation and inclusion decision.')
        if kind:
            if kind=='measurement':
                row.update(observed_point_cm=value,quantity_kind='measured_preserved_bed')
            else:
                row.update(bound_cm=value,bound_relation='<=' if kind=='upper_limit' else '>=',
                    bound_target='Original local climactic airborne bulk thickness under stated preservation conditions.',
                    quantity_kind='interpreted_original_ceiling' if kind=='upper_limit' else 'preserved_primary_component_minimum')
            row['fallout_model_status']='conditional_principal_'+kind
            selected[rid]=choose(row,kind,value,detail,assumption)
        row.update(fields)
        if rid.startswith('B2020-'):
            updates[str(int(rid[-3:]))]={k:v for k,v in row.items() if k not in ('record_id','parent_record_id','accepted_for_model','locality','source_excel_row','compiler_thickness_cm','source_description')}
        else:
            rows.append(row)
        if kind=='upper_limit':
            entry=archive(row,value,assumption)
            entry.update(url=url,control=detail,reading='occurrence_only')
            archives.append(entry)

    for rid,lake,position,basis,width,lab in [
        ('B2020-251','Marion Lake',(49+19/60,-(122+33/60)), 'Walker 1988, Chapter 2, study coordinates at one-minute precision. Mathewes Fig. 2 locates the hole in this lake.',5,'I-6821, 7645 ±340 radiocarbon BP below'),
        ('B2020-203','Surprise Lake',(49.319444,-122.561944),'Canadian Geographical Names Database JAZBL lake centre. Mathewes describes this small lake about 1 km northwest of Marion Lake.',3.5,'I-6966, 8275 ±135 radiocarbon BP below')]:
        detail=f'Mathewes reports a {width} cm distinct white silicic bed in the deep-water core. Marion Fig. 7C shows the compact unit and Fig. 7D its glass. The two different lake beds enter once each.'
        add(rid,lake,position,basis,MAT,'https://doi.org/10.1139/b73-271; marion1988.pdf; surprise-names.json; egan2016-thesis.pdf',
            'Mathewes pp. 2087–2091, Figs. 1–2, 7C–D and Table 1. Walker 1988, Chapter 2, Fig. 2.1.',
            'Regional Mazama correlation from single ash layer, microscopic glass, surrounding dates and stratigraphy. No site-specific shard chemistry presented.',
            lab+'. Stratigraphic constraint on the bed. The dated samples and the approximately 7.6 ka eruption age are distinct.',
            'Lake sediment core, UBC Research Forest, British Columbia.',detail,'measurement',width,LAKE,
            deposit_class='probable_primary_lake_fallout',thickness_quality='Primary paper reports the bed width.',
            verification='primary_published_log_and_figures_checked',core_depth_or_sample_interval='See Figure 2 for coring position and Table 1 for dated levels.')
    for rid,site,xy,outer,inner,anchors,dates in [
        ('B2020-141','Hager Pond',(48+36/60,-(116+57/60)),105,95,'Tx-2121 at 500–505 cm and Tx-2116 at 600–605 cm','6390 ±230 and 6630 ±80 radiocarbon BP'),
        ('B2020-129','Waits Lake',(48+11/60,-(117+48/60)),141,110,'Tx-2659 at 315–330 cm and Tx-2660 at 440–456 cm','7910 ±100 and 8480 ±260 radiocarbon BP; both appear too old, attributed to old-carbon contamination')]:
        detail=f'Mack identifies Mazama between {anchors} in core 1. The conservative outer sample edges enclose {outer} cm, with a {inner} cm inner gap. The outer width includes host sediment and permits uncertainty about where each sample fixes the contact.'
        add(rid,site,xy,'Primary date-list coordinates, rounded to the minute; mire or south-shore coring area. No GPS precision implied.',
            TEX,'valastro1980.pdf','Valastro et al. pp. 1096–1098, Hager Pond and Waits Lake series, core 1.',
            'Original collector explicitly brackets Mazama in named cores. No glass analyses in the radiocarbon list.',
            dates+'. Core-relative stratigraphy is the control.','Peat or gyttja core.',detail,'upper_limit',outer,CAP,url=TEX_URL,
            core_depth_or_sample_interval=anchors,deposit_class='dated_host_bracketing_correlated_ash')
    add('B2020-147','Huff Lake',(48.7407,-117.0625),'S1 lake coordinate on the Washington side of the state line. Original study cored the south end of the bog, about 10 m landward of open water.',
        HUF,'moseley1992.pdf','Huff Lake field account, pp. 2–4. Appendix 1 in the distributed report ends with its title page.',
        'Probable Mazama and separate older Glacier Peak horizons recognized by field position and appearance; laboratory confirmation was pending.',
        '6850 radiocarbon BP assigned by correlation, no independent date on the ash.','Bog margin, overlapping drives.',
        'Upper 0.5 m excavated in blocks and overlapping drives span 0.5 to approximately 10.8 m. Both ashes were recovered. The principal uses 1080 cm, the whole nominal surface-to-base depth. It retains 50 cm of headroom over the nominal 1030 cm span of the drives. No individual ash contacts are reported.',
        'upper_limit',1080,CAP+' The approximate basal depth and the probable identification are additional conditions.',url=HUF_URL,
        core_depth_or_sample_interval='0.5 to approximately 10.8 m; includes multiple lithologies and two distinct tephras.',
        deposit_class='probable_tephra_in_recovered_archive')
    geo=json.loads((AUDIT/'kootenai-section-positions.json').read_text())
    rep=(geo['representative_latitude'],geo['representative_longitude'])
    loc='Report p. 2.17 gives two adjacent PLSS sections, without explicitly assigning a coordinate to the named pit. The joint footprint midpoint is a section-scale proxy. Both section centres are retained.'
    add('R1982-KOOT-776','Kootenai canyon, 24LN1125 A-776',rep,loc,KOO,'koote1982.pdf; plss-koote-sections.json',
        'Chapter 2 pp. 2.17–2.21, Tables 2.1–2.2, Fig. 2.6. Area A, grid 776.',
        'Primary BCV1–3 have massive bedding and matching Mazama glass. BCV4 includes two chemical populations; BCV5 is a contaminated redeposit.',
        'Correlated Mazama, 6700 radiocarbon BP in the report. TX-3220, 8170 ±100 BP beneath ash, belongs to different site 24LN1046.',
        'Floodplain terrace cut and archaeological trench.',
        'BCV1 is 1.5–2.0 cm and BCV2 is 0.5–1.0 cm. Their combined preserved minimum is 1.5 + 0.5 = 2 cm. BCV3 is confirmed but has no component thickness. BCV4 (5–7 cm) and BCV5 (1.0–1.3 cm) do not enter this minimum.',
        'lower_limit',2,MIN,deposit_class='primary_fallout_components',
        interval_meaning='BCV1 and BCV2 together span 2–3 cm across the exposure. The securely identified bulk total also contains unquantified BCV3.',
        observed_min_cm=2,observed_max_cm=3,core_depth_or_sample_interval='Area A grid 776; 50–80 cm is regional common burial depth, not thickness.')
    for idx,quantity,value,maximum,context in [
        ('1','component_range',1.5,2,'Chemically identified massive basal silt loam.'),
        ('2','component_range',.5,1,'Chemically identified massive silt loam with worm burrows.'),
        ('3','unquantified','','','Coarsest primary unit; independent glass match and no numerical thickness in text.'),
        ('4','mixed_identity_range',5,7,'Airfall sedimentary structure, two different glass populations; Mazama fraction unresolved.'),
        ('5','secondary_range',1,1.3,'Mazama glass accompanied by abundant detrital minerals; redeposited.')]:
        add('R1982-BCV'+idx,'Kootenai 24LN1125 A-776 BCV'+idx,rep,loc,KOO,'koote1982.pdf','Chapter 2 pp. 2.17–2.21, Tables 2.1–2.2.',
            context,'Same section as A-776.','Co-located component.',context,parent_record_id='R1982-KOOT-776',
            observed_min_cm=value,observed_max_cm=maximum,quantity_kind=quantity,
            deposit_class='reworked' if idx=='5' else 'mixed_identity' if idx=='4' else 'primary_component',
            fallout_model_status='component_of_single_likelihood' if idx in ('1','2','3') else 'excluded_mixed_or_reworked')
    add('R1982-KOOT-1020','Kootenai canyon, 24LN1020 Jennings',rep,loc,KOO,'koote1982.pdf',
        'Chapter 2, Mazama field sections, Figs. 2.8–2.10.',
        'Primary regional marker recognized above early Holocene deposits at Jennings.','Regional correlation, approximately 7.6 ka cal BP.',
        'Terrace exposure.','Regional total 8–14 cm (about 11 cm) includes packages of unresolved component composition. No independent site-specific measurement was assigned.',
        fallout_model_status='mapped_occurrence_shared_footprint')
    for sid,site,lo,hi in [('VAN','Vantage region, Washington',5,8),('CHA','Challis area, Idaho',3,5)]:
        add('R1982-'+sid,site,('',''),'Regional statement only. No measured section position provided.',KOO,'koote1982.pdf',
            'Chapter 2, printed p. 2.21, regional comparison.','Authors call these primary Mazama thicknesses.',
            'Regional correlation only.','Unspecified sections.',f'Reported regional range {lo}–{hi} cm; individual outcrops and overlap with earlier work unresolved.',
            observed_min_cm=lo,observed_max_cm=hi,quantity_kind='regional_comparison_range',fallout_model_status='hold_section_location')
    add('TX1980-SIMP','Simpsons Flats, Washington, core 7',(48+25/60,-118.25),'Date-list location to the nearest minute. Closed mire on Colville Reservation.',
        TEX,'valastro1980.pdf','pp. 1096–1097, core 7.','Mack identifies Mazama at 680 cm in core 7.',
        'Author flags a stratigraphic time gap spanning approximately 9000–6700 radiocarbon BP.',
        'Mire gyttja.','680 cm is burial depth. Sample levels Tx-2220 (632–635 cm) and Tx-2418 (710 cm) straddle the horizon, a gross 78 cm enclosure. The documented discontinuity spans the eruption. Loss across the contact leaves original local thickness unbounded.',
        core_depth_or_sample_interval='680 cm, between a 632–635 cm sample and a 710 cm sample. Independent of Oregon Simpson Place Bog.',
        quantity_kind='stratigraphic_position_only',deposit_class='ash_adjacent_to_hiatus')
    original_checks = {
        47: ('long2010-tumalo.pdf', 'Long, Power and Bartlein, doi:10.1016/j.yqres.2010.08.010, Results p. 3, Table 1, Figs. 1–2. The 50 cm bed extends from 4.84 to 5.34 m. Its 7630 ±150 yr age tie is imported. McKay 2012 Table 3.2 gives 44°01.31′N 121°32.63′W for Tumalo, consistent with S1; the coordinates in the 2010 prose differ.', 'primary_measured_bed_checked'),
        54: ('mckay2012.pdf', 'McKay (2012), sec. 3.4.3.4, Fig. 3.8 and Table 3.2. Fourteen cm of silicic tephra at 267–281 cm, identified by stratigraphy and grain size. The 79 cm layer and very thin neighboring mafic ashes have other sources. Independent of identically named lakes.', 'primary_measured_bed_checked'),
        72: ('https://doi.org/10.1080/00040851.1977.12003929; plss-koote-sections.json', 'Mehringer et al. (1977), Tables 2–3, Fig. 5. Five adjacent holes support one composite. Contacts at 5.05 and 5.13 m give an 8 cm preserved ash bed. Separate ash-rich gyttja at 5.00–5.05 m. WSU-1552 above, 6700 ±100 BP; WSU-1553 below, 6720 ±120 BP. Petrography and trace elements confirm Mazama. The retained ≥4.9 cm seasonal primary component and 7 cm separate account have different provenance.', 'primary_composite_contacts_and_dates_checked'),
        148: ('valastro1980.pdf', 'Valastro et al. pp. 1098–1099. Two fen cores. Tx-2882 immediately above the upper Mazama ash, Tx-2881/2883 between two units, Tx-2884 below both. Depositional sequence corroborated independently. Neither component thickness nor equivalence of the two core depth origins is established by this date list. Its published 48°54′N 119°02′W differs from other reported positions.', 'primary_age_and_phase_sequence_compilation_thickness'),
        145: ('valastro1980.pdf', 'Valastro et al. p. 1098. Two cores on NW margin of Mud Lake, about 10 km northeast of Conconully. All dates listed below Mazama. The date list gives no upper bounding sample or individual core tie for the reported 40 cm. The Matz 80 cm upland correction is not a field measurement.', 'primary_location_and_context_thickness_pending'),
        146: ('valastro1980.pdf', 'Valastro et al. p. 1097. Two mire cores on Meadow Creek, 48°43′N 117°33′W. The date list establishes the coring area but supplies no independent thickness of the climactic bed. Keep corrected values and alternative unit splits separate.', 'primary_core_position_checked_thickness_pending'),
        142: ('moseley1992.pdf', 'Moseley et al. pp. 2–4. At locality 2 overlapping drives span roughly 12.10–17.4 m below the water surface. Several clean and mixed probable Mazama layers around 15.6 m indicate sinking into soft sediment. A 530 cm full-window cap is conditional on no net loss. The stronger linked Hager mire observation enters once. Locality 1 starts in ash and organics. Locality 3 is nearly continuously cored and has no ash contacts. Locality 4 was unopened and only suspected.', 'primary_field_account_reworking_checked'),
        212: ('https://emrlibrary.gov.yk.ca/gsc/papers/73-7.pdf; harper-names.json', 'Lowdon and Blake 1973, GSC Paper 73-7, Chase series. Two ashes at 45 and 70 cm in Harper Lake, near Chase. Marl and plant fractions dated at 82–86 cm. No glass correlation separates the two horizons in the logger’s description. Official Harper Lake position resolves the 54°/50° latitude discrepancy.', 'primary_date_list_identity_unresolved')}
    for n,(files,detail,status) in original_checks.items():
        merged = dict(decisions.get(str(n),{}))
        merged['evidence_source'] = '; '.join(filter(None,[merged.get('evidence_source','buckland2020-esm1.xlsx'),files]))
        merged['evidence_locator'] = '; '.join(filter(None,[merged.get('evidence_locator'),detail]))
        merged['verification'] = status
        merged['caveats'] = ' '.join(filter(None,[merged.get('caveats'),detail]))
        updates[str(n)]=merged
    updates['47'].update(observed_point_cm=50,core_depth_or_sample_interval='4.84–5.34 m reported in the archived 2010 paper.',
        identity='Distinct discrete lake tephra attributed to Mazama from chronology and stratigraphy.',thickness_quality='50 cm verified in Long et al. 2010.')
    updates['54'].update(observed_point_cm=14,core_depth_or_sample_interval='267–281 cm in McKay Fig. 3.8.',identity='Silicic bed assigned from thickness, grain size and stratigraphic position.',thickness_quality='14 cm verified in McKay 2012.')
    updates['72'].update(observed_point_cm=8,quantity_kind='primary_composite_bed_and_component',core_depth_or_sample_interval='5.05–5.13 m, below a separate ash-rich gyttja.',
        identity='Petrographic and trace-element identification by Okazaki and Smith.',local_age='WSU-1552 above 6700 ±100 and WSU-1553 below 6720 ±120 conventional radiocarbon BP.')
    updates['145'].update(latitude=48+34/60,longitude=-119.5,coordinate_basis='Valastro et al. original Mud Lake NW margin coordinates, rounded to one minute.',
        local_age='Earliest listed uppermost sample Tx-2686 at 254–260 cm is 8030 ±150 radiocarbon BP, below the ash.')
    updates['142'].update(bound_relation='<=',bound_cm=530,bound_target='Full recovered central-lake archive, conditional on complete containment and no net loss; grouped with Hager Pond.',
        quantity_kind='conditional_linked_archive_ceiling',parent_record_id='B2020-141',deposit_class='sunk_and_mixed_lake_tephra',fallout_model_status='linked_core_omit_duplicate_lake_likelihood')
    updates['212'].update(latitude=50.737222,longitude=-119.716944,coordinate_basis='Canadian Geographical Names Database JANPE, Harper Lake centre. Original study describes Harper Lake 10 km south of Chase.',
        identity='Two unnamed volcanic ash layers. Which was climactic Mazama remains unresolved.',local_age='GSC-1487 marl at 82–86 cm, 7190 ±150 radiocarbon BP; paired plant fraction GSC-1487-2, 7400 ±160 BP.')
    return rows,updates,archives
