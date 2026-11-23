"""Primary lake records and terrestrial sections checked after the Young transcription."""
from pyproj import Transformer

from benchmarks.mazama_young import archive, choose, observation

EVENT = 'Climactic Mazama correlation, approximately 7.6 ka cal BP. '
CORE = 'Assumes that the logged discrete bed captures local primary fall and that focusing, erosion, compaction and coring distortion do not materially change its thickness.'
CAP = 'Conditional upper endpoint. At least the original local bulk thickness must be represented within the recovered interval. Catchment input can enlarge the package. Net export, erosion, compaction or missing core can invalidate the ceiling.'
MIN = 'Conditional lower endpoint on in-place climactic fallout. Requires no local thickening by imported ash or deformation. Erosion and incomplete recovery can leave only part of the deposit.'
BRACKET = 'A geometric ceiling on the preserved horizon. Transferring it to original local fall requires continuous recovery across the bracket with no net thinning sufficient to violate it. Stratigraphic order and bed attribution must hold.'
SPOON = 'Spooner, Brubaker and Foit, North Cascades National Park Service Complex, task agreement J9W88030002.'
CHASE = 'Chase et al. 2008, Palaeogeography, Palaeoclimatology, Palaeoecology 257, 244–259. doi:10.1016/j.palaeo.2007.10.020.'
HERR = 'Herring 2014, Late Quaternary and Holocene paleoecology of interior mesic forests of northern Idaho, Chapters III–IV. Herring et al., Journal of Ecology, doi:10.1111/1365-2745.12826.'
ROSEN = 'Rosenberg et al. 2004, Midge-inferred Holocene climate history of two subalpine lakes in southern British Columbia, The Holocene 14. doi:10.1191/0959683604hl703rp.'
BAKER = 'Scott et al. 2020, Latest Pleistocene to Present Geology of Mount Baker Volcano, USGS Professional Paper 1865. doi:10.3133/pp1865.'
OSB = 'Osborn et al. 2012, Latest Pleistocene and Holocene glacier fluctuations on Mount Baker, Washington, Quaternary Science Reviews. doi:10.1016/j.quascirev.2012.06.004.'
MARC = 'Marcott 2005, A tale of three sisters, Portland State MS thesis. Marcott et al. 2009, Quaternary Research 71, 181–189, doi:10.1016/j.yqres.2008.09.002.'
UTM = Transformer.from_crs(26710,4326,always_xy=True)


def records(selected):
    rows, changes, archives = [], {}, []
    def add(rid, name, xy, ref, source, locator, coordinates, setting, ident, age,
            evidence, kind=None, value=None, assumption='', **details):
        row = observation(rid,name,*xy,primary_reference=ref,evidence_source=source,
            evidence_locator=locator,coordinate_basis=coordinates,coordinate_quality=coordinates,
            setting=setting,identity=ident,local_age=EVENT+age,
            age_basis='Event age imported unless an independent stratigraphic date is specified.',
            source_thickness_notes=evidence,deposit_class='identified_ash_horizon',
            quantity_kind='unquantified',fallout_model_status='occurrence_or_context',
            thickness_quality='Observed quantity and its original-to-preserved transfer are identified at the source locator.',
            caveats=assumption or evidence)
        if kind:
            if kind=='measurement':
                row.update(quantity_kind='reported_point',observed_point_cm=value)
            else:
                row.update(bound_relation='<=' if kind=='upper_limit' else '>=',bound_cm=value,
                           quantity_kind='interpreted_original_upper' if kind=='upper_limit' else 'conditional_primary_component',
                           bound_target='Original local climactic bulk fallout under the stated preservation assumptions.')
            row['fallout_model_status']='conditional_principal_'+kind
            selected[rid]=choose(row,kind,value,evidence,assumption)
        row.update(details)
        if rid.startswith('B2020-'):
            changes[str(int(rid.split('-')[1]))]={k:v for k,v in row.items() if v!='' and k not in ('record_id','accepted_for_model','parent_record_id','locality')}
        else:
            rows.append(row)
        if kind=='upper_limit':
            a=archive(row,value,assumption)
            a.update(url=urls[source.split('; ')[0]],reading='inspected_no_bed' if rid=='CH2008-RED' else 'occurrence_only')
            if rid=='CH2008-RED':
                a.update(reading='inspected_no_bed',horizon='Inferred eruption-age depth between independently dated levels. Target identification is an inspection window.')
            archives.append(a)
        return row
    urls={'thunder2007.pdf':'https://depts.washington.edu/pnwcesu/reports/J9W88030002_Thunder_Lake_July_31_2007.pdf',
          'ridley2008.pdf':'https://depts.washington.edu/pnwcesu/reports/J9W88030002_Final_Report_1.pdf',
          'chase2008.pdf':'https://doi.org/10.1016/j.palaeo.2007.10.020','herring2014.pdf':'https://scholarsbank.uoregon.edu/server/api/core/bitstreams/2dd2385a-b961-43da-9835-b610b93963b5/content',
          'baker-usgs2020.pdf':'https://doi.org/10.3133/pp1865','baker2012.pdf':'https://doi.org/10.1016/j.quascirev.2012.06.004',
          'marcott2005.pdf':'https://glaciers.pdx.edu/Thesis/Marcott/Marcott_2005.pdf',
          'shapley2015.html':'https://doi.org/10.1007/s10933-015-9826-6',
          'https://doi.org/10.1002/jqs.3554':'https://doi.org/10.1002/jqs.3554'}
    for rid,name,year,xy,utm,sample,depth,limit,age,detail in [
        ('SP2007-THU','Thunder Lake, North Cascades, Washington',2007,(48+41.81/60,-121-6.51/60),'639187 E, 5395477 N','TL1D 54',335,119.5,'CAMS-120907 at 320.5 cm, 5860 ± 130 14C BP. CAMS-123701 at 440 cm, 10190 ± 50 14C BP.','Entire visible horizon lies between two independently dated plant macrofossil levels. 440 − 320.5 = 119.5 cm.'),
        ('SP2008-RID','Ridley Lake, North Cascades, Washington',2008,(48.949,-121.030),'644259 E, 5423426 N','RY2B-4',206,111,'Distinct younger Glacier Peak Dusty Creek tephra at 166 cm (5830 cal BP by correlation).','The visible bed lies above the gray-clay contact at 277 cm and below the distinct Glacier Peak bed at 166 cm. 277 − 166 = 111 cm.')]:
        key='thunder2007.pdf' if year==2007 else 'ridley2008.pdf'
        add(rid,name,xy,f'{SPOON} {year}, {name.split(",")[0]} report.',key,
            'Study area and core methods, dated levels in Table 1, major oxides in Table 2, visible tephra stratigraphy.',
            f'Author-reported lake coordinates with companion UTM zone 10 coordinates {utm}. The individual borehole is unresolved.',
            'Two parallel, offset 7-cm piston cores in a lake.',f'{sample}, similarity coefficient >0.99 to USGS Wasco W-97 Mazama Climactic standard.',age,
            detail,'upper_limit',limit,BRACKET+' Several hundred mg of each tephra were prepared for analysis. Processing yield, ash purity and focusing prevent calibration of a precise original bulk thickness from mounted material.',
            core_depth_or_sample_interval=f'{depth} cm, source core sample {sample}.')
    for rid,name,xy,a,b,age in [
        ('CH2008-WIN','Windy Lake, British Columbia',(49+48/60+48/3600,-117-52/60-38/3600),287,373,'Wood at 223 cm, 4800 ± 60; bark below at 420 cm, 8350 ± 80 conventional 14C BP.'),
        ('CH2008-THU','Thunder Lake, Cariboo Mountains, British Columbia',(52+13/60+42/3600,-119-20/60-54/3600),170,202,'At 130 cm, 5330 ± 80; at 285 cm, 8630 ± 70 conventional 14C BP. The splice gap near 100 cm is younger.')]:
        add(rid,name,xy,CHASE,'chase2008.pdf','Study sites, Table 1, Fig. 2, core lithology sections 4.1–4.2.',
            'Author lake position to one arc-second. Individual core positions are not tabulated.','Paired offset lake cores, dark gyttja enclosing the bed.',
            'Author correlation by stratigraphic position and regional trajectory; a direct glass match is unreported.',age,
            f'Published layer contacts at {a}–{b} cm give {b-a} cm preserved ash.','measurement',b-a,CORE+' A steep catchment and aquatic deposition can increase the recorded bed. Test the complete package as a ceiling.',
            core_depth_or_sample_interval=f'{a}–{b} cm.')
    add('CH2008-RED','Redmountain Lake, British Columbia',(53+55/60+17/3600,-121-17/60-39/3600),CHASE,'chase2008.pdf',
        'Table 1, Fig. 2, section 4.3.1.','Author lake position to one arc-second.','Split cores through light-gray banded inorganic silts and clays.',
        'An independently bracketed target window. No volcanic tephra was visually evident in this core.','Cone scale at 409 cm, 4420 ± 80 14C BP. Abies needle at 520 cm, 8885 ± 100 14C BP.',
        'The event-age interval lies between 409 and 520 cm. Its entire surviving primary component cannot exceed their 111 cm separation.','upper_limit',111,
        BRACKET+' The 0.5 cm St Helens bed recognized in dark gyttja at Windy does not calibrate visibility in these pale inorganic sediments. No minimum ash or zero is observed.',
        deposit_class='dated_archive_inspection',core_depth_or_sample_interval='409–520 cm.')
    for rid,name,xy,a,b,age,note in [
        ('HER2014-DIS','Dismal Lake, Idaho',(47+7/60,-115-38/60),593,734,'Wood at 504–505 cm, 5640 ± 40 14C BP.','Chapter IV explicitly attributes excess thickness to slopewash and focusing. Below the bed are laminated gyttja and the separate Glacier Peak marker.'),
        ('HER2014-RRL','Rocky Ridge Lake, Idaho',(46+26/60+31/3600,-115-29/60-35/3600),278,439,'At 140–141 cm, 2965 ± 30 14C BP.','A separate uppermost St Helens layer is 0.5 cm. Massive gyttja contains aquatic vegetation mats.'),
        ('HER2014-HOR','Horseshoe Lake, Idaho',(46+32/60+59/3600,-115-4/60-6/3600),178,389,'At 68–69 cm, 3980 ± 30 14C BP.','The parallel cores could not be correlated below the ash. The authors separately report recovery of underlying sediment. Omit the ceiling if the splice missed part of the bed.')]:
        add(rid,name,xy,HERR,'herring2014.pdf','Tables 3.1–3.2, Fig. 3.4 and core results pp. 51–52. Dismal discussed separately in Chapter IV.',
            'Original Table 3.1 lake coordinates. Dismal rounded to minutes, other sites to seconds.','Lake-center sediment cores in northern Idaho.',
            'Named Mazama marker assigned 7630 (7480–7780) cal BP in Table 3.2. A local glass analysis is not supplied for these beds.',age,
            f'Mazama-bearing interval {a}–{b} cm measures {b-a} cm. '+note,'upper_limit',b-a,CAP+' '+note,
            observed_point_cm=b-a,interval_meaning='Preserved package total, conditional original upper endpoint.',core_depth_or_sample_interval=f'{a}–{b} cm.')
    for rid,name,xy,a,b,age,identity in [
        ('ROS2004-FRO','Lower Frozen Lake, British Columbia',(49+36/60,-121-28/60),155,163,'6170 ± 40 14C BP at 147–148 cm, 8180 ± 60 at 188–189 cm.','Mazama correlation in Table 2.'),
        ('ROS2004-EAG','Eagle Lake, Mount Revelstoke, British Columbia',(51+3/60,-118-10/60),81,96,'8330 ± 40 14C BP at 105–106 cm. Distinct younger Mount St Helens Yn.','Glass chemistry by Foit and event chronology, Table 4.')]:
        add(rid,name,xy,ROSEN,'https://doi.org/10.1191/0959683604hl703rp','Study area, Tables 2 and 4, stratigraphy.',
            'Original geographic coordinates to nearest minute. Position represents the lake.','Composite lake core.',identity,age,
            f'Core interval {a}–{b} cm gives a preserved bed {b-a} cm thick.','measurement',b-a,CORE,
            core_depth_or_sample_interval=f'{a}–{b} cm.')
    baker_sections(add)
    marcott_sections(add)
    idaho_sections(add)
    return rows,changes,archives



def baker_sections(add):
    coords = 'Original Appendix 1 reference-section UTM coordinates, NAD27 zone 10N (report p. 12). WGS84 decimal location converted with PROJ. The field section may extend along a cutbank.'
    cases = [
        (5,'Scott Paul Trail swale',587350,5397550,11,2,5,'lower_limit',2,'Layer O between organic-rich soils. The Baker OP tephra is higher.'),
        (15,'Pratt Creek upstream of Ridley Creek',583540,5396720,9,5,20,'upper_limit',20,'Ash in iron-stained soil beneath a 1.7 m lahar. The published 5–20 cm refers to the included ash.'),
        (16,'Middle fork Sulphur Creek',587290,5396380,8,1,1,'lower_limit',1,'The 1 cm discrete layer is directly below 3–8 cm of organic silt containing additional ash masses. Only the discrete base enters.'),
        (17,'Sulphur Creek cutbank by Scott Paul Trail',587350,5395350,5,2,4,'lower_limit',2,'Layer O is preserved in the logged section. Slumping later covered part of the exposure.'),
        (18,'Scott Paul Trail interfluve pit',586330,5397790,6,2,3,'lower_limit',2,'Excavated pit between tributaries. Appendix 1 repeats the reference coordinate of RS–6 at a different setting and altitude. Use the printed area position with that uncertainty. RS–6 has no second likelihood.'),
        (19,'West fork Pratt Creek',583440,5397490,1,3,3,'measurement',3,'Source gives approximately 3 cm. The recovered O horizon underlies a soil and Baker OP.'),
        (21,'Schriebers Meadow, Sulphur Creek',587690,5395230,6,1,4,'lower_limit',1,'Discrete ash between two fine organic units. Younger transported volcaniclastic layers and underlying lahar are excluded.'),
        (25,'Middle Fork Nooksack at Rutsatz Road',563620,5407690,4,2,5,None,None,'Downstream-pointing flame structure produced by lahar shear. Layer strain and undisturbed thickness are unknown.'),
        (12,'Dry Creek, Glacial Lake Baker',597980,5391640,5,0,20,None,None,'Ponded and redeposited ash in deformed lake beds. Original local deposition, capture and imported fractions are unknown. Zero is the observed pinch-out of a transported bed.'),
        (10,'Horseshoe Cove, Glacial Lake Baker',597550,5391340,5,None,None,None,None,'Ponded blue-weathered ash in mixed cross-laminated deltaic facies. The enclosing unit is 0–2.3 m. Sediment import, bypass and deformation preclude an original local endpoint.')]
    for n,name,east,north,unit,low,high,kind,x,note in cases:
        lon,lat=UTM.transform(east,north)
        src=f'Appendix 1, RS–{n} unit {unit}; Chapter E, pp. 60–61. UTM {east} E, {north} N.'
        raw = '' if low is None else (f'{low}–{high} cm. ' if low!=high else f'{"Approximately " if n==19 else ""}{low} cm. ')
        args = dict(core_depth_or_sample_interval=f'RS–{n} unit {unit}.')
        if low!=high:
            args.update(observed_min_cm=low,observed_max_cm=high,interval_meaning='Within-section preserved range; opposite endpoint subject to original bed capture.')
        elif low is not None:
            args['observed_point_cm']=low
        if kind is None:
            args.update(deposit_class='secondary_or_mixed_sedimentary_context',fallout_model_status='hold_reworked_or_ductile_strain',quantity_kind='deformed_or_redeposited_component')
        add(f'SC2020-RS{n:02}',f'Mount Baker RS–{n}, {name}',(round(lat,6),round(lon,6)),BAKER,'baker-usgs2020.pdf',src,coords,
            'Terrestrial reference section.' if n not in (10,12) else 'Glacial Lake Baker deltaic or ponded facies.',
            'Layer O is the climactic Mount Mazama fine glass ash. Baker set OP and Baker eruptive episodes have distinct origins.',
            'Field sequence and published event correlation.',raw+note,kind,x,{'lower_limit':MIN,'upper_limit':CAP,'measurement':CORE,None:note}[kind],**args)
    for rid,name,lat,lon,where,kind,x,note,qual in [
        ('OS2012-SFL','Mount Baker south-flank pits between Easton and Squak glaciers',48+43.9/60,-121-49.3/60,'Table 2, pp. 38–39, Fig. 6','upper_limit',30,'The pits contain up to 30 cm of O and a separate 3 cm of Baker OP. The reported maximum bounds the preserved group. Individual pits lack separate coordinates.',CAP),
        ('OS2012-POC','Pocket Lake, Mount Baker',48+42.7/60,-121-50.6/60,'Table 2, Fig. 10 and section 4.3','upper_limit',38,'Mazama at 0.93–0.55 m, hence 38 cm. A separate younger Baker bed at 0.48–0.41 m and reworked, deeper Baker SC material are excluded.',CAP+' A small moraine-dammed cirque can focus ash into the center core.'),
        ('OS2012-EAS','Easton Glacier left lateral moraine, Mount Baker',48+43.9/60,-121-50.0/60,'Table 2, section 4.2, Fig. 8 and Appendix A; Scott et al. 2020 RS–6',None,None,'OBM glass is Mazama. Deformed 1–3 cm pods and stringers in peat lie at the lower till contact. The other pale tephra is Baker OP.','Glacial shear, peat entrainment and folding preclude transfer from lens width to original local fallout.')]:
        add(rid,name,(lat,lon),OSB,'baker2012.pdf; baker-usgs2020.pdf',where,'Original Table 2 group coordinates to 0.1 arc-minute.',
            'Center lake cores.' if rid.endswith('POC') else 'Moraine, peat and terrestrial pits.',
            'Author-assigned Mazama layer O. Easton OBM has microprobe confirmation.','Event range reported as 7.48–7.46 ka. At Pocket Lake a macrofossil below at 1.06 m is 8.54–8.38 ka.',
            note,kind,x,qual,**(dict(observed_min_cm=1,observed_max_cm=3,quantity_kind='deformed_lenses',deposit_class='secondary_or_mixed_sedimentary_context',fallout_model_status='hold_reworked_or_ductile_strain',interval_meaning='Observed pods, not an original bed range.') if kind is None else {}))


# Marker centers and printed geographic ticks on the 1088 x 1450 Figure 2 reproduction.
# Format: (x1, x2, longitude1, longitude2, y1, y2, latitude1, latitude2).
PANELS = {
    'middle':(334,485,-121-46/60,-121-44/60,1099,1283,44+9/60,44+7/60),
    'south':(671,820,-121-46/60,-121-44/60,775,994,44+7/60,44+5/60),
    'broken':(639,859,-121-42/60,-121-40/60,1069,1209,44+6/60,44+5/60)}
MARKERS = {5:('middle',395,1124),15:('middle',448,1101),16:('middle',340,1203),32:('middle',420,1120),
           18:('south',765,801),20:('south',834,944),24:('south',832,838),27:('south',847,883),41:('south',880,931),43:('broken',692,1116)}


def marcott_position(n):
    panel,x,y=MARKERS[n]
    x1,x2,west,east,y1,y2,north,south=PANELS[panel]
    return (round(north+(south-north)*(y-y1)/(y2-y1),5),round(west+(east-west)*(x-x1)/(x2-x1),5))


def marcott_sections(add):
    for n,name,where,kind,x,low,high,note in [
        (5,'North Fork Whychus Creek, upstream cutbank','Marcott 2005 Fig. 41 and Marcott et al. 2009 Figs. 2–3, p. 185','lower_limit',4,4,9,'Two separately labeled beds (samples 02-MS-081504 and 03-MS-081504) are separated by sediment. The lower coarse band spans about 14 pixels at 360 cm per 763 pixels; allowing two pixels per edge gives an outward 4–9 cm reading. Use the lower bed alone under the in-place interpretation. The upper band is roughly 3–8 cm; an additional deep tephra is unassigned.'),
        (15,'North Fork Whychus Creek, downstream cutbank','2009 Fig. 3 and p. 185; 2005 Fig. 23 for the basal contact','lower_limit',50,50,66,'Ash above till and below sand. At 1450 × 1088 px, Fig. 3 shows the ash from y≈368 to 412 and 375 cm across y352–627. Approximately 60 cm, enclosed conservatively by 50–66 cm allowing for the drawn contacts.'),
        (41,'Soil pit downslope of Broken Top','Marcott 2005 Fig. 48, sample 65-BT-092304; 2009 Figs. 2–3','lower_limit',4,4,8,'Angular coarse sand-sized Mazama between modern soil and a buried silty soil. At 148 dpi Fig. 48 y≈247–263, at 90 cm per 230 pixels. Allowing two pixels per edge gives an outward 4–8 cm reading. Dispersed underlying ash is excluded.'),
        (43,'Broken Top northern-flank meadow core','2009 Fig. 2, Fig. 3 and p. 185; same-site 2004 auger and 2005 core','measurement',3,None,None,'A 3 cm primary glass-matched bed at 330 cm in former lake mud in the 4 m core. This is the meadow core, distinct from Camp Lake.'),
        (20,'Soil pit near Green Lake','2005 Fig. 44, sample 29-SS-090704',None,None,None,None,'Yellow tephra in rooted modern soil is possible reworked Mazama. Undefined source fraction, transport and soil truncation.'),
        (32,'Middle Sister auger pit','2005 Fig. 47, samples 55–58-MS-091704',None,None,None,None,'Reworked yellow ash in sand and trace grains in a lower interval. The top was not recovered. Original local thickness and retained fraction are unresolved.'),
        (18,'Pit downslope of Carver Lake','2005 Fig. 43',None,None,None,None,'Pumice mixed with sand and gravel; no confirmed Mazama horizon. Hole collapse at the water table.'),
        (24,'Pit below Prouty Glacier','2005 Fig. 45',None,None,None,None,'Bed descriptions and soil cobbles supply no identified Mazama unit.'),
        (27,'Saddle north of Green Lake','2005 Fig. 46',None,None,None,None,'Upper lapilli probably Rock Mesa or Devils Hill. Yellow vesicular grains in a lower gravel lack a diagnostic comparison.'),
        (16,'Auger beside Camp Lake and companion lake cores','2005 Fig. 42 and 2009 Figs. 2–3',None,None,None,None,'The site and companion lake record describe younger local rhyolites and Collier Cone. Young detrital dates or reworking compromise the auger chronology. No valid Mazama absence threshold.')]:
        panel,px,py=MARKERS[n]
        identifier='Field correlation by distinctive coarse yellow ash; chemistry is local to Site 43.' if kind else 'Attribution or stratigraphic preservation unresolved.'
        if n==43: identifier='WSU glass analysis, similarity coefficient 99%, and regional stratigraphy.'
        quality=f'2009 Figure 2 {panel} panel, marker pixel ({px}, {py}) on 1088 × 1450 plate. Approximate location. Original map ticks and registered close-ups are retained.'
        details=dict(observed_min_cm=low,observed_max_cm=high,interval_meaning='Outward graphical reading of the single preserved bed. Principal one-sided interpretation assumes an intact primary component.') if low else {}
        if not kind: details.update(deposit_class='secondary_or_mixed_sedimentary_context' if n in (20,32) else 'precursor_comparison',fallout_model_status='hold_reworked_or_unidentified',quantity_kind='reworked_or_unidentified_archive')
        add(f'MAR2009-S{n:02}',name+', Oregon',marcott_position(n),MARC,'marcott2005.pdf; marcott2009.pdf',where,quality,
            'Glacial terrain of the Three Sisters and Broken Top, Oregon.',identifier,'Basal charcoal in the Site 43 core is 8025–8175 cal BP. Other sites are dated by stratigraphic correlation.',
            note,kind,x,(CORE if n==43 else MIN) if kind else note,**details)


def idaho_sections(add):
    shap='Shapley et al. 2023, Journal of Quaternary Science, doi:10.1002/jqs.3554.'
    src='https://doi.org/10.1002/jqs.3554; csd-cores.csv'
    coord='University of Minnesota CSD Facility archive, GPS from the FISH-2007 drilling campaign; individual holes identified by Hole_ID.'
    origin='Facies T consists of 2–8 cm graded vitric ash beds. The authors distinguish airborne ash from remobilized sheets, event strata and repeated thin silt.'
    for rid,name,xy,extra in [
        ('SHP2023-RDF1A','Redfish Lake, RDF07-1A',(44.11202,-114.93026),'The primary horizon overlies Event A. The younger Event B is separately recognizable.'),
        ('SHP2023-RDF4A','Redfish Lake, RDF07-4A',(44.13096,-114.92577),'Microprobe-confirmed Redfish Mazama; the discrete horizon is used in the RDF-4A chronology. XRF also records prolonged remobilization above it.'),
        ('SHP2023-LRFL','Little Redfish Lake, LRFL08-02',(44.161,-114.909),'The authors use both the Mazama and older Glacier Peak horizons in the site age model. Three discrete gray tephras occur; the thin disturbance cap has a separate origin.')]:
        basis=coord if 'LRFL' not in rid else 'Lake-center representative in federal source table (2010 Federal Register 75/200, Idaho waters, 44.161, −114.909); the LRFL08-02 core coordinates are unresolved.'
        add(rid,name+', Idaho',xy,shap,src,'Tephra facies definition, Redfish and Little Redfish results, Figs. 2–4 and geochronology.',basis,'Proglacial lake sediment core.',
            origin+' '+extra,'Imported 7627 ± 150 cal BP. Event relationships independently constrain stratigraphic ordering.',
            'Group lithofacies range 2–8 cm. Use the largest observed facies thickness as a site ceiling; the publication gives no text thickness specific to this bed. '+extra,'upper_limit',8,CAP+' Correct inclusion of this bed in the published facies envelope is required.',
            observed_min_cm=2,observed_max_cm=8,interval_meaning='Author-reported facies-class envelope across beds, not a site measurement. Alternative original minimum requires absence of net thickening.')
    for rid,name,xy,note in [
        ('SHP2023-RDF2A','Redfish Lake, RDF07-2A',(44.10717,-114.93467),'The younger Event B contains an inclined coherent slab incorporating Mazama above sediments dated 6460 cal BP. Event A is below older deposits. An independent in-place observed thickness at this hole has not been separated from the displaced unit.'),
        ('SHP2023-RDF3A','Redfish Lake, RDF07-3A',(44.12295,-114.92855),'This core lacks the two primary Holocene tephra units found elsewhere. The event-bed unconformity truncates the Holocene section down to older glaciolacustrine sediment, so it supplies no absence limit.')]:
        add(rid,name+', Idaho',xy,shap,src,'Figs. 2–4, stratigraphy and fault interpretation.',coord,'Lake core through disturbance deposits.',note,'Contradictory 6460 versus 7627 cal BP discussed in the source.' if '2A' in rid else 'Age horizon missing at the event-bed unconformity.',note,deposit_class='secondary_or_mixed_sedimentary_context',quantity_kind='displaced_or_unreached',fallout_model_status='hold_reworked_or_missing_strata')
    add('B2020-045','Pettit Lake',(43.97954,-114.88334),'Shapley and Finney 2015, Journal of Paleolimnology, doi:10.1007/s10933-015-9826-6. '+shap,
        'shapley2015.html; https://doi.org/10.1002/jqs.3554; csd-cores.csv','2015 original author mass estimate; 2023 Figs. 2, 5, 8–9 and lithofacies definition; Buckland S1 row 57.',
        coord+' PTT07-1A represents the composite; PTT07-1C is at 43.97943, −114.88378, PTT07-2A at 43.97969, −114.87751. Individual holes do not supply three independent identical beds.',
        'Pettit Lake basin, microlaminated lake deposits.','Facies D airborne tephra, Mazama correlated by chemistry and stratigraphy. The succeeding <1 mm silt laminae derive from remobilization.',
        '7.63 ka by event correlation.','The compiler quotes a 6 cm bed. The original authors report approximately 50 kg/m² of fall and later a 2–8 cm tephra facies range. The thickness and mass imply approximately 833 kg/m³ if they correspond. Admit 6 cm as a conditional ceiling on original local thickness at the composite.',
        'upper_limit',6,CAP+' The precise 6 cm remains a compiled site reading. The independent article confirms the airfall facies and approximate mass. The mass alone requires an original bulk density for a thickness conversion.',observed_point_cm=6,verification='primary_facies_and_mass_compilation_thickness')
    for n,name,xy,depth,previous,caveat in [
        (175,'Cooley Lake',(49.491054,-117.645219),377,252,'Core fill, incomplete penetration and later redeposited beds prevent transferring the compiler’s 252 cm to a local minimum or maximum.'),
        (176,'Rockslide Lake',(49.550807,-117.521041),221,30,'The 30 cm secondary package and later millimetre laminae have ages inconsistent with primary deposition. No local bound survives that chronology and the open basal contact.')]:
        add(f'B2020-{n:03}',name,xy,'Foit, Gavin and Hu 2004, Can. J. Earth Sci. 41, 1401–1410; Gavin et al. 2006, Ecology 87, Appendix A.',
            'gavin2006-a1.html; buckland2020-esm1.xlsx','Original coring team’s Table A1 and 2004 identification; S1 and Buckland thesis, Chapter 2.',
            'Buckland S1; final precision and exact hole separation remain unresolved.','Composite lake core.',
            'Mazama O in original chronological table. Younger Glacier Peak and St Helens units are separate.',
            'Event assignment 7627 cal BP by correlation.',f'The original team gives >{depth} cm core DEPTH to Mazama, without a bed base. The compiler records {previous} cm as a deposit quantity. '+caveat,
            verification='original_core_chronology_compilation_thickness',quantity_kind='mixed_or_incomplete_core',deposit_class='secondary_or_mixed_sedimentary_context',
            fallout_model_status='hold_reworked_or_incomplete_core',observed_point_cm=previous,core_depth_or_sample_interval=f'>{depth} cm (depth, not ash thickness).')
