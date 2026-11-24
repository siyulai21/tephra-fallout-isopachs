"""Source-level follow-up to the Young distal citations and the regional lake and pumice search."""
import json

from pyproj import Transformer

from benchmarks.mazama import AUDIT

WEATHER = 'Weatherford 2015, Hydrogeologic investigation of a pumice aquifer, Appendix E. Weatherford and Cummings 2016, Round Meadow.'
CUMMINGS = 'Cummings et al. 2014, Hydrogeology of pumice-hosted fens, pp. 1764–1775.'
WET = 'The logged grains must belong to the in-place climactic component. Local grading, lithic/crystal content and stratigraphic contacts distinguish the residual bed from winnowed sand, float pumice and silt. Compaction and truncation can reduce recovered thickness. No uniform decompaction factor is applied.'
LAKE = 'Assumes that the recovered primary component was deposited locally without net lateral focusing. Coring loss and compaction can reduce its preserved thickness. The upper part and underlying older tephras are assessed separately.'
POINT = 'The principal fit treats the reported discrete horizon as local fallout. Reworking, catchment input, coring distortion and a mistaken stratigraphic attribution can alter that interpretation.'
UTM = Transformer.from_crs(32610,4326,always_xy=True)

def field_records(selected, points):
    from benchmarks.mazama_young import observation, choose, archive
    rows = []
    archives = []
    def add(rid,name,lat,lon,ref,source,where,setting,identity,coord,note,kind=None,x=None,assumption=None,**details):
        r=observation(rid,name,lat,lon,primary_reference=ref,evidence_source=source,
            evidence_locator=where,setting=setting,identity=identity,coordinate_basis=coord,
            coordinate_quality=coord,quantity_kind='unquantified',fallout_model_status='occurrence_or_context',
            source_thickness_notes=note,deposit_class='reported_airborne_ash_or_pumice',
            caveats=assumption or note)
        if kind:
            r.update(quantity_kind={'measurement':'reported_point','lower_limit':'conditional_primary_component','upper_limit':'interpreted_original_upper'}[kind],
                     fallout_model_status='conditional_principal_'+kind)
            if kind=='measurement':
                r['observed_point_cm']=x
            else:
                r.update(bound_relation='>=' if kind=='lower_limit' else '<=',bound_cm=x,
                    bound_target='Original combined local climactic airborne deposit under the stated transfer.')
            selected[rid]=choose(r,kind,x,note,assumption)
        r.update(details)
        if kind=='upper_limit':
            a=archive(r,x,assumption)
            a.update(reading='occurrence_only',url='https://pubs.usgs.gov/pp/pp1661f/')
            archives.append(a)
        rows.append(r)
        return r

    logs=[
        ('V1','vibracore 1',611803,4798804,118,None,'The log identifies original pumice below sand near 90–100 cm on the photo tape. It also describes a continuous 30–40 cm oxidized basal part of the recovered section.'),
        ('V2','vibracore 2',611706,4798785,120,80,'Paleosol at 290 cm and primary-pumice top at 195–210 cm. Use 290 − 210 = 80 cm under the shared recovered-depth interpretation. The alternate top gives 95 cm.'),
        ('V3','vibracore 3',611832,4798527,118,200,'330 − 130 = 200 cm on the recovered-core tape, including 8 cm of basal crystal/lithic sand. The overlying 10 cm lag and 20 cm floated pumice are excluded.'),
        ('SOUTH','south auger',611656,4798234,120,73,'The bit entered primary graded pumice at 94 cm and remained in the lower pumice at 167 cm. Observed section 167 − 94 = 73 cm.'),
        ('PAD','northwest fen pad',611237,4798935,121,16,'Yellow pumice below lag sand occupies 116–132 cm, giving 16 cm conditional on the basal facies being preserved primary pumice.'),
        ('30M','fen transect 30 m',611080,4798907,121,53,'A 58–111 cm lithic-bearing, moderately sorted basal pumice interval gives 53 cm if it is the in-place pumice underlying the silt. The base was not reached.'),
        ('60M','fen transect 60 m',611091,4798872,122,168,'Primary pumice starts at 70 cm. The basal sandy/pumice facies touches paleosol at 238 cm, giving 168 cm including the graded basal material.'),
        ('90M','fen transect 90 m',611102,4798853,122,73,'120–193 cm spans identified lower pumice through its first cemented pumice. Withhold the deeper material to refusal at 240 cm.'),
        ('CHEET','terrace southwest of outlet',611978,4799122,123,135,'Two accounts of one hole identify 38–173 cm as upper, middle and lower pumice. Preserve 135 cm. The 173–258 cm sandy material over bedrock is excluded.'),
        ('CHEET2','second description of the terrace hole',611977,4799126,123,None,'Author explicitly identifies this entry as the same hole. The two coordinates differ by about 4 m.'),
        ('CONFL','wet wedge at channel confluence',611780,4799037,124,None,'At 120–171 cm the hole contains moderately well-sorted pumice, beneath lithic sand with rounded pumice. The 51 cm is a facies interval. A primary fraction cannot be separated.'),
        ('OUT','outlet auger',612016,4799500,124,None,'31–120 cm includes sorted pumice. The log explicitly questions resedimentation. Rounded grains, dune sand and slough complicate recovery below 120 cm.')
    ]
    for tag,name,e,n,page,x,note in logs:
        lon,lat=UTM.transform(e,n)
        if tag=='V1':
            x=30
            note+=' Use only 30 cm if the continuous basal red zone belongs to the same primary bed. The reported 28% whole-core shortening is not a local correction.'
        rec=add('W2015-RM-'+tag,'Round Meadow, '+name,round(lat,7),round(lon,7),WEATHER,
            'weatherford2015.pdf; cummings2016.pdf; cummings2014.pdf',f'Weatherford Appendix E p. {page}, field log at {e} E, {n} N. Cummings et al. 2014 p. 1764 correlates both local internal divisions with Young upper pumice.',
            'Wetland, fen or terrace core/auger, Walker Rim, Oregon.',
            'Climactic single-vent Plinian fall. The local lower division in these field logs is the base of the climactic pumice.',
            f'Field easting {e} m, northing {n} m. UTM zone 10N converted as WGS84. Original log does not specify datum.',
            note,'lower_limit' if x else None,x,WET if x else note)
        if tag=='CHEET2':rec.update(parent_record_id='W2015-RM-CHEET',fallout_model_status='duplicate_field_log')
        if tag in {'CONFL','OUT'}:rec['deposit_class']='mixed_or_redistributed_wetland_pumice'

    # Fig. 2 printed p. 1764, 1199×1628 pixel rendering. Four graticule ticks per axis.
    for tag,name,xpx,ypx,x,note in [
        ('WIL','Wilshire fen, lower reach',590,861,87,'The ephemeral valley west of Wilshire fen retains 87 cm of lower pumice in its lower reach.'),
        ('DRY','Dry Meadow fen',745,815,None,'Piezometers intersect the pumice aquifer. Neither the isolated primary thickness nor the complete original surface is specified.'),
        ('CROOK','Crooked Meadow fen',653,799,None,'Pumice aquifer beneath the fen. Sheet discharge and peat thickness quantify different properties.'),
        ('460','FSLR 460 fen',485,835,None,'Mapped shallow piezometers are completed in the pumice. No complete local residual section is reported.'),
        ('JOHN','Johnson Meadow fen',614,1085,None,'Auger penetrated iron-cemented pumice beneath alluvium. The thickness of the peat, well screen and depth into underlying basalt cannot serve as local ash thickness.'),
        ('JACK','Jack Creek at Skookum Butte fen',553,1194,None,'Mapped piezometer in Jack Creek area. Fen-to-well linkage approximate. No numeric local ash thickness.')
    ]:
        lon=-(121+38/60)+(xpx-435)/144*(2/60)
        lat=43+20/60-(ypx-663)/197.33*(2/60)
        add('C2014-'+tag,name,round(lat,5),round(lon,5),CUMMINGS,'cummings2014.pdf',
            'Fig. 2 p. 1764 and site descriptions pp. 1765–1769. Fig. 4 and p. 1766 give the Wilshire lower-reach reading.',
            'Pumice aquifer and eroded valley or fen, Walker Rim.',
            'Locally preserved divisions of the single-vent climactic Plinian pumice.',
            f'Figure 2 graticule at PDF page width 1199 px. Feature ({xpx},{ypx}). Top ticks x=435,579,723,867 at 121°38′–32′W. Left ticks y=663,860,1057,1255 at 43°20′–14′N. Fen-scale proxy; exact point along the reach is unresolved.',
            note,'lower_limit' if x else None,x,WET if x else note)
    # The section geometry is supplied with the audit so no online service is needed to rebuild.
    geo=json.loads((AUDIT/'walker-rim-section5.json').read_text())
    lon,lat=geo['centroid_lon_lat']
    add('C2014-SEC5','Section 5 groundwater system, Walker Rim',lat,lon,CUMMINGS,'cummings2014.pdf; plss-walker5.json',
        'Fig. 9, p. 1770, non-fen piezometer. BLM PLSS Willamette Meridian, T27S R10E, section 5.',
        'Eroded pumice below seasonally flooded lodgepole forest.',
        'Preserved Mazama Plinian pumice with most of the local upper division eroded.',
        geo['coordinate_basis'],'Approximately 150 cm survives at the piezometer. Use the reported approximate residual as a conditional minimum.','lower_limit',150,WET)
    add('BC2010-8','Stillwater Point Reservoir diversion canal, POI 8',39.473802,-118.593227,
        'Bell, Caskey and House 2010, Nevada Bureau of Mines and Geology Map 168, second edition. Morrison 1964 USGS PP 401, section 22a.',
        'nevada168.pdf; bell2010-map.pdf; morrison1964.pdf','Text pp. 12–13, Fig. 12B and map point of interest 8. Morrison pp. 128–129.',
        'Auger in carbonaceous marsh or fluvial deltaic deposits over sorted sand. Older authors proposed a lake.',
        'Mapped and reoccupied Mazama marker, 6.85 ka conventional / 7.6 cal ka in NBMG report.',
        'Digitized square POI 8 from georeferenced NBMG plate, native NAD83 UTM11. Nearby Tmz triangle is about 57 m away. See stillwater-georeferencing.json.',
        'The auger intercepted a distinct 5 cm horizon at 70 cm below surface. Treat as conditional direct measurement. The adjacent older exposure is part of the same locality group.',
        'measurement',5,POINT,core_depth_or_sample_interval='70 cm in the auger. Morrison section 22a is a separate exposure within the group.',
        local_age='Mazama event correlation. 6.7 ka on organic sediment in the Macari Road auger is a separate sample, not the age of the tephra bed.')
    add('R1967-BB326-36','Willapa Canyon terrace, Brown Bear BB326-36-PC',46.48333,-124.70667,
        'Royse 1967, Northwest Science 41, 103–109. Williams and Goles 1968 pp. 37–38 and map. Goldfinger et al. 2012, USGS PP 1661-F.',
        'williams-goles1968.pdf; goldfinger2012.pdf; pac_ext.txt; royse1964-metadata.html',
        'Williams and Goles p. 37 give the 4 cm bed. Goldfinger et al. section on storm triggering identifies core and 240 cm depth. USGS usSEABED Pacific extracted table station BB326-36-PC.',
        'Marine sediment on a canyon-head terrace, offshore Washington.',
        'Mostly glass layer historically correlated to Mazama. A primary fall origin remains possible. The terrain and overlying section differ from nearby ash-bearing turbidites.',
        'USGS Pacific extracted sample table 16803 piston core, original cruise navigation. Source metadata lists nominal accuracy 0.5 nautical mile.',
        'Four cm of glass-rich sediment is a conditional ceiling if this horizon captured the complete local signal and redeposition or background sediment added thickness.',
        'upper_limit',4,'Possible canyon transport, compaction, winnowing or incomplete retention could invalidate transfer to original airborne bulk. Compare omission. The mixed marine blanket has no positive numerical primary minimum.',
        observed_point_cm=4,core_depth_or_sample_interval='240 cm core depth; published accounts give different water depths at the station.',
        verification='observer_report_via_two_published_accounts_and_original_station_archive')
    add('SW1995-IP','Indian Prairie fen, 1992 composite cores',44+38/60,-(122+34/60+30/3600),
        'Sea and Whitlock 1995, Quaternary Research 43, 370–381.','seawhitlock1995.pdf',
        'pp. 370, 372–373 and Table 1. The principal 92A and overlapping 92B holes lay within 2 m.',
        'Peat and lacustrine sediment, central Oregon.', 'Authors assign the discrete ash to Mazama layer O on stratigraphic grounds.',
        'Wetland coordinates printed p. 370, 44°38′N, 122°34′30″W.',
        'Reported discrete 2 cm layer near 3.35 m. Table 1 gives 3.32–3.34 m. A single composite-site measurement represents the two holes.',
        'measurement',2,POINT,core_depth_or_sample_interval='3.35 m in text and 3.32–3.34 m in Table 1.')
    add('SW1995-GLB','Gold Lake Bog, 1992 core 92B and overlaps',43+39/60,-(122+2/60+30/3600),
        'Sea and Whitlock 1995, Quaternary Research 43, 370–381.','seawhitlock1995.pdf',
        'pp. 370, 372–373 and Table 1. Separate fen north of the 2018 Gold Lake cored lake.',
        'Former lake to bog, central Oregon.', 'Investigators assume the thick pumiceous bed is Mazama layer O. No glass fingerprint or phase subdivision supplied.',
        'Bog coordinates printed p. 370, 43°39′N, 122°02′30″W.',
        '2.63–3.25 m of thick pumiceous tephra gives 62 cm of recovered bed. Conditional primary-component minimum; the adjacent lake contains a distinct Llao Rock layer beneath Mazama.',
        'lower_limit',62,LAKE+' If the nearby precursor is amalgamated here, the target minimum fails. Omit the bog in the source sensitivity.',
        observed_point_cm=62,core_depth_or_sample_interval='263–325 cm in composite core, primarily 92B.')
    add('J2025-NTL','North Twin Lake, Oregon',43.7273,-121.7658,
        'Johnson, G.M. 2025. Scale, Representation and a Utility Paradox for Applied Paleoecology, University of Oregon dissertation, chapter 4.',
        'northstudy.pdf','pp. 111–116 and 125–128, Fig. 19, Table 5 and Fig. 33.',
        'Maar-lake sediment, central Oregon.', 'Sharp-based coarse primary Mazama bed fines up into pumiceous sand and ash. Younger presumed Rock Mesa is a distinct horizon.',
        'Author gives lake coordinates 43.7273°N, 121.7658°W. Core location within the lake shown in Fig. 16.',
        'Logged 241–143 cm (98 cm) with an additional or terminal 5 cm finer interval. Optical description elsewhere uses 224 cm. Taking the shallower 224 cm base, 143 cm top and withholding all 5 cm yields 76 cm as a conservative primary minimum; Fig. 19 independently shows a thick masked interval.',
        'lower_limit',76,LAKE,core_depth_or_sample_interval='241–143 cm in lithology, 224 cm in color-profile commentary; 143 cm of overlying sapropel.',
        local_age='Bulk sediment ages predict 8184–8022 cal BP before the tephra age tie. Possible reservoir offset. The event age was incorporated separately.',
        observed_min_cm=81,observed_max_cm=98,interval_meaning='Sensitivity to the conflicting basal depth against the published 143 cm top. The extra 5 cm fine tail may be additional or included.')
    # No separate likelihood for these spatially ambiguous section components.
    for tag,name,lat,lon,note,x,detail in [
        ('NEW','Newberry caldera',43.72,-121.23,'7 cm lower dark sandy ash plus 3 cm upper vitric ash. Upper material with 33% mafic scoria interpreted as co-ignimbrite.',3,'Caldera-scale coordinates from Oregon State University Volcano World, https://volcano.oregonstate.edu/newberry. USGS describes a 4 by 5 mile caldera. Exact section unresolved.'),
        ('FR','Fort Rock Basin','','','The 4 cm ash above UPU has a 2.5 cm crystal-rich base and 1.5 cm vitric top containing 9% mafic glass.',1.5,'Broad basin and distance only. The sample was communicated by P. Mehringer. No unique core or exposure.'),
        ('MB','Mount Bachelor fine ash',points[103]['latitude'],points[103]['longitude'],'Approximately half as thick as the 10 cm Newberry ash. Its upper unit contains 11% mafic glass. The thickness of the identified upper portion was not measured.',None,'Mount Bachelor site group proxy; sampled ash cannot be tied uniquely to section 103, 128 or B2020-046.'),
        ('ELK','Elkhorn Mountains bog section','','','Young pp. 131, 214 describes 20–22 cm of homogeneous basal Plinian ash and about 10 cm of overlying banded or contaminated material. Candidate Anthony Lake area.',20,'Elkhorn Mountains bog, northeastern Oregon. No unambiguous independent section coordinate.'),
        ('PINK','Pink ash over proximal ignimbrite','','','The source describes locally preserved ash above flow deposits. Identified primary and reworked fractions vary across exposures; the flow thickness cannot be added to airborne bulk.',None,'Regional sections over ignimbrite, no unique source measurement position for this generalization.')]:
        from benchmarks.mazama_young import THESIS
        note2='The preserved identified airborne component supplies a minimum conditional on in-place deposition. Other components and locally reworked ash are excluded.'
        r=add('Y1990-C-'+tag,name,lat,lon,THESIS,'young1990.pdf','Thesis pp. 93–95, 130–132 and 214. Compare sample-specific glass compositions.',
            'Bog or terrestrial ash above climactic deposits.', 'Climactic Plinian/co-ignimbrite component as described in the section.',detail,note,
            'lower_limit' if tag=='NEW' else None,x,note2)
        if x and tag!='NEW':r.update(quantity_kind='component_lower_bound_unlocated',bound_relation='>=',bound_cm=x,bound_target='Preserved primary component in reported section, conditional on attribution; original position unresolved.')
        if tag=='MB':r.update(parent_record_id='B2020-046',observed_point_cm=5,quantity_kind='approximate_mixed_package')
    add('L1985-VIR','Arroyo near Last Supper Cave, Virgin Creek group','','','Layton 1985, Invaders from the South?, p. 191. Davis 1978, Virgin Creek.',
        'https://escholarship.org/uc/item/4gj347dt','Layton p. 191, Tx-2597, 1.5 km from cave.',
        'Arroyo wall, northwest Nevada.','Mazama correlation in dated sediment section.','Relative location only. The named compilation site B2020-013 and Young Table 1.1 site 23 may refer to this drainage.',
        'A 30 cm bed was reported. The creek setting, unresolved relationship to the 10 cm equivalent and other arroyo exposures, and lack of a measured in-place basal fraction prevent a second original-fallout observation.',
        parent_record_id='B2020-013',observed_point_cm=30,quantity_kind='reported_bed_uncertain_transport',local_age='Tx-2597 6710 ± 110 radiocarbon years BP as reported by Layton.')
    add('LSC1978','Last Supper Cave, interior, northwestern Nevada','','','Layton and Davis 1978 field excavation, summarized by Layton 1985 and Smith et al. 2015.',
        'https://escholarship.org/uc/item/6cr1409z','Excavation stratigraphy, upper shell horizon; compare adjacent arroyo separately.',
        'Sheltered archaeological deposit, Nevada.','Attributed Mazama tephra, rear and center of cave.','Cave-area descriptions only.',
        'Ash enters a sheltered, locally disturbed environment. Without a capture fraction or independently measured external bed this preserves occurrence.')
    shared='Local ash seam recorded without a direct bed thickness. The archive supplies chronological or stratigraphic control. A sample age or burial depth does not measure the original ash bed.'
    revisions={}
    for n,name,lat,lon,source,where,ident,age in [
        (115,'Arrow Lake',47+25/60,-122-21/60,'Rubin and Alexander 1960, p. 164. Powers and Wilcox 1964 is the later attribution.','Arrow Lake series W-776, W-777, NE¼ section 6 T22N R4E.',
         'The original collectors assigned this ash to Glacier Peak. The catalogue cites the later Mazama reinterpretation. The two radiocarbon estimates overlap the Mazama age; they cannot independently discriminate source.', 'Peat immediately above 6630 ± 400 and below 6600 ± 400 14C yr BP.'),
        (116,'Bow Lake',47+26/60,-122-18/60,'Rubin and Alexander 1960, p. 164.','W-779, east side of section 33 T23N R4E, Des Moines quadrangle.',
         'The original date list labels the ash as Glacier Peak. Later compilations index this horizon as Mazama. Direct glass data matching this exact sample are unresolved.', 'Overlying peat 5950 ± 400 14C yr BP.'),
        (179,'Milliken',49+33/60,-121-24/60,'McCallum and Dyck 1960 p. 78 and Lowdon et al. 1969 p. 30.','DjRi-3, Zone D; GSC-459 (NMC-69), S-47 (UBC-7).',
         'Lowdon et al. report a discontinuous ash seam assigned to Mazama in part of the gravel zone. The archaeological label Mazama phase is independent of a volcanic glass fingerprint.', 'GSC-459 charcoal in Zone D 7190 ± 150 14C yr BP. S-47 culturally associated charcoal 8150 ± 300 14C yr BP.'),
        (208,'Drynoch',50+20/60,-121-24/60,'Lowdon et al. 1969 pp. 31–32.','EcRi:1; sample GSC-530 (NMC-84).',
         'The investigators explicitly treat the ash above the midden as possible Mazama. No bed thickness or physical preservation constraint is supplied.', 'Composite charcoal below ash 7530 ± 270 14C yr BP. A younger slide overlies the ash, dated 3175 ± 150 14C yr BP (I-462).')
    ]:
        revisions[str(n)]=dict(verification='original_date_list_and_site_description_checked',primary_reference=source,evidence_locator=where,identity=ident,local_age=age,age_basis='Primary conventional radiocarbon sample context. Date on organic material, not glass.',
            evidence_source='rubin1960.pdf' if n in (115,116) else ('lowdon1969.pdf; mccallum1960.pdf' if n==179 else 'lowdon1969.pdf'),
            latitude=lat,longitude=lon,coordinate_basis='Primary field location published to one arc-minute. Original compilation coordinates remain in the source table.',
            coordinate_quality='Original date-list location; round-number coordinates are not hole-scale survey control.',
            quantity_kind='unquantified',fallout_model_status='occurrence_only_no_measured_or_geometric_bed_limit',
            caveats=shared,thickness_quality='No ash thickness or calibrated detection threshold was reported.')
    return rows,revisions,archives

