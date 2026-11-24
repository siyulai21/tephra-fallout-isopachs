"""Register the published Young (1990) site symbols to WGS84 and retain alternate positions."""
import csv
import json
from collections import defaultdict

import numpy as np
from pyproj import Geod, Proj
from scipy.optimize import minimize_scalar

from benchmarks.mazama import AUDIT, write_csv

GEOD = Geod(ellps="WGS84")
# Page numbers count from the first PDF page. Values are x,y from its top left.
MAPS = {
    30: dict(dpi=150, scale=[[239, 1269], [472, 1267]], km=5, north=[[258,758],[257,666]], origin=[514.5,1021.5]),
    321: dict(dpi=128, scale=[[286,1007],[436,1005]], km=5, north=[[800,521],[798,438]], origin=[585,699]),
    323: dict(dpi=128, scale=[[578,567.2],[708,568.3]], km=5, north=[[818.1,409],[819.4,326]], origin=[490.17,864.6]),
    324: dict(dpi=128, scale=[[258.5,395.4],[379,394.8]], km=20, north=[[776.8,398.8],[776.6,317]], origin=[285.9,997.9]),
}
SHORE = [-122.16466983577116,42.904194137012205,-122.05011517700126,42.97875910240582]
FIG_V = [609,968]
CONTROL_VENTS = {
    "north_rim_west": ([416,860],[-122.12342252416533,42.978753643323365]),
    "north_rim_east": ([686,805],[-122.06825786793826,42.982804003685835]),
}
DISTAL_POINTS = [(263,820),(266,796),(475,758),(525,782),(477,806),(472,968),(302,692),(295,655),
    (422,557),(453,632),(469,681),(324,413),(521,361),(645,254),(651,181),(802,424),(773,555),
    (658,752),(797,636),(882,838),(746,1057),(568,1236),(422,1021),(485,1123),(402,1217)]
# Border intersections read from Figure 1.14 at 150 dpi.
BORDERS = [(396,1000,-120,42),(535,999,-117.02777,42),(678,989,-114.04172,42),(526,740,-117.03994,46),
    (515,551,-117.03,49),(557,551,-116.049,49),(808,531,-110,49),(402,1194,-120,39),(193,1002,-124.21,42)]
TABLE = ['T','T','35','40','30','4','5–10','2.5','5','<50','2','≤1','<25','<5','≤0.2','<5','T','<10','T','T','T','T','10','<5','T']
CITATIONS = ['Powers and Wilcox 1964']*2 + ['Harward and Youngberg 1969, Borchardt et al. 1973 and D.R. Mullineaux written communication 1989']*3 + ['Blinman et al. 1979','Mullineaux 1974','D.R. Mullineaux written communication 1989','Crandell and Mullineaux 1967','Czamanske and Porter 1965','Blinman et al. 1979','Williams and Goles 1968','Westgate and Dreimanis 1967','Westgate et al. 1969','Lichti-Federovich 1970','David 1970','Horberg and Robie 1955','Blinman et al. 1979'] + ['Mullineaux 1974']*3 + ['J.O. Davis personal communication 1989','Davis 1978','Davis 1978','Davis 1978 and personal communication 1989']
MATCHES = [60,65,76,68,63,21,104,117,156,120,108,229,264,290,291,232,135,72,112,51,7,2,13,4,3]


def displacement(page, xy, origin=None):
    spec = MAPS[page]
    scale = spec['km'] / np.linalg.norm(np.diff(spec['scale'], axis=0))
    north = np.diff(spec['north'], axis=0)[0]
    north = north / np.linalg.norm(north)
    east = np.array([-north[1], north[0]])
    delta = np.array(xy) - np.array(origin if origin is not None else spec['origin'])
    return np.array([np.dot(delta, east), np.dot(delta, north)]) * scale


def offset(lonlat, east_north):
    east, north = east_north
    lon, lat, _ = GEOD.fwd(*lonlat, np.degrees(np.arctan2(east, north)), 1000*np.hypot(east, north))
    return [lon, lat]


def km(a,b):
    return GEOD.inv(*a,*b)[2] / 1000


def distal_fit(controls):
    control = np.asarray(controls)
    design = np.column_stack([control[:, :2], np.ones(len(control))])
    def solve(lon0):
        projection = Proj(proj='tmerc', lon_0=lon0, ellps='WGS84')
        projected = np.column_stack(projection(control[:,2],control[:,3]))
        matrix = np.linalg.lstsq(design, projected, rcond=None)[0]
        return projection, matrix, (design @ matrix - projected)
    best = minimize_scalar(lambda x: np.sum(solve(x)[2]**2), bounds=(-135., -80.), method='bounded')
    projection, matrix, errors = solve(best.x)
    def position(xy):
        x,y = np.array([*xy,1.]) @ matrix
        return list(projection(x,y,inverse=True))
    return position, float(best.x), matrix, np.linalg.norm(errors,axis=1)/1000


def generate():
    anchor = [(SHORE[0]+SHORE[2])/2,(SHORE[1]+SHORE[3])/2]
    vents = {'shoreline_centre':offset(anchor,displacement(30,FIG_V))}
    for title,(pixel, fixed) in CONTROL_VENTS.items():
        vents[title] = offset(fixed,displacement(30,FIG_V,pixel))
    vents['catalogue_vent'] = [-122.1,42.95]
    # The 2024 authors call 47 an equivalent section. Its actual sampling coordinates
    # give an independent translation check but do not survey the 1990 pit.
    picks = list(csv.DictReader((AUDIT/'young1990-map-picks.csv').open()))
    by_site = defaultdict(dict)
    for p in picks:
        by_site[int(p['site'])][int(p['pdf_page'])] = (float(p['pixel_x']),float(p['pixel_y']))
    shift47=displacement(324,by_site[47][324])
    vents['upper_kirk_equivalent'] = offset([-121.8017461,42.75185019],-shift47)
    points = {}
    def source_position(page, pixel, vent):
        if page==321:
            anchor106=offset(vent,displacement(323,by_site[106][323]))
            return offset(anchor106,displacement(321,pixel))
        return offset(vent,displacement(page,pixel))
    for site, pages in by_site.items():
        page=next(p for p in (323,324,321) if p in pages)
        xy=pages[page]
        positions={k:source_position(page,xy,v) for k,v in vents.items()}
        nominal=positions['shoreline_centre']
        distances=[km(nominal, source_position(p,px,vents['shoreline_centre'])) for p,px in pages.items() if p!=page]
        points[site]=dict(site=site,map_pdf_page=page,longitude=round(nominal[0],6),latitude=round(nominal[1],6),
            pixel_x=xy[0],pixel_y=xy[1],one_pixel_km=round(np.linalg.norm(displacement(page,[xy[0]+1,xy[1]],xy)),4),
            alternate_registration_max_displacement_km=round(max(km(nominal,p) for p in positions.values()),3),
            cross_map_difference_km=round(max(distances),3) if distances else '',
            upper_kirk_anchor_longitude=round(positions['upper_kirk_equivalent'][0],6),
            upper_kirk_anchor_latitude=round(positions['upper_kirk_equivalent'][1],6),
            marker_note=next((p['note'] for p in picks if int(p['site'])==site and p['note']),''))

    lookup={int(r['buckland_id']):r for r in csv.DictReader((AUDIT/'buckland2020-293-localities.csv').open())}
    position, lon0, matrix, err=distal_fit(BORDERS)
    loo=[km(distal_fit(BORDERS[:i]+BORDERS[i+1:])[0](c[:2]),c[2:]) for i,c in enumerate(BORDERS)]
    distal=[]
    for n,(xy,reading,citation,mid) in enumerate(zip(DISTAL_POINTS,TABLE,CITATIONS,MATCHES,strict=True),1):
        lonlat=position(xy)
        row=lookup[mid]
        target=[float(row['longitude_raw']),float(row['latitude_raw'])]
        distal.append(dict(table_site=n,field_checked_by_young=n in {3,4,5},source_reference=citation,upland_equivalent_cm=reading,
            reference_density_kg_m3=700,pixel_x=xy[0],pixel_y=xy[1],map_latitude=round(lonlat[1],5),map_longitude=round(lonlat[0],5),
            candidate_catalogue_id=f'B2020-{mid:03}',candidate_name=row['locality'],candidate_latitude=target[1],candidate_longitude=target[0],
            distance_to_candidate_km=round(km(lonlat,target),2),match_note={
            3:'Tollgate area, Dick Springs and Meadowwood both plausible.',
            7:'Park-wide comparison. Carbon River is a separate named section.',
            8:'Compilation identifies Young site 8. Personal communication, field station unknown.',
            9:'Geographic candidate only. Figure places symbol in southern British Columbia but citation describes Mount Rainier, Washington.',
            10:'Creston Bog cores and Creston lake or marsh records have conflicting thicknesses. Compare IDs 120 and 121.',
            12:'Named bog and original Mathews letter take precedence over small-scale map.',
            15:'Named lake confirmed independently. Use author coordinates.',
            16:'Multiple South Saskatchewan River exposures. Compare compilation IDs 230–232.',
            17:'Historic Galata attribution, large symbol-to-catalogue offset.',
            25:'Davis distinguishes more than one Holocene Mazama-source eruption. Climactic identity of the Osgood candidate is uncertain.'
            }.get(n,'Candidate from locality description, citation or compiler numbering. See discrepancy column.')))
    write_csv(AUDIT/'young1990-distal.csv',distal)
    write_csv(AUDIT/'young1990-section-positions.csv',[points[k] for k in sorted(points)])
    report=dict(source='Young 1990. Separate numbering systems: Fig. 1.14/Table 1.1 and Appendix 1d.',maps=MAPS,
        geologic_map_url='https://pubs.usgs.gov/sim/2832/',shoreline_wgs84_bounds=SHORE,
        shoreline_control='Largest merged Crater Lake water polygon of the eastern and western geologic map coverages. A modern shoreline bounding box centres the schematic caldera rim in Fig. 1.9.',
        fig19_rim_pixel_bounds=[278,814,751,1229],fig19_climactic_vent_pixel=FIG_V,candidate_north_rim_controls=CONTROL_VENTS,
        alternate_origin_lon_lat=vents,appendix321_anchor='Shared source site 106 mapped on pp.321 and 323. The precursor map draws different vents.',
        coordinate_method='Map scale and north orientation, WGS84 forward geodesic from the mapped origin. Digits retain computational reproducibility. Registration and individual symbol/label uncertainty exceed pixel precision.',
        distal=dict(dpi=150,control_pixel_x_y_lon_lat=BORDERS,central_meridian_fitted_deg=lon0,affine_pixel_to_projected_m=matrix.tolist(),
            residual_km=err.tolist(),leave_one_control_out_geodesic_km=loo,
            qualification='Transverse Mercator is printed on the figure. Central meridian and affine page deformation are estimated. Named localities can differ by much more than control residuals.'),
        uncertainty='Alternate origin offsets, cross-map discrepancies and one-pixel resolution are diagnostics. None is a statistical confidence interval or universal limit on location error.')
    (AUDIT/'young1990-georeferencing.json').write_text(json.dumps(report,indent=2)+'\n')
    return points,distal


if __name__ == '__main__':
    generate()

