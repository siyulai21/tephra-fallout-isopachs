"""Cross-check the factual age entries in Egan's Tables 6.1 and 12.2.

Usage: python extract_egan_dates.py PATH_TO_THESIS_PDF. Requires pdfplumber.
The spatial links identify candidate records. They never create model observations.
"""
import argparse
import bisect
import csv
import hashlib
import math
from pathlib import Path
import re

import pdfplumber

HERE = Path(__file__).resolve().parent
DIGEST = '33a66ff8514258a1ef29d5e48c938bc124d0d579c2b92b4cd541c8bfeaa737f0'
# Link by named locality, sample and investigator. Several records share names,
# so retain both possible catalogue IDs when the core cannot be distinguished.
LINKS = [
    '022','114','119','115','116','206','168','164','003','208',
    '266','179','205','139','174','212','212','251','203','072',
    '259','141','148','099','118','027','165','165','088','088',
    '','','','028','028','034','264','028','272','028','013','103',
    '021','148','148','148','030|031','030|031','030|031','030|031',
    '030|031','227','221','276|277','088','088','088','115','206',
    '266','134','272','244','104','072','141','103','099','118',
    '279','165','165','252','252','020','020','020','020','020','088','',
]
RELATIONS = {'Below Mazama':'below', 'Syn-Mazama':'associated', 'Above Mazama':'above'}


def read_table(pdf, pages, boundaries):
    rows, current, relation = [], None, ''
    for page_number in pages:
        words = sorted(pdf.pages[page_number-1].extract_words(), key=lambda w: (w['top'], w['x0']))
        lines = []
        for word in words:
            if word['top'] >= 504:  # Printed page footer
                continue
            if lines and abs(word['top'] - lines[-1][0]['top']) <= 2.5:
                lines[-1].append(word)
            else:
                lines.append([word])
        for line in lines:
            columns = [''] * (len(boundaries) + 1)
            for word in sorted(line, key=lambda w: w['x0']):
                index = bisect.bisect_right(boundaries, word['x0'])
                columns[index] = (columns[index] + ' ' + word['text']).strip()
            description = ' '.join(columns).strip()
            if description.startswith(('Ice-core ages', 'Ice core dates')):
                return rows
            if description in RELATIONS:
                relation = RELATIONS[description]
                continue
            if re.match(r'^\d{4}\s*±', columns[2]):
                current = dict(page=page_number, relation=relation, columns=columns.copy())
                rows.append(current)
            elif current:
                for i, value in enumerate(columns):
                    current['columns'][i] = (current['columns'][i] + ' ' + value).strip()
    return rows


def distance(lat, lon, record):
    y, x = map(float, (record['latitude'], record['longitude']))
    dlat, dlon = math.radians(y-lat), math.radians(x-lon)
    a = math.sin(dlat/2)**2 + math.cos(math.radians(y))*math.cos(math.radians(lat))*math.sin(dlon/2)**2
    return round(12742.0176 * math.asin(min(1, math.sqrt(a))), 2)


def material(text):
    terms = ('pollen concentrate', 'rat dung', 'collagen', 'bone', 'peat', 'gyttja', 'marl',
             'organic muck', 'plant detritus', 'organic matter', 'bulk sediment', 'charcoal',
             'twig', 'wood', 'log', 'branch', 'paleosol', 'organic material', 'lake sediment',
             'total organic carbon', 'reduced carbon', 'fibrous organic', 'plant')
    return ', '.join(term for term in terms if term in text.lower()) or 'see source description'


def build(path):
    assert hashlib.sha256(path.read_bytes()).hexdigest() == DIGEST
    with pdfplumber.open(path) as pdf:
        main = read_table(pdf, range(167,175), [230,300,369,449,569])
        appendix = read_table(pdf, range(411,427), [141,228,307,390,470,546,616,715])
    assert len(main) == len(appendix) == len(LINKS) == 81
    # The Quesnel row begins slightly higher in the locality columns than in
    # the lab-ID column on p.421. Check alignment against the actual page.
    a, b = appendix[52]['columns'], appendix[53]['columns']
    assert a[1] == 'TO-5196' and b[1] == 'Beta 315832' and '-121.11409' in a[5]
    a[5], b[5] = a[5].split()
    a[6], b[6] = a[6].split()
    a[8], b[8] = 'Yes', 'Yes'
    ledger = {r['record_id']:r for r in csv.DictReader((HERE/'revised-locality-ledger.csv').open())}
    rows = []
    for i, (row, associated, ids) in enumerate(zip(main, appendix, LINKS, strict=True), 1):
        m, a = row['columns'], associated['columns']
        normal = lambda text: re.sub(r'[\s-]+', '', text).upper()
        assert normal(m[1]) == normal(a[1]), (i, m[1], a[1])
        assert row['relation'] == associated['relation']
        lon = a[5].replace(' ', '')
        lat = a[6].replace(' ', '')
        lab = m[1].replace('- ', '-')
        link = ['B2020-' + id_ for id_ in ids.split('|') if id_]
        coords = lon not in ('', '-') and lat not in ('', '-')
        separations = [distance(float(lat),float(lon),ledger[id_]) for id_ in link] if coords else []
        relation = row['relation']
        note = ''
        if lab in ('W-776','W-777'):
            note = 'Primary Rubin and Alexander date list takes precedence for the above/below assignments. The two appendix positions conflict.'
        elif lab.upper() in ('TX-2116','TX-2121'):
            note = 'Appendix repeats Deep Creek coordinates for Idaho. Refer to Mack core 1 in Valastro et al. 1980. Tx-2121 there is 6390 ±230 BP.'
        elif lab in ('WSU-1552','WSU-1553'):
            note = 'Use Mehringer et al. 1977, Table 3 and Montana survey section. The appendix places the bog in Idaho at a different position.'
        elif lab.upper() in ('TX-2881','TX-2882','TX-2883','TX-2884'):
            note = 'Valastro et al. 1980 place Tx-2882 above the upper ash, Tx-2881 and 2883 between two units, and Tx-2884 below both.'
        elif lab in ('GaK-1124','Tx-487','C-247','C-247b'):
            note = 'Muir Creek contains pumice/flow-associated dated wood and laboratory replication. See original site evidence.'
        elif lab.startswith('CAMS'):
            note = 'Bulk organic carbon ages. Source prints an age corrected for older carbon. See Colman et al. for the core-specific chronology.'
        elif len(link) > 1:
            note = 'Shared locality name. The exact section or core is unresolved in these age tables.'
        elif not coords:
            note = 'Regional provenance only in the age table.'
        elif separations and max(separations)>20:
            note = 'Large coordinate disagreement. Check original site and datum before spatial use.'
        matches = re.fullmatch(r'(\d+)\s*±\s*(\d+)',m[2])
        app = re.fullmatch(r'(\d+)\s*±\s*(\d+)',a[2])
        assert matches and app and matches[1] == app[1] and relation
        filtering = a[8].strip().lower()
        rows.append(dict(source_row=i,table_6_1_page=row['page'],table_12_2_page=associated['page'],
            original_citation=m[0],lab_id=lab,source_locality=m[4],age_system='conventional radiocarbon BP',
            age_bp=matches[1],published_plusminus_table_6_1=matches[2],published_plusminus_table_12_2=app[2],
            calibrated_entry_table_6_1=m[3].rstrip('*'),table_stratigraphic_group=relation,
            dated_material=material(m[5]),appendix_filtered_model=('yes' if filtering.startswith('yes') else
                'no' if filtering.startswith('no') else 'unspecified'),
            appendix_latitude=lat if coords else '',appendix_longitude=lon if coords else '',
            candidate_ledger_ids='|'.join(link),km_to_each_ledger_position='|'.join(map(str,separations)),
            interpretation=note))
    for n, (id_, site, ref, age, err) in enumerate([
            ('B2020-293','Camp Century','Hammer et al. (1980)',6350,110),
            ('B2020-292','GISP2','Zdanowicz et al. (1999)',7627,150)],82):
        rows.append(dict.fromkeys(rows[0],''))
        rows[-1].update(source_row=n,table_6_1_page=174,table_12_2_page=426,source_locality=site,
            original_citation=ref,age_system='ice-core layer-counted BP',age_bp=age,
            published_plusminus_table_6_1=err,published_plusminus_table_12_2=err,
            candidate_ledger_ids=id_,interpretation='Independent time scale. Consult the site identity and deposition evidence separately.')
    with (HERE/'egan-age-crosswalk.csv').open('w',newline='') as out:
        writer = csv.DictWriter(out,fieldnames=list(rows[0]),lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('thesis', type=Path)
    build(parser.parse_args().thesis)
