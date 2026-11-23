"""Reproduce the reviewed column and row transcription; original file hash is in the manifest."""
from pathlib import Path
import csv, re
import xml.etree.ElementTree as ET
import argparse
import subprocess
import tempfile

parser=argparse.ArgumentParser(description='Extract Appendix 1a–c field tables from the 363-page Young (1990) thesis.')
parser.add_argument('thesis',type=Path)
parser.add_argument('--output',type=Path,default=Path('data/audits/Mazama/young1990-appendix1.csv'))
args=parser.parse_args()

ns='{http://www.w3.org/1999/xhtml}'
with tempfile.TemporaryDirectory() as temp:
    xml=Path(temp)/'appendix.xml'
    subprocess.run(['pdftotext','-f','313','-l','320','-bbox',str(args.thesis),str(xml)],check=True)
    ps=ET.parse(xml).findall('.//'+ns+'page')
boundaries={
 313:[121,207,295,382,469,556,644,731,817],314:[123,211,298,385,472,560,647,734,820],
 315:[114,224,361,477,594,725,855],316:[105,214,352,469,586,717,846],317:[123,232,369,486,602,733,864],
 318:[98,237,373,491,606,723,838],319:[116,254,391,508,624,741,858],320:[125,263,401,517,633,750,866]}
labels={'A':['llao_rock_th_cm','llao_mp_cm','llao_ml_cm','lower_cleetwood_th_cm','lower_cleetwood_mp_cm','lower_cleetwood_ml_cm','upper_cleetwood_th_cm'],
 'B':['lower_pumice_th_cm','lower_mp_cm','lower_ml_cm','basal_ash_th_cm','divider_ash_th_cm'],
 'C':['upper_pumice_th_cm','upper_3c_mp_cm','upper_3c_ml_cm','upper_1_mp_cm','upper_1_ml_cm']}
allrows=[]
for n,page in enumerate(ps,313):
 grp='A' if n<315 else 'B' if n<318 else 'C'
 edges=boundaries[n]
 cells=[]
 for word in list(page):
  t=word.text or ''
  b=word.attrib
  x=(float(b['xMin'])+float(b['xMax']))/2*128/72
  y=(float(b['yMin'])+float(b['yMax']))/2*128/72
  cells.append((x,y,t))
 ids=[(int(t),y) for x,y,t in cells if edges[0]<x<edges[1] and re.fullmatch(r'\d+',t) and y>200]
 for rid,y in ids:
  row=dict(pdf_page=n,appendix_table=grp,site=rid)
  for i,col in enumerate(labels[grp],1):
   vs=[(x,t) for x,py,t in cells if edges[i]<x<edges[i+1] and abs(py-y)<12]
   value=' '.join(t for x,t in sorted(vs))
   if grp in ('B','C') and i==1 and rid in (86,135,136,146,147,149):
    if value:value='J '+re.sub(r'^[^0-9]*','',value)
   value={'o':'0','O':'0','I':'1','I+':'1 +','T.':'T'}.get(value,value)
   row[col]=value
  allrows.append(row)
cols=['pdf_page','appendix_table','site',*labels['A'],*labels['B'],*labels['C']]
with args.output.open('w') as o:
 w=csv.DictWriter(o,fieldnames=cols,lineterminator='\n');w.writeheader();w.writerows(allrows)
for grp in labels:
 rr=[r for r in allrows if r['appendix_table']==grp]
 print(grp,len(rr),[r['site'] for r in rr])
for r in allrows:
 for k,v in r.items():
  if k.endswith('_cm') and v and not re.fullmatch(r'(J )?([<≤] ?)?(\d+(\.\d+)?|T)( ?\+)?',v):print('CHECK',r['pdf_page'],r['site'],k,v)

