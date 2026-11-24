"""Identify the 2016 photographed Jenny Lake drive using the labelled 2024 line scans.

Requires the original larsen2016.pdf and jenny2024-supplement.pdf in the case
source directory, Poppler pdfimages, numpy, scipy and Pillow.
"""
import csv
from pathlib import Path
import subprocess
import tempfile

import numpy as np
from PIL import Image
from scipy import ndimage, signal

ROOT = Path(__file__).resolve().parents[4]
SOURCES = ROOT/'data/sources/mazama'
DEST = Path(__file__).with_name('jenny-registration.csv')
SECTIONS = {'13-1 first':(139,190),'13-1 second':(229,273),'13-2 first':(373,417),
            '13-2 second':(459,503),'15-3 first':(611,667),'15-3 second':(710,763),
            '15-4 first':(965,1022),'15-4 second':(1061,1117)}
WINDOWS = {'upper outside ash':(150,125,524,385),'lower outside ash':(150,810,524,1190),
           'upper and contact':(150,150,524,655)}


def highpass(image):
    a=np.asarray(image.convert('L'),dtype=float)
    return a-ndimage.gaussian_filter(a,2.2)


def normalized_correlation(field, patch):
    patch=patch-patch.mean()
    ones=np.ones_like(patch)
    sums=signal.correlate(field,ones,mode='valid',method='fft')
    squares=signal.correlate(field*field,ones,mode='valid',method='fft')
    return signal.correlate(field,patch,mode='valid',method='fft')/np.sqrt(
        np.maximum(squares-sums**2/patch.size,1e-6)*np.sum(patch**2))


def match(plate, scan, name, box, section, x_bounds):
    left,right=x_bounds
    field=highpass(scan.crop((left,170,right,1080)))
    original=plate.crop(box)
    best=None
    for sx in (.085,.095,.105,.12):
        for sy in (.09,.095,.1):
            size=(round(original.width*sx),round(original.height*sy))
            patch=highpass(original.resize(size,Image.Resampling.LANCZOS))
            if patch.shape[1]>field.shape[1]:
                continue
            for flip in (False,True):
                grid=normalized_correlation(field,patch[:,::-1] if flip else patch)
                y,x=np.unravel_index(grid.argmax(),grid.shape)
                score=grid[y,x]
                if best is None or score>best['correlation']:
                    best=dict(photo_window=name,photo_box_px=','.join(map(str,box)),candidate=section,
                              scan_x=int(x+left),scan_y=int(y+170),scale_x=sx,scale_y=sy,
                              flipped=flip,correlation=round(float(score),9))
    return best


def main():
    with tempfile.TemporaryDirectory() as temp:
        temp=Path(temp)
        subprocess.run(['pdfimages','-f','5','-l','5','-j',str(SOURCES/'larsen2016.pdf'),str(temp/'plate')],check=True)
        subprocess.run(['pdfimages','-f','5','-l','5','-png',str(SOURCES/'jenny2024-supplement.pdf'),str(temp/'scan')],check=True)
        with Image.open(temp/'plate-001.jpg') as plate, Image.open(temp/'scan-000.png') as scan:
            assert plate.size==(1028,1298) and scan.size==(1431,1852)
            rows=[match(plate,scan,name,box,section,xb) for name,box in WINDOWS.items() for section,xb in SECTIONS.items()]
    with DEST.open('w',newline='') as output:
        writer=csv.DictWriter(output,fieldnames=list(rows[0]),lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)
    for name in WINDOWS:
        candidates=sorted((r for r in rows if r['photo_window']==name),key=lambda r:r['correlation'],reverse=True)
        assert candidates[0]['candidate']=='13-2 first' and candidates[0]['correlation']>2*candidates[1]['correlation']


if __name__=='__main__':
    main()

