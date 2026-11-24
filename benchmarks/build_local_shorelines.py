"""Build offline, scale-dependent local basemaps from the pinned GSHHG 2.3.7 shapefile archive."""
import argparse
from collections import defaultdict
from hashlib import sha256
import io
import json
from pathlib import Path
import struct
from zipfile import ZipFile, ZipInfo

import numpy as np
import shapely

ARCHIVE_SHA256 = "8dbbe7e071e77e9e75f2d639239099ebca8d5c16d6a07df8169729d49f15cf41"
SOURCE = "https://github.com/GenericMappingTools/gshhg-gmt/releases/download/2.3.7/gshhg-shp-2.3.7.zip"


def pieces(stream):
    """GSHHG's split WGS84 shapefiles store each polygon as a single closed ring."""
    stream.read(100)
    while header := stream.read(8):
        size = struct.unpack(">II",header)[1]*2
        record = stream.read(size)
        if len(record)!=size or struct.unpack_from("<I",record)[0]!=5:
            raise ValueError("Invalid GSHHG polygon record")
        bbox = struct.unpack_from("<4d",record,4)
        parts,points = struct.unpack_from("<II",record,36)
        if parts!=1:
            raise ValueError("Unexpected multipart polygon in the pinned source")
        yield bbox,shapely.Polygon(np.frombuffer(record,dtype="<f8",count=2*points,offset=48).reshape(-1,2))


def tile_index(x,origin,count):
    return min(count-1,max(0,int(np.floor((x-origin)/10.))))


def tile_blocks(stream):
    groups = defaultdict(list)
    for (west,south,east,north),poly in pieces(stream):
        left,right = tile_index(west,-180,36),tile_index(east,-180,36)
        bottom,top = tile_index(south,-90,18),tile_index(north,-90,18)
        if left==right and bottom==top:
            groups[(left,bottom)].append(poly)
            continue
        for x in range(left,right+1):
            w = 10*x-180
            for y in range(bottom,top+1):
                s = 10*y-90
                cut = shapely.clip_by_rect(poly,w,s,w+10,s+10)
                groups[(x,y)].extend(p for p in shapely.get_parts(cut) if p.geom_type=="Polygon" and not p.is_empty)
    for tile,polys in sorted(groups.items()):
        if not polys:
            continue
        _,coords,(rings,polygons) = shapely.to_ragged_array(polys)
        micro = np.rint(coords*1e6).astype("<i4")
        delta = np.diff(micro,axis=0,prepend=np.zeros((1,2),dtype="<i4"))
        buffer = io.BytesIO()
        np.savez_compressed(buffer,delta=delta,rings=np.asarray(rings,dtype="<i4"),polygons=np.asarray(polygons,dtype="<i4"))
        yield tile,buffer.getvalue()


def build(archive,dest):
    if sha256(archive.read_bytes()).hexdigest()!=ARCHIVE_SHA256:
        raise ValueError("GSHHG source archive SHA-256 differs")
    dest.mkdir(parents=True,exist_ok=True)
    index = {}
    group = None
    group_no = 0
    with ZipFile(archive) as source:
        for scale in ("f","h","i"):
            for level in (1,2,3,4,5):
                name = f"GSHHS_shp/{scale}/GSHHS_{scale}_L{level}.shp"
                if name not in source.namelist():
                    continue
                with source.open(name) as stream:
                    for (x,y),block in tile_blocks(stream):
                        if group is None or group.fp.tell()+len(block)>1_200_000:
                            if group:group.close()
                            group_no += 1
                            group = ZipFile(dest/f"shore-{group_no:02d}.zip","w")
                        key = f"{scale}/{x}/{y}/{level}"
                        group.writestr(ZipInfo(key),block)
                        index[key]=group_no
                print(scale,level,flush=True)
        if group:group.close()
        for name in ("LICENSE.TXT","COPYING.LESSERv3"):
            (dest/name).write_bytes(source.read(name))
    receipt = dict(version="GSHHG 2.3.7", source=SOURCE, source_sha256=ARCHIVE_SHA256,
                   tile_degrees=10, coordinate_quantum_degrees=1e-6, members=index,
                   packs_sha256={p.name:sha256(p.read_bytes()).hexdigest() for p in sorted(dest.glob("shore-*.zip"))})
    (dest/"index.json").write_text(json.dumps(receipt,sort_keys=True,separators=(",",":"))+"\n")


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive",type=Path)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    build(args.archive,args.output)
