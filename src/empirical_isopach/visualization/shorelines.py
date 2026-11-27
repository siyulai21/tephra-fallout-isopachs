"""Scale-dependent GSHHG land and inland water for near-vent panels."""
from functools import lru_cache
from hashlib import sha256
import io
import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import shapely

from src.Laea import make_laea_transform

ASSETS = Path(__file__).parents[1]/"assets/shorelines"
TIERS = {"f":"full","h":"high","i":"intermediate"}


@lru_cache(maxsize=1)
def index():
    return json.loads((ASSETS/"index.json").read_text())


@lru_cache(maxsize=16)
def pack(number):
    name = f"shore-{number:02d}.zip"
    payload = (ASSETS/name).read_bytes()
    if sha256(payload).hexdigest()!=index()["packs_sha256"][name]:
        raise ValueError(f"Shoreline pack checksum differs: {name}")
    return ZipFile(io.BytesIO(payload))


@lru_cache(maxsize=96)
def geometry(key):
    number = index()["members"].get(key)
    if number is None:
        return None
    with np.load(io.BytesIO(pack(number).read(key)),allow_pickle=False) as data:
        coords = np.cumsum(data["delta"],dtype=np.int64,axis=0)*1e-6
        polygons = shapely.from_ragged_array(shapely.GeometryType.POLYGON,coords,(data["rings"],data["polygons"]))
    result = shapely.geometrycollections(polygons)
    shapely.prepare(result)
    return result


def detail(bounds, *, display_pixels=500):
    # A model close-up occupies approximately 500 pixels. The atlas supplies its own grid width.
    width = float(max(bounds[2:]-bounds[:2]))
    pixel = width/display_pixels
    tier = "f" if pixel<=.3 else "h" if pixel<=1.5 else "i"
    return dict(source=index()["version"],archive_sha256=index()["source_sha256"],
                tier=tier,resolution=TIERS[tier],km_per_display_pixel=pixel,raster_long_side=901)


def local_mask(x,y,vent,tier):
    xx,yy = np.meshgrid(x,y)
    _,inverse,_ = make_laea_transform(*vent)
    lon,lat = inverse.transform(xx,yy)
    ok = np.isfinite(lon)&np.isfinite(lat)
    mask = np.full(xx.shape,-1,dtype=np.int8)
    if not ok.any():
        return mask
    lng = (lon[ok]+180.)%360.-180.
    phi = lat[ok]
    tx = np.clip(np.floor((lng+180.)/10).astype(int),0,35)
    ty = np.clip(np.floor((phi+90.)/10).astype(int),0,17)
    tiles = 36*ty+tx
    surface = np.zeros(len(tiles),dtype=np.int8)
    for tile in np.unique(tiles):
        ids = np.flatnonzero(tiles==tile)
        part = np.zeros(len(ids),dtype=np.int8)
        for level in (1,5,2,3,4):
            shape = geometry(f"{tier}/{tile%36}/{tile//36}/{level}")
            if shape is not None:
                inside = shapely.intersects_xy(shape,lng[ids],phi[ids])
                part[inside] = level in (1,3,5)
        surface[ids] = part
    mask[ok] = surface
    return mask
