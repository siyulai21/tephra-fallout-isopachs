"""Offline NOAA ETOPO terrain for the observation atlas."""
from functools import lru_cache
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter, map_coordinates

ASSETS = Path(__file__).resolve().parent / "assets/observation-relief"
RECEIPT = ASSETS / "source.json"
SOURCE = "NOAA NCEI ETOPO 2022"
DOI = "https://doi.org/10.25921/fd45-gt74"


@lru_cache(maxsize=1)
def inventory():
    return json.loads(RECEIPT.read_text())


def relief_provenance():
    return {
        "source": SOURCE,
        "doi": DOI,
        "style": "land elevation and northwest hillshade",
        "assets_sha256": sha256(RECEIPT.read_bytes()).hexdigest(),
    }


def verify_assets():
    hashes = {tile["file"]: tile["sha256"] for tile in inventory()["tiles"]}
    for name, expected in hashes.items():
        path = ASSETS / name
        if sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f"Elevation tile differs from the source receipt: {path.name}")


@lru_cache(maxsize=24)
def grid(name, key):
    with np.load(ASSETS / name) as data:
        return data[key]


def sample(tile, lon, lat):
    west, _, _, north = tile["bounds"]
    spacing = tile["arcseconds"] / 3600
    y = (north - lat) / spacing - .5
    x = (lon - west) / spacing - .5
    return map_coordinates(grid(tile["file"], tile["key"]), [y, x], order=1,
                           mode="nearest", output=np.float32)


def elevation(lon, lat, valid, span):
    """Sample the fine crop if it covers the frame, else use the global grid."""
    tiles = inventory()["tiles"]
    lon = (lon + 180) % 360 - 180
    z = np.zeros(lon.shape, dtype=np.float32)
    if not valid.any():
        return z
    longitude, latitude = lon[valid], lat[valid]
    if span <= inventory()["fine_detail_max_km"]:
        lon_min, lon_max = longitude.min(), longitude.max()
        lat_min, lat_max = latitude.min(), latitude.max()
        for tile in tiles:
            if tile["arcseconds"] != 15:
                continue
            west, south, east, north = tile["bounds"]
            padding = .5 * tile["arcseconds"] / 3600
            if (west + padding <= lon_min and lon_max <= east - padding
                    and south + padding <= lat_min and lat_max <= north - padding):
                z[valid] = sample(tile, longitude, latitude)
                return z
    covered = np.zeros_like(valid)
    for tile in tiles:
        if tile["arcseconds"] == 15:
            continue
        west, south, east, north = tile["coverage"]
        inside = valid & (lat >= south) & ((lat < north) | ((north == 90) & (lat == 90)))
        inside &= (lon >= west) & (lon < east)
        if inside.any():
            z[inside] = sample(tile, lon[inside], lat[inside])
            covered |= inside
    if (valid & ~covered).any():
        raise ValueError("Global relief grid does not cover a valid map coordinate")
    return z


def terrain_rgb(lon, lat, mask, x, y):
    """Shade projected land and keep oceans and lakes in the map's blue."""
    valid = mask >= 0
    land = mask == 1
    canvas = np.ones((*mask.shape, 3), dtype=np.float32)
    canvas[mask == 0] = (.898, .937, .953)
    if not land.any():
        return canvas
    z = elevation(lon, lat, valid, max(np.ptp(x), np.ptp(y)))
    z = gaussian_filter(z, .65)
    north, east = np.gradient(z, (y[1] - y[0]) * 1000, (x[1] - x[0]) * 1000)
    # Light from 315° at 45° altitude. Keep the pale terrain subordinate to markers.
    exaggeration = 2.1
    light = (.7071 + .5 * exaggeration * (east - north)) / np.sqrt(
        1 + exaggeration**2 * (east**2 + north**2)
    )
    altitude = np.sqrt(np.clip(z / 3700, 0, 1))
    lightness = np.clip(.93 + .21 * (light - .7071) - .17 * altitude, .67, .973)
    canvas[land] = lightness[land, None] * np.array([1., .988, .952])
    return canvas
