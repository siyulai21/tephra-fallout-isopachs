"""Source-centred WGS84 Lambert azimuthal equal-area projection in kilometres."""
import pyproj


def make_laea_transform(vent_lon: float, vent_lat: float):
    params = {
        "proj": "laea",
        "lat_0": vent_lat,
        "lon_0": vent_lon,
        "x_0": 0,
        "y_0": 0,
        "datum": "WGS84",
        "units": "km",
        "no_defs": True
    }
    laea_crs = pyproj.CRS.from_user_input(params)
    wgs84 = pyproj.CRS.from_epsg(4326)
    fwd = pyproj.Transformer.from_crs(wgs84, laea_crs, always_xy=True)
    inv = pyproj.Transformer.from_crs(laea_crs, wgs84, always_xy=True)
    return fwd, inv, laea_crs
