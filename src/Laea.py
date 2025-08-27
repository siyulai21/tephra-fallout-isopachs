import pyproj
import pandas as pd
import numpy as np


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


def prepare_inputs_from_csv(csv_path: str, lon_col: str, lat_col: str, thickness_cm_col: str,
                            vent_lon_col: str = "vent_lon", vent_lat_col: str = "vent_lat"):
    df = pd.read_csv(csv_path)
    vent_lon = df[vent_lon_col].iloc[0]
    vent_lat = df[vent_lat_col].iloc[0]
    fwd, inv, laea_crs = make_laea_transform(vent_lon, vent_lat)
    lon = df[lon_col].to_numpy(dtype=float)
    lat = df[lat_col].to_numpy(dtype=float)
    t_cm = df[thickness_cm_col].to_numpy(dtype=float)
    x_km, y_km = fwd.transform(lon, lat)
    # t_km = t_cm / 100000
    t_km = t_cm
    meta = {
        "vent_lon": vent_lon,
        "vent_lat": vent_lat,
        "laea_crs": laea_crs,
        "transform_fwd": fwd,
        "transform_inv": inv,
        "n_points": len(t_km),
    }
    return x_km, y_km, t_km, meta


def suggested_knot_spacing(x_km: np.ndarray, y_km: np.ndarray):
    Lx = np.max(x_km) - np.min(x_km)
    Ly = np.max(y_km) - np.min(y_km)
    spacing = np.sqrt(Lx * Ly / len(x_km)) *2
    return spacing
