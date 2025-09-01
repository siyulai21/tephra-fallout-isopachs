# regional_laea_bbox_demo.py
import os
import numpy as np
import matplotlib.pyplot as plt
from pyproj import Geod, CRS, Transformer

try:
    import geopandas as gpd
    SHAPEFILE = gpd.datasets.get_path("naturalearth_lowres")
except Exception:
    SHAPEFILE = os.environ.get("NE_SHAPEFILE", "ne_110m_admin_0_countries.shp")

def geodesic_segment(lon1, lat1, lon2, lat2, npts=100, geod=None):
    if geod is None:
        geod = Geod(ellps="WGS84")
    if npts < 2:
        return [(lon1, lat1), (lon2, lat2)]
    inter = geod.npts(lon1, lat1, lon2, lat2, npts - 2)
    return [(lon1, lat1)] + inter + [(lon2, lat2)]

def geodesic_path_through_waypoints(waypoints, per_segment_points=80, geod=None):
    if geod is None:
        geod = Geod(ellps="WGS84")
    pts = []
    for (lon1, lat1), (lon2, lat2) in zip(waypoints[:-1], waypoints[1:]):
        seg = geodesic_segment(lon1, lat1, lon2, lat2, npts=per_segment_points, geod=geod)
        if pts:
            pts += seg[1:]
        else:
            pts += seg
    return pts

def spherical_centroid(lonlat_list):
    lons = np.radians([p[0] for p in lonlat_list]); lats = np.radians([p[1] for p in lonlat_list])
    x = np.cos(lats) * np.cos(lons); y = np.cos(lats) * np.sin(lons); z = np.sin(lats)
    X = x.mean(); Y = y.mean(); Z = z.mean()
    lon0 = np.degrees(np.arctan2(Y, X)); hyp = float(np.sqrt(X*X + Y*Y)); lat0 = np.degrees(np.arctan2(Z, hyp))
    return float(lon0), float(lat0)

def deg_pad_for_km(lat0_deg, pad_km):
    lat_rad = np.radians(lat0_deg)
    dlat = pad_km / 110.574
    dlon = pad_km / (111.320 * max(np.cos(lat_rad), 1e-6))
    return float(dlon), float(dlat)

def normalize_lon_relative(lon, lon0):
    return ((lon - lon0 + 180.0) % 360.0) - 180.0 + lon0

def wrap180(lon):
    return ((lon + 180.0) % 360.0) - 180.0

def lonlat_bbox_for_dataset(points_lonlat, lon0, lat0, pad_km=200.0):
    lons = np.array([p[0] for p in points_lonlat], dtype=float)
    lats = np.array([p[1] for p in points_lonlat], dtype=float)
    lons_norm = normalize_lon_relative(lons, lon0)
    lon_min = float(lons_norm.min()); lon_max = float(lons_norm.max())
    lat_min = float(lats.min());      lat_max = float(lats.max())
    dlon_pad, dlat_pad = deg_pad_for_km(lat0, pad_km)
    lon_min -= dlon_pad; lon_max += dlon_pad
    lat_min -= dlat_pad; lat_max += dlat_pad
    lon_min_wrapped = wrap180(lon_min); lon_max_wrapped = wrap180(lon_max)
    if lon_min <= -180 or lon_max > 180:
        b1 = (max(-180.0, lon_min_wrapped), lat_min, 180.0, lat_max)
        b2 = (-180.0, lat_min, min(180.0, lon_max_wrapped), lat_max)
        return [b1, b2]
    else:
        return [(lon_min, lat_min, lon_max, lat_max)]


def draw_country_borders_projected_bbox(ax, shp_path, transformer, bboxes, linewidth=0.4):
    with fiona.open(shp_path) as src:
        for bbox in bboxes:
            for feat in src.filter(bbox=bbox):
                geom = feat["geometry"]
                if geom is None:
                    continue
                gtype = geom["type"]
                if gtype == "Polygon":
                    for ring in geom["coordinates"]:
                        xs = [pt[0] for pt in ring]; ys = [pt[1] for pt in ring]
                        X, Y = transformer.transform(xs, ys)
                        ax.plot(X, Y, linewidth=linewidth)
                elif gtype == "MultiPolygon":
                    for poly in geom["coordinates"]:
                        for ring in poly:
                            xs = [pt[0] for pt in ring]; ys = [pt[1] for pt in ring]
                            X, Y = transformer.transform(xs, ys)
                            ax.plot(X, Y, linewidth=linewidth)


def auto_extent(ax, Xs, Ys, pad_ratio=0.1):
    Xcat = np.concatenate([np.asarray(x).ravel() for x in Xs])
    Ycat = np.concatenate([np.asarray(y).ravel() for y in Ys])
    xmin, xmax = float(Xcat.min()), float(Xcat.max())
    ymin, ymax = float(Ycat.min()), float(Ycat.max())
    dx = xmax - xmin; dy = ymax - ymin
    pad_x = dx * pad_ratio if dx > 0 else 1.0
    pad_y = dy * pad_ratio if dy > 0 else 1.0
    ax.set_xlim(xmin - pad_x, xmax + pad_x)
    ax.set_ylim(ymin - pad_y, ymax + pad_y)

def main():
    if not os.path.exists(SHAPEFILE):
        raise FileNotFoundError("Set NE_SHAPEFILE to a Natural Earth 'countries' shapefile.")
    # Replace with your data:
    scatter_lonlat = [
        (139.6917, 35.6895),  # Tokyo
        (141.3545, 43.0621),  # Sapporo
        (130.4017, 33.5902),  # Fukuoka
        (121.5654, 25.0330),  # Taipei
        (126.9780, 37.5665),  # Seoul
    ]
    curveA_waypoints = [(121.5654, 25.0330), (139.6917, 35.6895), (141.3545, 43.0621)]
    curveB_waypoints = [(130.4017, 33.5902), (126.9780, 37.5665)]
    PAD_KM = 250.0

    geod = Geod(ellps="WGS84")
    curveA_pts = geodesic_path_through_waypoints(curveA_waypoints, per_segment_points=120, geod=geod)
    curveB_pts = geodesic_path_through_waypoints(curveB_waypoints, per_segment_points=120, geod=geod)
    all_lonlat = scatter_lonlat + curveA_pts + curveB_pts
    lon0, lat0 = spherical_centroid(all_lonlat)

    laea_crs = CRS.from_proj4(f"+proj=laea +lat_0={lat0} +lon_0={lon0} +datum=WGS84 +units=m +no_defs")
    transformer = Transformer.from_crs("EPSG:4326", laea_crs, always_xy=True)
    bboxes = lonlat_bbox_for_dataset(all_lonlat, lon0, lat0, pad_km=PAD_KM)

    scatter_X, scatter_Y = transformer.transform([p[0] for p in scatter_lonlat], [p[1] for p in scatter_lonlat])
    curveA_X, curveA_Y = transformer.transform([p[0] for p in curveA_pts], [p[1] for p in curveA_pts])
    curveB_X, curveB_Y = transformer.transform([p[0] for p in curveB_pts], [p[1] for p in curveB_pts])

    fig = plt.figure(figsize=(7, 7), dpi=150)
    ax = plt.gca()
    draw_country_borders_projected_bbox(ax, SHAPEFILE, transformer, bboxes, linewidth=0.5)
    ax.plot(scatter_X, scatter_Y, linestyle='none', marker='o')
    ax.plot(curveA_X, curveA_Y, linewidth=1.5)
    ax.plot(curveB_X, curveB_Y, linewidth=1.5)
    ax.set_title("Regional LAEA (bbox-filtered borders)")
    ax.set_aspect('equal', adjustable='box')
    auto_extent(ax, [scatter_X, curveA_X, curveB_X], [scatter_Y, curveA_Y, curveB_Y], pad_ratio=0.12)
    ax.set_axis_off()
    plt.tight_layout()
    plt.savefig("regional_laea_bbox_map.png", bbox_inches="tight")

if __name__ == "__main__":
    main()
