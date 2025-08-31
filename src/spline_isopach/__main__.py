import numpy as np
from lsf_bsplines2d import LsfBsplines2d
from src import Laea
import pandas as pd


def contours_xy_to_lonlat(contour_set, inv_transform):
    lonlat_by_level = []
    for level, seglists in zip(contour_set.levels, contour_set.allsegs):
        out = []
        for seg in seglists:               # seg is an (N,2) array of [x, y] vertices
            xs, ys = seg[:,0], seg[:,1]
            lon, lat = inv_transform.transform(xs, ys)  # returns degrees
            out.append(np.column_stack([lon, lat]))
        lonlat_by_level.append((level, out))
    return lonlat_by_level


if __name__ == "__main__":
    xd, yd, fd, meta = Laea.prepare_inputs_from_csv("Mazama_tephra.csv",
                                                    "Lon",
                                                    "Lat",
                                                    "Thickness",
                                                    "Vent Lon",
                                                    "Vent Lat"
                                                    )
    # df = pd.read_csv("datas.csv")
    # xd = df["xd"].to_numpy(float)
    # yd = df["yd"].to_numpy(float)
    # fd = df["fd"].to_numpy(float)
    #
    # inv = meta["transform_inv"]
    # lon, lat = inv.transform(xd, yd)
    #
    # X = np.column_stack((lon, lat, fd))
    # np.savetxt("Mazama_tephra.csv", X, delimiter=",", header="Lon,Lat,Thickness", fmt="%.6f")

    # tau_v = [0.25, 0.75]
    # rou_v = [3.16, 1.33]

    tau_v = [0.25]
    rou_v = [3.16]
    for i in range(len(tau_v)):
        tau = tau_v[i]
        rou = rou_v[i]
        isopach = LsfBsplines2d(xd, yd, fd, np.zeros_like(fd), tau, rou)
        isopach.fit()
        contour_set = isopach.graph_contour()

        # inv = meta["transform_inv"]
        # lonlat_lines = contours_xy_to_lonlat(contour_set, inv)
