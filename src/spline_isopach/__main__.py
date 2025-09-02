import numpy as np
from lsf_bsplines2d import LsfBsplines2d
from src import Laea
import pandas as pd
from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt


if __name__ == "__main__":
    xd, yd, fd, meta = Laea.prepare_inputs_from_csv("../../examples/Data/Aso-4_tephra.csv",
                                                    "Lon",
                                                    "Lat",
                                                    "Thickness",
                                                    "Vent Lon",
                                                    "Vent Lat"
                                                    )

    # tau_v = [0.25, 0.75]
    # rou_v = [3.16, 1.33]
    # xd = np.append(xd, 625)
    # yd = np.append(yd, -800)
    # fd = np.append(fd, 2)

    tau_v = [0.25]
    rou_v = [6]
    for i in range(len(tau_v)):
        tau = tau_v[i]
        rou = rou_v[i]
        isopach = LsfBsplines2d(xd, yd, fd, np.zeros_like(fd), tau, rou)
        isopach.fit()
        cs = isopach.graph_contour(meta['vent'], [ 5,10,20], 150, 150)
        print(isopach.area_compute(cs))
