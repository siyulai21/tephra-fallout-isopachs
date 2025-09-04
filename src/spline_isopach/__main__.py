import numpy as np
from lsf_bsplines2d import LsfBsplines2d
from src import Laea
import pandas as pd
from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt

if __name__ == "__main__":
    path = input("Path to your data sheet (No input defaults to Mazama tephra datasheet): ")
    if path == '':
        path = "../../examples/Data/Aso-4_tephra.csv"
    xd, yd, fd, meta = Laea.prepare_inputs_from_csv(path,
                                                    "Lon",
                                                    "Lat",
                                                    "Thickness",
                                                    "Vent Lon",
                                                    "Vent Lat",
                                                    )

    # tau_v = [0.25, 0.75]
    # rou_v = [3.16, 1.33]
    tau_v = [0.25]
    rou_v = [3.1]
    for i in range(len(tau_v)):
        tau = tau_v[i]
        rou = rou_v[i]
        isopach = LsfBsplines2d(xd, yd, fd, np.zeros_like(fd), tau, rou, meta)
        isopach.fit()
        cs = isopach.graph_contour((meta['vent_lonlat']), [5, 10, 20], 150, 150)
        print("Thickness (cm), Area square root (km): ", isopach.area_compute(cs))
