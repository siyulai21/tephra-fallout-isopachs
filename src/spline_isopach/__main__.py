import numpy as np
from lsf_bsplines2d import LsfBsplines2d
from src import Laea
import pandas as pd
from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt

if __name__ == "__main__":
    print("To use this program, put your data sheet within the path: tephra-fallout-isopachs/examples/Data")
    data_name = input("Input the name of your data sheet, no input defaults to Mazama tephra datasheet: ")
    if data_name == '':
        data_name = "../../examples/Data/Mazama_tephra.csv"
    xd, yd, fd, meta = Laea.prepare_inputs_from_csv(data_name,
                                                    "Lon",
                                                    "Lat",
                                                    "Thickness",
                                                    "Vent Lon",
                                                    "Vent Lat",
                                                    )

    # tau_v = [0.25, 0.75]
    # rou_v = [3.16, 1.33]
    tau_v = [0.25]
    rou_v = [3.2]
    for i in range(len(tau_v)):
        tau_rou = input("Input your tau and rou, no input defaults to 0.25, 3.2 [x1, x2]: ")
        if tau_rou == '':
            tau = tau_v[i]
            rou = rou_v[i]
        else:
            tau = float(tau_rou.split(',')[0])
            rou = float(tau_rou.split(',')[1])
        isopach = LsfBsplines2d(xd, yd, fd, np.zeros_like(fd), tau, rou, meta)
        isopach.fit()
        level_str = input("Input your isopach levels in ascending order, no input "
                          "defaults to 2, 4, 8, 16, 32 [x1, x2,..., xn]: ")
        if level_str == '':
            levels = [2, 4, 8, 16, 32]
        else:
            levels = [float(x) for x in level_str.split(',')]
        print("Once you're done viewing isopach map, close the map window to get thickness-area data")
        cs = isopach.graph_contour((meta['vent_lonlat']), levels)
        print("Thickness (cm), Area square root (km): ", isopach.area_compute(cs))