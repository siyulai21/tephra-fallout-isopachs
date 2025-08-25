import numpy as np
from scipy.interpolate import BSpline
import pandas as pd

def cubic_B_vals(r):
    r2 = r*r
    r3 = r2*r
    B1 = r3/6
    B2 = (-3*r3 + 3*r2 + 3*r + 1)/6
    B3 = (3*r3 - 6*r2 + 4)/6
    B4 = (-1*r3 + 3*r2 - 3*r + 1)/6
    return B1, B2, B3, B4

def cubic_B_d1(r):
    r2 = r*r
    B1p = (r2)/2
    B2p = (-9*r2 + 6*r + 3)/6
    B3p = (9*r2 - 12*r)/6
    B4p = (-3*r2 + 6*r - 3)/6
    return B1p, B2p, B3p, B4p

def cubic_B_d2(r):
    B2pp = (-18*r + 6)/6
    B3pp = (18*r - 12)/6
    B4pp = (-6*r + 6)/6
    return r, B2pp, B3pp, B4pp

