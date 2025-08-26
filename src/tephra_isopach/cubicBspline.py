import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix, csr_matrix, csc_matrix, diags
from scipy.sparse.linalg import spsolve
from typing import Tuple, List, Optional
import KnotAxis
import FitParams


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
    B1p = r2/2
    B2p = (-9*r2 + 6*r + 3)/6
    B3p = (9*r2 - 12*r)/6
    B4p = (-3*r2 + 6*r - 3)/6
    return B1p, B2p, B3p, B4p


def cubic_B_d2(r):
    B2pp = (-18*r + 6)/6
    B3pp = (18*r - 12)/6
    B4pp = (-6*r + 6)/6
    return r, B2pp, B3pp, B4pp


def assemble_data_matrix(x, y, xa: KnotAxis, ya: KnotAxis):
    n = len(x)
    Nx, Ny = xa.nbasis, ya.nbasis
    m = Nx * Ny
    rows, cols, vals = [], [], []
    for p in range(n):
        ix, vx, _, _ = xa.local_basis(x[p])
        iy, vy, _, _ = ya.local_basis(y[p])
        for a in range(4):
            for b in range(4):
                rows.append(p)
                cols.append(ix[a]*Ny+iy[b])
                vals.append(vx[a]*vy[b])

    return coo_matrix((vals, (rows, cols)), shape=(n, m)).tocsr()


def assemble_derivative_mats(Xg, Yg, xa: KnotAxis, ya: KnotAxis, wq: np.array):
    Nx, Ny = xa.nbasis, ya.nbasis
    m = Nx * Ny
    G = len(Xg)

    def build_Q(kind):
        rows, cols, vals = [], [], []
        for g in range(G):
            ix, vx, dx, d2x = xa.local_basis(Xg[g])
            iy, vy, dy, d2y = ya.local_basis(Yg[g])
            if kind == "x":
                wx = dx
                wy = vy
            elif kind == "y":
                wx = vx
                wy = dy
            elif kind == "xx":
                wx = d2x
                wy = vy
            elif kind == "xy":
                wx = dx
                wy = dy
            else:
                wx = vx
                wy = d2y
            for a in range(4):
                for b in range(4):
                    col = ix[a] * Ny + iy[b]
                    rows.append(g)
                    cols.append(col)
                    vals.append(wx[a] * wy[b])
        return coo_matrix((vals, (rows, cols)), shape=(G, m)).tocsr()

    Qx = build_Q("x")
    Qy = build_Q("y")
    Qxx = build_Q("xx")
    Qxy = build_Q("xy")
    Qyy = build_Q("yy")
    Wq = diags(wq)
    R1 = (Qx.T @ Wq @ Qx) + (Qy.T @ Wq @ Qy)
    R2 = (Qxx.T @ Wq @ Qxx) + 2.0 * (Qxy.T @ Wq @ Qxy) + (Qyy.T @ Wq @ Qyy)
    return R1.tocsc(), R2.tocsc()


def fit_spline_under_tension(x, y, t, wp=None, params: FitParams=FitParams(), quad_nx=80, quad_ny=80):
    xa = KnotAxis(float(x.min()), float(x.max()), params.x_spacing)
    ya = KnotAxis(float(y.min()), float(y.max()), params.y_spacing)
    Nx, Ny = xa.nbasis, ya.nbasis
    m = Nx*Ny
    A = assemble_data_matrix(x, y, xa, ya)
    if wp is None:
        W = diags(np.ones_like(t))
    else:
        W = diags(wp)
    Xg = np.linspace(xa.start, xa.end, quad_nx)
    Yg = np.linspace(ya.start, ya.end, quad_ny)
    XG, YG = np.meshgrid(Xg, Yg, indexing="ij")
    dx = (xa.start - xa.end) / (quad_nx-1)
    dy = (ya.start - ya.end) / (quad_ny-1)
    wq = dx * dy * np.ones(XG.size)
    R1, R2 = assemble_data_matrix(XG.ravel(), YG.ravel(), xa, ya, wq)
    ATA = A.T @ W @ A
    rhs = A.T @ W @ t
    K = ATA + params.lambda_tension * R1 + params.lambda_rough * R2
    c = spsolve(K.tocsc(), rhs)
    model = {"xa": xa, "ya": ya, "coeffs": c.reshape((Nx, Ny)), "params": params}
    return model


if __name__ == "__main__":
    print("main")