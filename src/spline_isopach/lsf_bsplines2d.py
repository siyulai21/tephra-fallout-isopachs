import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import cKDTree
import matplotlib as mpl
from pathlib import Path

try:
    import scipy.linalg as _spl
    _HAVE_SCIPY = True
except Exception:
    _HAVE_SCIPY = False


class LsfBsplines2d:

    def __init__(self, xd, yd, fd, sd, tau, rou, meta):
        xd, yd, fd, sd = (np.asarray(a, dtype=float) for a in (xd, yd, fd, sd))
        if any(a.ndim != 1 or len(a) != len(fd) for a in (xd, yd, fd, sd)):
            raise ValueError("Spline inputs must be matching one-dimensional arrays.")
        if len(fd) < 4 or not all(np.isfinite(a).all() for a in (xd, yd, fd, sd)):
            raise ValueError("Spline requires at least four finite observations.")
        if np.any(fd <= 0) or np.any(sd < 0):
            raise ValueError("Thickness must be positive and log-scale SD nonnegative.")
        if np.ptp(xd) == 0 or np.ptp(yd) == 0 or not 0 <= tau <= 1 or not np.isfinite(rou) or rou <= 0:
            raise ValueError("Require spatial extent in both axes, 0<=tau<=1 and rou>0.")
        self.xd = xd
        self.yd = yd
        self.fd = np.log(fd)
        self.sd = sd
        self.xmin = xd.min()
        self.ymin = yd.min()
        self.xmax = xd.max()
        self.ymax = yd.max()
        self.tau = tau
        self.rou = rou
        self.meta = meta
        self.unl, self.und, self.mx0, self.my0, self.n2 = self.domain_divisions()
        self.cij = np.zeros((self.my0 + 3) * (self.mx0 + 3))

    def fit(self):
        self.fit_bsplines2d()

    def area_compute(self, cs) -> list:
        """Area within closed contours in map metres; reject clipped/open curves.

        Matplotlib orients exterior and interior rings oppositely, so signed
        areas preserve holes. Do not round thickness levels or areas here.
        """
        areas = []
        for i in range(len(cs.allsegs)):
            area = 0
            levelxsegs = cs.allsegs[i]
            for polygonx in levelxsegs:
                if len(polygonx) < 3:
                    continue
                if not np.allclose(polygonx[0], polygonx[-1], rtol=0, atol=1e-7):
                    raise ValueError("An isopach crosses the map boundary; its enclosed area is not identified.")
                x = polygonx[:,0]
                y = polygonx[:,1]
                area += 0.5 * (np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))
            area = np.sqrt(abs(area)/10**6) # from m2 to km2
            areas.append((float(area), float(cs.levels[i])))
        return areas  # [(thickness_1, area_1^0.5), (thickness_2, area_2^0.5), ...]

    def graph_contour(self, vent, levels, nxg=150, nyg=150) -> "mpl.QuadContourSet":
        from mpl_toolkits.basemap import Basemap
        Z_fit = self.eval_surface_grid(nx=nxg, ny=nyg)
        self.levels = levels
        height = max(abs(self.ymin), abs(self.ymax))*2*1.25
        width = max(abs(self.xmin), abs(self.xmax))*2*1.25
        offset_x = width / 2
        offset_y = height / 2
        X = np.linspace(self.xmin, self.xmax, nxg, endpoint=False)+offset_x
        Y = np.linspace(self.ymin, self.ymax, nyg, endpoint=False)+offset_y

        lon_0, lat_0 = vent
        m = Basemap(resolution='l', projection='laea',
                    height=height*1000,width=width*1000,
                    lat_0=lat_0, lon_0=lon_0)
        m.fillcontinents()
        ax = plt.gca()
        fig = plt.gcf()
        ax.set_title(f"Fitted surface (rou={self.rou:.2f}, tau={self.tau:.2f})")
        CS = ax.contour(X*1000, Y*1000, Z_fit, levels=self.levels, colors='k', linewidths=1)
        ax.clabel(CS, inline=True, fontsize=8, fmt="%g")

        masks = []
        for i in range(len(levels)+2):
            if i == len(levels):
                masks.append((f"t ≥ {t} cm", (self.fd >= np.log(t)) &
                              (self.meta['pseudo'] != "Yes"))
                             )
                continue
            elif i == len(levels)+1:
                masks.append(('Pseudo data', self.meta['pseudo'] == "Yes")
                             )
                continue
            t = levels[i]
            if i == 0:
                masks.append((f"{t} > t cm", (self.fd < np.log(t)) &
                             (self.meta['pseudo'] != "Yes"))
                             )
            else:
                masks.append((f"{t} > t ≥ {levels[i-1]} cm",
                             (np.log(t) > self.fd) &
                             (self.fd >= np.log(levels[i-1])) &
                             (self.meta['pseudo'] != "Yes"))
                             )

        colors = mpl.colormaps['tab10'].colors  # take first 5
        for i, (label, m) in enumerate(masks):
            x = self.xd[m]+offset_x
            y = self.yd[m]+offset_y
            if i == len(masks)-1:
                ax.scatter(x * 1000, y * 1000, label=label, s=20,
                           facecolors='none', edgecolor="k")
            else:
                ax.scatter(x*1000, y*1000, label=label, s=20,
                        color=colors[i], edgecolor="k", alpha=0.65)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        pad_x = 0.25 * (self.xmax-self.xmin)
        pad_y = 0.25 * (self.ymax-self.ymin)
        left_x, right_x = (self.xmin + offset_x - pad_x)*1000, (self.xmax+offset_x+pad_x)*1000
        bottom_y, top_y = (self.ymin + offset_y-pad_y) * 1000, (self.ymax + offset_y+pad_y) * 1000
        ax.set_xlim(left_x, right_x)
        ax.set_ylim(bottom_y, top_y)
        fig.tight_layout()
        ax.legend(title="Thickness range", fontsize="small", title_fontsize="small",
                   scatterpoints=1, markerscale=1.2, frameon=False)
        plt.show()
        saving = input("Do you want to save this map to results/legacy/Result.png [y/n]: ")
        if saving == "y":
            destination = Path(__file__).parents[2]/"results/legacy/Result.png"
            destination.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(destination, dpi=240)
        plt.close()
        return CS

    def eval_surface_grid(self, nx=100, ny=100):
        xs = np.linspace(self.xmin, self.xmax, nx, endpoint=False)
        ys = np.linspace(self.ymin, self.ymax, ny, endpoint=False)
        Z = np.zeros((ny, nx)) # type: np.ndarray
        for j, y in enumerate(ys):
            for i, x in enumerate(xs):
                Z[j, i] = np.exp(self.eval_cubic_bspline_surface(x, y))
        return Z

    def eval_cubic_bspline_surface(self, x, y):
        mx = self.mx0 * (2 ** self.n2)
        my = self.my0 * (2 ** self.n2)
        nx = mx + 3
        hx = (self.xmax - self.xmin) / mx
        hy = (self.ymax - self.ymin) / my
        x = np.minimum(np.maximum(x, self.xmin), np.nextafter(self.xmax, self.xmin))
        y = np.minimum(np.maximum(y, self.ymin), np.nextafter(self.ymax, self.ymin))
        ip = int(np.floor((x - self.xmin) / hx)) + 1
        jp = int(np.floor((y - self.ymin) / hy)) + 1
        ip = 1 if ip < 1 else (mx if ip > mx else ip)
        jp = 1 if jp < 1 else (my if jp > my else jp)
        rx = (x - self.xmin) / hx - (ip - 1)
        ry = (y - self.ymin) / hy - (jp - 1)
        rx = 0.0 if rx < 0.0 else (1.0 if rx > 1.0 else rx)
        ry = 0.0 if ry < 0.0 else (1.0 if ry > 1.0 else ry)
        bx = np.zeros(4)
        by = np.zeros(4)
        BSP3(float(rx), float(hx), 0, bx) # type: np.ndarray
        BSP3(float(ry), float(hy), 0, by) # type: np.ndarray
        s = 0.0
        for j1 in range(0, 4):
            for i1 in range(0, 4):
                s += bx[3 - i1] * by[3 - j1] * self.cij[(jp + j1 - 1) * nx + (ip + i1 - 1)]
        return s

    def fit_bsplines2d(self, nit=200, omg=1.2, use_scipy=True, seed=None):
        """Solve the complete normal equations on the final knot grid.

        The historical refinement stencil loses mixed-sign cross terms. Direct
        banded assembly includes every basis pair and avoids iteration-order
        dependence. nit/omg/seed remain accepted for API compatibility.
        """
        mx, my = self.mx0 * 2**self.n2, self.my0 * 2**self.n2
        A, b, width, n = self._assemble_coarse(mx=mx, my=my, rou=self.rou)
        if use_scipy and _HAVE_SCIPY:
            # A already stores lower diagonals by COLUMN: ab[i-j,j]=A[i,j].
            self.cij = _spl.solveh_banded(A.reshape((width, n), order="F"), b, lower=True)
        else:
            self.cij = b.copy()
            MCHLBA_exact(A.copy(), width, n, width, self.cij)

    def _assemble_coarse(self, mx=None, my=None, rou=None):
        mx = self.mx0 if mx is None else mx
        my = self.my0 if my is None else my
        rou = self.rou / (10.0 ** self.n2) if rou is None else rou
        neq = (mx + 3) * (my + 3)
        nband = mx * 3 + 13
        hx = (self.xmax - self.xmin) / mx
        hy = (self.ymax - self.ymin) / my
        A = np.zeros(nband * neq, dtype=float)
        b = np.zeros(neq, dtype=float)

        bx = np.zeros(4);
        by = np.zeros(4)
        s10 = np.zeros((4, 4));
        s11 = np.zeros((4, 4));
        s12 = np.zeros((4, 4))
        for k in range(len(self.xd)):
            x = float(self.xd[k]);
            y = float(self.yd[k])
            if (x < self.xmin) or (x > self.xmax) or (y < self.ymin) or (y > self.ymax): continue
            ip = int(np.floor((x - self.xmin) / hx)) + 1
            jp = int(np.floor((y - self.ymin) / hy)) + 1
            ip = 1 if ip < 1 else (mx if ip > mx else ip)
            jp = 1 if jp < 1 else (my if jp > my else jp)
            rx = (x - self.xmin) / hx - (ip - 1)
            ry = (y - self.ymin) / hy - (jp - 1)
            rx = 0.0 if rx < 0.0 else (1.0 if rx > 1.0 else rx)
            ry = 0.0 if ry < 0.0 else (1.0 if ry > 1.0 else ry)
            s = float(self.sd[k]);
            wd = 1.0 if s <= 0.0 else 1.0 / (s * s)
            BSP3(rx, hx, 0, bx);
            BSP3(ry, hy, 0, by)
            for j1 in range(0, 4):
                for i1 in range(0, 4):
                    ij1 = (ip + i1) + (mx + 3) * (jp + j1 - 1)
                    b[ij1 - 1] += bx[3 - i1] * by[3 - j1] * float(self.fd[k]) * wd
                    for j2 in range(j1, 4):
                        for i2 in range(i1 if j2 == j1 else 0, 4):
                            ij2 = (ip + i2) + (mx + 3) * (jp + j2 - 1)
                            ij2b = ij2 - ij1 + 1
                            A[(ij1 - 1) * nband + (ij2b - 1)] += bx[3 - i1] * by[3 - j1] * bx[3 - i2] * by[3 - j2] * wd
        SHFTI(0, s10);
        SHFTI(1, s11);
        SHFTI(2, s12)
        w1 = self.tau / rou * (self.unl / self.und) ** 2
        w2 = (1.0 - self.tau) / rou * (self.unl ** 2 / self.und) ** 2
        wx = (hy / hx) * (w1 / (self.unl ** 2))
        wy = (hx / hy) * (w1 / (self.unl ** 2))
        wkx = (hy / (hx ** 3)) * (w2 / (self.unl ** 2))
        wxy = (1.0 / (hx * hy)) * (w2 / (self.unl ** 2)) * 2.0
        wyy = (hx / (hy ** 3)) * (w2 / (self.unl ** 2))
        for j1 in range(1, my + 3 + 1):
            for i1 in range(1, mx + 3 + 1):
                for js in range(0, 4):
                    for is_ in range(-3, 3 + 1):
                        if js == 0 and is_ < 0: continue
                        i2 = i1 + is_;
                        j2 = j1 + js
                        if i2 < 1 or i2 > (mx + 3) or j2 < 1 or j2 > (my + 3): continue
                        lx1 = max(1, is_ + 1, 5 - i1);
                        lx2 = min(4, is_ + 4, mx + 4 - i1)
                        s0x = s1x = s2x = 0.0
                        for lx in range(lx1, lx2 + 1):
                            s0x += s10[lx - 1, (lx - is_) - 1]
                            s1x += s11[lx - 1, (lx - is_) - 1]
                            s2x += s12[lx - 1, (lx - is_) - 1]
                        ly1 = max(1, js + 1, 5 - j1);
                        ly2 = min(4, js + 4, my + 4 - j1)
                        s0y = s1y = s2y = 0.0
                        for ly in range(ly1, ly2 + 1):
                            s0y += s10[ly - 1, (ly - js) - 1]
                            s1y += s11[ly - 1, (ly - js) - 1]
                            s2y += s12[ly - 1, (ly - js) - 1]
                        ij1 = i1 + (mx + 3) * (j1 - 1)
                        ij2 = i2 + (mx + 3) * (j2 - 1)
                        ij2b = ij2 - ij1 + 1
                        if 1 <= ij2b <= nband:
                            A[(ij1 - 1) * nband + (ij2b - 1)] += (
                                    wx * (s1x * s0y) + wy * (s0x * s1y) + wkx * (s2x * s0y) + wxy * (
                                        s1x * s1y) + wyy * (s0x * s2y)
                            )
        return A, b, nband, (mx + 3) * (my + 3)

    def domain_divisions(self):
       Lx = max(self.xd) - min(self.xd)
       Ly = max(self.yd) - min(self.yd)
       N = len(self.xd)
       domain_A = Lx * Ly
       s_uni = np.sqrt(domain_A / N)
       pts = np.c_[self.xd, self.yd]
       dists_fi, _ = cKDTree(pts).query(pts, k=2)
       dists_th, _ = cKDTree(pts).query(pts, k=4)
       d1 = np.median(dists_fi[:, 1])
       d3 = np.median(dists_th[:, 3])
       d_rob = np.sqrt(d1 * d3)
       unl = np.clip(np.sqrt(s_uni * d_rob), 0.7 * s_uni, 1.5 * s_uni)

       X = np.c_[np.ones_like(self.xd), self.xd, self.yd]
       beta, *_ = np.linalg.lstsq(X, self.fd, rcond=None)
       resid = self.fd - X.dot(beta)
       mad = np.median(np.abs(resid - np.median(resid)))
       und = max(1.4826 * mad, 1e-6)

       c0 = int(np.clip(N / 8, 50, 300))
       mx0 = max(4, round(np.sqrt(c0) * np.sqrt(Lx/Ly)))
       my0 = max(4, round(np.sqrt(c0) * np.sqrt(Ly / Lx)))
       h0x = Lx / max(mx0, 1)
       h0y = Ly / max(my0, 1)
       h_fine_star = max(0.5 * d1, 0.5 * unl)
       n2 = int(np.ceil(max(np.log2(max(h0x / h_fine_star, 1.0)), np.log2(max(h0y / h_fine_star, 1.0)))))
       while mx0 * (2 ** n2) > 128 or my0 * (2 ** n2) > 128:
           n2 -= 1
           if n2 <= 0: break

       return unl, und, mx0, my0, n2


def _AD_offset(ib, jb, i, j, nx):
    return (ib - 1) + 4 * (jb - 1) + 16 * ((i - 1) + nx * (j - 1))


def BSP3(r, h, idef, bn_out):
    if idef == 3:
        a = 1.0 / (h * h * h)
        bn_out[0] =  1.0 * a
        bn_out[1] = -3.0 * a
        bn_out[2] =  3.0 * a
        bn_out[3] = -1.0 * a
        return
    if idef == 2:
        a = 1.0 / (h * h)
        bn_out[0] = r * a
        bn_out[1] = (-3.0 * r + 1.0) * a
        bn_out[2] = ( 3.0 * r - 2.0) * a
        bn_out[3] = (-r + 1.0) * a
        return
    r2 = r * r
    if idef == 1:
        a = 0.5 / h
        bn_out[0] = r2 * a
        bn_out[1] = (-3.0 * r2 + 2.0 * r + 1.0) * a
        bn_out[2] = ( 3.0 * r2 - 4.0 * r        ) * a
        bn_out[3] = (-r2 + 2.0 * r - 1.0) * a
        return
    r3 = r2 * r
    a = 1.0 / 6.0
    bn_out[0] =  r3 * a
    bn_out[1] = (-3.0 * r3 + 3.0 * r2 + 3.0 * r + 1.0) * a
    bn_out[2] = ( 3.0 * r3 - 6.0 * r2 + 4.0            ) * a
    bn_out[3] = (-r3 + 3.0 * r2 - 3.0 * r + 1.0) * a


def SHFTI(idef, S_out, npts=2001):
    r = np.linspace(0.0, 1.0, npts)
    B = np.zeros((4, npts))
    tmp = np.zeros(4)
    for k, rv in enumerate(r):
        BSP3(rv, 1.0, idef, tmp)
        B[:, k] = tmp[::-1]
    for i in range(4):
        for j in range(4):
            S_out[i, j] = np.trapezoid(B[i, :] * B[j, :], r)


def MCHLBA_exact(A, KA, NEQ, NB, B):
    def Aget(r, c):
        return A[(c - 1) * KA + (r - 1)]
    def Aset(r, c, v):
        A[(c - 1) * KA + (r - 1)] = v

    for I in range(1, NEQ + 1):
        K0 = max(1, I - NB + 1)
        for K in range(K0, I):
            Aset(1, I, Aget(1, I) - Aget(I-K+1, K) * Aget(I-K+1, K) / Aget(1, K))
        for K in range(K0, I):
            Aset(I-K+1, K, Aget(I-K+1, K) / Aget(1, K))
        for JB in range(2, NB + 1):
            for K in range(max(1, I - NB + JB), I):
                r = I - K + JB
                if r <= KA:
                    Aset(JB, I, Aget(JB, I) - Aget(r, K) * Aget(I-K+1, K))
        for K in range(K0, I):
            B[I-1] -= Aget(I-K+1, K) * B[K-1]
    B[NEQ-1] /= Aget(1, NEQ)
    for I in range(NEQ - 1, 0, -1):
        B[I-1] /= Aget(1, I)
        for JB in range(2, NB + 1):
            if I + JB - 1 <= NEQ:
                B[I-1] -= Aget(JB, I) * B[I + JB - 2]
