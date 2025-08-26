import numpy as np
from dataclasses import dataclass
from cubicBspline import cubic_B_vals, cubic_B_d1, cubic_B_d2

@dataclass
class KnotAxis:
    start: float
    end: float
    spacing: float

    def __post_init__(self):
        L = self.end - self.start
        M = int(np.floor(L / self.spacing))
        self.M = max(M, 1)
        self.f = L / self.M
        self.knots = self.start + np.arange(self.M+1)*self.f
        self.nbasis = self.M + 3

    def local_basis(self, x):
        eps = 1e-12
        x = np.clip(x, self.start+eps, self.end-eps)
        k = int(np.floor((x-self.start)/self.f))
        k = min(max(k, 0), self.M-1)
        r = (x - self.knots[k]) / self.f
        B = cubic_B_vals(r)
        Bp = cubic_B_d1(r)
        Bpp = cubic_B_d2(r)
        idx = np.array([k+3, k+2, k+1, k], dtype=int)
        vals = np.array(B)
        d1 = np.array(Bp) / self.f
        d2 = np.array(Bpp) / (self.f**2)
        return idx, vals, d1, d2