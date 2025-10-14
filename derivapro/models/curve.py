# derivapro/models/curve.py
import bisect, math
from typing import List

class Curve:
    """
    Piecewise-linear zero-rate curve on T (years).
    Rates are continuously-compounded by default.
    """
    def __init__(self, t: List[float], z: List[float], comp: str = "cont"):
        assert len(t) == len(z) and len(t) > 0
        pts = sorted(zip(t, z))
        self.t = [p[0] for p in pts]
        self.z = [p[1] for p in pts]
        self.comp = comp

    def _z(self, T: float) -> float:
        if T <= self.t[0]: return self.z[0]
        if T >= self.t[-1]: return self.z[-1]
        i = bisect.bisect_left(self.t, T)
        t0,t1 = self.t[i-1], self.t[i]
        z0,z1 = self.z[i-1], self.z[i]
        w = (T - t0) / (t1 - t0)
        return z0 + w*(z1 - z0)

    def df(self, T: float) -> float:
        if T <= 0: return 1.0
        zT = self._z(T)
        if self.comp == "cont":
            return math.exp(-zT*T)
        elif self.comp == "simple":
            return 1.0/(1.0 + zT*T)
        else:                       # annual comp
            return (1.0 + zT) ** (-T)
