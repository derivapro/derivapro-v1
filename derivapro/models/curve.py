# derivapro/models/curve.py
import bisect
import math
from typing import List


class Curve:
    """
    One-dimensional interest-rate curve on T (years).
    Rates are continuously-compounded by default.
    """

    def __init__(
        self,
        t: List[float],
        z: List[float],
        comp: str = "cont",
        interpolation: str = "linear_zero",
    ):
        assert len(t) == len(z) and len(t) > 0
        pts = sorted(zip(t, z))
        self.t = [p[0] for p in pts]
        self.z = [p[1] for p in pts]
        self.comp = comp
        self.interpolation = self._normalize_interpolation(interpolation)
        self._spline_second_derivatives = self._natural_cubic_second_derivatives()

    @staticmethod
    def _normalize_interpolation(interpolation: str) -> str:
        value = (interpolation or "linear_zero").strip().lower().replace("-", "_")
        aliases = {
            "linear": "linear_zero",
            "linear_spot": "linear_zero",
            "spot_linear": "linear_zero",
            "zero": "linear_zero",
            "zero_rate": "linear_zero",
            "zero_rates": "linear_zero",
            "linear_df": "linear_discount",
            "discount": "linear_discount",
            "discount_factor": "linear_discount",
            "log_df": "exponential",
            "log_discount": "exponential",
            "exponential_discount": "exponential",
            "cubic": "cubic_spline",
            "spline": "cubic_spline",
        }
        return aliases.get(value, value)

    def _natural_cubic_second_derivatives(self) -> List[float]:
        n = len(self.t)
        if n < 3:
            return [0.0] * n

        second = [0.0] * n
        u = [0.0] * (n - 1)
        for i in range(1, n - 1):
            span = self.t[i + 1] - self.t[i - 1]
            if span <= 0:
                continue
            sig = (self.t[i] - self.t[i - 1]) / span
            p = sig * second[i - 1] + 2.0
            second[i] = (sig - 1.0) / p
            left_slope = (self.z[i] - self.z[i - 1]) / (self.t[i] - self.t[i - 1])
            right_slope = (self.z[i + 1] - self.z[i]) / (self.t[i + 1] - self.t[i])
            u[i] = (6.0 * (right_slope - left_slope) / span - sig * u[i - 1]) / p

        for k in range(n - 2, -1, -1):
            second[k] = second[k] * second[k + 1] + u[k]
        return second

    def _z(self, T: float) -> float:
        if T <= self.t[0]:
            return self.z[0]
        if T >= self.t[-1]:
            return self.z[-1]
        i = bisect.bisect_left(self.t, T)
        t0, t1 = self.t[i - 1], self.t[i]
        z0, z1 = self.z[i - 1], self.z[i]
        w = (T - t0) / (t1 - t0)
        if self.interpolation == "cubic_spline" and len(self.t) >= 3:
            a = (t1 - T) / (t1 - t0)
            b = (T - t0) / (t1 - t0)
            return (
                a * z0
                + b * z1
                + ((a**3 - a) * self._spline_second_derivatives[i - 1] + (b**3 - b) * self._spline_second_derivatives[i])
                * (t1 - t0) ** 2
                / 6.0
            )
        return z0 + w*(z1 - z0)

    def df(self, T: float) -> float:
        if T <= 0:
            return 1.0
        if self.interpolation in {"exponential", "linear_discount"} and len(self.t) > 1:
            return self._interpolated_df(T)
        zT = self._z(T)
        if self.comp == "cont":
            return math.exp(-zT*T)
        elif self.comp == "simple":
            return 1.0/(1.0 + zT*T)
        else:                       # annual comp
            return (1.0 + zT) ** (-T)

    def _node_df(self, index: int) -> float:
        T = self.t[index]
        zT = self.z[index]
        if self.comp == "cont":
            return math.exp(-zT * T)
        if self.comp == "simple":
            return 1.0 / (1.0 + zT * T)
        return (1.0 + zT) ** (-T)

    def _interpolated_df(self, T: float) -> float:
        if T <= self.t[0]:
            zT = self.z[0]
            if self.comp == "cont":
                return math.exp(-zT * T)
            if self.comp == "simple":
                return 1.0 / (1.0 + zT * T)
            return (1.0 + zT) ** (-T)
        if T >= self.t[-1]:
            zT = self.z[-1]
            if self.comp == "cont":
                return math.exp(-zT * T)
            if self.comp == "simple":
                return 1.0 / (1.0 + zT * T)
            return (1.0 + zT) ** (-T)
        i = bisect.bisect_left(self.t, T)
        t0, t1 = self.t[i - 1], self.t[i]
        df0, df1 = self._node_df(i - 1), self._node_df(i)
        w = (T - t0) / (t1 - t0)
        if self.interpolation == "exponential":
            return math.exp(math.log(df0) + w * (math.log(df1) - math.log(df0)))
        return df0 + w * (df1 - df0)
