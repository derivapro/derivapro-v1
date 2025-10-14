# derivapro/models/swaptions.py
import math, datetime as dt
from typing import Tuple, Dict, Any, Optional
import numpy as np

from .curve import Curve
from .daycount import DayCount
from .schedule import build_schedule
from .vol_surface import VolSurface

SQRT2 = math.sqrt(2.0)
INV_SQRT_2PI = 1.0 / math.sqrt(2.0 * math.pi)

def _N(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / SQRT2))

def _n(x: float) -> float:
    return INV_SQRT_2PI * math.exp(-0.5 * x * x)

def annuity_and_forward(
    start: dt.date,
    end: dt.date,
    pay_freq_per_year: int,
    dc_fixed: str,
    disc_curve: Curve,
    valuation_date: Optional[dt.date] = None
) -> Tuple[float, float]:
    """
    Returns (A0, F0) where A0 is the fixed-leg annuity and F0 is the par swap rate,
    computed with discount factors from disc_curve as-of valuation_date.
    """
    pays = build_schedule(start, end, pay_freq_per_year)
    val_date = valuation_date or dt.date.today()

    # accruals & discount factors to each payment date
    accr: list[float] = []
    D: list[float] = []
    prev = start
    for d in pays:
        accr.append(DayCount.year_frac(prev, d, dc_fixed))
        prev = d
        T = DayCount.year_frac(val_date, d, "ACT/365")
        D.append(disc_curve.df(T))

    A0 = sum(a * df for a, df in zip(accr, D))

    # float leg PV via forward rates implied from discount factors
    pv_flt, D_prev = 0.0, 1.0
    for a_f, Df in zip(accr, D):
        # simple-forward over the period
        F = (D_prev / Df - 1.0) / max(1e-12, a_f)
        pv_flt += F * a_f * Df
        D_prev = Df

    F0 = pv_flt / max(1e-12, A0)
    return A0, F0


def price_european(
    *,
    method: str,            # "black" or "bachelier"
    is_payer: bool,
    strike: float,
    option_expiry: dt.date,
    swap_start: dt.date,
    swap_end: dt.date,
    pay_freq_per_year: int,
    dc_fixed: str,
    disc_curve: Curve,
    vol_surf: VolSurface,
    notional: float = 1.0,
    valuation_date: Optional[dt.date] = None
) -> Dict[str, Any]:
    """
    European swaption pricer (analytic Black/Bachelier).
    All times are measured from valuation_date (defaults to today).
    """
    val_date = valuation_date or dt.date.today()

    # time to option expiry
    T = DayCount.year_frac(val_date, option_expiry, "ACT/365")
    if T <= 0:
        raise ValueError("Option expiry must be after valuation_date.")

    # annuity and forward swap rate as-of valuation date
    A0, F0 = annuity_and_forward(
        swap_start, swap_end, pay_freq_per_year, dc_fixed, disc_curve, valuation_date=val_date
    )

    # tenor for vol lookup is underlying swap length
    tenor_y = DayCount.year_frac(swap_start, swap_end, "ACT/365")
    vol = vol_surf.get(T, tenor_y)

    if method == "black":
        sigT = vol * math.sqrt(T)
        if sigT <= 0:
            raise ValueError("Vol must be positive for Black.")
        d1 = (math.log(max(1e-18, F0) / max(1e-18, strike)) + 0.5 * sigT * sigT) / sigT
        d2 = d1 - sigT
        pv = A0 * (F0 * _N(d1) - strike * _N(d2)) if is_payer else A0 * (strike * _N(-d2) - F0 * _N(-d1))
        vega = A0 * math.sqrt(T) * F0 * _n(d1)
        delta_F = A0 * (_N(d1) if is_payer else (_N(d1) - 1.0))
        gamma_F = A0 * _n(d1) / (max(1e-18, F0) * sigT)
        model_name = "black"

    elif method == "bachelier":
        sigT = vol * math.sqrt(T)
        if sigT <= 0:
            raise ValueError("Vol must be positive for Bachelier.")
        d = (F0 - strike) / sigT
        call = (F0 - strike) * _N(d) + sigT * _n(d)
        put  = (strike - F0) * _N(-d) + sigT * _n(d)
        pv = A0 * (call if is_payer else put)
        vega = A0 * math.sqrt(T) * _n(d)
        delta_F = A0 * (_N(d) if is_payer else (_N(d) - 1.0))
        gamma_F = A0 * _n(d) / sigT
        model_name = "bachelier"

    else:
        raise ValueError("Unsupported method. Use 'black' or 'bachelier'.")

    return {
        "model": model_name,
        "pv": float(pv) * notional,
        "vega": float(vega) * notional,
        "delta_F": float(delta_F) * notional,
        "gamma_F": float(gamma_F) * notional,
        "A0": float(A0) * notional,
        "F0": float(F0),
        "strike": float(strike),
        "expiry_years": float(T),
        "vol_used": float(vol)
    }


# ----- Hull–White 1F Monte Carlo -----
def _inst_fwd_from_curve(curve: Curve, t: float, h: float = 1e-4) -> float:
    t0 = max(1e-8, t - h); t1 = t + h
    p0, p1 = curve.df(t0), curve.df(t1)
    return -(math.log(max(1e-18, p1)) - math.log(max(1e-18, p0))) / (t1 - t0)

def _theta_HW(curve: Curve, a: float, sigma: float, t: float) -> float:
    f0 = _inst_fwd_from_curve(curve, t)
    h = 1e-3
    f0m, f0p = _inst_fwd_from_curve(curve, max(1e-8, t - h)), _inst_fwd_from_curve(curve, t + h)
    dft = (f0p - f0m) / (2 * h)
    adj = (sigma * sigma) / (2.0 * a) * (1.0 - math.exp(-2.0 * a * t))
    return dft + a * f0 + adj

def price_hw1f_mc(
    *,
    disc_curve: Curve,
    start: dt.date,
    end: dt.date,
    expiry: dt.date,
    strike: float,
    is_payer: bool,
    pay_freq_per_year: int,
    dc_fixed: str,
    dc_float: str,
    a: float,
    sigma: float,
    n_paths: int,
    steps_per_year: int,
    notional: float = 1.0,
    valuation_date: Optional[dt.date] = None
) -> Dict[str, Any]:
    """
    Hull–White 1F MC swaption pricer.
    Times are measured from valuation_date (defaults to today).
    """
    pays = build_schedule(start, end, pay_freq_per_year)
    if not pays:
        raise ValueError("Empty payment schedule.")

    # accruals per leg period
    accr_fix, accr_flt, prev = [], [], start
    for d in pays:
        accr_fix.append(DayCount.year_frac(prev, d, dc_fixed))
        accr_flt.append(DayCount.year_frac(prev, d, dc_float))
        prev = d

    val_date = valuation_date or dt.date.today()

    # map key times from valuation_date
    T = DayCount.year_frac(val_date, expiry, "ACT/365")
    times_pay = [DayCount.year_frac(val_date, d, "ACT/365") for d in pays]
    t_last = times_pay[-1]
    if t_last <= 0 or T <= 0:
        raise ValueError("Non-positive times detected (ensure expiry and payment dates are after valuation_date).")

    # simulation grid
    steps = max(1, int(math.ceil(t_last * steps_per_year)))
    dtau = t_last / steps
    grid = np.linspace(0.0, t_last, steps + 1)

    def _nearest_idx(t: float) -> int:
        return min(steps, max(0, int(round(t / dtau))))

    idx_T = _nearest_idx(T)
    idx_pay = [_nearest_idx(tp) for tp in times_pay]

    # theta(t) from the discount curve
    theta = np.array([_theta_HW(disc_curve, a, sigma, t) for t in grid], dtype=float)

    # simulate short rate and its time-integral
    r0 = _inst_fwd_from_curve(disc_curve, 1e-4)
    r = np.full(n_paths, r0, dtype=float)
    I = np.zeros(n_paths, dtype=float)
    I_hist = np.zeros((len(idx_pay), n_paths), dtype=float)
    I_T = np.zeros(n_paths, dtype=float)

    rng = np.random.default_rng()
    pay_marks = set(idx_pay)

    for k in range(1, steps + 1):
        z = rng.standard_normal(n_paths)
        r += a * (theta[k - 1] - r) * dtau + sigma * np.sqrt(dtau) * z
        I += r * dtau
        if k == idx_T:
            I_T[:] = I
        if k in pay_marks:
            j = idx_pay.index(k)
            I_hist[j, :] = I

    # P(T, t_i) and P(0, T)
    P0T = np.exp(-I_T)
    PTti = [np.exp(-(I_hist[j, :] - I_T)) for j in range(len(pays))]

    # swap PV at T along each path
    PT_prev = np.ones(n_paths, dtype=float)
    pv_flt_T = np.zeros(n_paths, dtype=float)
    pv_fix_T = np.zeros(n_paths, dtype=float)
    for i in range(len(pays)):
        PT_i = PTti[i]
        af_f = max(1e-12, float(accr_flt[i]))
        fwd_i = (PT_prev / PT_i - 1.0) / af_f
        pv_flt_T += (fwd_i * af_f) * PT_i
        pv_fix_T += (strike * float(accr_fix[i])) * PT_i
        PT_prev = PT_i

    swap_T = pv_flt_T - pv_fix_T
    if not is_payer:
        swap_T = -swap_T
    payoff_T = np.maximum(swap_T, 0.0)

    # discount payoff back to 0
    pv0_paths = payoff_T * P0T
    pv0 = float(np.mean(pv0_paths)) * notional
    stderr = float(np.std(pv0_paths, ddof=1) / math.sqrt(n_paths)) * notional

    A0, F0 = annuity_and_forward(start, end, pay_freq_per_year, dc_fixed, disc_curve, valuation_date=val_date)
    return {
        "model": "hull-white-1f (mc)",
        "pv": pv0,
        "stderr": stderr,
        "A0": A0 * notional,
        "F0": F0
    }
