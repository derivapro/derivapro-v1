# derivapro/models/swaps.py
import pandas as pd
import os
import datetime as dt
from typing import Dict, Any
from .curve import Curve
from .daycount import DayCount
from .schedule import build_schedule

def _read_curve_csv(path: str, col_t="tenor_years", col_r="zero_rate") -> Curve:
    if not path or not os.path.exists(path):
        raise FileNotFoundError(f"Curve file not found: {path}")
    df = pd.read_csv(path)
    if col_t not in df or col_r not in df:
        raise ValueError(f"Curve CSV missing required columns: {col_t}, {col_r}")
    df = df[[col_t, col_r]].dropna().sort_values(col_t)
    return Curve(df[col_t].tolist(), df[col_r].tolist(), comp="cont")

def _read_curve_file(file_obj, col_t="tenor_years", col_r="zero_rate") -> Curve:
    df = pd.read_csv(file_obj)
    df = df[[col_t, col_r]].dropna().sort_values(col_t)
    return Curve(df[col_t].tolist(), df[col_r].tolist(), comp="cont")

def price_plain_swap(
    start: dt.date,
    end: dt.date,
    notional: float,
    fixed_rate: float,
    side: str,                # "pay_fixed" or "receive_fixed"
    pay_freq_per_year: int,
    dc_fixed: str,
    dc_float: str,
    disc_curve: Curve,
    fwd_curve: Curve
) -> Dict[str, Any]:
    # schedule + accruals
    pays = build_schedule(start, end, pay_freq_per_year)
    accr_fix, accr_flt, prev = [], [], start
    for d in pays:
        accr_fix.append(DayCount.year_frac(prev, d, dc_fixed))
        accr_flt.append(DayCount.year_frac(prev, d, dc_float))
        prev = d

    # time-to-pay from today
    today = dt.date.today()
    Ts = [ DayCount.year_frac(today, d, "ACT/365") for d in pays ]
    D = [ disc_curve.df(T) for T in Ts ]

    # annuity
    A = sum(af * df for af, df in zip(accr_fix, D))

    # float leg via discount factors (single-curve)
    pv_flt, D_prev = 0.0, 1.0
    for af, Df in zip(accr_flt, D):
        F = (D_prev / Df - 1.0) / max(1e-12, af)
        pv_flt += F * af * Df
        D_prev = Df

    par_rate = pv_flt / max(1e-12, A)
    pv_fixed = fixed_rate * A
    pv_unit  = pv_flt - pv_fixed
    if side == "receive_fixed":
        pv_unit = -pv_unit

    legs = pd.DataFrame({
        "pay_date": pays,
        "T_years": Ts,
        "DF": D,
        "accr_fix": accr_fix,
        "accr_flt": accr_flt
    })
    return {
        "par_rate": par_rate,
        "npv": pv_unit * notional,
        "annuity": A * notional,
        "legs": legs
    }
