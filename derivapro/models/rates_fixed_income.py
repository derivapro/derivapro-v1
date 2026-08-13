import datetime as dt
import math
from dataclasses import dataclass
from typing import Any

from .curve import Curve
from .daycount import DayCount
from .schedule import build_schedule


def parse_curve(tenors_raw: str, rates_raw: str) -> Curve:
    tenors = [float(item.strip()) for item in tenors_raw.split(",") if item.strip()]
    rates = [float(item.strip()) for item in rates_raw.split(",") if item.strip()]
    if len(tenors) != len(rates) or not tenors:
        raise ValueError("Curve tenors and rates must be comma-separated lists with the same nonzero length.")
    return Curve(tenors, rates, comp="cont")


def parse_date(value: str) -> dt.date:
    return dt.date.fromisoformat(value)


def shifted_curve(curve: Curve, shock_bp: float) -> Curve:
    return Curve(curve.t, [rate + shock_bp / 10000.0 for rate in curve.z], curve.comp)


def _normal_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _black_rate_option(forward: float, strike: float, vol: float, expiry: float, option_type: str) -> float:
    forward = max(forward, 1e-8)
    strike = max(strike, 1e-8)
    vol = max(vol, 0.0)
    expiry = max(expiry, 0.0)
    intrinsic = max(forward - strike, 0.0) if option_type == "cap" else max(strike - forward, 0.0)
    if vol == 0.0 or expiry == 0.0:
        return intrinsic
    total_vol = vol * math.sqrt(expiry)
    d1 = (math.log(forward / strike) + 0.5 * total_vol * total_vol) / total_vol
    d2 = d1 - total_vol
    if option_type == "cap":
        return forward * _normal_cdf(d1) - strike * _normal_cdf(d2)
    return strike * _normal_cdf(-d2) - forward * _normal_cdf(-d1)


def _forward_rate(curve: Curve, start_t: float, end_t: float, accrual: float) -> float:
    if accrual <= 0:
        raise ValueError("Accrual period must be positive.")
    return (curve.df(start_t) / curve.df(end_t) - 1.0) / accrual


def _money(value: float) -> str:
    return f"${value:,.2f}"


def _pct(value: float) -> str:
    return f"{value * 100:.4f}%"


@dataclass
class FraTerms:
    valuation_date: dt.date
    start_date: dt.date
    end_date: dt.date
    notional: float
    strike_rate: float
    position: str
    day_count: str
    discount_curve: Curve
    forward_curve: Curve
    scenario_shock_bp: float


def price_fra(terms: FraTerms) -> dict[str, Any]:
    start_t = DayCount.year_frac(terms.valuation_date, terms.start_date, "ACT/365")
    end_t = DayCount.year_frac(terms.valuation_date, terms.end_date, "ACT/365")
    accrual = DayCount.year_frac(terms.start_date, terms.end_date, terms.day_count)
    if start_t < 0 or end_t <= start_t:
        raise ValueError("FRA start/end dates must be after valuation date and ordered correctly.")

    def run(disc: Curve, fwd: Curve, label: str) -> dict[str, Any]:
        forward = _forward_rate(fwd, start_t, end_t, accrual)
        payoff = (forward - terms.strike_rate) * accrual * terms.notional
        if terms.position == "receive_fixed":
            payoff *= -1.0
        pv = payoff * disc.df(end_t)
        return {
            "label": label,
            "forward_rate": forward,
            "discount_factor": disc.df(end_t),
            "pv": pv,
            "payoff": payoff,
        }

    base = run(terms.discount_curve, terms.forward_curve, "Base")
    up = run(
        shifted_curve(terms.discount_curve, terms.scenario_shock_bp),
        shifted_curve(terms.forward_curve, terms.scenario_shock_bp),
        f"+{terms.scenario_shock_bp:.0f} bp",
    )
    down = run(
        shifted_curve(terms.discount_curve, -terms.scenario_shock_bp),
        shifted_curve(terms.forward_curve, -terms.scenario_shock_bp),
        f"-{terms.scenario_shock_bp:.0f} bp",
    )

    return {
        "primary_metrics": [
            {"label": "Present Value", "value": _money(base["pv"])},
            {"label": "Forward Rate", "value": _pct(base["forward_rate"])},
            {"label": "Strike Rate", "value": _pct(terms.strike_rate)},
            {"label": "Discount Factor", "value": f"{base['discount_factor']:.6f}"},
        ],
        "scenarios": [base, up, down],
        "cashflows": [
            {
                "period": f"{terms.start_date.isoformat()} to {terms.end_date.isoformat()}",
                "accrual": accrual,
                "projected_payoff": base["payoff"],
                "discounted_pv": base["pv"],
            }
        ],
        "summary": (
            "FRA PV is calculated from the difference between projected forward rate and contract rate, "
            "discounted on the supplied discount curve."
        ),
    }


@dataclass
class CapFloorTerms:
    valuation_date: dt.date
    start_date: dt.date
    maturity_date: dt.date
    notional: float
    strike_rate: float
    option_type: str
    volatility: float
    payments_per_year: int
    day_count: str
    discount_curve: Curve
    forward_curve: Curve
    scenario_rate_shock_bp: float
    scenario_vol_shock: float


def price_cap_floor(terms: CapFloorTerms) -> dict[str, Any]:
    if terms.maturity_date <= terms.start_date:
        raise ValueError("Maturity date must be after start date.")
    pay_dates = build_schedule(terms.start_date, terms.maturity_date, terms.payments_per_year)
    period_start = terms.start_date
    rows = []
    total_pv = 0.0
    total_vega = 0.0

    def price_with(curve_shift_bp: float, vol_shift: float) -> float:
        disc = shifted_curve(terms.discount_curve, curve_shift_bp)
        fwd = shifted_curve(terms.forward_curve, curve_shift_bp)
        pv = 0.0
        local_start = terms.start_date
        for pay_date in pay_dates:
            start_t = max(DayCount.year_frac(terms.valuation_date, local_start, "ACT/365"), 0.0)
            end_t = max(DayCount.year_frac(terms.valuation_date, pay_date, "ACT/365"), 0.0)
            accrual = DayCount.year_frac(local_start, pay_date, terms.day_count)
            forward = _forward_rate(fwd, start_t, end_t, accrual)
            option_value = _black_rate_option(
                forward,
                terms.strike_rate,
                max(terms.volatility + vol_shift, 0.0),
                start_t,
                terms.option_type,
            )
            pv += terms.notional * accrual * disc.df(end_t) * option_value
            local_start = pay_date
        return pv

    for pay_date in pay_dates:
        start_t = max(DayCount.year_frac(terms.valuation_date, period_start, "ACT/365"), 0.0)
        end_t = max(DayCount.year_frac(terms.valuation_date, pay_date, "ACT/365"), 0.0)
        accrual = DayCount.year_frac(period_start, pay_date, terms.day_count)
        forward = _forward_rate(terms.forward_curve, start_t, end_t, accrual)
        unit_value = _black_rate_option(forward, terms.strike_rate, terms.volatility, start_t, terms.option_type)
        caplet_pv = terms.notional * accrual * terms.discount_curve.df(end_t) * unit_value
        bumped_value = _black_rate_option(forward, terms.strike_rate, terms.volatility + 0.0001, start_t, terms.option_type)
        vega = terms.notional * accrual * terms.discount_curve.df(end_t) * (bumped_value - unit_value) / 0.0001
        total_pv += caplet_pv
        total_vega += vega
        rows.append(
            {
                "period": f"{period_start.isoformat()} to {pay_date.isoformat()}",
                "forward_rate": forward,
                "accrual": accrual,
                "discount_factor": terms.discount_curve.df(end_t),
                "pv": caplet_pv,
            }
        )
        period_start = pay_date

    return {
        "primary_metrics": [
            {"label": "Present Value", "value": _money(total_pv)},
            {"label": "Strike Rate", "value": _pct(terms.strike_rate)},
            {"label": "Volatility", "value": _pct(terms.volatility)},
            {"label": "Vega / 100 Vol Pts", "value": _money(total_vega)},
        ],
        "scenarios": [
            {"label": "Base", "pv": total_pv},
            {"label": f"Rates +{terms.scenario_rate_shock_bp:.0f} bp", "pv": price_with(terms.scenario_rate_shock_bp, 0.0)},
            {"label": f"Rates -{terms.scenario_rate_shock_bp:.0f} bp", "pv": price_with(-terms.scenario_rate_shock_bp, 0.0)},
            {"label": f"Vol +{terms.scenario_vol_shock * 100:.1f} pp", "pv": price_with(0.0, terms.scenario_vol_shock)},
            {"label": f"Vol -{terms.scenario_vol_shock * 100:.1f} pp", "pv": price_with(0.0, -terms.scenario_vol_shock)},
        ],
        "cashflows": rows,
        "summary": (
            "Cap/floor PV is the sum of Black caplet/floorlet values using the supplied forward curve, "
            "discount curve, strike, volatility, and payment schedule."
        ),
    }


@dataclass
class CallableBondTerms:
    valuation_date: dt.date
    maturity_date: dt.date
    notional: float
    coupon_rate: float
    payments_per_year: int
    day_count: str
    discount_curve: Curve
    option_type: str
    call_or_put_price: float
    first_exercise_year: float
    short_rate_volatility: float
    scenario_shock_bp: float


def price_callable_putable_bond(terms: CallableBondTerms) -> dict[str, Any]:
    if terms.maturity_date <= terms.valuation_date:
        raise ValueError("Maturity date must be after valuation date.")
    pay_dates = build_schedule(terms.valuation_date, terms.maturity_date, terms.payments_per_year)
    maturity_years = max(DayCount.year_frac(terms.valuation_date, terms.maturity_date, "ACT/365"), 1e-8)
    steps = max(1, int(round(maturity_years * terms.payments_per_year)))
    dt_step = maturity_years / steps
    flat_rate = -math.log(terms.discount_curve.df(maturity_years)) / maturity_years
    exercise_step = max(1, int(math.ceil(terms.first_exercise_year / dt_step)))
    coupon = terms.notional * terms.coupon_rate / terms.payments_per_year
    redemption = terms.notional

    cashflow_by_step = {step: coupon for step in range(1, steps + 1)}
    cashflow_by_step[steps] = cashflow_by_step.get(steps, 0.0) + redemption

    def straight_pv(curve: Curve) -> float:
        pv = 0.0
        previous = terms.valuation_date
        for pay_date in pay_dates:
            t = DayCount.year_frac(terms.valuation_date, pay_date, "ACT/365")
            accrual = DayCount.year_frac(previous, pay_date, terms.day_count)
            pv += terms.notional * terms.coupon_rate * accrual * curve.df(t)
            previous = pay_date
        pv += terms.notional * curve.df(maturity_years)
        return pv

    def lattice_value(curve_shift_bp: float = 0.0) -> float:
        r0 = flat_rate + curve_shift_bp / 10000.0
        values = [cashflow_by_step[steps] for _ in range(steps + 1)]
        for step in range(steps - 1, -1, -1):
            next_values = []
            for node in range(step + 1):
                short_rate = max(r0 + (2 * node - step) * terms.short_rate_volatility * math.sqrt(dt_step), -0.02)
                continuation = 0.5 * (values[node] + values[node + 1]) * math.exp(-short_rate * dt_step)
                continuation += cashflow_by_step.get(step, 0.0)
                if step >= exercise_step and step < steps:
                    exercise_value = terms.call_or_put_price / 100.0 * terms.notional + cashflow_by_step.get(step, 0.0)
                    if terms.option_type == "callable":
                        continuation = min(continuation, exercise_value)
                    elif terms.option_type == "putable":
                        continuation = max(continuation, exercise_value)
                next_values.append(continuation)
            values = next_values
        return values[0]

    straight = straight_pv(terms.discount_curve)
    option_adjusted = lattice_value(0.0)
    option_value = straight - option_adjusted if terms.option_type == "callable" else option_adjusted - straight
    up_value = lattice_value(terms.scenario_shock_bp)
    down_value = lattice_value(-terms.scenario_shock_bp)
    effective_duration = (down_value - up_value) / (2 * option_adjusted * terms.scenario_shock_bp / 10000.0)

    return {
        "primary_metrics": [
            {"label": "Option-Adjusted PV", "value": _money(option_adjusted)},
            {"label": "Straight Bond PV", "value": _money(straight)},
            {"label": "Embedded Option Value", "value": _money(option_value)},
            {"label": "Effective Duration", "value": f"{effective_duration:.4f}"},
        ],
        "scenarios": [
            {"label": "Base", "pv": option_adjusted},
            {"label": f"Rates +{terms.scenario_shock_bp:.0f} bp", "pv": up_value},
            {"label": f"Rates -{terms.scenario_shock_bp:.0f} bp", "pv": down_value},
        ],
        "cashflows": [
            {
                "period": f"Step {step}",
                "coupon": cashflow_by_step.get(step, 0.0) - (redemption if step == steps else 0.0),
                "principal": redemption if step == steps else 0.0,
                "exercise_allowed": step >= exercise_step and step < steps,
            }
            for step in range(1, steps + 1)
        ],
        "summary": (
            "Callable/putable bond PV uses a recombining short-rate lattice for the embedded exercise feature "
            "and a deterministic curve PV for the straight-bond benchmark."
        ),
    }
