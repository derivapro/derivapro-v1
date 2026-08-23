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


def _parse_dated_amounts(raw: str | None, value_type: str = "amount") -> list[tuple[dt.date, float]]:
    """Parse comma-separated date:value entries used by first-pass schedule tables."""
    if not raw:
        return []
    rows = []
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        if ":" not in item:
            raise ValueError(f"Schedule entry '{item}' must use YYYY-MM-DD:value format.")
        date_raw, value_raw = item.split(":", 1)
        value = float(value_raw.strip())
        if value_type == "pct" and abs(value) > 1.0:
            value = value / 100.0
        rows.append((parse_date(date_raw.strip()), value))
    return sorted(rows, key=lambda row: row[0])


def _value_effective_on(schedule: list[tuple[dt.date, float]], as_of: dt.date, default: float) -> float:
    value = default
    for effective_date, candidate in schedule:
        if effective_date <= as_of:
            value = candidate
        else:
            break
    return value


def _principal_due_between(schedule: list[tuple[dt.date, float]], start: dt.date, end: dt.date, original_notional: float) -> float:
    return sum(pct * original_notional for event_date, pct in schedule if start < event_date <= end)


def _fixed_payments_between(schedule: list[tuple[dt.date, float]], start: dt.date, end: dt.date) -> float:
    return sum(amount for event_date, amount in schedule if start < event_date <= end)


def _solve_yield_from_cashflows(
    valuation_date: dt.date,
    cashflows: list[tuple[dt.date, float]],
    target_price: float,
    day_count: str = "ACT/365",
) -> float | None:
    if target_price <= 0 or not cashflows:
        return None

    def pv_at(rate: float) -> float:
        total = 0.0
        for pay_date, amount in cashflows:
            t = max(DayCount.year_frac(valuation_date, pay_date, day_count), 0.0)
            total += amount / ((1.0 + rate) ** t)
        return total

    low, high = -0.95, 1.00
    pv_low = pv_at(low) - target_price
    pv_high = pv_at(high) - target_price
    if pv_low * pv_high > 0:
        return None
    for _ in range(100):
        mid = (low + high) / 2.0
        pv_mid = pv_at(mid) - target_price
        if abs(pv_mid) < 1e-8:
            return mid
        if pv_low * pv_mid <= 0:
            high = mid
            pv_high = pv_mid
        else:
            low = mid
            pv_low = pv_mid
    return (low + high) / 2.0


def _rate_from_curve(curve: Curve, maturity_years: float) -> float:
    maturity_years = max(maturity_years, 1e-8)
    return -math.log(curve.df(maturity_years)) / maturity_years


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
    market_clean_price_pct: float
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
    market_price = terms.notional * terms.market_clean_price_pct / 100.0
    yield_rows = []
    for step, pay_date in enumerate(pay_dates, start=1):
        if step < exercise_step or step >= steps:
            continue
        call_or_put_cashflows = []
        for idx, candidate_date in enumerate(pay_dates[:step], start=1):
            amount = coupon
            if idx == step:
                amount += terms.call_or_put_price / 100.0 * terms.notional
            call_or_put_cashflows.append((candidate_date, amount))
        exercise_yield = _solve_yield_from_cashflows(
            terms.valuation_date,
            call_or_put_cashflows,
            market_price,
            terms.day_count,
        )
        if exercise_yield is not None:
            yield_rows.append(
                {
                    "period": pay_date.isoformat(),
                    "yield": exercise_yield,
                    "exercise_price": terms.call_or_put_price,
                    "cashflow_count": len(call_or_put_cashflows),
                }
            )
    maturity_cashflows = []
    previous = terms.valuation_date
    for pay_date in pay_dates:
        accrual = DayCount.year_frac(previous, pay_date, terms.day_count)
        amount = terms.notional * terms.coupon_rate * accrual
        if pay_date == pay_dates[-1]:
            amount += terms.notional
        maturity_cashflows.append((pay_date, amount))
        previous = pay_date
    yield_to_maturity = _solve_yield_from_cashflows(
        terms.valuation_date,
        maturity_cashflows,
        market_price,
        terms.day_count,
    )
    if yield_to_maturity is not None:
        yield_rows.append(
            {
                "period": terms.maturity_date.isoformat(),
                "yield": yield_to_maturity,
                "exercise_price": 100.0,
                "cashflow_count": len(maturity_cashflows),
            }
        )
    if yield_rows:
        yields = [row["yield"] for row in yield_rows]
        yield_to_worst = min(yields)
        yield_to_best = max(yields)
    else:
        yield_to_worst = None
        yield_to_best = None

    return {
        "primary_metrics": [
            {"label": "Option-Adjusted PV", "value": _money(option_adjusted)},
            {"label": "Straight Bond PV", "value": _money(straight)},
            {"label": "Embedded Option Value", "value": _money(option_value)},
            {"label": "Effective Duration", "value": f"{effective_duration:.4f}"},
            {"label": "Yield to Worst", "value": _pct(yield_to_worst) if yield_to_worst is not None else "n/a"},
            {"label": "Yield to Best", "value": _pct(yield_to_best) if yield_to_best is not None else "n/a"},
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
        "diagnostics": yield_rows,
        "summary": (
            "Callable/putable bond PV uses a recombining short-rate lattice for the embedded exercise feature "
            "and a deterministic curve PV for the straight-bond benchmark. Yield-to-best/worst is calculated "
            "from maturity and eligible exercise-date cash-flow cases using the supplied market clean-price reference."
        ),
    }


@dataclass
class GenericBondTerms:
    valuation_date: dt.date
    maturity_date: dt.date
    notional: float
    coupon_rate: float
    payments_per_year: int
    day_count: str
    discount_curve: Curve
    market_clean_price_pct: float
    scenario_shock_bp: float
    coupon_schedule: str = ""
    principal_schedule: str = ""
    fixed_payment_schedule: str = ""
    amortization_style: str = "bullet"
    redemption_pct: float = 100.0


def _generic_bond_run(terms: GenericBondTerms, curve: Curve, label: str) -> dict[str, Any]:
    if terms.maturity_date <= terms.valuation_date:
        raise ValueError("Maturity date must be after valuation date.")
    if terms.notional <= 0:
        raise ValueError("Notional must be positive.")

    pay_dates = build_schedule(terms.valuation_date, terms.maturity_date, terms.payments_per_year)
    coupon_schedule = _parse_dated_amounts(terms.coupon_schedule, "pct")
    principal_schedule = _parse_dated_amounts(terms.principal_schedule, "pct")
    fixed_schedule = _parse_dated_amounts(terms.fixed_payment_schedule, "amount")

    previous = terms.valuation_date
    outstanding = terms.notional
    total_pv = 0.0
    weighted_time_pv = 0.0
    convexity_numerator = 0.0
    rows = []
    cashflows_for_yield = []

    for idx, pay_date in enumerate(pay_dates, start=1):
        t = DayCount.year_frac(terms.valuation_date, pay_date, "ACT/365")
        accrual = DayCount.year_frac(previous, pay_date, terms.day_count)
        coupon_rate = _value_effective_on(coupon_schedule, pay_date, terms.coupon_rate)
        coupon = outstanding * coupon_rate * accrual

        principal = 0.0
        if terms.amortization_style == "straight_line":
            principal = terms.notional / len(pay_dates)
        elif terms.amortization_style == "sinking_schedule":
            principal = _principal_due_between(principal_schedule, previous, pay_date, terms.notional)
        elif pay_date == pay_dates[-1]:
            principal = terms.notional * terms.redemption_pct / 100.0

        if pay_date == pay_dates[-1] and terms.amortization_style in {"straight_line", "sinking_schedule"}:
            principal = min(max(outstanding, 0.0), max(principal, outstanding))
        principal = min(max(principal, 0.0), max(outstanding, 0.0))
        fixed_payment = _fixed_payments_between(fixed_schedule, previous, pay_date)
        cashflow = coupon + principal + fixed_payment
        df = curve.df(t)
        pv = cashflow * df
        total_pv += pv
        weighted_time_pv += t * pv
        convexity_numerator += t * (t + 1.0) * pv
        cashflows_for_yield.append((pay_date, cashflow))
        rows.append(
            {
                "period": f"{previous.isoformat()} to {pay_date.isoformat()}",
                "coupon_rate": coupon_rate,
                "opening_notional": outstanding,
                "coupon": coupon,
                "principal": principal,
                "fixed_payment": fixed_payment,
                "discount_factor": df,
                "discounted_pv": pv,
            }
        )
        outstanding -= principal
        previous = pay_date

    market_price = terms.notional * terms.market_clean_price_pct / 100.0
    ytm = _solve_yield_from_cashflows(
        terms.valuation_date,
        cashflows_for_yield,
        market_price,
        terms.day_count,
    )
    duration = weighted_time_pv / max(total_pv, 1e-12)
    convexity = convexity_numerator / max(total_pv, 1e-12)
    dv01 = duration * total_pv / 10000.0
    return {
        "label": label,
        "pv": total_pv,
        "model_price_pct": total_pv / terms.notional * 100.0,
        "market_price": market_price,
        "ytm": ytm,
        "duration": duration,
        "convexity": convexity,
        "dv01": dv01,
        "cashflows": rows,
    }


def price_generic_bond(terms: GenericBondTerms) -> dict[str, Any]:
    base = _generic_bond_run(terms, terms.discount_curve, "Base")
    up = _generic_bond_run(terms, shifted_curve(terms.discount_curve, terms.scenario_shock_bp), f"Rates +{terms.scenario_shock_bp:.0f} bp")
    down = _generic_bond_run(terms, shifted_curve(terms.discount_curve, -terms.scenario_shock_bp), f"Rates -{terms.scenario_shock_bp:.0f} bp")

    return {
        "primary_metrics": [
            {"label": "Model PV", "value": _money(base["pv"])},
            {"label": "Model Price", "value": f"{base['model_price_pct']:.4f}%"},
            {"label": "Yield to Maturity", "value": _pct(base["ytm"]) if base["ytm"] is not None else "n/a"},
            {"label": "Modified Duration", "value": f"{base['duration']:.4f}"},
            {"label": "DV01", "value": _money(base["dv01"])},
            {"label": "Convexity", "value": f"{base['convexity']:.4f}"},
        ],
        "scenarios": [base, up, down],
        "cashflows": base["cashflows"],
        "summary": (
            "Bond PV is calculated as the discounted value of generated coupon, principal, sinking, and fixed-payment "
            "cash flows. Yield is solved against the supplied market clean-price reference."
        ),
    }


@dataclass
class BondSeriesTerms:
    valuation_date: dt.date
    series_table: str
    day_count: str
    discount_curve: Curve
    scenario_shock_bp: float


def _parse_bond_series_table(raw: str) -> list[dict[str, Any]]:
    if not raw.strip():
        raise ValueError("Bond series table is required.")
    bonds = []
    for item in raw.split(";"):
        item = item.strip()
        if not item:
            continue
        parts = [part.strip() for part in item.split("|")]
        if len(parts) != 4:
            raise ValueError("Each bond series row must use maturity|principal|coupon_rate|redemption_pct.")
        maturity, principal, coupon_rate, redemption_pct = parts
        coupon = float(coupon_rate)
        if abs(coupon) > 1.0:
            coupon = coupon / 100.0
        bonds.append(
            {
                "maturity_date": parse_date(maturity),
                "principal": float(principal),
                "coupon_rate": coupon,
                "redemption_pct": float(redemption_pct),
            }
        )
    return bonds


def _bond_series_run(terms: BondSeriesTerms, curve: Curve, label: str) -> dict[str, Any]:
    bonds = _parse_bond_series_table(terms.series_table)
    total_pv = 0.0
    rows = []
    cashflow_map: dict[dt.date, float] = {}
    for idx, bond in enumerate(bonds, start=1):
        generic_terms = GenericBondTerms(
            valuation_date=terms.valuation_date,
            maturity_date=bond["maturity_date"],
            notional=bond["principal"],
            coupon_rate=bond["coupon_rate"],
            payments_per_year=2,
            day_count=terms.day_count,
            discount_curve=curve,
            market_clean_price_pct=100.0,
            scenario_shock_bp=terms.scenario_shock_bp,
            redemption_pct=bond["redemption_pct"],
        )
        run = _generic_bond_run(generic_terms, curve, f"Bond {idx}")
        total_pv += run["pv"]
        rows.append(
            {
                "period": f"Series bond {idx}",
                "maturity_date": bond["maturity_date"].isoformat(),
                "principal": bond["principal"],
                "coupon_rate": bond["coupon_rate"],
                "model_pv": run["pv"],
                "model_price_pct": run["model_price_pct"],
            }
        )
        for cashflow in run["cashflows"]:
            end_date = parse_date(cashflow["period"].split(" to ")[1])
            cashflow_map[end_date] = cashflow_map.get(end_date, 0.0) + cashflow["coupon"] + cashflow["principal"]

    aggregate_cashflows = sorted(cashflow_map.items())
    aggregate_yield = _solve_yield_from_cashflows(
        terms.valuation_date,
        aggregate_cashflows,
        sum(bond["principal"] for bond in bonds),
        terms.day_count,
    )
    maturity_years = [
        DayCount.year_frac(terms.valuation_date, bond["maturity_date"], "ACT/365") for bond in bonds
    ]
    weighted_average_maturity = sum(
        year * bond["principal"] for year, bond in zip(maturity_years, bonds)
    ) / max(sum(bond["principal"] for bond in bonds), 1e-12)
    return {
        "label": label,
        "pv": total_pv,
        "aggregate_yield": aggregate_yield,
        "weighted_average_maturity": weighted_average_maturity,
        "cashflows": rows,
    }


def price_bond_series(terms: BondSeriesTerms) -> dict[str, Any]:
    base = _bond_series_run(terms, terms.discount_curve, "Base")
    up = _bond_series_run(terms, shifted_curve(terms.discount_curve, terms.scenario_shock_bp), f"Rates +{terms.scenario_shock_bp:.0f} bp")
    down = _bond_series_run(terms, shifted_curve(terms.discount_curve, -terms.scenario_shock_bp), f"Rates -{terms.scenario_shock_bp:.0f} bp")
    return {
        "primary_metrics": [
            {"label": "Series PV", "value": _money(base["pv"])},
            {"label": "Aggregate Yield", "value": _pct(base["aggregate_yield"]) if base["aggregate_yield"] is not None else "n/a"},
            {"label": "Weighted Avg Maturity", "value": f"{base['weighted_average_maturity']:.2f} years"},
            {"label": "Number of Bonds", "value": str(len(base["cashflows"]))},
        ],
        "scenarios": [base, up, down],
        "cashflows": base["cashflows"],
        "summary": (
            "Bond series analytics value each maturity in the series and aggregate PV, yield, and maturity exposure "
            "at the issuer/program level."
        ),
    }


@dataclass
class LoanLeaseAnnuityTerms:
    valuation_date: dt.date
    start_date: dt.date
    maturity_date: dt.date
    principal: float
    contract_rate: float
    payments_per_year: int
    day_count: str
    structure_type: str
    discount_curve: Curve
    scenario_shock_bp: float


def _loan_lease_annuity_run(terms: LoanLeaseAnnuityTerms, curve: Curve, label: str) -> dict[str, Any]:
    if terms.maturity_date <= terms.start_date:
        raise ValueError("Maturity date must be after start date.")
    dates = build_schedule(terms.start_date, terms.maturity_date, terms.payments_per_year)
    periods = len(dates)
    period_rate = terms.contract_rate / terms.payments_per_year
    if terms.structure_type == "level_payment":
        payment = terms.principal * period_rate / max(1.0 - (1.0 + period_rate) ** -periods, 1e-12)
    else:
        payment = 0.0
    previous = terms.start_date
    outstanding = terms.principal
    total_pv = 0.0
    rows = []
    for idx, pay_date in enumerate(dates, start=1):
        accrual = DayCount.year_frac(previous, pay_date, terms.day_count)
        interest = outstanding * terms.contract_rate * accrual
        if terms.structure_type == "interest_only" and idx < periods:
            principal_payment = 0.0
            scheduled_payment = interest
        elif terms.structure_type == "equal_principal":
            principal_payment = terms.principal / periods
            scheduled_payment = principal_payment + interest
        elif terms.structure_type == "interest_only":
            principal_payment = outstanding
            scheduled_payment = interest + principal_payment
        else:
            scheduled_payment = payment
            principal_payment = max(payment - interest, 0.0)
            if idx == periods:
                principal_payment = outstanding
                scheduled_payment = interest + principal_payment
        principal_payment = min(principal_payment, outstanding)
        t = DayCount.year_frac(terms.valuation_date, pay_date, "ACT/365")
        df = curve.df(t)
        pv = scheduled_payment * df
        total_pv += pv
        rows.append(
            {
                "period": f"{previous.isoformat()} to {pay_date.isoformat()}",
                "opening_balance": outstanding,
                "scheduled_payment": scheduled_payment,
                "interest": interest,
                "principal": principal_payment,
                "discount_factor": df,
                "discounted_pv": pv,
            }
        )
        outstanding -= principal_payment
        previous = pay_date
    yield_value = _solve_yield_from_cashflows(
        terms.valuation_date,
        [(parse_date(row["period"].split(" to ")[1]), row["scheduled_payment"]) for row in rows],
        terms.principal,
        terms.day_count,
    )
    return {
        "label": label,
        "pv": total_pv,
        "yield": yield_value,
        "cashflows": rows,
    }


def price_loan_lease_annuity(terms: LoanLeaseAnnuityTerms) -> dict[str, Any]:
    base = _loan_lease_annuity_run(terms, terms.discount_curve, "Base")
    up = _loan_lease_annuity_run(terms, shifted_curve(terms.discount_curve, terms.scenario_shock_bp), f"Rates +{terms.scenario_shock_bp:.0f} bp")
    down = _loan_lease_annuity_run(terms, shifted_curve(terms.discount_curve, -terms.scenario_shock_bp), f"Rates -{terms.scenario_shock_bp:.0f} bp")
    return {
        "primary_metrics": [
            {"label": "Present Value", "value": _money(base["pv"])},
            {"label": "Price / Principal", "value": f"{base['pv'] / terms.principal * 100.0:.4f}%"},
            {"label": "Implied Yield", "value": _pct(base["yield"]) if base["yield"] is not None else "n/a"},
            {"label": "Payment Count", "value": str(len(base["cashflows"]))},
        ],
        "scenarios": [base, up, down],
        "cashflows": base["cashflows"],
        "summary": (
            "Loan, lease, and annuity PV is calculated from the scheduled payment stream discounted on the supplied "
            "curve, with level-payment, equal-principal, and interest-only patterns supported."
        ),
    }


@dataclass
class AssetSwapTerms:
    valuation_date: dt.date
    maturity_date: dt.date
    notional: float
    coupon_rate: float
    market_clean_price_pct: float
    quoted_asset_swap_spread_bp: float
    payments_per_year: int
    day_count: str
    discount_curve: Curve
    scenario_rate_shock_bp: float
    scenario_spread_shock_bp: float


def price_asset_swap(terms: AssetSwapTerms) -> dict[str, Any]:
    if terms.maturity_date <= terms.valuation_date:
        raise ValueError("Maturity date must be after valuation date.")

    pay_dates = build_schedule(terms.valuation_date, terms.maturity_date, terms.payments_per_year)
    maturity_years = DayCount.year_frac(terms.valuation_date, terms.maturity_date, "ACT/365")
    market_dirty_price = terms.notional * terms.market_clean_price_pct / 100.0

    def leg_stats(curve: Curve) -> dict[str, float]:
        previous = terms.valuation_date
        annuity = 0.0
        fixed_coupon_pv = 0.0
        rows = []
        for pay_date in pay_dates:
            t = DayCount.year_frac(terms.valuation_date, pay_date, "ACT/365")
            accrual = DayCount.year_frac(previous, pay_date, terms.day_count)
            df = curve.df(t)
            coupon = terms.notional * terms.coupon_rate * accrual
            fixed_coupon_pv += coupon * df
            annuity += accrual * df
            rows.append(
                {
                    "period": f"{previous.isoformat()} to {pay_date.isoformat()}",
                    "accrual": accrual,
                    "discount_factor": df,
                    "fixed_coupon": coupon,
                    "fixed_coupon_pv": coupon * df,
                }
            )
            previous = pay_date
        maturity_df = curve.df(maturity_years)
        bond_model_pv = fixed_coupon_pv + terms.notional * maturity_df
        par_swap_rate = (1.0 - maturity_df) / max(annuity, 1e-12)
        # First-pass par asset-swap spread: coupon-versus-par-swap plus price pull-to-par.
        par_asset_swap_spread = (
            terms.coupon_rate
            - par_swap_rate
            + (terms.notional - market_dirty_price) / (terms.notional * max(annuity, 1e-12))
        )
        quoted_spread = terms.quoted_asset_swap_spread_bp / 10000.0
        package_pv = terms.notional * annuity * (quoted_spread - par_asset_swap_spread)
        return {
            "annuity": annuity,
            "par_swap_rate": par_swap_rate,
            "par_asset_swap_spread": par_asset_swap_spread,
            "bond_model_pv": bond_model_pv,
            "package_pv": package_pv,
            "cashflows": rows,
        }

    base = leg_stats(terms.discount_curve)
    up = leg_stats(shifted_curve(terms.discount_curve, terms.scenario_rate_shock_bp))
    down = leg_stats(shifted_curve(terms.discount_curve, -terms.scenario_rate_shock_bp))
    spread_up_pv = terms.notional * base["annuity"] * (
        (terms.quoted_asset_swap_spread_bp + terms.scenario_spread_shock_bp) / 10000.0
        - base["par_asset_swap_spread"]
    )
    spread_down_pv = terms.notional * base["annuity"] * (
        (terms.quoted_asset_swap_spread_bp - terms.scenario_spread_shock_bp) / 10000.0
        - base["par_asset_swap_spread"]
    )

    return {
        "primary_metrics": [
            {"label": "Asset-Swap PV", "value": _money(base["package_pv"])},
            {"label": "Par ASW Spread", "value": f"{base['par_asset_swap_spread'] * 10000:.2f} bp"},
            {"label": "Par Swap Rate", "value": _pct(base["par_swap_rate"])},
            {"label": "Bond Model PV", "value": _money(base["bond_model_pv"])},
        ],
        "scenarios": [
            {"label": "Base", "pv": base["package_pv"]},
            {"label": f"Rates +{terms.scenario_rate_shock_bp:.0f} bp", "pv": up["package_pv"]},
            {"label": f"Rates -{terms.scenario_rate_shock_bp:.0f} bp", "pv": down["package_pv"]},
            {"label": f"Spread +{terms.scenario_spread_shock_bp:.0f} bp", "pv": spread_up_pv},
            {"label": f"Spread -{terms.scenario_spread_shock_bp:.0f} bp", "pv": spread_down_pv},
        ],
        "cashflows": base["cashflows"],
        "summary": (
            "Asset-swap analytics compare the bond coupon stream with the par swap rate and solve a "
            "first-pass par asset-swap spread using the supplied market clean price and discount curve."
        ),
    }


@dataclass
class InflationLinkedBondTerms:
    valuation_date: dt.date
    maturity_date: dt.date
    notional: float
    real_coupon_rate: float
    base_cpi: float
    current_cpi: float
    annual_inflation_rate: float
    indexation_lag_months: int
    principal_floor: str
    payments_per_year: int
    day_count: str
    real_discount_curve: Curve
    nominal_curve: Curve
    scenario_real_rate_shock_bp: float
    scenario_inflation_shock_bp: float


def price_inflation_linked_bond(terms: InflationLinkedBondTerms) -> dict[str, Any]:
    if terms.maturity_date <= terms.valuation_date:
        raise ValueError("Maturity date must be after valuation date.")
    if terms.base_cpi <= 0 or terms.current_cpi <= 0:
        raise ValueError("Base CPI and current CPI must be positive.")

    pay_dates = build_schedule(terms.valuation_date, terms.maturity_date, terms.payments_per_year)
    maturity_years = DayCount.year_frac(terms.valuation_date, terms.maturity_date, "ACT/365")
    current_index_ratio = terms.current_cpi / terms.base_cpi
    lag_years = max(terms.indexation_lag_months, 0) / 12.0

    def projected_index_ratio(t: float, inflation_shift_bp: float = 0.0) -> float:
        projected_years = max(t - lag_years, 0.0)
        inflation = terms.annual_inflation_rate + inflation_shift_bp / 10000.0
        ratio = current_index_ratio * math.exp(inflation * projected_years)
        if terms.principal_floor == "yes":
            ratio = max(ratio, 1.0)
        return ratio

    def run(curve: Curve, inflation_shift_bp: float, label: str) -> dict[str, Any]:
        previous = terms.valuation_date
        total_pv = 0.0
        rows = []
        for pay_date in pay_dates:
            t = DayCount.year_frac(terms.valuation_date, pay_date, "ACT/365")
            accrual = DayCount.year_frac(previous, pay_date, terms.day_count)
            df = curve.df(t)
            index_ratio = projected_index_ratio(t, inflation_shift_bp)
            coupon = terms.notional * index_ratio * terms.real_coupon_rate * accrual
            principal = terms.notional * index_ratio if pay_date == pay_dates[-1] else 0.0
            pv = (coupon + principal) * df
            total_pv += pv
            rows.append(
                {
                    "period": f"{previous.isoformat()} to {pay_date.isoformat()}",
                    "index_ratio": index_ratio,
                    "coupon": coupon,
                    "principal": principal,
                    "discounted_pv": pv,
                }
            )
            previous = pay_date
        return {"label": label, "pv": total_pv, "cashflows": rows}

    base = run(terms.real_discount_curve, 0.0, "Base")
    real_up = run(shifted_curve(terms.real_discount_curve, terms.scenario_real_rate_shock_bp), 0.0, f"Real rates +{terms.scenario_real_rate_shock_bp:.0f} bp")
    real_down = run(shifted_curve(terms.real_discount_curve, -terms.scenario_real_rate_shock_bp), 0.0, f"Real rates -{terms.scenario_real_rate_shock_bp:.0f} bp")
    inflation_up = run(terms.real_discount_curve, terms.scenario_inflation_shock_bp, f"Inflation +{terms.scenario_inflation_shock_bp:.0f} bp")
    inflation_down = run(terms.real_discount_curve, -terms.scenario_inflation_shock_bp, f"Inflation -{terms.scenario_inflation_shock_bp:.0f} bp")
    nominal_zero = -math.log(terms.nominal_curve.df(maturity_years)) / max(maturity_years, 1e-12)
    real_zero = -math.log(terms.real_discount_curve.df(maturity_years)) / max(maturity_years, 1e-12)

    return {
        "primary_metrics": [
            {"label": "Real Discounted PV", "value": _money(base["pv"])},
            {"label": "Current Index Ratio", "value": f"{current_index_ratio:.6f}"},
            {"label": "Projected Maturity Ratio", "value": f"{projected_index_ratio(maturity_years):.6f}"},
            {"label": "Breakeven Proxy", "value": _pct(nominal_zero - real_zero)},
        ],
        "scenarios": [base, real_up, real_down, inflation_up, inflation_down],
        "cashflows": base["cashflows"],
        "summary": (
            "Inflation-linked bond PV projects indexed coupons and principal from CPI assumptions, applies the "
            "selected indexation lag, and discounts real cash flows on the supplied real discount curve."
        ),
    }


@dataclass
class BondForwardTreasuryLockTerms:
    valuation_date: dt.date
    delivery_date: dt.date
    bond_maturity_date: dt.date
    notional: float
    coupon_rate: float
    spot_dirty_price_pct: float
    financing_rate: float
    locked_forward_yield: float
    modified_duration: float
    position: str
    payments_per_year: int
    day_count: str
    discount_curve: Curve
    scenario_rate_shock_bp: float


def _bond_forward_stats(terms: BondForwardTreasuryLockTerms, curve: Curve, label: str) -> dict[str, Any]:
    if terms.delivery_date <= terms.valuation_date:
        raise ValueError("Delivery date must be after valuation date.")
    if terms.bond_maturity_date <= terms.delivery_date:
        raise ValueError("Bond maturity date must be after delivery date.")

    spot_dirty_price = terms.notional * terms.spot_dirty_price_pct / 100.0
    delivery_t = DayCount.year_frac(terms.valuation_date, terms.delivery_date, "ACT/365")
    pay_dates = build_schedule(terms.valuation_date, terms.bond_maturity_date, terms.payments_per_year)
    previous = terms.valuation_date
    income_pv = 0.0
    rows = []
    for pay_date in pay_dates:
        accrual = DayCount.year_frac(previous, pay_date, terms.day_count)
        coupon = terms.notional * terms.coupon_rate * accrual
        if pay_date <= terms.delivery_date:
            t = DayCount.year_frac(terms.valuation_date, pay_date, "ACT/365")
            income_pv += coupon * curve.df(t)
            rows.append(
                {
                    "period": f"{previous.isoformat()} to {pay_date.isoformat()}",
                    "coupon_date": pay_date.isoformat(),
                    "coupon_before_delivery": coupon,
                    "coupon_pv": coupon * curve.df(t),
                }
            )
        previous = pay_date

    forward_dirty_price = (spot_dirty_price - income_pv) * math.exp(terms.financing_rate * delivery_t)
    forward_price_pct = forward_dirty_price / terms.notional * 100.0
    remaining_years = DayCount.year_frac(terms.delivery_date, terms.bond_maturity_date, "ACT/365")
    forward_yield = max(
        (terms.coupon_rate * 100.0 + (100.0 - forward_price_pct) / max(remaining_years, 1e-12))
        / ((100.0 + forward_price_pct) / 2.0),
        -0.99,
    )
    lock_sign = 1.0 if terms.position == "receive_fixed" else -1.0
    treasury_lock_pv = (
        lock_sign
        * -terms.modified_duration
        * terms.notional
        * (forward_yield - terms.locked_forward_yield)
        * curve.df(delivery_t)
    )
    return {
        "label": label,
        "pv": treasury_lock_pv,
        "forward_dirty_price": forward_dirty_price,
        "forward_price_pct": forward_price_pct,
        "forward_yield": forward_yield,
        "income_pv": income_pv,
        "cashflows": rows,
    }


def price_bond_forward_treasury_lock(terms: BondForwardTreasuryLockTerms) -> dict[str, Any]:
    base = _bond_forward_stats(terms, terms.discount_curve, "Base")
    up = _bond_forward_stats(terms, shifted_curve(terms.discount_curve, terms.scenario_rate_shock_bp), f"Rates +{terms.scenario_rate_shock_bp:.0f} bp")
    down = _bond_forward_stats(terms, shifted_curve(terms.discount_curve, -terms.scenario_rate_shock_bp), f"Rates -{terms.scenario_rate_shock_bp:.0f} bp")

    return {
        "primary_metrics": [
            {"label": "Treasury Lock PV", "value": _money(base["pv"])},
            {"label": "Forward Dirty Price", "value": _money(base["forward_dirty_price"])},
            {"label": "Forward Price", "value": f"{base['forward_price_pct']:.4f}%"},
            {"label": "Implied Forward Yield", "value": _pct(base["forward_yield"])},
        ],
        "scenarios": [base, up, down],
        "cashflows": base["cashflows"],
        "summary": (
            "Bond forward analytics use cost-of-carry pricing from spot dirty price, pre-delivery coupon income, "
            "and financing assumptions; the treasury-lock PV is a duration-based approximation versus the locked yield."
        ),
    }
