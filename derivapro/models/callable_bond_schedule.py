"""Contractual schedules for callable bonds, separate from lattice time steps."""

import datetime as dt
import math

import QuantLib as ql


def qdate(date):
    return ql.Date(date.day, date.month, date.year)


def pydate(date):
    return dt.date(date.year(), date.month(), date.dayOfMonth())


def accrual_counter(convention, maturity):
    name = convention.upper().replace(" ", "")
    counters = {
        # The local reference's 30/360 (ISDA) is BondBasis, not QL's
        # similarly named 30E/360 ISDA (German end-of-February treatment).
        "30/360ISDA": ql.Thirty360(ql.Thirty360.BondBasis),
        "30/360": ql.Thirty360(ql.Thirty360.USA),
        "ACT/360": ql.Actual360(),
        "ACT/365": ql.Actual365Fixed(),
        "ACT/ACTISMA": ql.ActualActual(ql.ActualActual.ISMA),
    }
    if name not in counters:
        raise ValueError("Unsupported callable bond accrual convention.")
    return counters[name]


def bond_cashflows(terms, input_rows, curve):
    effective = terms.effective_date or terms.dated_date or terms.valuation_date
    dated = terms.dated_date or effective
    if not dated <= effective < terms.maturity_date or terms.valuation_date < effective:
        raise ValueError("Require dated date <= effective date <= settlement < maturity.")
    if terms.payments_per_year not in {1, 2, 4, 12}:
        raise ValueError("Unsupported coupon frequency.")
    calendar = ql.BespokeCalendar("Instrument holidays")
    calendar.addWeekend(ql.Saturday)
    calendar.addWeekend(ql.Sunday)
    for value in terms.holiday_dates.replace("\n", ";").split(";"):
        if value.strip():
            calendar.addHoliday(qdate(dt.date.fromisoformat(value.strip())))
    conventions = {"none": ql.Unadjusted, "following": ql.Following, "modified_following": ql.ModifiedFollowing}
    if terms.business_day_convention not in conventions:
        raise ValueError("Unsupported business-day convention.")
    convention = conventions[terms.business_day_convention]
    tenor = ql.Period(12 // terms.payments_per_year, ql.Months)
    schedule = ql.Schedule(qdate(effective), qdate(terms.maturity_date), tenor, ql.NullCalendar(),
                           ql.Unadjusted, ql.Unadjusted, ql.DateGeneration.Backward, True,
                           qdate(terms.first_coupon_date) if terms.first_coupon_date else ql.Date(),
                           qdate(terms.last_coupon_date) if terms.last_coupon_date else ql.Date())
    dates = [pydate(date) for date in list(schedule)[1:]]
    if terms.schedule_mode not in {"explicit", "period_terms"}:
        raise ValueError("Unsupported coupon table interpretation.")
    if input_rows:
        if len({row["payment_date"] for row in input_rows}) != len(input_rows):
            raise ValueError("Coupon table dates must be unique.")
        if any(row["payment_date"] <= effective for row in input_rows):
            raise ValueError("Coupon table dates must follow the effective date.")
        if terms.schedule_mode == "period_terms":
            if input_rows[-1]["payment_date"] < terms.maturity_date:
                raise ValueError("Coupon terms must cover maturity.")
            rows = []
            previous_cutoff = effective
            for source in input_rows:
                period_dates = [date for date in dates if previous_cutoff < date <= source["payment_date"]]
                if not period_dates:
                    raise ValueError("Each coupon terms row must cover at least one generated coupon date.")
                for date in period_dates:
                    rows.append({**source, "payment_date": date,
                                 "principal": source["principal"] if date == period_dates[-1] else 0.0,
                                 "fixed_payment": source["fixed_payment"] if date == period_dates[-1] else 0.0})
                previous_cutoff = source["payment_date"]
        else:
            rows = input_rows
            if rows[-1]["payment_date"] != terms.maturity_date:
                raise ValueError("Explicit payment dates must end at maturity. Use coupon terms mode for cutoff dates.")
    else:
        outstanding = terms.notional
        rows = []
        for date in dates:
            principal = terms.notional / len(dates) if terms.amortization_style == "straight_line" else 0.0
            if date == dates[-1]:
                principal = outstanding
            rows.append(dict(payment_date=date, opening_notional=outstanding, coupon_rate=terms.coupon_rate,
                             principal=principal, fixed_payment=0.0))
            outstanding -= principal
    counter = accrual_counter(terms.day_count, terms.maturity_date)
    previous = dated
    outstanding = terms.notional
    result = []
    accrued = 0.0
    for row in rows:
        date = row["payment_date"]
        n, rate, principal, fixed = (float(row[key]) for key in ("opening_notional", "coupon_rate", "principal", "fixed_payment"))
        if not all(math.isfinite(x) for x in (n, rate, principal, fixed)) or min(n, principal, fixed) < 0:
            raise ValueError("Coupon table amounts must be finite; principal and notionals cannot be negative.")
        if not math.isclose(n, outstanding, rel_tol=1e-9, abs_tol=1e-7):
            raise ValueError(f"Opening notional on {date} must equal prior remaining principal ({outstanding:g}).")
        reduction = principal + (fixed if terms.fixed_payment_treatment == "principal_redemption" else 0.0)
        if reduction > n + 1e-7:
            raise ValueError(f"Principal repayment on {date} exceeds outstanding principal.")
        reference_end = qdate(date)
        reference_start = reference_end - tenor
        fraction = counter.yearFraction(qdate(previous), qdate(date), reference_start, reference_end)
        coupon = n * rate * fraction
        payment = pydate(calendar.adjust(qdate(date), convention))
        if previous < terms.valuation_date < date:
            accrued = n * rate * counter.yearFraction(qdate(previous), qdate(terms.valuation_date), reference_start, reference_end)
        if payment > terms.valuation_date:
            time = (payment - terms.valuation_date).days / 365.0
            result.append(dict(payment_date=payment.isoformat(), accrual_start=previous.isoformat(),
                               accrual_end=date.isoformat(), opening_notional=n, closing_notional=n-reduction,
                               coupon_rate=rate, accrual_factor=fraction, coupon=coupon, principal=principal,
                               fixed_payment=fixed, cashflow=coupon+principal+fixed, discount_factor=curve.df(time),
                               pv=(coupon+principal+fixed)*curve.df(time)))
        outstanding = n - reduction
        previous = date
    if outstanding > 1e-7:
        raise ValueError("Coupon schedule must repay the outstanding principal at maturity.")
    dirty = sum(row["pv"] for row in result)
    return {"cashflows": result, "dirty_pv": dirty, "pv": dirty-accrued, "accrued_interest": accrued}
