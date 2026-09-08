"""Notice-aware callable cashflow rollback on a curve-fitted short-rate tree."""

import datetime as dt

import numpy as np

from .callable_bond_schedule import accrual_counter, qdate
from .short_rate_lattice import ShortRateLattice


class CallableBondTree:
    def __init__(self, terms, curve, cashflows, exercise_rows):
        self.terms = terms
        self.cashflows = cashflows
        self.events = []
        self.time = lambda date: (date - terms.valuation_date).days / 365.0
        if terms.notification_days < 0:
            raise ValueError("Notification days cannot be negative.")
        if terms.exercise_style not in {"bermudan", "american_grid"}:
            raise ValueError("Unsupported exercise style.")
        counter = accrual_counter(terms.day_count, terms.maturity_date)
        step_days = max(1, int(365 / (terms.payments_per_year * terms.lattice_steps_per_period)))
        for row_index, row in enumerate(exercise_rows):
            start, end = row["start_date"], row["end_date"]
            if end < start:
                raise ValueError("Exercise window end must not precede its start.")
            if end > terms.maturity_date:
                raise ValueError("Exercise dates cannot follow maturity.")
            dates = [end]
            if terms.exercise_style == "american_grid":
                first = max(start, terms.valuation_date + dt.timedelta(days=terms.notification_days))
                dates = [first + dt.timedelta(days=i) for i in range(0, max((end-first).days, 0)+1, step_days)] + [end]
            for date in sorted(set(dates)):
                notice = date - dt.timedelta(days=terms.notification_days)
                if notice < terms.valuation_date or date >= terms.maturity_date:
                    continue
                call = row["call_price_pct"] if terms.option_rights in {"callable", "callable_putable"} else 0.0
                put = row["put_price_pct"] if terms.option_rights in {"putable", "callable_putable"} else 0.0
                if min(call, put) < 0:
                    raise ValueError("Exercise prices cannot be negative.")
                if call == put == 0:
                    continue
                applicable = next((cf for cf in cashflows if dt.date.fromisoformat(cf["payment_date"]) >= date), None)
                if applicable is None:
                    continue
                on_coupon = applicable["payment_date"] == date.isoformat()
                outstanding = applicable["closing_notional"] if on_coupon else applicable["opening_notional"]
                accrued = 0.0
                accrual_start = dt.date.fromisoformat(applicable["accrual_start"])
                if not on_coupon and date > accrual_start:
                    accrued = applicable["opening_notional"] * applicable["coupon_rate"] * counter.yearFraction(qdate(accrual_start), qdate(date))
                self.events.append(dict(date=date, notice=notice, call=call, put=put,
                                        outstanding=outstanding, accrued=accrued, row_index=row_index))
        mandatory = [self.time(dt.date.fromisoformat(cf["payment_date"])) for cf in cashflows]
        mandatory.extend(self.time(date) for event in self.events for date in (event["date"], event["notice"]))
        self.lattice = ShortRateLattice(curve, mandatory, terms.payments_per_year * terms.lattice_steps_per_period,
                                        terms.short_rate_mean_reversion, terms.short_rate_volatility, terms.short_rate_model)
        self.cash_by_step = {}
        self.payment_step_by_date = {}
        for cf in cashflows:
            date = dt.date.fromisoformat(cf["payment_date"])
            index = self.lattice.index(self.time(date))
            self.payment_step_by_date[date] = index
            self.cash_by_step[index] = self.cash_by_step.get(index, 0.0) + cf["cashflow"]
        self.events_by_step = {}
        for event in self.events:
            event["step"] = self.lattice.index(self.time(event["notice"]))
            event["payment_step"] = self.lattice.index(self.time(event["date"]))
            self.events_by_step.setdefault(event["step"], []).append(event)
        # Conditional tail values must retain rate dependence even when exercise
        # rollback stops early. Tail construction uses the same calibrated tree.
        self.exercise_horizon = max((event["step"] for event in self.events), default=0)

    def settlement_leg(self, event, strike, spread):
        tree = self.lattice
        end = event["payment_step"]
        values = np.full(len(tree.states[end]), event["outstanding"] * strike / 100 + event["accrued"]
                         + self.cash_by_step.get(end, 0.0))
        for i in range(end-1, event["step"]-1, -1):
            values = tree.rollback(values, i, spread) + self.cash_by_step.get(i, 0.0)
        return values

    def value(self, spread_bp=0.0, exercise=True):
        tree = self.lattice
        spread = spread_bp / 10000
        last = len(tree.times)-1
        values = np.full(len(tree.states[last]), self.cash_by_step.get(last, 0.0))
        horizon = self.exercise_horizon if self.terms.tree_generation == "last_callable_date" and self.events else last
        for i in range(last-1, horizon-1, -1):
            values = tree.rollback(values, i, spread) + self.cash_by_step.get(i, 0.0)
        decisions = {}
        for i in range(horizon, -1, -1):
            if i < horizon:
                values = tree.rollback(values, i, spread) + self.cash_by_step.get(i, 0.0)
            chosen = np.full(len(values), -1, dtype=int)
            if exercise:
                for event in self.events_by_step.get(i, []):
                    event_index = self.events.index(event)
                    if event["put"]:
                        put = self.settlement_leg(event, event["put"], spread)
                        mask = put > values + 1e-12
                        values = np.maximum(values, put)
                        chosen[mask] = event_index * 2 + 1
                    if event["call"]:
                        call = self.settlement_leg(event, event["call"], spread)
                        mask = call < values - 1e-12
                        values = np.minimum(values, call)
                        chosen[mask] = event_index * 2
            decisions[i] = chosen
        active = np.array([1.0])
        call_by_step, put_by_step = {}, {}
        row_probabilities = {}
        expected_exercise = 0.0
        for i in range(horizon+1):
            chosen = decisions[i]
            for code in np.unique(chosen[chosen >= 0]):
                event = self.events[int(code)//2]
                mass = float(active[chosen == code].sum())
                target = put_by_step if code % 2 else call_by_step
                target[event["step"]] = target.get(event["step"], 0.0) + mass
                key = (event["row_index"], "put" if code % 2 else "call")
                row_probabilities[key] = row_probabilities.get(key, 0.0) + mass
                expected_exercise += mass * self.time(event["date"])
            active[chosen >= 0] = 0
            if i < horizon:
                active = tree.propagate(active, i)
        total_exercise = sum(call_by_step.values()) + sum(put_by_step.values())
        return dict(dirty_value=float(values[0]), call_probability_by_step=call_by_step,
                    put_probability_by_step=put_by_step, maturity_probability=float(active.sum()),
                    row_probabilities=row_probabilities,
                    expected_exercise_time=expected_exercise / total_exercise if total_exercise else None)
