import datetime as dt
import unittest

from derivapro.models.curve import Curve
from derivapro.models.rates_fixed_income import GenericBondTerms, price_generic_bond


def _add_months(d: dt.date, months: int) -> dt.date:
    year = d.year + (d.month - 1 + months) // 12
    month = (d.month - 1 + months) % 12 + 1
    return dt.date(year, month, d.day)


def _benchmark_terms(amortization_style: str) -> GenericBondTerms:
    return GenericBondTerms(
        valuation_date=dt.date(2026, 8, 30),
        dated_date=dt.date(2026, 6, 20),
        first_coupon_date=dt.date(2026, 12, 20),
        last_coupon_date=dt.date(2040, 12, 20),
        maturity_date=dt.date(2041, 6, 20),
        notional=1_000_000,
        coupon_rate=0.05,
        payments_per_year=2,
        day_count="ACT/ACT ISMA",
        discount_curve=Curve([1, 30], [0.06, 0.06]),
        market_clean_price_pct=100.0,
        scenario_shock_bp=1.0,
        amortization_style=amortization_style,
        pricing_basis="yield",
        yield_to_maturity=0.06,
    )


def _metric(results: dict, label: str) -> str:
    for item in results["primary_metrics"]:
        if item["label"] == label:
            return item["value"]
    raise AssertionError(f"Missing metric: {label}")


class StructuredAmortizingBondBenchmarkTest(unittest.TestCase):
    def test_bullet_bond_matches_external_calculator_benchmark(self):
        results = price_generic_bond(_benchmark_terms("bullet"))

        self.assertEqual(_metric(results, "Fair Value (Clean)"), "$902,702.15")
        self.assertEqual(_metric(results, "Accrued Interest"), "$9,699.45")
        self.assertEqual(_metric(results, "Fair Value + Accrued"), "$912,401.60")
        self.assertEqual(_metric(results, "Duration"), "10.2737")
        self.assertEqual(_metric(results, "Modified Duration"), "9.9745")
        self.assertEqual(_metric(results, "Convexity"), "129.8359")
        self.assertEqual(_metric(results, "BPV (+1bp Price Change)"), "$-910.07")

    def test_straight_line_amortizing_bond_matches_external_calculator_benchmark(self):
        results = price_generic_bond(_benchmark_terms("straight_line"))

        self.assertEqual(_metric(results, "Fair Value (Clean)"), "$943,393.01")
        self.assertEqual(_metric(results, "Accrued Interest"), "$9,699.45")
        self.assertEqual(_metric(results, "Fair Value + Accrued"), "$953,092.46")
        self.assertEqual(_metric(results, "Duration"), "5.8385")
        self.assertEqual(_metric(results, "Modified Duration"), "5.6684")
        self.assertEqual(_metric(results, "Convexity"), "50.7291")
        self.assertEqual(_metric(results, "BPV (+1bp Price Change)"), "$-540.25")

    def test_schedule_opening_notional_is_derived_from_principal_roll_forward(self):
        terms = _benchmark_terms("straight_line")
        dates = []
        current = terms.first_coupon_date
        while current < terms.maturity_date:
            dates.append(current)
            current = _add_months(current, 6)
        dates.append(terms.maturity_date)

        principal = terms.notional / len(dates)
        terms.cashflow_schedule = ";".join(
            f"{payment_date.isoformat()}|999999999|0.05|{principal}"
            for payment_date in dates
        )
        results = price_generic_bond(terms)

        self.assertEqual(_metric(results, "Fair Value (Clean)"), "$943,393.01")
        self.assertEqual(results["cashflows"][1]["opening_notional"], 966666.6666666666)


if __name__ == "__main__":
    unittest.main()
