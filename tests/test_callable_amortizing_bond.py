import datetime as dt
import unittest

from derivapro.models.curve import Curve
from derivapro.models.rates_fixed_income import (
    CallableAmortizingBondTerms,
    parse_discount_factor_curve,
    price_callable_amortizing_bond,
)


def _terms(option_rights: str = "callable", exercise_schedule: str | None = None) -> CallableAmortizingBondTerms:
    return CallableAmortizingBondTerms(
        valuation_date=dt.date(2026, 8, 30),
        dated_date=dt.date(2026, 6, 20),
        first_coupon_date=dt.date(2026, 12, 20),
        last_coupon_date=dt.date(2040, 12, 20),
        maturity_date=dt.date(2041, 6, 20),
        notional=1_000_000,
        coupon_rate=0.05,
        market_clean_price_pct=100.0,
        payments_per_year=2,
        day_count="ACT/ACT ISMA",
        discount_curve=Curve([1, 30], [0.06, 0.06]),
        option_rights=option_rights,
        exercise_style="bermudan",
        first_exercise_date=dt.date(2028, 6, 20),
        exercise_price_pct=100.0,
        short_rate_model="hull_white",
        short_rate_volatility=0.015,
        short_rate_mean_reversion=0.03,
        lattice_steps_per_period=2,
        scenario_shock_bp=25.0,
        amortization_style="straight_line",
        exercise_schedule=exercise_schedule or "",
    )


def _metric(results: dict, label: str) -> str:
    for item in results["primary_metrics"]:
        if item["label"] == label:
            return item["value"]
    raise AssertionError(f"Missing metric: {label}")


def _money_to_float(value: str) -> float:
    return float(value.replace("$", "").replace(",", ""))


class CallableAmortizingBondTest(unittest.TestCase):
    def test_methodology_page_is_registered_and_renderable(self):
        from derivapro import create_app
        from derivapro.routes.index import METHODOLOGY_DOCS, methodology_doc

        self.assertEqual(
            METHODOLOGY_DOCS["callable_amortizing_bond"],
            "callable_amortizing_bond.md",
        )
        app = create_app()
        with app.test_request_context("/methodology/callable_amortizing_bond"):
            response = methodology_doc("callable_amortizing_bond")
        self.assertIn("Callable and Putable Amortizing Bonds", response)

    def test_disabled_exercise_collapses_to_straight_amortizing_value(self):
        results = price_callable_amortizing_bond(_terms(exercise_schedule="2028-06-20|2028-06-20|0|0"))

        self.assertEqual(_metric(results, "Option-Adjusted Clean PV"), _metric(results, "Straight Bond Clean PV"))
        self.assertEqual(_metric(results, "Call / Put Probability"), "0.00% / 0.00%")

    def test_callable_value_is_capped_below_straight_bond_value(self):
        results = price_callable_amortizing_bond(_terms("callable"))

        option_adjusted = _money_to_float(_metric(results, "Option-Adjusted Clean PV"))
        straight = _money_to_float(_metric(results, "Straight Bond Clean PV"))
        self.assertLessEqual(option_adjusted, straight)

    def test_putable_value_is_floored_above_straight_bond_value(self):
        results = price_callable_amortizing_bond(_terms("putable"))

        option_adjusted = _money_to_float(_metric(results, "Option-Adjusted Clean PV"))
        straight = _money_to_float(_metric(results, "Straight Bond Clean PV"))
        self.assertGreaterEqual(option_adjusted, straight)

    def test_fixed_payment_column_is_included_in_cashflows(self):
        terms = _terms(exercise_schedule="2028-06-20|2028-06-20|0|0")
        terms.cashflow_schedule = "2027-08-30|1000000|0.05|0|1250;2028-08-30|1000000|0.05|1000000|0"
        terms.maturity_date = dt.date(2028, 8, 30)
        terms.first_coupon_date = dt.date(2027, 8, 30)
        terms.last_coupon_date = dt.date(2027, 8, 30)

        results = price_callable_amortizing_bond(terms)

        self.assertEqual(results["cashflows"][0]["fixed_payment"], 1250.0)
        self.assertGreater(results["cashflows"][0]["cashflow"], results["cashflows"][0]["coupon"])

    def test_fixed_payment_can_be_principal_redemption_without_double_counting(self):
        terms = _terms(exercise_schedule="2028-06-20|2028-06-20|0|0")
        terms.valuation_date = dt.date(2019, 9, 24)
        terms.notional = 100
        terms.dated_date = dt.date(2019, 9, 24)
        terms.maturity_date = dt.date(2025, 6, 20)
        terms.first_coupon_date = dt.date(2020, 12, 20)
        terms.last_coupon_date = dt.date(2022, 12, 20)
        terms.cashflow_schedule = (
            "2020-12-20|100|0.0500|0|0;"
            "2022-06-20|100|0.0550|0|20;"
            "2022-12-20|80|0.0550|0|0;"
            "2025-06-20|80|0.0550|0|80"
        )
        terms.fixed_payment_treatment = "principal_redemption"
        terms.exercise_schedule = "2022-06-20|2022-06-20|0|0"

        results = price_callable_amortizing_bond(terms)
        cashflows = results["cashflows"]

        self.assertEqual(cashflows[1]["opening_notional"], 100.0)
        self.assertEqual(cashflows[1]["fixed_payment"], 20.0)
        self.assertEqual(cashflows[1]["principal"], 0.0)
        self.assertEqual(cashflows[2]["opening_notional"], 80.0)
        self.assertEqual(cashflows[-1]["fixed_payment"], 80.0)

    def test_discount_factor_curve_parser_accepts_dated_curve_table(self):
        curve = parse_discount_factor_curve(
            dt.date(2019, 9, 24),
            "2020-09-24|0.95276295;2024-09-24|0.78342144",
        )

        self.assertAlmostEqual(curve.df(5.005479452054795), 0.78342144, places=6)

    def test_discount_factor_curve_parser_preserves_interpolation_choice(self):
        curve = parse_discount_factor_curve(
            dt.date(2019, 9, 24),
            "2020-09-24|0.95276295;2024-09-24|0.78342144",
            "exponential",
        )

        self.assertEqual(curve.interpolation, "exponential")

    def test_black_karasinski_model_runs(self):
        terms = _terms("callable")
        terms.short_rate_model = "black_karasinski"
        terms.short_rate_volatility = 0.20

        results = price_callable_amortizing_bond(terms)

        self.assertIn("primary_metrics", results)
        self.assertIn("benchmark_metrics", results)

    def test_tree_can_be_truncated_at_last_callable_date(self):
        terms = _terms("callable")
        terms.tree_generation = "last_callable_date"
        terms.exercise_schedule = "2028-06-20|2028-06-20|100|0"

        results = price_callable_amortizing_bond(terms)
        metrics = {item["label"]: item["value"] for item in results["benchmark_metrics"]}

        self.assertLess(metrics["Tree Horizon Years"], 10.0)

    def test_curve_fit_and_no_option_tree_match_discounted_cashflows(self):
        from derivapro.models.callable_bond_schedule import bond_cashflows
        from derivapro.models.callable_bond_tree import CallableBondTree
        for model, sigma in [("hull_white", .2), ("black_karasinski", .2), ("hull_white", 0.0)]:
            terms = _terms()
            terms.short_rate_model, terms.short_rate_volatility = model, sigma
            terms.discount_curve = Curve([.1, 1, 5, 20], [.01, .025, .04, .05])
            base = bond_cashflows(terms, [], terms.discount_curve)
            tree = CallableBondTree(terms, terms.discount_curve, base["cashflows"], [])
            self.assertLess(tree.lattice.curve_fit_error, 1e-11)
            self.assertAlmostEqual(tree.value()["dirty_value"], base["dirty_pv"], places=6)
            self.assertAlmostEqual(tree.value()["maturity_probability"], 1.0, places=12)

    def test_single_call_matches_quantlib_analytic_zero_coupon_bond_option(self):
        import QuantLib as ql
        from derivapro.models.callable_bond_schedule import bond_cashflows
        from derivapro.models.callable_bond_tree import CallableBondTree
        terms = _terms()
        terms.valuation_date = terms.dated_date = dt.date(2019, 9, 24)
        terms.maturity_date = dt.date(2024, 9, 24)
        terms.first_coupon_date = terms.last_coupon_date = None
        terms.notional, terms.coupon_rate = 100, 0
        terms.amortization_style = "bullet"
        terms.short_rate_volatility, terms.short_rate_mean_reversion = .01, .03
        terms.lattice_steps_per_period = 80
        terms.discount_curve = Curve([1, 10], [.04, .04])
        exercise = dt.date(2021, 9, 24)
        base = bond_cashflows(terms, [], terms.discount_curve)
        tree = CallableBondTree(terms, terms.discount_curve, base["cashflows"],
                                [dict(start_date=exercise, end_date=exercise, call_price_pct=90, put_price_pct=0)])
        curve = ql.YieldTermStructureHandle(ql.FlatForward(ql.Date(24, 9, 2019), .04, ql.Actual365Fixed()))
        model = ql.HullWhite(curve, .03, .01)
        t, maturity = (exercise-terms.valuation_date).days/365, (terms.maturity_date-terms.valuation_date).days/365
        expected = 100 * (curve.discount(maturity) - model.discountBondOption(ql.Option.Call, .9, t, maturity))
        self.assertAlmostEqual(tree.value()["dirty_value"], expected, delta=.01)

    def test_benchmark_terms_generate_ten_coupons_and_match_straight_pv(self):
        from derivapro.routes.bonds import FIXED_INCOME_EXTENSION_CONFIGS, _default_form_data, _price_fixed_income_extension
        form = _default_form_data(FIXED_INCOME_EXTENSION_CONFIGS["callable-amortizing-bond"])
        expected_defaults = {
            "valuation_date": "2019-09-24",
            "effective_date": "2019-09-24",
            "dated_date": "2019-09-24",
            "first_coupon_date": "",
            "last_coupon_date": "",
            "maturity_date": "2024-09-24",
            "schedule_mode": "period_terms",
            "notional": "100",
            "payments_per_year": "2",
            "day_count": "30/360 ISDA",
            "business_day_convention": "none",
            "notification_days": "30",
            "option_rights": "callable",
            "exercise_style": "bermudan",
            "exercise_schedule_source": "custom",
            "short_rate_model": "hull_white",
            "short_rate_volatility_pct": "20.00",
            "short_rate_mean_reversion_pct": "0.50",
            "lattice_steps_per_period": "5",
            "tree_generation": "maturity",
            "discount_curve_input_type": "discount_factors",
            "interpolation_method": "linear_discount",
        }
        for field, expected in expected_defaults.items():
            self.assertEqual(form[field], expected, field)
        form["short_rate_volatility_pct"] = "10"
        result = _price_fixed_income_extension("callable-amortizing-bond", form)
        rows = result["cashflows"]
        self.assertEqual(len(rows), 10)
        self.assertEqual(rows[-1]["payment_date"], "2024-09-24")
        self.assertEqual(rows[4]["payment_date"], "2022-03-24")
        self.assertEqual(rows[4]["fixed_payment"], 20)
        self.assertEqual(rows[5]["opening_notional"], 80)
        self.assertEqual(rows[-1]["fixed_payment"], 80)
        metrics = {m["label"]: m["value"] for m in result["benchmark_metrics"]}
        self.assertAlmostEqual(metrics["Straight Bond Clean Price"], 101.7476542, delta=1e-6)
        self.assertLess(abs(metrics["Straight Bond Tree vs Cashflow PV Error"]), 1e-8)
        self.assertAlmostEqual(sum(metrics[k] for k in ("Probability of Call", "Probability of Put", "Probability of No Exercise")), 1, places=12)

    def test_notice_period_reduces_issuer_option_and_event_dates_are_exact(self):
        from derivapro.models.callable_bond_schedule import bond_cashflows
        from derivapro.models.callable_bond_tree import CallableBondTree
        terms = _terms()
        date = dt.date(2028, 6, 20)
        exercise = [dict(start_date=date, end_date=date, call_price_pct=100, put_price_pct=0)]
        base = bond_cashflows(terms, [], terms.discount_curve)
        immediate = CallableBondTree(terms, terms.discount_curve, base["cashflows"], exercise).value()["dirty_value"]
        terms.notification_days = 30
        tree = CallableBondTree(terms, terms.discount_curve, base["cashflows"], exercise)
        notice = date-dt.timedelta(days=30)
        self.assertAlmostEqual(tree.lattice.times[tree.events[0]["step"]], (notice-terms.valuation_date).days/365)
        self.assertGreaterEqual(tree.value()["dirty_value"], immediate-50)

    def test_invalid_explicit_maturity_and_notional_are_rejected(self):
        from derivapro.models.callable_bond_schedule import bond_cashflows
        terms = _terms()
        row = dict(payment_date=terms.maturity_date+dt.timedelta(days=1), opening_notional=terms.notional,
                   coupon_rate=.05, principal=terms.notional, fixed_payment=0)
        with self.assertRaisesRegex(ValueError, "end at maturity"):
            bond_cashflows(terms, [row], terms.discount_curve)
        row["payment_date"] = terms.maturity_date
        row["opening_notional"] = 80
        with self.assertRaisesRegex(ValueError, "Opening notional"):
            bond_cashflows(terms, [row], terms.discount_curve)

    def test_reference_isda_daycount_is_not_german_february_rule(self):
        import QuantLib as ql
        from derivapro.models.callable_bond_schedule import accrual_counter
        counter = accrual_counter("30/360 ISDA", dt.date(2030, 1, 1))
        self.assertAlmostEqual(counter.yearFraction(ql.Date(28, 2, 2025), ql.Date(31, 3, 2025)), 33/360)


if __name__ == "__main__":
    unittest.main()
