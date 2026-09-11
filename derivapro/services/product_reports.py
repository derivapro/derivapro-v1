"""Generic pricing-report assembly for product pages that don't have a
bespoke report pipeline (see ``routes/vanilla_options.py::_build_report_template``
for the one page that does).

``build_product_report`` reads the most recent :class:`Instrument` /
:class:`PricingResult` / :class:`AnalysisResult` rows for the current user and
a given ``product_type`` and maps them onto a :class:`ProductReport`. Fields
that were never populated (no plot, no analysis run yet) are simply omitted.
"""

from __future__ import annotations

from datetime import datetime
from typing import Iterable, Optional

from flask_login import current_user

from ..models.db_models import AnalysisResult, Instrument, PricingResult
from .report_builder import ProductReport

# Human-friendly labels for the params_json / result_json keys that show up
# across product pages. Anything not listed here falls back to a titleized
# version of the raw key so new fields never silently disappear from reports.
_LABEL_OVERRIDES = {
    "spot_price": "Spot Price",
    "strike_price": "Strike Price",
    "risk_free_rate": "Risk-Free Rate",
    "dividend_yield": "Dividend Yield",
    "volatility": "Volatility",
    "notional": "Notional",
    "day_count": "Day Count Convention",
    "option_type": "Option Type",
    "pricing_model": "Pricing Model",
    "model_name": "Model",
    "num_steps": "Number of Steps",
    "num_paths": "Number of Paths",
    "valuation_date": "Settlement / Value Date",
    "effective_date": "Effective Date",
    "dated_date": "Dated Date / Accrual Start",
    "maturity_date": "Maturity / Terminating Date",
    "schedule_mode": "Payment Schedule Source",
    "market_clean_price_pct": "OAS Target Clean Price (% of Par)",
    "payments_per_year": "Coupon Frequency (Payments / Year)",
    "option_rights": "Embedded Option Rights",
    "exercise_style": "Exercise Style",
    "short_rate_model": "Short-Rate Model",
    "short_rate_volatility_pct": "Short-Rate Volatility (%)",
    "short_rate_mean_reversion_pct": "Mean Reversion (%)",
    "lattice_steps_per_period": "Tree Steps per Coupon Period",
    "interpolation_method": "Curve Interpolation",
    "notification_days": "Notification Days (Calendar)",
    "scenario_shock_bp": "Scenario Shock (bp)",
    "benchmark_preset": "Benchmark Scenario",
    "pricing_basis": "Pricing Basis",
    "yield_to_maturity": "Yield to Maturity",
    "amortization_style": "Principal Schedule Type",
    "first_coupon_date": "First Coupon Date",
    "last_coupon_date": "Penultimate Coupon Date",
}

# Fields that are noisy/internal and should not clutter the Inputs table.
_SKIP_INPUT_KEYS = {"simulation_configuration", "random_seed"}

_RESULT_METRIC_LABELS = [
    ("price", "Price"),
    ("delta", "Delta"),
    ("gamma", "Gamma"),
    ("vega", "Vega"),
    ("theta", "Theta"),
    ("rho", "Rho"),
]


def _label_for(key: str) -> str:
    if key in _LABEL_OVERRIDES:
        return _LABEL_OVERRIDES[key]
    return key.replace("_", " ").strip().title()


def get_latest_pricing_result(product_types: Iterable[str]) -> Optional[PricingResult]:
    if not current_user.is_authenticated:
        return None
    return (
        PricingResult.query.join(Instrument, PricingResult.instrument_id == Instrument.id)
        .filter(
            PricingResult.user_id == current_user.id,
            Instrument.user_id == current_user.id,
            Instrument.product_type.in_(list(product_types)),
        )
        .order_by(PricingResult.created_at.desc())
        .first()
    )


def get_latest_analysis_result(product_types: Iterable[str]) -> Optional[AnalysisResult]:
    if not current_user.is_authenticated:
        return None
    return (
        AnalysisResult.query.join(Instrument, AnalysisResult.instrument_id == Instrument.id)
        .filter(
            AnalysisResult.user_id == current_user.id,
            Instrument.user_id == current_user.id,
            Instrument.product_type.in_(list(product_types)),
        )
        .order_by(AnalysisResult.created_at.desc())
        .first()
    )


_CALLABLE_BENCHMARKS = {
    "callable_benchmark": {
        "label": "Callable benchmark - 20% volatility",
        "values": {
            "Clean Price Including Option": 70.8033979,
            "Straight Bond Clean Price": 101.7476542,
            "Embedded Option Value": 30.9442562,
            "Probability of Call": 0.33939017,
            "Effective Duration": 2.40067326,
            "Effective Convexity": 9.15839942,
        },
    },
    "putable_benchmark": {
        "label": "Puttable benchmark - 10% volatility",
        "values": {
            "Clean Price Including Option": 119.9377674,
            "Straight Bond Clean Price": 101.7476542,
            "Embedded Option Value": 18.19011324,
            "Probability of Put": 0.808024308,
            "Effective Duration": 2.52476305,
            "Effective Convexity": 10.1268701,
        },
    },
    "faster_amortization": {
        "label": "Faster amortization - 15% volatility",
        "values": {
            "Clean Price Including Option": 87.51319389,
            "Straight Bond Clean Price": 100.9864534,
            "Embedded Option Value": 13.47325948,
            "Probability of Call": 0.485842967,
            "Effective Duration": 1.540990157,
            "Effective Convexity": 4.431304727,
        },
    },
}

_STRUCTURED_AMORTIZING_BENCHMARKS = {
    "bullet": {
        "label": "External GenericBonds benchmark - bullet principal",
        "values": {
            "Fair Value (Clean)": 902702.15,
            "Accrued Interest": 9699.45,
            "Fair Value + Accrued": 912401.60,
            "Duration": 10.2737,
            "Modified Duration": 9.9745,
            "BPV (+1bp Price Change)": -910.07,
            "Convexity": 129.8359,
        },
    },
    "straight_line": {
        "label": "External GenericBonds benchmark - straight-line amortization",
        "values": {
            "Fair Value (Clean)": 943393.01,
            "Accrued Interest": 9699.45,
            "Fair Value + Accrued": 953092.46,
            "Duration": 5.8385,
            "Modified Duration": 5.6684,
            "BPV (+1bp Price Change)": -540.25,
            "Convexity": 50.7291,
        },
    },
}


def _metric_map(result_json: dict) -> dict:
    return {
        item.get("label"): item.get("value")
        for item in result_json.get("benchmark_metrics", [])
        if isinstance(item, dict) and item.get("label")
    }


def _fmt_validation_value(label: str, value) -> str:
    if value is None:
        return "N/A"
    if label.startswith("Probability"):
        return f"{float(value) * 100:.4f}%"
    if "Error" in label:
        return f"{float(value):.3e}"
    return f"{float(value):.6f}"


def _callable_amortizing_report_content(params: dict, result_json: dict) -> dict:
    """Build documented methodology and run-specific validation evidence."""
    metrics = _metric_map(result_json)
    call_probability = float(metrics.get("Probability of Call") or 0.0)
    put_probability = float(metrics.get("Probability of Put") or 0.0)
    no_exercise_probability = float(metrics.get("Probability of No Exercise") or 0.0)
    probability_error = abs(call_probability + put_probability + no_exercise_probability - 1.0)
    curve_fit_error = abs(float(metrics.get("Maximum Discount Factor Fit Error") or 0.0))
    straight_tree_error = abs(float(metrics.get("Straight Bond Tree vs Cashflow PV Error") or 0.0))

    straight_value = float(metrics.get("Straight Bond Clean Price") or 0.0)
    option_value = float(metrics.get("Clean Price Including Option") or 0.0)
    rights = params.get("option_rights", "")
    if rights == "callable":
        boundary_ok = option_value <= straight_value + 1e-10
        boundary_criterion = "Option-adjusted PV <= straight PV"
    elif rights == "putable":
        boundary_ok = option_value + 1e-10 >= straight_value
        boundary_criterion = "Option-adjusted PV >= straight PV"
    else:
        boundary_ok = True
        boundary_criterion = "Joint call/put priority applied"

    testing_results = [
        (
            "Discount-curve fit",
            _fmt_validation_value("Maximum Discount Factor Fit Error", curve_fit_error),
            "Maximum residual <= 1e-10",
            "Pass" if curve_fit_error <= 1e-10 else "Review",
        ),
        (
            "No-exercise tree reconciliation",
            _fmt_validation_value("Straight Bond Tree vs Cashflow PV Error", straight_tree_error),
            "Absolute PV error <= 1e-8",
            "Pass" if straight_tree_error <= 1e-8 else "Review",
        ),
        (
            "Exercise-probability conservation",
            f"Residual {probability_error:.3e}",
            "Call + put + no exercise = 100%",
            "Pass" if probability_error <= 1e-10 else "Review",
        ),
        (
            "Embedded-option value boundary",
            f"Option-adjusted {option_value:.6f}; straight {straight_value:.6f}",
            boundary_criterion,
            "Pass" if boundary_ok else "Review",
        ),
    ]

    benchmark_comparison = []
    benchmark = _CALLABLE_BENCHMARKS.get(params.get("benchmark_preset"))
    if benchmark:
        for label, external_value in benchmark["values"].items():
            deriva_value = metrics.get(label)
            if deriva_value is None:
                continue
            difference = float(deriva_value) - external_value
            if label.startswith("Probability"):
                difference_text = f"{difference * 100:+.4f} pp"
            else:
                difference_text = f"{difference:+.6f}"
            status = (
                "Reconciled"
                if label == "Straight Bond Clean Price" and abs(difference) <= 1e-5
                else "Comparison"
            )
            benchmark_comparison.append(
                (
                    label,
                    _fmt_validation_value(label, deriva_value),
                    _fmt_validation_value(label, external_value),
                    difference_text,
                    status,
                )
            )

    return {
        "methodology_doc": "callable_amortizing_bond",
        "methodology_summary": (
            "Contractual coupon and principal cashflows are generated independently from the embedded option. "
            "A one-factor Hull-White or Black-Karasinski recombining trinomial lattice is fitted to the supplied "
            "discount curve, and call or put decisions are applied by backward induction on scheduled Bermudan "
            "dates or sampled American windows. Notice-period cashflows are retained. OAS, effective duration, "
            "effective convexity, exercise probabilities and deterministic yield diagnostics are derived by "
            "consistent repricing of the same contractual schedule."
        ),
        "methodology_steps": [
            ("Cashflow construction", "Generate coupon, principal and accrued-interest amounts under the selected day-count and calendar conventions."),
            ("Curve and lattice", "Interpolate the supplied zero-rate or discount-factor curve and fit state-price shifts at every tree step."),
            ("Exercise valuation", "Compare continuation with state-dependent call and put settlement values, including the notification period."),
            ("Risk and diagnostics", "Reprice shifted curves for BPV, duration and convexity; solve OAS to the stated clean-price target."),
        ],
        "methodology_references": [
            "docs/methodology/callable_amortizing_bond.md",
            "Math/Callablebond.html",
            "Math/TermStructureCalibration.html",
            "Math/GenericBonds.html",
            "Math/Daycount.html",
            "Math/Dategen.html",
        ],
        "testing_scope": [
            ("Contract and cashflows", "Coupon-term and explicit schedules, notional roll-forward, fixed principal, maturity coverage and day-count behavior."),
            ("Curve construction", "Dated discount factors and zero rates, supported interpolation choices and lattice discount-factor fit."),
            ("Lattice valuation", "Hull-White and Black-Karasinski trees, callable and puttable boundaries, notification timing and probability conservation."),
            ("Independent checks", "Direct discounted-cashflow reconciliation, analytical Hull-White zero-coupon bond option and supplied external examples."),
            ("Risk outputs", "Parallel-rate scenarios, BPV, effective duration, effective convexity, OAS and yield diagnostics."),
        ],
        "testing_results": testing_results,
        "benchmark_name": benchmark["label"] if benchmark else "",
        "benchmark_comparison": benchmark_comparison,
        "testing_note": (
            "Internal numerical checks use the values from this pricing run. External option-sensitive outputs are "
            "reported as comparisons, not pass/fail conclusions, because exercise end dates are partially obscured "
            "in the supplied source and proprietary grid, notice and output conventions may differ."
        ),
    }


def _structured_amortizing_benchmark(params: dict) -> dict | None:
    expected = {
        "valuation_date": "2026-08-30",
        "dated_date": "2026-06-20",
        "first_coupon_date": "2026-12-20",
        "last_coupon_date": "2040-12-20",
        "maturity_date": "2041-06-20",
        "notional": 1_000_000.0,
        "coupon_rate": 0.05,
        "payments_per_year": 2.0,
        "yield_to_maturity": 0.06,
    }
    for key, expected_value in expected.items():
        actual = params.get(key)
        if isinstance(expected_value, str):
            if actual != expected_value:
                return None
        else:
            try:
                if abs(float(actual) - expected_value) > 1e-12:
                    return None
            except (TypeError, ValueError):
                return None
    if params.get("pricing_basis") != "yield" or params.get("day_count") != "ACT/ACT ISMA":
        return None
    return _STRUCTURED_AMORTIZING_BENCHMARKS.get(params.get("amortization_style"))


def _fmt_structured_metric(label: str, value: float) -> str:
    if label in {"Fair Value (Clean)", "Accrued Interest", "Fair Value + Accrued", "BPV (+1bp Price Change)"}:
        return f"${float(value):,.2f}"
    return f"{float(value):,.4f}"


def _structured_amortizing_report_content(params: dict, result_json: dict) -> dict:
    """Build deterministic cashflow validation and benchmark evidence."""
    metrics = _metric_map(result_json)
    cashflows = [row for row in result_json.get("cashflows", []) if isinstance(row, dict)]
    scenarios = [row for row in result_json.get("scenarios", []) if isinstance(row, dict)]

    cashflow_error = max(
        (
            abs(
                float(row.get("cashflow", 0.0))
                - float(row.get("coupon", 0.0))
                - float(row.get("principal", 0.0))
                - float(row.get("fixed_payment", 0.0))
            )
            for row in cashflows
        ),
        default=0.0,
    )
    rollforward_error = max(
        (
            abs(float(current.get("opening_notional", 0.0)) - float(previous.get("closing_notional", 0.0)))
            for previous, current in zip(cashflows, cashflows[1:])
        ),
        default=0.0,
    )
    terminal_residual = abs(float(cashflows[-1].get("closing_notional", 0.0))) if cashflows else 0.0
    clean_dirty_error = abs(
        float(metrics.get("Fair Value (Clean)") or 0.0)
        + float(metrics.get("Accrued Interest") or 0.0)
        - float(metrics.get("Fair Value + Accrued") or 0.0)
    )
    scenario_ok = len(scenarios) >= 3 and float(scenarios[1]["pv"]) <= float(scenarios[0]["pv"]) <= float(scenarios[2]["pv"])

    testing_results = [
        (
            "Cashflow identity",
            f"Maximum residual {cashflow_error:.3e}",
            "Coupon + principal + fixed payment = total cashflow",
            "Pass" if cashflow_error <= 1e-8 else "Review",
        ),
        (
            "Principal roll-forward",
            f"Maximum link error {rollforward_error:.3e}; terminal residual {terminal_residual:.3e}",
            "Consecutive notionals reconcile and principal is fully repaid",
            "Pass" if rollforward_error <= 1e-8 and terminal_residual <= 1e-8 else "Review",
        ),
        (
            "Clean/dirty reconciliation",
            f"Residual {clean_dirty_error:.3e}",
            "Clean value + accrued interest = dirty value",
            "Pass" if clean_dirty_error <= 1e-8 else "Review",
        ),
        (
            "Parallel-rate direction",
            "Up-rate PV <= base PV <= down-rate PV" if scenario_ok else "Scenario ordering requires review",
            "Positive-rate shock lowers value; negative-rate shock raises value",
            "Pass" if scenario_ok else "Review",
        ),
    ]

    benchmark_comparison = []
    benchmark = _structured_amortizing_benchmark(params)
    if benchmark:
        for label, external_value in benchmark["values"].items():
            deriva_value = metrics.get(label)
            if deriva_value is None:
                continue
            difference = float(deriva_value) - external_value
            tolerance = 0.011 if label in {"Fair Value (Clean)", "Accrued Interest", "Fair Value + Accrued", "BPV (+1bp Price Change)"} else 0.00011
            benchmark_comparison.append(
                (
                    label,
                    _fmt_structured_metric(label, deriva_value),
                    _fmt_structured_metric(label, external_value),
                    f"{difference:+,.6f}",
                    "Reconciled" if abs(difference) <= tolerance else "Review",
                )
            )

    return {
        "methodology_doc": "amortizing_stepup_sinking_bond",
        "methodology_summary": (
            "The engine generates deterministic coupon, principal and fixed-payment cashflows from the contractual "
            "schedule. Each cashflow is discounted using either the supplied yield to maturity or the interpolated "
            "zero curve. Accrued interest is separated from dirty value to produce clean value, and duration, modified "
            "duration, convexity and BPV are calculated consistently from the same cashflows and pricing basis."
        ),
        "methodology_steps": [
            ("Schedule construction", "Generate contractual payment dates and apply the coupon and principal terms in effect for each period."),
            ("Cashflow calculation", "Calculate accrual, coupon, principal redemption and any additional fixed payment, then roll opening notional forward."),
            ("Valuation", "Discount each cashflow using the selected yield or continuously compounded zero curve and separate accrued interest."),
            ("Risk diagnostics", "Reprice under parallel shocks and calculate yield, duration, modified duration, convexity, BPV and principal runoff."),
        ],
        "methodology_references": [
            "docs/methodology/amortizing_stepup_sinking_bond.md",
            "Math/GenericBonds.html",
            "Math/lcb.html",
            "Math/BondTable.html",
            "Math/Daycount.html",
            "Math/Dategen.html",
        ],
        "testing_scope": [
            ("Contract dates", "Settlement, dated date, first and penultimate coupon dates, maturity and stub-period generation."),
            ("Cashflow schedules", "Bullet, straight-line and custom sinking principal; level, step-up and step-down coupons; fixed payments."),
            ("Valuation", "Yield-based and curve-based discounting, accrued interest, clean/dirty reconciliation and yield solving."),
            ("Risk outputs", "Parallel-rate scenarios, duration, modified duration, convexity, BPV and principal runoff."),
            ("Independent benchmark", "Published GenericBonds bullet and straight-line examples under the documented default terms."),
        ],
        "testing_results": testing_results,
        "benchmark_name": benchmark["label"] if benchmark else "",
        "benchmark_comparison": benchmark_comparison,
        "testing_note": (
            "Internal checks are calculated from this saved pricing run. The external comparison is included only when "
            "the saved dates, notional, coupon, frequency, day count, yield and principal style match the documented benchmark."
        ),
    }


def build_product_report(product_types: Iterable[str], title: str) -> Optional[ProductReport]:
    """Assemble a :class:`ProductReport` from the latest run, or ``None`` if
    the user has never priced this product (so the caller can show an empty
    state instead of a report with nothing in it)."""
    product_types = list(product_types)
    pricing_result = get_latest_pricing_result(product_types)
    if pricing_result is None:
        return None

    instrument = pricing_result.instrument
    params = instrument.params_json or {} if instrument else {}
    is_structured_amortizing = "fixed_income_amortizing-stepup-sinking-bond" in product_types
    skipped_inputs = set(_SKIP_INPUT_KEYS)
    if is_structured_amortizing:
        skipped_inputs.update({"coupon_schedule", "principal_schedule"})
        if params.get("pricing_basis") == "yield":
            skipped_inputs.update(
                {
                    "discount_curve_input_type",
                    "discount_curve_tenors",
                    "discount_curve_rates",
                    "discount_factor_curve",
                    "interpolation_method",
                }
            )
        else:
            skipped_inputs.update({"yield_to_maturity", "discount_curve_input_type", "discount_factor_curve", "interpolation_method"})

    inputs = []
    for key, value in params.items():
        if key in skipped_inputs or value is None or value == "":
            continue
        if is_structured_amortizing and key == "cashflow_schedule":
            value = f"{len([row for row in str(value).replace(chr(10), ';').split(';') if row.strip()])} payment rows"
        label = _label_for(key)
        if is_structured_amortizing and key == "market_clean_price_pct":
            label = "Market Clean Price (% of Par)"
        inputs.append((label, value))
    if instrument:
        if instrument.ticker:
            inputs.insert(0, ("Ticker", instrument.ticker))
        if instrument.model_name:
            inputs.insert(0, ("Model", instrument.model_name))

    results = [
        (label, getattr(pricing_result, field))
        for field, label in _RESULT_METRIC_LABELS
        if getattr(pricing_result, field) is not None
    ]
    result_json = pricing_result.result_json or {}
    # Several pricing pipelines (first-wave exotics, autocallables, fixed-income
    # extensions) share a "primary_metrics": [{"label", "value"}, ...] shape
    # with pre-formatted display strings - surface those directly.
    primary_metrics = result_json.get("primary_metrics")
    if primary_metrics:
        formatted_results = [(m["label"], m["value"]) for m in primary_metrics if "label" in m]
        results = formatted_results if is_structured_amortizing else results + formatted_results
    elif not results and result_json:
        results = [
            (_label_for(key), value)
            for key, value in result_json.items()
            if isinstance(value, (int, float, str)) and key != "run_summary"
        ][:12]  # keep the PDF table from running away on verbose result payloads

    analysis = get_latest_analysis_result(product_types)
    plot_filename = None
    analysis_note = ""
    if analysis and analysis.result_json:
        plot_filename = analysis.result_json.get("plot_filename")
        variable = analysis.result_json.get("variable")
        if variable:
            analysis_note = f"Latest analysis: {analysis.analysis_type} on {variable}."
        else:
            analysis_note = f"Latest analysis type: {analysis.analysis_type}."

    report_content = {}
    if "fixed_income_callable-amortizing-bond" in product_types:
        report_content = _callable_amortizing_report_content(params, result_json)
        if not analysis_note:
            analysis_note = (
                "The latest run includes parallel-rate PV scenarios, principal-runoff analysis, "
                "and exercise probabilities by scheduled date."
            )
    elif "fixed_income_amortizing-stepup-sinking-bond" in product_types:
        report_content = _structured_amortizing_report_content(params, result_json)
        if not analysis_note:
            analysis_note = (
                "The latest run includes parallel-rate clean-value scenarios, principal runoff, "
                "and the contractual coupon profile by payment date."
            )

    return ProductReport(
        title=title,
        subtitle=instrument.product_type.replace("_", " ").title() if instrument else "",
        generated_at=datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC"),
        inputs=inputs,
        results=results,
        plot_filename=plot_filename,
        analysis_note=analysis_note,
        **report_content,
    )
