from __future__ import annotations

from dataclasses import fields
from typing import Any, Dict

from flask import Blueprint, abort, render_template, request, url_for
from flask_login import current_user

from ..extensions import db
from ..models.db_models import Instrument, PricingResult
from ..models.mdls_structured_first_wave import (
    StructuredNoteTerms,
    price_structured_note,
    structured_note_scenarios,
)

structured_products_bp = Blueprint("structured_products", __name__)


PRODUCT_CONFIGS: Dict[str, Dict[str, Any]] = {
    "barrier-reverse-convertible": {
        "product_type": "barrier_reverse_convertible",
        "title": "Barrier Reverse Convertible",
        "subtitle": "Price a coupon-enhanced note with conditional downside exposure through a protection barrier.",
        "description_title": "A yield-enhancement note with principal at risk if the underlying breaches the downside condition.",
        "description_body": (
            "The first-pass model pays a fixed coupon and returns full notional unless the final underlying level is below "
            "the protection barrier. If the barrier condition is breached, redemption participates in the underlying downside."
        ),
        "chips": ["Reverse convertible", "Barrier protection", "Coupon income", "Principal at risk", "Monte Carlo"],
        "methodology_doc": "barrier_reverse_convertible",
        "defaults": {
            "notional": "1000000",
            "spot_price": "100",
            "maturity": "1.0",
            "risk_free_rate": "0.045",
            "dividend_yield": "0.000",
            "volatility": "0.24",
            "coupon_rate": "0.10",
            "protection_barrier": "0.70",
            "num_paths": "10000",
            "num_steps": "252",
            "random_seed": "42",
        },
        "sections": [
            ("Product Terms", ["notional", "spot_price", "maturity", "coupon_rate", "protection_barrier"]),
            ("Market Assumptions", ["risk_free_rate", "dividend_yield", "volatility"]),
            ("Simulation Settings", ["num_paths", "num_steps", "random_seed"]),
        ],
        "metrics": [
            ("Present Value", "price", "currency"),
            ("PV / Notional", "price_pct_notional", "percent"),
            ("Barrier Breach Probability", "breach_probability", "percent"),
            ("Expected Redemption", "expected_redemption", "currency"),
            ("Expected Coupon", "expected_coupon", "currency"),
            ("Standard Error", "standard_error", "currency"),
        ],
    },
    "principal-protected-note": {
        "product_type": "principal_protected_note",
        "title": "Principal-Protected Market-Linked Note",
        "subtitle": "Value a protected note with upside participation and optional cap.",
        "description_title": "A market-linked note combining discounted principal protection with equity upside participation.",
        "description_body": (
            "The first-pass model protects a stated percentage of notional at maturity and adds payoff from positive "
            "underlying performance, subject to participation and cap assumptions."
        ),
        "chips": ["Principal protection", "Market-linked note", "Upside participation", "Capped return", "Monte Carlo"],
        "methodology_doc": "principal_protected_note",
        "defaults": {
            "notional": "1000000",
            "spot_price": "100",
            "maturity": "2.0",
            "risk_free_rate": "0.045",
            "dividend_yield": "0.000",
            "volatility": "0.22",
            "principal_protection": "1.00",
            "participation_rate": "0.80",
            "cap_return": "0.30",
            "num_paths": "10000",
            "num_steps": "252",
            "random_seed": "42",
        },
        "sections": [
            ("Product Terms", ["notional", "spot_price", "maturity", "principal_protection", "participation_rate", "cap_return"]),
            ("Market Assumptions", ["risk_free_rate", "dividend_yield", "volatility"]),
            ("Simulation Settings", ["num_paths", "num_steps", "random_seed"]),
        ],
        "metrics": [
            ("Present Value", "price", "currency"),
            ("PV / Notional", "price_pct_notional", "percent"),
            ("Protection Floor", "protection_floor", "currency"),
            ("Upside Participation Probability", "upside_participation_probability", "percent"),
            ("Cap Hit Probability", "cap_hit_probability", "percent"),
            ("Expected Note Return", "expected_note_return", "percent"),
        ],
    },
    "enhanced-participation-note": {
        "product_type": "enhanced_participation_note",
        "title": "Enhanced Participation / Buffered Note",
        "subtitle": "Price a note with leveraged upside, capped return, and a downside buffer.",
        "description_title": "A return-enhancement note that reshapes upside and downside participation.",
        "description_body": (
            "The first-pass model applies leveraged upside participation up to a cap and absorbs downside only after "
            "the stated buffer is exhausted."
        ),
        "chips": ["Enhanced participation", "Buffered downside", "Capped upside", "Equity-linked payoff", "Monte Carlo"],
        "methodology_doc": "enhanced_participation_note",
        "defaults": {
            "notional": "1000000",
            "spot_price": "100",
            "maturity": "1.5",
            "risk_free_rate": "0.045",
            "dividend_yield": "0.000",
            "volatility": "0.24",
            "participation_rate": "1.50",
            "cap_return": "0.25",
            "buffer": "0.15",
            "downside_participation": "1.00",
            "num_paths": "10000",
            "num_steps": "252",
            "random_seed": "42",
        },
        "sections": [
            ("Product Terms", ["notional", "spot_price", "maturity", "participation_rate", "cap_return", "buffer", "downside_participation"]),
            ("Market Assumptions", ["risk_free_rate", "dividend_yield", "volatility"]),
            ("Simulation Settings", ["num_paths", "num_steps", "random_seed"]),
        ],
        "metrics": [
            ("Present Value", "price", "currency"),
            ("PV / Notional", "price_pct_notional", "percent"),
            ("Buffer Breach Probability", "buffer_breach_probability", "percent"),
            ("Cap Hit Probability", "cap_hit_probability", "percent"),
            ("Expected Note Return", "expected_note_return", "percent"),
            ("Standard Error", "standard_error", "currency"),
        ],
    },
    "contingent-income-note": {
        "product_type": "contingent_income_note",
        "title": "Digital Coupon / Contingent Income Note",
        "subtitle": "Price scheduled conditional coupons with optional memory and downside protection.",
        "description_title": "An income note whose coupons depend on observation-date barrier tests.",
        "description_body": (
            "The first-pass model pays coupons on observation dates when the underlying is above the coupon barrier. "
            "At maturity, principal is protected unless the final level breaches the protection barrier."
        ),
        "chips": ["Contingent coupon", "Digital coupon", "Memory coupon", "Protection barrier", "Monte Carlo"],
        "methodology_doc": "contingent_income_note",
        "defaults": {
            "notional": "1000000",
            "spot_price": "100",
            "maturity": "1.0",
            "risk_free_rate": "0.045",
            "dividend_yield": "0.000",
            "volatility": "0.24",
            "coupon_rate": "0.12",
            "coupon_frequency": "4",
            "coupon_barrier": "0.70",
            "protection_barrier": "0.60",
            "memory_coupon": "on",
            "num_paths": "10000",
            "num_steps": "252",
            "random_seed": "42",
        },
        "sections": [
            ("Product Terms", ["notional", "spot_price", "maturity", "coupon_rate", "coupon_frequency", "coupon_barrier", "protection_barrier", "memory_coupon"]),
            ("Market Assumptions", ["risk_free_rate", "dividend_yield", "volatility"]),
            ("Simulation Settings", ["num_paths", "num_steps", "random_seed"]),
        ],
        "metrics": [
            ("Present Value", "price", "currency"),
            ("PV / Notional", "price_pct_notional", "percent"),
            ("Coupon Payment Probability", "coupon_payment_probability", "percent"),
            ("Average Coupon Count", "average_coupon_count", "number"),
            ("Protection Breach Probability", "protection_breach_probability", "percent"),
            ("Expected Redemption", "expected_redemption", "currency"),
        ],
    },
    "credit-linked-note": {
        "product_type": "credit_linked_note",
        "title": "Credit-Linked Note",
        "subtitle": "Value a coupon note with principal exposed to reference credit default.",
        "description_title": "A funded credit-risk instrument linked to a reference entity default event.",
        "description_body": (
            "The first-pass model uses a flat reduced-form default intensity. Coupons are paid while the note survives; "
            "if default occurs, the investor receives a recovery-based redemption."
        ),
        "chips": ["Credit-linked note", "Default intensity", "Recovery rate", "Coupon income", "Reduced-form credit"],
        "methodology_doc": "credit_linked_note_structured",
        "defaults": {
            "notional": "1000000",
            "maturity": "3.0",
            "risk_free_rate": "0.045",
            "coupon_rate": "0.07",
            "coupon_frequency": "4",
            "hazard_rate": "0.025",
            "recovery_rate": "0.40",
            "num_paths": "10000",
            "num_steps": "252",
            "random_seed": "42",
        },
        "sections": [
            ("Product Terms", ["notional", "maturity", "coupon_rate", "coupon_frequency"]),
            ("Credit Assumptions", ["hazard_rate", "recovery_rate", "risk_free_rate"]),
            ("Simulation Settings", ["num_paths", "random_seed"]),
        ],
        "metrics": [
            ("Present Value", "price", "currency"),
            ("PV / Notional", "price_pct_notional", "percent"),
            ("Default Probability", "default_probability", "percent"),
            ("Survival Probability", "survival_probability", "percent"),
            ("Expected Loss", "expected_loss", "currency"),
            ("Average Coupon Count", "average_coupon_count", "number"),
        ],
    },
}


FIELD_META = {
    "notional": ("Notional", "number", "10000", "Principal amount used for payoff scaling."),
    "spot_price": ("Initial Spot / Index Level", "number", "0.01", "Initial reference level for normalized payoff simulation."),
    "maturity": ("Maturity (Years)", "number", "0.01", "Time from valuation date to final redemption."),
    "risk_free_rate": ("Risk-Free Rate", "number", "0.001", "Flat continuously compounded discount rate."),
    "dividend_yield": ("Dividend Yield", "number", "0.001", "Flat continuous dividend or carry assumption."),
    "volatility": ("Volatility", "number", "0.001", "Flat annualized volatility assumption."),
    "coupon_rate": ("Annual Coupon Rate", "number", "0.001", "Annualized coupon rate paid by the note."),
    "coupon_frequency": ("Coupon Frequency", "number", "1", "Coupon observations/payments per year."),
    "participation_rate": ("Participation Rate", "number", "0.01", "Multiplier applied to positive underlying performance."),
    "cap_return": ("Return Cap", "number", "0.01", "Maximum note return from the payoff component."),
    "buffer": ("Downside Buffer", "number", "0.01", "Initial loss absorption before investor downside applies."),
    "downside_participation": ("Downside Participation", "number", "0.01", "Multiplier applied to losses beyond the buffer."),
    "principal_protection": ("Principal Protection", "number", "0.01", "Protected notional percentage at maturity."),
    "protection_barrier": ("Protection Barrier", "number", "0.01", "Final level threshold for principal-at-risk payoff."),
    "coupon_barrier": ("Coupon Barrier", "number", "0.01", "Observation level required for coupon payment."),
    "memory_coupon": ("Memory Coupon", "checkbox", None, "Missed coupons accrue until a later coupon condition is met."),
    "hazard_rate": ("Hazard Rate", "number", "0.001", "Flat annual default intensity for the reference entity."),
    "recovery_rate": ("Recovery Rate", "number", "0.01", "Notional recovery percentage after default."),
    "num_paths": ("Number of Paths", "number", "100", "Monte Carlo path count."),
    "num_steps": ("Number of Time Steps", "number", "1", "Simulation time discretization."),
    "random_seed": ("Random Seed", "number", "1", "Seed used for repeatable first-pass simulations."),
}


def _format_value(value, value_type):
    if value is None:
        return "N/A"
    if value_type == "currency":
        return "${:,.2f}".format(float(value))
    if value_type == "percent":
        return "{:.2f}%".format(float(value) * 100.0)
    if value_type == "number":
        return "{:,.2f}".format(float(value))
    return str(value)


def _field_config(field_name, defaults):
    label, field_type, step, hint = FIELD_META[field_name]
    return {
        "name": field_name,
        "label": label,
        "type": field_type,
        "step": step,
        "hint": hint,
        "value": defaults.get(field_name, ""),
    }


def _build_form_sections(config, form_data):
    return [
        {
            "title": title,
            "fields": [_field_config(field_name, form_data) for field_name in field_names],
        }
        for title, field_names in config["sections"]
    ]


def _parse_form_data(config):
    form_data = {**config["defaults"], **dict(request.form)}
    if "memory_coupon" in config["defaults"] and "memory_coupon" not in request.form:
        form_data["memory_coupon"] = "off"
    return form_data


def _build_terms(config, form_data):
    term_values = {
        "product_type": config["product_type"],
        "spot_price": 100.0,
        "volatility": 0.20,
        "dividend_yield": 0.0,
    }
    valid_fields = {field.name for field in fields(StructuredNoteTerms)}
    for key, value in form_data.items():
        if key not in valid_fields or key == "product_type":
            continue
        if key == "memory_coupon":
            term_values[key] = value == "on"
        elif key in {"num_paths", "num_steps", "random_seed", "coupon_frequency"}:
            term_values[key] = int(float(value))
        else:
            term_values[key] = float(value)
    return StructuredNoteTerms(**term_values)


def _format_results(config, raw_results):
    primary_metrics = [
        {
            "label": label,
            "value": _format_value(raw_results.get(key), value_type),
        }
        for label, key, value_type in config["metrics"]
    ]
    distribution = []
    for label, key in [
        ("Final Level Mean", "final_level_mean"),
        ("Final Level 5th Percentile", "final_level_p05"),
        ("Final Level Median", "final_level_p50"),
        ("Final Level 95th Percentile", "final_level_p95"),
        ("Discounted Payoff 5th Percentile", "p05"),
        ("Discounted Payoff Median", "p50"),
        ("Discounted Payoff 95th Percentile", "p95"),
    ]:
        if key in raw_results:
            value_type = "percent" if key.startswith("final_level") else "currency"
            distribution.append({"label": label, "value": _format_value(raw_results[key], value_type)})
    return {"primary_metrics": primary_metrics, "distribution": distribution, "raw": raw_results}


def _format_scenarios(scenarios):
    return [
        {
            "name": row["name"],
            "price": _format_value(row["price"], "currency"),
            "change": _format_value(row["change"], "currency"),
            "change_pct_notional": _format_value(row.get("change_pct_notional", 0.0), "percent"),
        }
        for row in scenarios
    ]


def _save_pricing_result(config, terms, results):
    if not current_user.is_authenticated:
        return

    instrument = Instrument(
        user_id=current_user.id,
        product_type=config["product_type"],
        ticker=None,
        model_name="structured_products_first_wave",
        start_date=None,
        end_date=None,
        params_json=terms.__dict__,
    )
    db.session.add(instrument)
    db.session.flush()

    pricing_result = PricingResult(
        user_id=current_user.id,
        instrument_id=instrument.id,
        price=float(results["price"]),
        delta=None,
        gamma=None,
        vega=None,
        theta=None,
        rho=None,
        result_json=results,
    )
    db.session.add(pricing_result)
    db.session.commit()


@structured_products_bp.route("/", methods=["GET"])
def structured_products_home():
    return render_template(
        "structured_products_home.html",
        products=PRODUCT_CONFIGS,
    )


@structured_products_bp.route("/<product_slug>", methods=["GET", "POST"])
def structured_product(product_slug):
    config = PRODUCT_CONFIGS.get(product_slug)
    if not config:
        abort(404)

    form_data = config["defaults"]
    formatted_results = None
    scenario_results = None
    pricing_error = None

    if request.method == "POST":
        form_data = _parse_form_data(config)
        try:
            terms = _build_terms(config, form_data)
            raw_results = price_structured_note(terms)
            _save_pricing_result(config, terms, raw_results)
            formatted_results = _format_results(config, raw_results)
            scenario_results = _format_scenarios(structured_note_scenarios(terms))
        except Exception as exc:
            pricing_error = str(exc)

    methodology_url = url_for(
        "index.methodology_doc",
        doc_name=config["methodology_doc"],
    )
    return render_template(
        "structured_product_standard.html",
        product_slug=product_slug,
        config=config,
        form_sections=_build_form_sections(config, form_data),
        form_data=form_data,
        results=formatted_results,
        scenarios=scenario_results,
        pricing_error=pricing_error,
        methodology_url=methodology_url,
    )
