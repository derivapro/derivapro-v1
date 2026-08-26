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


def build_product_report(product_types: Iterable[str], title: str) -> Optional[ProductReport]:
    """Assemble a :class:`ProductReport` from the latest run, or ``None`` if
    the user has never priced this product (so the caller can show an empty
    state instead of a report with nothing in it)."""
    pricing_result = get_latest_pricing_result(product_types)
    if pricing_result is None:
        return None

    instrument = pricing_result.instrument
    params = instrument.params_json or {} if instrument else {}

    inputs = [
        (_label_for(key), value)
        for key, value in params.items()
        if key not in _SKIP_INPUT_KEYS and value is not None
    ]
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
        results += [(m["label"], m["value"]) for m in primary_metrics if "label" in m]
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

    return ProductReport(
        title=title,
        subtitle=instrument.product_type.replace("_", " ").title() if instrument else "",
        generated_at=datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC"),
        inputs=inputs,
        results=results,
        plot_filename=plot_filename,
        analysis_note=analysis_note,
    )
