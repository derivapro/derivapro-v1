"""Generic pricing-report preview/download for product pages that use
``build_product_report`` instead of a bespoke report pipeline.

Registered products are looked up by a short ``product_key`` used in URLs
(e.g. ``/reports/barrier-reverse-convertible/preview``); each key maps to the
``Instrument.product_type`` value(s) that page persists under.
"""

import io
import os
import uuid

from flask import Blueprint, abort, render_template, send_file
from flask_login import current_user, login_required

from ..extensions import db
from ..models.db_models import Report
from ..services.product_reports import build_product_report, get_latest_pricing_result
from ..services.report_builder import render_product_report_pdf

product_reports_bp = Blueprint("product_reports", __name__)

# product_key -> (title, [Instrument.product_type values that count as this product])
PRODUCT_REPORT_REGISTRY = {
    "barrier-reverse-convertible": ("Barrier Reverse Convertible", ["barrier_reverse_convertible"]),
    "principal-protected-note": ("Principal-Protected Market-Linked Note", ["principal_protected_note"]),
    "enhanced-participation-note": ("Enhanced Participation / Buffered Note", ["enhanced_participation_note"]),
    "contingent-income-note": ("Digital Coupon / Contingent Income Note", ["contingent_income_note"]),
    "credit-linked-note": ("Credit-Linked Note", ["credit_linked_note"]),
    "autocallable": ("Autocallable Structured Note", ["autocallable_structured_note", "autocallable_option"]),
    "digital": ("Digital Option", ["first_wave_digital"]),
    "lookback": ("Lookback Option", ["first_wave_lookback"]),
    "basket": ("Basket Option", ["first_wave_basket"]),
    "cliquet": ("Cliquet / Ratchet Option", ["first_wave_cliquet"]),
    "quanto": ("Quanto Option", ["first_wave_quanto"]),
    "fixed-rate-bond": ("Fixed-Rate Bond", ["fixed_rate_bond"]),
    "credit-default-swap": ("Credit Default Swap", ["credit_default_swap"]),
    "forward-contract": ("Forward Contract", ["forward_contract"]),
    "variance-swap": ("Variance Swap", ["variance_swap"]),
    "interest-rate-swap": ("Interest Rate Swap", ["interest_rate_swap"]),
    "swaption": ("European Swaption", ["swaption"]),
    "fra": ("Forward Rate Agreement", ["fixed_income_fra"]),
    "cap-floor": ("Cap / Floor", ["fixed_income_cap-floor"]),
    "callable-putable-bond": ("Callable / Putable Bond", ["fixed_income_callable-putable-bond"]),
    "level-coupon-bond": ("Level Coupon Bond", ["fixed_income_level-coupon-bond"]),
    "amortizing-stepup-sinking-bond": ("Amortizing / Step-Up / Sinking Bond", ["fixed_income_amortizing-stepup-sinking-bond"]),
    "custom-structured-bond": ("Custom Structured Bond", ["fixed_income_custom-structured-bond"]),
    "bond-series": ("Bond Series", ["fixed_income_bond-series"]),
    "loan-lease-annuity": ("Loan / Lease / Annuity", ["fixed_income_loan-lease-annuity"]),
    "asset-swap": ("Asset Swap", ["fixed_income_asset-swap"]),
    "inflation-linked-bond": ("Inflation-Linked Bond", ["fixed_income_inflation-linked-bond"]),
    "bond-forward-treasury-lock": ("Bond Forward / Treasury Lock", ["fixed_income_bond-forward-treasury-lock"]),
}


def _resolve(product_key):
    entry = PRODUCT_REPORT_REGISTRY.get(product_key)
    if not entry:
        abort(404)
    return entry


@product_reports_bp.route("/reports/<product_key>/preview", methods=["GET"])
@login_required
def preview(product_key):
    title, product_types = _resolve(product_key)
    report = build_product_report(product_types, title)
    return render_template("generic_report_preview.html", report=report, title=title)


@product_reports_bp.route("/reports/<product_key>/download", methods=["GET"])
@login_required
def download(product_key):
    title, product_types = _resolve(product_key)
    report = build_product_report(product_types, title)
    if report is None:
        abort(404)

    base_dir = os.path.dirname(os.path.abspath(__file__))
    static_dir = os.path.join(base_dir, "..", "static")
    reports_dir = os.path.join(static_dir, "reports")
    os.makedirs(reports_dir, exist_ok=True)

    pdf_bytes = render_product_report_pdf(report, static_dir)

    report_filename = f"{product_key}_report_{uuid.uuid4().hex}.pdf"
    pdf_output_path = os.path.join(reports_dir, report_filename)
    with open(pdf_output_path, "wb") as pdf_file:
        pdf_file.write(pdf_bytes)

    latest_pricing_result = get_latest_pricing_result(product_types)
    report_row = Report(
        user_id=current_user.id,
        instrument_id=latest_pricing_result.instrument_id if latest_pricing_result else None,
        pricing_result_id=latest_pricing_result.id if latest_pricing_result else None,
        analysis_result_id=None,
        report_type=f"{product_key}_report",
        filename=report_filename,
        filepath=os.path.join("derivapro", "static", "reports", report_filename),
        pdf_data=pdf_bytes,
    )
    db.session.add(report_row)
    db.session.commit()

    return send_file(
        io.BytesIO(pdf_bytes),
        as_attachment=True,
        download_name=f"{title.replace(' ', '_')}_Report.pdf",
        mimetype="application/pdf",
    )
