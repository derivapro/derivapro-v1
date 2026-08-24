from pathlib import Path
import re

import markdown
from flask import Blueprint, abort, render_template

index_bp = Blueprint("index", __name__)

METHODOLOGY_DOCS = {
    "european_option": "european_option.md",
    "american_option": "american_option.md",
    "barrier_option": "barrier_option.md",
    "asian_option": "asian_option.md",
    "autocallable_note": "autocallable_note.md",
    "barrier_reverse_convertible": "barrier_reverse_convertible.md",
    "principal_protected_note": "principal_protected_note.md",
    "enhanced_participation_note": "enhanced_participation_note.md",
    "contingent_income_note": "contingent_income_note.md",
    "credit_linked_note_structured": "credit_linked_note_structured.md",
    "forward_rate_agreement": "forward_rate_agreement.md",
    "cap_floor": "cap_floor.md",
    "callable_putable_bond": "callable_putable_bond.md",
    "level_coupon_bond": "level_coupon_bond.md",
    "amortizing_stepup_sinking_bond": "amortizing_stepup_sinking_bond.md",
    "custom_structured_bond": "custom_structured_bond.md",
    "bond_series": "bond_series.md",
    "loan_lease_annuity": "loan_lease_annuity.md",
    "asset_swap": "asset_swap.md",
    "inflation_linked_bond": "inflation_linked_bond.md",
    "bond_forward_treasury_lock": "bond_forward_treasury_lock.md",
    "digital_option": "digital_option.md",
    "lookback_option": "lookback_option.md",
    "basket_option": "basket_option.md",
    "cliquet_option": "cliquet_option.md",
    "quanto_option": "quanto_option.md",
}


DISPLAY_MATH_PATTERN = re.compile(r"(\\\[.*?\\\]|\$\$.*?\$\$)", re.DOTALL)


def _protect_display_math(markdown_source):
    math_blocks = []

    def replace_math_block(match):
        math_blocks.append(match.group(0))
        return f"\n\nDERIVAPRO_MATH_BLOCK_{len(math_blocks) - 1}\n\n"

    protected_source = DISPLAY_MATH_PATTERN.sub(replace_math_block, markdown_source)
    return protected_source, math_blocks


def _restore_display_math(html_content, math_blocks):
    for index, math_block in enumerate(math_blocks):
        placeholder = f"DERIVAPRO_MATH_BLOCK_{index}"
        html_content = html_content.replace(
            f"<p>{placeholder}</p>",
            f'<div class="methodology-math-display">{math_block}</div>',
        )
        html_content = html_content.replace(
            placeholder,
            f'<div class="methodology-math-display">{math_block}</div>',
        )
    return html_content


@index_bp.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@index_bp.route("/products", methods=["GET"])
def products():
    return render_template("products.html")


@index_bp.route("/methodology/<doc_name>", methods=["GET"])
def methodology_doc(doc_name):
    filename = METHODOLOGY_DOCS.get(doc_name)
    if not filename:
        abort(404)

    repo_root = Path(__file__).resolve().parents[2]
    doc_path = repo_root / "docs" / "methodology" / filename
    if not doc_path.exists():
        abort(404)

    markdown_source, math_blocks = _protect_display_math(
        doc_path.read_text(encoding="utf-8")
    )
    html_content = markdown.markdown(
        markdown_source,
        extensions=["tables", "fenced_code"],
    )
    html_content = _restore_display_math(html_content, math_blocks)
    return render_template(
        "methodology_doc.html",
        title=doc_name.replace("_", " ").title(),
        html_content=html_content,
        source_filename=filename,
    )
