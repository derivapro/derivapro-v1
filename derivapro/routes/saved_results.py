from flask import Blueprint, render_template
from flask_login import current_user, login_required

from ..models.db_models import Portfolio, PricingResult

saved_results_bp = Blueprint("saved_results", __name__)


@saved_results_bp.route("/", methods=["GET"])
@login_required
def saved_results():
    results = (
        PricingResult.query
        .filter_by(user_id=current_user.id)
        .order_by(PricingResult.created_at.desc())
        .all()
    )

    portfolios = (
        Portfolio.query
        .filter_by(user_id=current_user.id)
        .order_by(Portfolio.created_at.desc())
        .all()
    )

    return render_template(
        "saved_results.html",
        results=results,
        portfolios=portfolios,
    )


@saved_results_bp.route("/<int:result_id>", methods=["GET"])
@login_required
def saved_result_detail(result_id):
    result = PricingResult.query.filter_by(
        id=result_id,
        user_id=current_user.id,
    ).first_or_404()

    return render_template("saved_result_detail.html", result=result)
