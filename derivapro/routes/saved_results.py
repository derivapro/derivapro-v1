from flask import Blueprint, render_template
from flask_login import current_user, login_required

from ..models.db_models import PricingResult

saved_results_bp = Blueprint("saved_results", __name__)


@saved_results_bp.route("/", methods=["GET"])
@login_required
def saved_results():
    results = (
        PricingResult.query.filter_by(user_id=current_user.id)
        .order_by(PricingResult.created_at.desc())
        .all()
    )

    return render_template("saved_results.html", results=results)