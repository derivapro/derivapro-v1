from flask import Blueprint, render_template
from flask_login import current_user, login_required

from ..models.db_models import AnalysisResult

analysis_history_bp = Blueprint("analysis_history", __name__)


@analysis_history_bp.route("/", methods=["GET"])
@login_required
def analysis_history():
    analyses = (
        AnalysisResult.query.filter_by(user_id=current_user.id)
        .order_by(AnalysisResult.created_at.desc())
        .all()
    )

    return render_template("analysis_history.html", analyses=analyses)