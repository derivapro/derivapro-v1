from flask import Blueprint, render_template, redirect, url_for
from flask_login import current_user, login_required
from ..models.db_models import Report

reports_generated_bp = Blueprint('reports_generated', __name__)

@reports_generated_bp.route('/reports-generated', methods=['GET'])
@login_required
def reports_generated():
    reports = Report.query.filter_by(user_id=current_user.id).order_by(Report.created_at.desc()).all()
    return render_template('reports_generated.html', reports=reports)

