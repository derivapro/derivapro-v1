from flask import Blueprint, flash, redirect, render_template, request, url_for
from flask_login import current_user, login_required

from ..extensions import db
from ..models.db_models import Portfolio, Position, PricingResult

portfolios_bp = Blueprint("portfolios", __name__)


@portfolios_bp.route("/", methods=["GET", "POST"])
@login_required
def portfolios():
    if request.method == "POST":
        name = request.form.get("name", "").strip()
        description = request.form.get("description", "").strip()

        if not name:
            flash("Portfolio name is required.", "error")
            return redirect(url_for("portfolios.portfolios"))

        portfolio = Portfolio(
            user_id=current_user.id,
            name=name,
            description=description or None,
        )
        db.session.add(portfolio)
        db.session.commit()

        flash("Portfolio created successfully.", "success")
        return redirect(url_for("portfolios.portfolios"))

    portfolios_list = (
        Portfolio.query
        .filter_by(user_id=current_user.id)
        .order_by(Portfolio.created_at.desc())
        .all()
    )

    return render_template("portfolios.html", portfolios=portfolios_list)


@portfolios_bp.route("/<int:portfolio_id>", methods=["GET"])
@login_required
def portfolio_detail(portfolio_id):
    portfolio = Portfolio.query.filter_by(
        id=portfolio_id, user_id=current_user.id
    ).first_or_404()

    positions = (
        Position.query
        .filter_by(portfolio_id=portfolio.id, user_id=current_user.id)
        .order_by(Position.created_at.desc())
        .all()
    )

    return render_template(
        "portfolio_detail.html",
        portfolio=portfolio,
        positions=positions,
    )


@portfolios_bp.route("/add-position", methods=["POST"])
@login_required
def add_position():
    portfolio_id = request.form.get("portfolio_id", type=int)
    pricing_result_id = request.form.get("pricing_result_id", type=int)
    quantity = request.form.get("quantity", type=float, default=1.0)
    notional = request.form.get("notional", type=float)

    if not portfolio_id or not pricing_result_id:
        flash("Portfolio and pricing result are required.", "error")
        return redirect(url_for("saved_results.saved_results"))

    portfolio = Portfolio.query.filter_by(
        id=portfolio_id, user_id=current_user.id
    ).first()

    if not portfolio:
        flash("Portfolio not found.", "error")
        return redirect(url_for("saved_results.saved_results"))

    pricing_result = PricingResult.query.filter_by(
        id=pricing_result_id, user_id=current_user.id
    ).first()

    if not pricing_result:
        flash("Pricing result not found.", "error")
        return redirect(url_for("saved_results.saved_results"))

    if quantity is None:
        quantity = 1.0

    position = Position(
        portfolio_id=portfolio.id,
        user_id=current_user.id,
        instrument_id=pricing_result.instrument_id,
        pricing_result_id=pricing_result.id,
        quantity=quantity,
        notional=notional,
    )
    db.session.add(position)
    db.session.commit()

    flash("Position added to portfolio successfully.", "success")
    return redirect(url_for("portfolios.portfolio_detail", portfolio_id=portfolio.id))
