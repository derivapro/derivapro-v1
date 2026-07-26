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

    greek_fields = ["delta", "gamma", "vega", "theta", "rho"]
    summary = {field: 0.0 for field in greek_fields}
    asset_class_summary = {}

    for position in positions:
        pricing_result = position.pricing_result
        if not pricing_result:
            continue

        multiplier = (
            position.notional if position.notional is not None else position.quantity
        )
        if multiplier is None:
            multiplier = 1.0

        asset_class = (
            position.instrument.product_type
            if position.instrument and position.instrument.product_type
            else "Unknown"
        )

        by_class = asset_class_summary.setdefault(
            asset_class,
            {"name": asset_class, **{field: 0.0 for field in greek_fields}},
        )

        for field in greek_fields:
            value = getattr(pricing_result, field) or 0.0
            exposure = value * multiplier
            summary[field] += exposure
            by_class[field] += exposure

    max_greek = max(abs(value) for value in summary.values()) or 1.0

    return render_template(
        "portfolio_detail.html",
        portfolio=portfolio,
        positions=positions,
        summary=summary,
        asset_class_summary=asset_class_summary,
        max_greek=max_greek,
    )


@portfolios_bp.route("/<int:portfolio_id>/update-position", methods=["POST"])
@login_required
def update_position(portfolio_id):
    position_id = request.form.get("position_id", type=int)
    quantity = request.form.get("quantity", type=float)
    notional = request.form.get("notional", type=float)

    position = Position.query.filter_by(
        id=position_id,
        portfolio_id=portfolio_id,
        user_id=current_user.id,
    ).first()

    if not position:
        flash("Position not found.", "error")
        return redirect(
            url_for("portfolios.portfolio_detail", portfolio_id=portfolio_id)
        )

    if quantity is not None:
        position.quantity = quantity
    if notional is not None:
        position.notional = notional

    db.session.commit()
    flash("Position updated successfully.", "success")
    return redirect(url_for("portfolios.portfolio_detail", portfolio_id=portfolio_id))


@portfolios_bp.route("/<int:portfolio_id>/delete-position", methods=["POST"])
@login_required
def delete_position(portfolio_id):
    position_id = request.form.get("position_id", type=int)

    position = Position.query.filter_by(
        id=position_id,
        portfolio_id=portfolio_id,
        user_id=current_user.id,
    ).first()

    if not position:
        flash("Position not found.", "error")
        return redirect(
            url_for("portfolios.portfolio_detail", portfolio_id=portfolio_id)
        )

    db.session.delete(position)
    db.session.commit()

    flash("Position deleted successfully.", "success")
    return redirect(url_for("portfolios.portfolio_detail", portfolio_id=portfolio_id))


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
