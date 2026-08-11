import json
from pathlib import Path

from flask import (
    Blueprint,
    abort,
    flash,
    redirect,
    render_template,
    request,
    send_file,
    url_for,
)
from flask_login import current_user, login_required

from ..extensions import db
from ..models.db_models import Instrument, Portfolio, Position, PricingResult
from ..services.portfolio_store import (
    classify_asset_class,
    infer_underlying,
    list_local_portfolio_copies,
    load_portfolio_snapshot,
    portfolio_snapshot_path,
    user_portfolio_dir,
    write_portfolio_snapshot,
)

portfolios_bp = Blueprint("portfolios", __name__)


def _position_sign(position: Position) -> float:
    return -1.0 if position.side == "short" else 1.0


def _position_multiplier(position: Position) -> float:
    multiplier = position.notional if position.notional is not None else position.quantity
    if multiplier is None:
        multiplier = 1.0
    return float(multiplier) * _position_sign(position)


def _optional_float(value):
    if value in {None, ""}:
        return None
    return float(value)


def _portfolio_snapshot_or_none(portfolio: Portfolio):
    try:
        return write_portfolio_snapshot(portfolio, current_user)
    except OSError as exc:
        flash(f"Portfolio saved, but local JSON copy could not be written: {exc}", "error")
        return None


def _position_defaults(pricing_result: PricingResult):
    instrument = pricing_result.instrument
    product_type = instrument.product_type if instrument else None
    return {
        "asset_class": classify_asset_class(product_type),
        "product_category": product_type or "Unknown",
        "underlying": infer_underlying(instrument),
        "currency": "USD",
    }


def _create_imported_instrument(position_payload):
    instrument_payload = position_payload.get("instrument") or {}
    instrument = Instrument(
        user_id=current_user.id,
        product_type=instrument_payload.get("product_type") or "imported_position",
        ticker=instrument_payload.get("ticker"),
        model_name=instrument_payload.get("model_name") or "imported_snapshot",
        start_date=None,
        end_date=None,
        params_json=instrument_payload.get("params_json") or {},
    )
    db.session.add(instrument)
    db.session.flush()
    return instrument


def _create_imported_pricing_result(position_payload, instrument):
    pricing_payload = position_payload.get("pricing_result") or {}
    if not pricing_payload:
        return None
    pricing_result = PricingResult(
        user_id=current_user.id,
        instrument_id=instrument.id,
        price=_optional_float(pricing_payload.get("price")),
        delta=_optional_float(pricing_payload.get("delta")),
        gamma=_optional_float(pricing_payload.get("gamma")),
        vega=_optional_float(pricing_payload.get("vega")),
        theta=_optional_float(pricing_payload.get("theta")),
        rho=_optional_float(pricing_payload.get("rho")),
        result_json=pricing_payload.get("result_json") or {},
    )
    db.session.add(pricing_result)
    db.session.flush()
    return pricing_result


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
        _portfolio_snapshot_or_none(portfolio)

        flash("Portfolio created successfully.", "success")
        return redirect(url_for("portfolios.portfolios"))

    portfolios_list = (
        Portfolio.query
        .filter_by(user_id=current_user.id)
        .order_by(Portfolio.created_at.desc())
        .all()
    )

    return render_template(
        "portfolios.html",
        portfolios=portfolios_list,
        local_copies=list_local_portfolio_copies(current_user),
    )


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
    portfolio_metrics = {
        "position_count": len(positions),
        "total_market_value": 0.0,
        "long_notional": 0.0,
        "short_notional": 0.0,
        "priced_position_count": 0,
    }
    asset_class_summary = {}
    underlying_summary = {}

    for position in positions:
        pricing_result = position.pricing_result
        if not pricing_result:
            continue

        multiplier = _position_multiplier(position)
        absolute_notional = abs(
            position.notional if position.notional is not None else position.quantity
        )
        if position.side == "short":
            portfolio_metrics["short_notional"] += absolute_notional
        else:
            portfolio_metrics["long_notional"] += absolute_notional
        portfolio_metrics["priced_position_count"] += 1
        portfolio_metrics["total_market_value"] += (pricing_result.price or 0.0) * multiplier

        asset_class = (
            position.asset_class
            or classify_asset_class(
                position.instrument.product_type if position.instrument else None
            )
        )
        underlying = position.underlying or infer_underlying(position.instrument) or "Unspecified"

        by_class = asset_class_summary.setdefault(
            asset_class,
            {
                "name": asset_class,
                "market_value": 0.0,
                "position_count": 0,
                **{field: 0.0 for field in greek_fields},
            },
        )
        by_underlying = underlying_summary.setdefault(
            underlying,
            {"name": underlying, "market_value": 0.0, "position_count": 0},
        )
        by_class["market_value"] += (pricing_result.price or 0.0) * multiplier
        by_class["position_count"] += 1
        by_underlying["market_value"] += (pricing_result.price or 0.0) * multiplier
        by_underlying["position_count"] += 1

        for field in greek_fields:
            value = getattr(pricing_result, field) or 0.0
            exposure = value * multiplier
            summary[field] += exposure
            by_class[field] += exposure

    max_greek = max(abs(value) for value in summary.values()) or 1.0
    local_snapshot_path = portfolio_snapshot_path(portfolio, current_user)

    return render_template(
        "portfolio_detail.html",
        portfolio=portfolio,
        positions=positions,
        summary=summary,
        portfolio_metrics=portfolio_metrics,
        asset_class_summary=asset_class_summary,
        underlying_summary=underlying_summary,
        max_greek=max_greek,
        local_snapshot_path=local_snapshot_path,
        local_snapshot_exists=local_snapshot_path.exists(),
    )


@portfolios_bp.route("/<int:portfolio_id>/update-position", methods=["POST"])
@login_required
def update_position(portfolio_id):
    position_id = request.form.get("position_id", type=int)
    quantity = request.form.get("quantity", type=float)
    notional = _optional_float(request.form.get("notional"))
    side = request.form.get("side", "long")
    position_label = request.form.get("position_label", "").strip()
    currency = request.form.get("currency", "USD").strip().upper() or "USD"
    asset_class = request.form.get("asset_class", "").strip()
    underlying = request.form.get("underlying", "").strip()
    notes = request.form.get("notes", "").strip()

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
    position.notional = notional
    position.side = "short" if side == "short" else "long"
    position.position_label = position_label or None
    position.currency = currency
    position.asset_class = asset_class or position.asset_class
    position.underlying = underlying or None
    position.notes = notes or None

    db.session.commit()
    _portfolio_snapshot_or_none(position.portfolio)
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

    portfolio = position.portfolio
    db.session.delete(position)
    db.session.commit()
    _portfolio_snapshot_or_none(portfolio)

    flash("Position deleted successfully.", "success")
    return redirect(url_for("portfolios.portfolio_detail", portfolio_id=portfolio_id))


@portfolios_bp.route("/add-position", methods=["POST"])
@login_required
def add_position():
    portfolio_id = request.form.get("portfolio_id", type=int)
    pricing_result_id = request.form.get("pricing_result_id", type=int)
    quantity = request.form.get("quantity", type=float, default=1.0)
    notional = _optional_float(request.form.get("notional"))
    side = request.form.get("side", "long")
    position_label = request.form.get("position_label", "").strip()
    currency = request.form.get("currency", "USD").strip().upper() or "USD"
    notes = request.form.get("notes", "").strip()

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

    defaults = _position_defaults(pricing_result)
    position = Position(
        portfolio_id=portfolio.id,
        user_id=current_user.id,
        instrument_id=pricing_result.instrument_id,
        pricing_result_id=pricing_result.id,
        quantity=quantity,
        notional=notional,
        side="short" if side == "short" else "long",
        position_label=position_label or None,
        currency=currency,
        asset_class=defaults["asset_class"],
        product_category=defaults["product_category"],
        underlying=defaults["underlying"],
        notes=notes or None,
    )
    db.session.add(position)
    db.session.commit()
    _portfolio_snapshot_or_none(portfolio)

    flash("Position added to portfolio successfully.", "success")
    return redirect(url_for("portfolios.portfolio_detail", portfolio_id=portfolio.id))


@portfolios_bp.route("/<int:portfolio_id>/export-json", methods=["GET"])
@login_required
def export_portfolio_json(portfolio_id):
    portfolio = Portfolio.query.filter_by(
        id=portfolio_id, user_id=current_user.id
    ).first_or_404()
    snapshot_path = write_portfolio_snapshot(portfolio, current_user)
    return send_file(
        snapshot_path,
        as_attachment=True,
        download_name=snapshot_path.name,
        mimetype="application/json",
    )


@portfolios_bp.route("/import-local-copy", methods=["POST"])
@login_required
def import_local_copy():
    filename = Path(request.form.get("filename", "")).name
    if not filename:
        flash("Select a local portfolio JSON file to import.", "error")
        return redirect(url_for("portfolios.portfolios"))

    local_path = user_portfolio_dir(current_user) / filename
    if not local_path.exists() or local_path.suffix.lower() != ".json":
        abort(404)

    return _import_portfolio_payload(load_portfolio_snapshot(local_path))


@portfolios_bp.route("/import-upload", methods=["POST"])
@login_required
def import_upload():
    uploaded_file = request.files.get("portfolio_file")
    if not uploaded_file or not uploaded_file.filename:
        flash("Choose a portfolio JSON file to import.", "error")
        return redirect(url_for("portfolios.portfolios"))

    try:
        payload = json.load(uploaded_file.stream)
        if not isinstance(payload, dict) or "portfolio" not in payload:
            raise ValueError("Invalid portfolio file. Expected a DerivaPro portfolio JSON payload.")
    except Exception as exc:
        flash(f"Portfolio import failed: {exc}", "error")
        return redirect(url_for("portfolios.portfolios"))

    return _import_portfolio_payload(payload)


def _import_portfolio_payload(payload):
    portfolio_payload = payload.get("portfolio") or {}
    name = portfolio_payload.get("name") or "Imported Portfolio"
    portfolio = Portfolio(
        user_id=current_user.id,
        name=f"{name} (Imported)",
        description=portfolio_payload.get("description"),
    )
    db.session.add(portfolio)
    db.session.flush()

    imported_count = 0
    for position_payload in payload.get("positions", []):
        instrument = _create_imported_instrument(position_payload)
        pricing_result = _create_imported_pricing_result(position_payload, instrument)
        product_type = instrument.product_type
        position = Position(
            portfolio_id=portfolio.id,
            user_id=current_user.id,
            instrument_id=instrument.id,
            pricing_result_id=pricing_result.id if pricing_result else None,
            quantity=float(position_payload.get("quantity") or 1.0),
            notional=_optional_float(position_payload.get("notional")),
            side="short" if position_payload.get("side") == "short" else "long",
            position_label=position_payload.get("position_label"),
            currency=(position_payload.get("currency") or "USD").upper(),
            asset_class=position_payload.get("asset_class")
            or classify_asset_class(product_type),
            product_category=position_payload.get("product_category") or product_type,
            underlying=position_payload.get("underlying") or infer_underlying(instrument),
            notes=position_payload.get("notes"),
        )
        db.session.add(position)
        imported_count += 1

    db.session.commit()
    _portfolio_snapshot_or_none(portfolio)
    flash(
        f"Imported portfolio '{portfolio.name}' with {imported_count} positions.",
        "success",
    )
    return redirect(url_for("portfolios.portfolio_detail", portfolio_id=portfolio.id))
