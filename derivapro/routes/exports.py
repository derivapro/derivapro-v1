from flask import Blueprint, send_file, request, abort
from flask_login import login_required, current_user
from io import BytesIO

from ..models.db_models import PricingResult, Portfolio, Position
from ..utils.export_utils import dicts_to_xlsx_bytes, dicts_to_csv_bytes

exports_bp = Blueprint("exports", __name__)


@exports_bp.route("/export")
@login_required
def export():
    kind = request.args.get("type")
    fmt = request.args.get("format", "xlsx").lower()

    if kind == "saved_results":
        results = (
            PricingResult.query
            .filter_by(user_id=current_user.id)
            .order_by(PricingResult.created_at.desc())
            .all()
        )
        rows = []
        for r in results:
            rows.append({
                "id": r.id,
                "created_at": r.created_at,
                "product_type": r.instrument.product_type if r.instrument else "",
                "ticker": r.instrument.ticker if r.instrument else "",
                "model": r.instrument.model_name if r.instrument else "",
                "price": r.price,
                "delta": r.delta,
                "gamma": r.gamma,
                "vega": r.vega,
                "theta": r.theta,
                "rho": r.rho,
            })
        filename = f"saved_results_{current_user.id}"

    elif kind == "portfolio_positions":
        portfolio_id = request.args.get("portfolio_id", type=int)
        if not portfolio_id:
            abort(400)
        portfolio = Portfolio.query.filter_by(
            id=portfolio_id, user_id=current_user.id
        ).first()
        if not portfolio:
            abort(404)

        rows = []
        for p in portfolio.positions:
            rows.append({
                "position_id": p.id,
                "added_at": p.created_at,
                "product_type": p.instrument.product_type if p.instrument else "",
                "ticker": p.instrument.ticker if p.instrument else "",
                "model": p.instrument.model_name if p.instrument else "",
                "price": p.pricing_result.price if p.pricing_result else None,
                "quantity": p.quantity,
                "notional": p.notional,
            })
        filename = f"portfolio_{portfolio.id}_positions"

    elif kind == "portfolios_list":
        from ..models.db_models import Portfolio as PortfolioModel

        portfolios = (
            PortfolioModel.query
            .filter_by(user_id=current_user.id)
            .order_by(PortfolioModel.created_at.desc())
            .all()
        )
        rows = [
            {
                "id": p.id,
                "name": p.name,
                "description": p.description,
                "created_at": p.created_at,
            }
            for p in portfolios
        ]
        filename = f"portfolios_{current_user.id}"

    elif kind == "portfolio_summary":
        portfolio_id = request.args.get("portfolio_id", type=int)
        if not portfolio_id:
            abort(400)
        portfolio = Portfolio.query.filter_by(
            id=portfolio_id, user_id=current_user.id
        ).first()
        if not portfolio:
            abort(404)

        # Build greek summary by asset class similar to portfolio_detail view
        # Recompute locally for export
        rows = []
        from collections import defaultdict

        greek_fields = ["delta", "gamma", "vega", "theta", "rho"]
        asset_class_summary = defaultdict(lambda: {f: 0.0 for f in greek_fields})

        for p in portfolio.positions:
            pr = p.pricing_result
            if not pr:
                continue
            multiplier = p.notional if p.notional is not None else p.quantity
            if multiplier is None:
                multiplier = 1.0
            asset_class = (
                p.instrument.product_type
                if p.instrument and p.instrument.product_type
                else "Unknown"
            )
            for f in greek_fields:
                val = getattr(pr, f) or 0.0
                asset_class_summary[asset_class][f] += val * multiplier

        for asset, data in asset_class_summary.items():
            row = {"asset_class": asset}
            row.update(data)
            rows.append(row)

        filename = f"portfolio_{portfolio.id}_summary"

    else:
        abort(400)

    if fmt == "xlsx":
        bio = dicts_to_xlsx_bytes(rows)
        mimetype = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        out_name = f"{filename}.xlsx"
    else:
        bio = dicts_to_csv_bytes(rows)
        mimetype = "text/csv"
        out_name = f"{filename}.csv"

    return send_file(bio, as_attachment=True, download_name=out_name, mimetype=mimetype)
