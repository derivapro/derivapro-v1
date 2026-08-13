from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any

from flask import current_app

from ..models.db_models import Instrument, Portfolio, Position, PricingResult


ASSET_CLASS_BY_PRODUCT = {
    "european_option": "Equity Derivatives",
    "american_option": "Equity Derivatives",
    "barrier_option": "Equity Derivatives",
    "asian_option": "Equity Derivatives",
    "barrier_reverse_convertible": "Structured Products",
    "principal_protected_note": "Structured Products",
    "enhanced_participation_note": "Structured Products",
    "contingent_income_note": "Structured Products",
    "credit_linked_note": "Structured Products",
    "autocallable_note": "Structured Products",
    "bond": "Fixed Income",
    "fixed_rate_bond": "Fixed Income",
    "floating_rate_bond": "Fixed Income",
    "callable_putable_bond": "Fixed Income",
    "forward_rate_agreement": "Rates",
    "fra": "Rates",
    "cap_floor": "Rates",
    "interest_rate_cap": "Rates",
    "interest_rate_floor": "Rates",
    "swap": "Rates",
    "swaption": "Rates",
    "forward": "Forwards",
    "future": "Forwards",
    "credit_default_swap": "Credit",
}


def classify_asset_class(product_type: str | None) -> str:
    if not product_type:
        return "Other"
    normalized = product_type.lower().replace("-", "_").replace(" ", "_")
    for key, asset_class in ASSET_CLASS_BY_PRODUCT.items():
        if key in normalized:
            return asset_class
    return "Other"


def infer_underlying(instrument: Instrument | None) -> str | None:
    if not instrument:
        return None
    if instrument.ticker:
        return instrument.ticker
    params = instrument.params_json or {}
    for key in ["ticker", "symbol", "underlying", "underlying_ticker"]:
        value = params.get(key)
        if value:
            return str(value)
    return None


def portfolio_local_root() -> Path:
    configured = current_app.config.get("PORTFOLIO_LOCAL_STORE_DIR")
    if configured:
        return Path(configured)
    return Path(current_app.root_path).parent / "local_data" / "portfolios"


def user_portfolio_dir(user) -> Path:
    safe_user = re.sub(r"[^A-Za-z0-9_.-]+", "_", user.username or f"user_{user.id}")
    return portfolio_local_root() / safe_user


def _safe_file_stem(name: str) -> str:
    stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", name.strip().lower())
    return stem.strip("._-") or "portfolio"


def portfolio_snapshot_path(portfolio: Portfolio, user) -> Path:
    filename = f"{portfolio.id}_{_safe_file_stem(portfolio.name)}.json"
    return user_portfolio_dir(user) / filename


def _pricing_payload(pricing_result: PricingResult | None) -> dict[str, Any] | None:
    if not pricing_result:
        return None
    return {
        "price": pricing_result.price,
        "delta": pricing_result.delta,
        "gamma": pricing_result.gamma,
        "vega": pricing_result.vega,
        "theta": pricing_result.theta,
        "rho": pricing_result.rho,
        "result_json": pricing_result.result_json,
        "created_at": pricing_result.created_at.isoformat()
        if pricing_result.created_at
        else None,
    }


def serialize_portfolio(portfolio: Portfolio) -> dict[str, Any]:
    positions = []
    for position in portfolio.positions:
        instrument = position.instrument
        positions.append(
            {
                "position_label": position.position_label,
                "trade_id": position.trade_id,
                "side": position.side,
                "quantity": position.quantity,
                "notional": position.notional,
                "currency": position.currency,
                "asset_class": position.asset_class,
                "product_category": position.product_category,
                "underlying": position.underlying,
                "valuation_status": position.valuation_status,
                "notes": position.notes,
                "instrument": {
                    "product_type": instrument.product_type if instrument else None,
                    "ticker": instrument.ticker if instrument else None,
                    "model_name": instrument.model_name if instrument else None,
                    "params_json": instrument.params_json if instrument else None,
                },
                "pricing_result": _pricing_payload(position.pricing_result),
                "added_at": position.created_at.isoformat()
                if position.created_at
                else None,
            }
        )

    return {
        "schema_version": "1.0",
        "exported_at": datetime.utcnow().isoformat() + "Z",
        "portfolio": {
            "name": portfolio.name,
            "description": portfolio.description,
            "created_at": portfolio.created_at.isoformat()
            if portfolio.created_at
            else None,
        },
        "positions": positions,
        "notice": (
            "Local DerivaPro portfolio copy. User portfolio files are private local "
            "artifacts and should not be committed to source control."
        ),
    }


def write_portfolio_snapshot(portfolio: Portfolio, user) -> Path:
    output_path = portfolio_snapshot_path(portfolio, user)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(serialize_portfolio(portfolio), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return output_path


def list_local_portfolio_copies(user) -> list[dict[str, Any]]:
    directory = user_portfolio_dir(user)
    if not directory.exists():
        return []
    copies = []
    for path in sorted(directory.glob("*.json"), key=lambda item: item.stat().st_mtime, reverse=True):
        stat = path.stat()
        copies.append(
            {
                "filename": path.name,
                "path": str(path),
                "modified_at": datetime.fromtimestamp(stat.st_mtime),
                "size_kb": round(stat.st_size / 1024, 1),
            }
        )
    return copies


def load_portfolio_snapshot(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or "portfolio" not in payload:
        raise ValueError("Invalid portfolio file. Expected a DerivaPro portfolio JSON payload.")
    return payload
