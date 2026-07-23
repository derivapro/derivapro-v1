from typing import Iterable, Optional

from flask_login import current_user

from ..models.db_models import AnalysisResult, Instrument, PricingResult


def get_latest_pricing_result_for_user(
    product_type: str,
    user_id: int,
) -> Optional[PricingResult]:
    return (
        PricingResult.query
        .join(Instrument, PricingResult.instrument_id == Instrument.id)
        .filter(
            PricingResult.user_id == user_id,
            Instrument.user_id == user_id,
            Instrument.product_type == product_type,
        )
        .order_by(PricingResult.created_at.desc())
        .first()
    )


def get_latest_pricing_result_for_user_product_types(
    product_types: Iterable[str],
    user_id: int,
) -> Optional[PricingResult]:
    return (
        PricingResult.query
        .join(Instrument, PricingResult.instrument_id == Instrument.id)
        .filter(
            PricingResult.user_id == user_id,
            Instrument.user_id == user_id,
            Instrument.product_type.in_(list(product_types)),
        )
        .order_by(PricingResult.created_at.desc())
        .first()
    )


def get_latest_analysis_result_for_user(
    product_type: str,
    analysis_type: str,
    user_id: int,
) -> Optional[AnalysisResult]:
    return (
        AnalysisResult.query
        .join(Instrument, AnalysisResult.instrument_id == Instrument.id)
        .filter(
            AnalysisResult.user_id == user_id,
            Instrument.user_id == user_id,
            Instrument.product_type == product_type,
            AnalysisResult.analysis_type == analysis_type,
        )
        .order_by(AnalysisResult.created_at.desc())
        .first()
    )


def resolve_current_pricing_result_id(product_type: str) -> Optional[int]:
    if not current_user.is_authenticated:
        return None
    latest = get_latest_pricing_result_for_user(product_type, current_user.id)
    return latest.id if latest else None
