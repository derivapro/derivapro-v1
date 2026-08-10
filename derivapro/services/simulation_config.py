from __future__ import annotations

from typing import Any, MutableMapping

from flask_login import current_user
from sqlalchemy.exc import SQLAlchemyError

from ..extensions import db
from ..models.db_models import User, UserSimulationConfig
from ..models.simulation_settings import (
    CONVERGENCE_MODE_CHOICES,
    RANDOM_SEQUENCE_CHOICES,
    RUNTIME_PROFILE_CHOICES,
    SIMULATION_METHOD_CHOICES,
    VARIANCE_REDUCTION_CHOICES,
    SimulationSettings,
    default_simulation_settings,
)


def get_effective_simulation_settings(user: User | None = None) -> SimulationSettings:
    target_user = user if user is not None else current_user
    if not getattr(target_user, "is_authenticated", False):
        return default_simulation_settings()

    try:
        record = UserSimulationConfig.query.filter_by(user_id=target_user.id).first()
    except SQLAlchemyError:
        db.session.rollback()
        return default_simulation_settings()
    if not record:
        return default_simulation_settings()

    return SimulationSettings.from_mapping(record.settings_json)


def save_simulation_settings(user: User, form_data: dict[str, Any]) -> SimulationSettings:
    settings = SimulationSettings.from_mapping(
        {
            "simulation_method": form_data.get("simulation_method"),
            "random_sequence": form_data.get("random_sequence"),
            "num_paths": form_data.get("num_paths"),
            "num_steps": form_data.get("num_steps"),
            "random_seed": form_data.get("random_seed"),
            "variance_reduction": form_data.get("variance_reduction"),
            "convergence_mode": form_data.get("convergence_mode"),
            "target_standard_error": form_data.get("target_standard_error"),
            "runtime_profile": form_data.get("runtime_profile"),
            "apply_globally": form_data.get("apply_globally") == "on",
            "allow_product_override": form_data.get("allow_product_override") == "on",
        }
    )

    record = UserSimulationConfig.query.filter_by(user_id=user.id).first()
    if not record:
        record = UserSimulationConfig(user_id=user.id, settings_json=settings.to_dict())
        db.session.add(record)
    else:
        record.settings_json = settings.to_dict()

    db.session.commit()
    return settings


def reset_simulation_settings(user: User) -> SimulationSettings:
    settings = default_simulation_settings()
    record = UserSimulationConfig.query.filter_by(user_id=user.id).first()
    if not record:
        record = UserSimulationConfig(user_id=user.id, settings_json=settings.to_dict())
        db.session.add(record)
    else:
        record.settings_json = settings.to_dict()
    db.session.commit()
    return settings


def simulation_settings_choices() -> dict[str, dict[str, str]]:
    return {
        "simulation_method": SIMULATION_METHOD_CHOICES,
        "random_sequence": RANDOM_SEQUENCE_CHOICES,
        "variance_reduction": VARIANCE_REDUCTION_CHOICES,
        "convergence_mode": CONVERGENCE_MODE_CHOICES,
        "runtime_profile": RUNTIME_PROFILE_CHOICES,
    }


def apply_simulation_defaults(
    form_data: MutableMapping[str, Any],
    settings: SimulationSettings | None = None,
    *,
    prefix: str = "",
    paths_key: str = "num_paths",
    steps_key: str | None = "num_steps",
    random_key: str | None = "random_type",
    seed_key: str | None = "random_seed",
) -> MutableMapping[str, Any]:
    active_settings = settings or get_effective_simulation_settings()
    if not active_settings.apply_globally:
        return form_data

    key_prefix = f"{prefix}_" if prefix else ""

    form_data[f"{key_prefix}{paths_key}"] = active_settings.num_paths
    if steps_key:
        form_data[f"{key_prefix}{steps_key}"] = active_settings.num_steps
    if random_key:
        form_data[f"{key_prefix}{random_key}"] = active_settings.random_sequence
    if seed_key:
        form_data[f"{key_prefix}{seed_key}"] = active_settings.random_seed
    return form_data


def simulation_audit_payload(settings: SimulationSettings | None = None) -> dict[str, Any]:
    active_settings = settings or get_effective_simulation_settings()
    return {
        **active_settings.to_dict(),
        "display_rows": active_settings.display_rows(),
    }
