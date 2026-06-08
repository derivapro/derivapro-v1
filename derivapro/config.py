# -*- coding: utf-8 -*-
"""
Application configuration.
"""

import os

from .secret_key_utils import ensure_local_secret_key


def _env_flag(name: str, default: str = "false") -> bool:
    return os.getenv(name, default).strip().lower() in {"1", "true", "yes", "on"}


def _get_flask_env() -> str:
    return os.getenv("FLASK_ENV", "development").strip().lower()


def _get_secret_key() -> str:
    secret_key = os.getenv("SECRET_KEY")
    if secret_key:
        return secret_key

    flask_env = _get_flask_env()
    if flask_env == "development":
        return ensure_local_secret_key()

    raise RuntimeError(
        "SECRET_KEY is not set. Configure it in the environment for non-development environments."
    )


class Config:
    SECRET_KEY = _get_secret_key()
    TESTING = False
    DEBUG = False


class DevelopmentConfig(Config):
    DEBUG = _env_flag("FLASK_DEBUG", "true")


class ProductionConfig(Config):
    DEBUG = False


config_by_name = {
    "development": DevelopmentConfig,
    "production": ProductionConfig,
}
