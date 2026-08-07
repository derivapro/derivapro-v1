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
    SQLALCHEMY_DATABASE_URI = os.getenv("DATABASE_URL", "sqlite:///derivapro.db")
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    SESSION_COOKIE_HTTPONLY = True
    SESSION_COOKIE_SAMESITE = os.getenv("SESSION_COOKIE_SAMESITE", "Lax")
    SESSION_COOKIE_SECURE = _env_flag("SESSION_COOKIE_SECURE", "false")
    REMEMBER_COOKIE_HTTPONLY = True
    REMEMBER_COOKIE_SAMESITE = os.getenv("REMEMBER_COOKIE_SAMESITE", "Lax")
    REMEMBER_COOKIE_SECURE = _env_flag("REMEMBER_COOKIE_SECURE", "false")
    MAX_CONTENT_LENGTH = int(os.getenv("MAX_CONTENT_LENGTH", str(16 * 1024 * 1024)))
    PASSWORD_MIN_LENGTH = int(os.getenv("PASSWORD_MIN_LENGTH", "12"))
    AUTH_RATE_LIMIT_ATTEMPTS = int(os.getenv("AUTH_RATE_LIMIT_ATTEMPTS", "5"))
    AUTH_RATE_LIMIT_WINDOW_SECONDS = int(
        os.getenv("AUTH_RATE_LIMIT_WINDOW_SECONDS", "900")
    )
    PASSWORD_RESET_TOKEN_MAX_AGE_SECONDS = int(
        os.getenv("PASSWORD_RESET_TOKEN_MAX_AGE_SECONDS", "1800")
    )
    PREPAYMENT_UPLOAD_ROOT = os.getenv(
        "PREPAYMENT_UPLOAD_ROOT",
        "derivapro/static/uploads",
    )
    PREPAYMENT_TEMP_MODEL_DIR = os.getenv(
        "PREPAYMENT_TEMP_MODEL_DIR",
        "derivapro/static/temp_models",
    )
    PREPAYMENT_MODEL_REGISTRY_DIR = os.getenv(
        "PREPAYMENT_MODEL_REGISTRY_DIR",
        "derivapro/static/model_registry",
    )
    PREPAYMENT_MODEL_STORAGE_BACKEND = os.getenv(
        "PREPAYMENT_MODEL_STORAGE_BACKEND",
        "local",
    )
    PREPAYMENT_S3_BUCKET = os.getenv("PREPAYMENT_S3_BUCKET", "")
    PREPAYMENT_S3_PREFIX = os.getenv("PREPAYMENT_S3_PREFIX", "prepayment-models/")
    PREPAYMENT_S3_ENDPOINT_URL = os.getenv("PREPAYMENT_S3_ENDPOINT_URL", "")
    PREPAYMENT_S3_REGION = os.getenv("PREPAYMENT_S3_REGION", "")

    # ---------- Flask-Caching ----------
    # Configurable TTL for external market-data fetches (yfinance / FRED / SOFR).
    # Set MARKET_DATA_CACHE_TTL=0 in the environment to disable caching.
    MARKET_DATA_CACHE_TTL: int = int(os.getenv("MARKET_DATA_CACHE_TTL", "300"))  # 5 min
    CACHE_DEFAULT_TIMEOUT: int = int(os.getenv("CACHE_DEFAULT_TIMEOUT", "300"))


class DevelopmentConfig(Config):
    DEBUG = _env_flag("FLASK_DEBUG", "true")

    # Simple in-process memory cache – no external dependency in dev
    CACHE_TYPE: str = "SimpleCache"
    CACHE_THRESHOLD: int = 2000  # max items in the SimpleCache store


class ProductionConfig(Config):
    DEBUG = False

    # Redis in production. Set REDIS_URL in the environment (e.g. redis://localhost:6379/0).
    CACHE_TYPE: str = "RedisCache"
    CACHE_REDIS_URL: str = os.getenv("REDIS_URL", "redis://localhost:6379/0")
    CACHE_KEY_PREFIX: str = "derivapro:"
    SESSION_COOKIE_SECURE = _env_flag("SESSION_COOKIE_SECURE", "true")
    REMEMBER_COOKIE_SECURE = _env_flag("REMEMBER_COOKIE_SECURE", "true")


config_by_name = {
    "development": DevelopmentConfig,
    "production": ProductionConfig,
}
