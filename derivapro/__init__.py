# -*- coding: utf-8 -*-
"""
Created on Fri Jun  7 21:27:49 2024

@author: minwuu01
"""

import os
import logging

from dotenv import load_dotenv
from curl_cffi.requests.exceptions import RequestException as CurlRequestException
from flask import Flask, jsonify, request
from werkzeug.exceptions import BadRequestKeyError

load_dotenv()

from .config import config_by_name
from .extensions import bcrypt, cache, db, login_manager, migrate
from .logging_config import configure_logging

logger = logging.getLogger(__name__)


def create_app():
    configure_logging()

    app = Flask(__name__)

    flask_env = os.getenv("FLASK_ENV", "development").strip().lower()
    config_class = config_by_name.get(flask_env, config_by_name["development"])
    app.config.from_object(config_class)

    db.init_app(app)
    migrate.init_app(app, db)
    login_manager.init_app(app)
    bcrypt.init_app(app)
    cache.init_app(app)
    from .extensions import csrf
    from .routes import register_routes

    csrf.init_app(app)

    from .models import db_models  # noqa: F401

    register_routes(app)

    @app.errorhandler(CurlRequestException)
    def handle_external_market_data_error(error):
        logger.exception("External market data request failed")

        payload = {
            "error": "External market data is currently unavailable.",
            "detail": "Please verify the ticker or try again later.",
        }
        if request.accept_mimetypes.best == "application/json":
            return jsonify(payload), 503

        return (
            "External market data is currently unavailable. "
            "Please verify the ticker or try again later.",
            503,
        )

    @app.errorhandler(BadRequestKeyError)
    def handle_bad_request_key(error):
        missing_field = getattr(error, "args", [None])[0]
        logger.warning("Missing required form field: %s", missing_field)

        payload = {
            "error": "Missing required form field.",
            "field": missing_field,
        }
        if request.accept_mimetypes.best == "application/json":
            return jsonify(payload), 400

        return (
            f"Missing required form field: {missing_field}. "
            "Please complete the required inputs and submit again.",
            400,
        )

    return app


def launch():
    app = create_app()
    debug_enabled = app.config.get("DEBUG", False)
    app.run(
        debug=debug_enabled,
        use_reloader=debug_enabled,
    )
