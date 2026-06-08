# -*- coding: utf-8 -*-
"""
Created on Fri Jun  7 21:27:49 2024

@author: minwuu01
"""

import os

from flask import Flask

from .config import config_by_name
from .logging_config import configure_logging
from .routes import register_routes


def create_app():
    configure_logging()

    app = Flask(__name__)

    flask_env = os.getenv("FLASK_ENV", "development").strip().lower()
    config_class = config_by_name.get(flask_env, config_by_name["development"])
    app.config.from_object(config_class)

    register_routes(app)

    return app


def launch():
    app = create_app()
    app.run(use_reloader=False, debug=app.config.get("DEBUG", False))