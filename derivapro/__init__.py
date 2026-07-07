# -*- coding: utf-8 -*-
"""
Created on Fri Jun  7 21:27:49 2024

@author: minwuu01
"""

import os

from dotenv import load_dotenv
from flask import Flask

load_dotenv()

from .config import config_by_name
from .extensions import bcrypt, db, login_manager, migrate
from .logging_config import configure_logging


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
    from .extensions import csrf
    from .routes import register_routes

    csrf.init_app(app)

    from .models import db_models  # noqa: F401

    register_routes(app)

    return app


def launch():
    app = create_app()
    app.run(use_reloader=False, debug=app.config.get("DEBUG", False))
