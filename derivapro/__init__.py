# -*- coding: utf-8 -*-
"""
Created on Fri Jun  7 21:27:49 2024

@author: minwuu01
"""

from flask import Flask
from .routes import register_routes

def create_app():
    app = Flask(__name__)
    app.config.from_object('derivapro.config.Config')  # updated package name

    register_routes(app)

    return app

def launch():
    app = create_app()
    app.run(debug=True)

