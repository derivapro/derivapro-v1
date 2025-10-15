# -*- coding: utf-8 -*-
"""
Created on Mon Oct 13 17:47:41 2025

@author: minwuu01
"""

from flask import Blueprint, render_template

rates_bp = Blueprint("rates", __name__, url_prefix="/rates")

@rates_bp.get("/swap")
def swap_page():
    return render_template("rates/swap.html", title="Interest Rate Swap (New)")

@rates_bp.get("/swaption")
def swaption_page():
    return render_template("rates/swaption.html", title="European Swaption (New)")
