# -*- coding: utf-8 -*-
"""
Created on Sun Jun  9 00:47:59 2024

@author: minwuu01
"""

from flask import Blueprint, render_template

equity_derivatives_bp = Blueprint('equity_derivatives', __name__)

@equity_derivatives_bp.route('/', methods=['GET', 'POST'])
def equity_derivatives():
    return render_template('equity_derivatives.html')
