
from flask import Blueprint, render_template

swap_swaptions_bp = Blueprint('swap_swaptions', __name__, url_prefix='/swap_swaptions')

@swap_swaptions_bp.route('/')
def swap_swaptions():
    return render_template('swaps_swaptions.html')