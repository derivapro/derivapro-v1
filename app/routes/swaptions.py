
from flask import Blueprint, render_template

swaptions_bp = Blueprint('swaptions', __name__)

@swaptions_bp.route('/', methods=['GET', 'POST'])
def swaptions():
    return render_template('swaptions.html')
