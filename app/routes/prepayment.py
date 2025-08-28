from ..models.mdls_prepayment import Prepayment
from flask import Blueprint, render_template, request, session
import os
import markdown
import matplotlib.pyplot as plt

prepayment_bp = Blueprint('prepayment', __name__)

@prepayment_bp.route('/prepayment-probability-calculator', methods=['GET', 'POST'])

def prepayment():
    readme_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'prepayment.md')
    with open(readme_path, 'r') as readme_file:
        content = readme_file.read()
    md_content = markdown.markdown(content)

    prepayment_results = None

    form_data = {}

    if request.method == 'POST':
        form_data = {
            'orig_rate': float(request.form.get('orig_rate', 0)),
            'market_rate': float(request.form.get('market_rate', 0)),
            'orig_fico': int(request.form.get('orig_fico', 0)),
            'loan_age': int(request.form.get('loan_age', 0)),
            'orig_ltv': int(request.form.get('orig_ltv', 0)),
            'intercept': float(request.form.get('intercept', 0)),
            'beta_spread': float(request.form.get('beta_spread', 0)),
            'beta_fico': float(request.form.get('beta_fico', 0)),
            'beta_loan_age': float(request.form.get('beta_loan_age', 0)),
            'beta_ltv': float(request.form.get('beta_ltv', 0)),
        }

        orig_rate = float(form_data['orig_rate'])
        market_rate = float(form_data['market_rate'])
        orig_fico = int(form_data['orig_fico'])
        loan_age = int(form_data['loan_age'])
        orig_ltv = int(form_data['orig_ltv'])
        intercept = float(form_data['intercept'])
        beta_spread = float(form_data['beta_spread'])
        beta_fico = float(form_data['beta_fico'])
        beta_loan_age = float(form_data['beta_loan_age'])
        beta_ltv = float(form_data['beta_ltv'])

        prepay = Prepayment(
            orig_rate = orig_rate,
            market_rate = market_rate,
            orig_fico = orig_fico,
            loan_age = loan_age,
            orig_ltv = orig_ltv,
            intercept = intercept,
            beta_spread = beta_spread,
            beta_fico = beta_fico,
            beta_loan_age = beta_loan_age,
            beta_ltv = beta_ltv
        )

        prepayment_results = prepay.prepayment_probability()

    return render_template(
        'prepayment_probability_calculator.html', 
        form_data = form_data, 
        prepayment_results = prepayment_results, 
        md_content = md_content
        )
