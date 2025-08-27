from app.models.mdls_variance_volatility_swaps import varianceSwaps
from flask import Blueprint, render_template, request
import QuantLib as ql
import os
import markdown

# Initialize Flask app
variance_swaps_bp = Blueprint('variance_swaps', __name__)

@variance_swaps_bp.route('/variance-swaps', methods=['GET', 'POST'])

def variance_swaps():
    readme_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'variance_swaps.md')
    with open(readme_path, 'r') as readme_file:
        content = readme_file.read()
    md_content = markdown.markdown(content)

    variance_notional=realized_variance=settlement_amount=simulated_settlement_amount=expected_real_variance=current_value=None

    form_data = {}

    if request.method == 'POST':
        form_data = {
        'ticker': request.form['ticker'],
        'start_date': request.form['start_date'],
        'end_date': request.form['end_date'],
        'as_of_date': request.form['as_of_date'],
        'strike_vol': float(request.form['strike_vol']),
        'new_strike_vol': float(request.form['new_strike_vol']),
        'vega_notional': float(request.form['vega_notional']),
        'risk_free_rate': float(request.form['risk_free_rate']),
        'position': request.form['position'],
        'rho': float(request.form['rho']),
        'kappa': float(request.form['kappa']),
        'theta': float(request.form['theta']),
        'sigma': float(request.form['sigma']),
        'calendar_val': request.form['calendar'],
        }
        
        ticker = form_data['ticker']
        start_date = form_data['start_date']
        end_date = form_data['end_date']
        as_of_date = form_data['as_of_date']
        strike_vol = form_data['strike_vol']
        new_strike_vol = form_data['new_strike_vol']
        vega_notional = form_data['vega_notional']
        risk_free_rate = form_data['risk_free_rate']
        position = form_data['position']
        rho = form_data['rho']
        kappa = form_data['kappa']
        theta = form_data['theta']
        sigma = form_data['sigma']
        
        calendar_val = form_data['calendar_val']
        # Map the calendar value to a QuantLib Calendar
        if calendar_val == 'UnitedStates':
            calendar = ql.UnitedStates(ql.UnitedStates.NYSE)
        elif calendar_val == 'TARGET':
            calendar = ql.TARGET()
        elif calendar_val == 'UnitedKingdom':
            calendar = ql.UnitedKingdom()
        elif calendar_val == 'China':
            calendar = ql.China()
        
        variance_swap = varianceSwaps(ticker, start_date, end_date, as_of_date, strike_vol, new_strike_vol, vega_notional, risk_free_rate, position, 
                                      rho, kappa, theta, sigma, calendar)
        
        variance_notional = variance_swap.variance_notional()
        realized_variance = variance_swap.realized_variance()
        
        settlement_amount = variance_swap.settlement_amount(position)
        simulated_settlement_amount, expected_real_variance = variance_swap.simulated_settlement_amount(position)
        current_value = variance_swap.current_value(position)


    return render_template('variance_swaps.html', form_data=form_data, content=md_content, variance_notional=variance_notional, realized_variance=realized_variance, settlement_amount=settlement_amount,
                           simulated_settlement_amount=simulated_settlement_amount, expected_real_variance=expected_real_variance, current_value=current_value)


