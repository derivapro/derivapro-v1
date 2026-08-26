from ..models.mdls_variance_volatility_swaps import varianceSwaps
from flask import Blueprint, render_template, request
from flask_login import current_user
import QuantLib as ql
import os
import markdown
from ..extensions import db
from ..models.db_models import Instrument, PricingResult
from ..services.validation import check_positive

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
    validation_errors = []

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

        validation_errors = check_positive(
            {"strike_vol": form_data["strike_vol"], "new_strike_vol": form_data["new_strike_vol"],
             "vega_notional": form_data["vega_notional"], "sigma": form_data["sigma"],
             "kappa": form_data["kappa"], "theta": form_data["theta"]},
            {"strike_vol": "Strike volatility", "new_strike_vol": "New strike volatility",
             "vega_notional": "Vega notional", "sigma": "Volatility of volatility (sigma)",
             "kappa": "Mean reversion speed (kappa)", "theta": "Long-run variance (theta)"},
        )
        if validation_errors:
            return render_template('variance_swaps.html', form_data=form_data, content=md_content,
                                   variance_notional=None, realized_variance=None, settlement_amount=None,
                                   simulated_settlement_amount=None, expected_real_variance=None,
                                   current_value=None, validation_errors=validation_errors)

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

        if current_user.is_authenticated:
            instrument = Instrument(
                user_id=current_user.id,
                product_type="variance_swap",
                ticker=ticker,
                model_name="varianceSwaps",
                start_date=str(start_date),
                end_date=str(end_date),
                params_json={
                    "strike_vol": strike_vol,
                    "new_strike_vol": new_strike_vol,
                    "vega_notional": vega_notional,
                    "risk_free_rate": risk_free_rate,
                    "position": position,
                    "rho": rho,
                    "kappa": kappa,
                    "theta": theta,
                    "sigma": sigma,
                },
            )
            db.session.add(instrument)
            db.session.flush()

            pricing_result = PricingResult(
                user_id=current_user.id,
                instrument_id=instrument.id,
                price=current_value,
                delta=None,
                gamma=None,
                vega=None,
                theta=None,
                rho=None,
                result_json={
                    "variance_notional": variance_notional,
                    "realized_variance": realized_variance,
                    "settlement_amount": settlement_amount,
                    "simulated_settlement_amount": simulated_settlement_amount,
                    "expected_real_variance": expected_real_variance,
                    "current_value": current_value,
                },
            )
            db.session.add(pricing_result)
            db.session.commit()


    return render_template('variance_swaps.html', form_data=form_data, content=md_content, variance_notional=variance_notional, realized_variance=realized_variance, settlement_amount=settlement_amount,
                           simulated_settlement_amount=simulated_settlement_amount, expected_real_variance=expected_real_variance, current_value=current_value, validation_errors=validation_errors)

