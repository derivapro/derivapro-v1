from ..models.mdls_bonds import NCFixedBonds, NCFloatingBonds
from flask import Blueprint, render_template, request
import QuantLib as ql
import os
import markdown

# Initialize Flask app
swaps_bp = Blueprint('swaps', __name__)

@swaps_bp.route('/', methods=['GET', 'POST'])
def swaps():

    readme_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'swaps.md')
    with open(readme_path, 'r') as readme_file:
        content = readme_file.read()
    md_content = markdown.markdown(content)
    
    pay_results = pay_leg = rec_results = rec_leg = swap_price = None
    pay_results = rec_results = pay_leg = rec_leg = swap_price = []  # Initialize as empty lists --> Edward's Edit

    form_data = {}
    
    if request.method == 'POST':     
        # Retrieve form data from URL parameters
        form_data = {
            'pay_flag': request.form['pay_flag'],
            'rec_flag': request.form['rec_flag'],
            'value_date': request.form['value_date'],
            'calendar_val': request.form['calendar'],
            'interpolation_val': request.form['interpolation'],
            'compounding_val': request.form['compounding'],
            'currency_val': request.form['currency'],
            'shocks': request.form['shocks'],
            'pay_spot_dates': request.form['pay_spot_dates'],
            'pay_spot_rates': request.form['pay_spot_rates'],
            'pay_index_dates': request.form['pay_index_dates'],
            'pay_index_rates': request.form['pay_index_rates'],
            'pay_day_count_val': request.form['pay_day_count'],
            'pay_compounding_frequency_val': request.form['pay_compounding_frequency'],
            'pay_issue_date': request.form['pay_issue_date'],
            'pay_maturity_date': request.form['pay_maturity_date'],
            'pay_tenor_val': request.form['pay_tenor'],
            'pay_coupon_rate': request.form['pay_coupon_rate'],
            'pay_spread': request.form['pay_spread'],
            'pay_notional': request.form['pay_notional'],
            'rec_spot_dates': request.form['rec_spot_dates'],
            'rec_spot_rates': request.form['rec_spot_rates'],
            'rec_index_dates': request.form['rec_index_dates'],
            'rec_index_rates': request.form['rec_index_rates'],
            'rec_day_count_val': request.form['rec_day_count'],
            'rec_compounding_frequency_val': request.form['rec_compounding_frequency'],
            'rec_issue_date': request.form['rec_issue_date'],
            'rec_maturity_date': request.form['rec_maturity_date'],
            'rec_tenor_val': request.form['rec_tenor'],
            'rec_coupon_rate': request.form['rec_coupon_rate'],
            'rec_spread': request.form['rec_spread'],
            'rec_notional': request.form['rec_notional']
        }

        pay_flag = form_data['pay_flag']
        rec_flag = form_data['rec_flag']

        value_date = form_data['value_date']
        calendar_val = form_data['calendar_val']
        interpolation_val = form_data['interpolation_val']
        compounding_val = form_data['compounding_val']
        currency_val = form_data['currency_val']
        shocks = form_data['shocks']
        
        pay_spot_dates = form_data['pay_spot_dates']
        pay_spot_rates = form_data['pay_spot_rates']
        pay_index_dates = form_data['pay_index_dates']
        pay_index_rates = form_data['pay_index_rates']
        pay_issue_date = form_data['pay_issue_date']
        pay_maturity_date = form_data['pay_maturity_date']
        pay_coupon_rate = form_data['pay_coupon_rate']
        pay_spread = form_data['pay_spread']
        pay_notional = form_data['pay_notional']
        
        rec_spot_dates = form_data['rec_spot_dates']
        rec_spot_rates = form_data['rec_spot_rates']
        rec_index_dates = form_data['rec_index_dates']
        rec_index_rates = form_data['rec_index_rates']
        rec_issue_date = form_data['rec_issue_date']
        rec_maturity_date = form_data['rec_maturity_date']
        rec_coupon_rate = form_data['rec_coupon_rate']
        rec_spread = form_data['rec_spread']
        rec_notional = form_data['rec_notional']
               
        rec_compounding_frequency_val = form_data['rec_compounding_frequency_val']
        pay_compounding_frequency_val = form_data['pay_compounding_frequency_val']
        pay_tenor_val = form_data['pay_tenor_val']
        rec_tenor_val = form_data['rec_tenor_val']
        pay_day_count_val = form_data['pay_day_count_val']
        rec_day_count_val = form_data['rec_day_count_val']
        
        # Map the calendar value to a QuantLib Calendar
        if calendar_val == 'UnitedStates':
            calendar = ql.UnitedStates(ql.UnitedStates.NYSE)
        elif calendar_val == 'TARGET':
            calendar = ql.TARGET()
        elif calendar_val == 'UnitedKingdom':
            calendar = ql.UnitedKingdom()
        elif calendar_val == 'China':
            calendar = ql.China()
               
        # Map interpolation value to a QuantLib interpolation type
        if interpolation_val == 'Linear':
            interpolation = ql.Linear()
        elif interpolation_val == 'LogLinear':
            interpolation = ql.LogLinear()
        elif interpolation_val == 'Cubic':
            interpolation = ql.Cubic()
        
        # Map compounding value to a QuantLib Compounding type
        if compounding_val == 'Compounded':
            compounding = ql.Compounded
        elif compounding_val == 'Simple':
            compounding = ql.Simple
        elif compounding_val == 'Continuous':
            compounding = ql.Continuous
            
        # Map the currency value to a QuantLib Currency
        if currency_val == 'USD':
            currency = ql.USDCurrency()
        elif currency_val == 'EUR':
            currency = ql.EURCurrency()
        elif currency_val == 'CNY':
            currency = ql.CNYCurrency()
        elif currency_val == 'GBP':
            currency = ql.GBPCurrency()
        elif currency_val == 'JPY':
            currency = ql.JPYCurrency()
        
        # Map comp_freq value to QuantLib
        if pay_compounding_frequency_val == 'Annual':
            pay_compounding_frequency = ql.Annual
        elif pay_compounding_frequency_val == 'Semiannual':
            pay_compounding_frequency = ql.Semiannual
        elif pay_compounding_frequency_val == 'Quarterly':
            pay_compounding_frequency = ql.Quarterly
        elif pay_compounding_frequency_val == 'Monthly':
            pay_compounding_frequency = ql.Monthly
        elif pay_compounding_frequency_val == 'Daily':
            pay_compounding_frequency = ql.Daily
        
        # Map comp_freq value to QuantLib 
        if rec_compounding_frequency_val == 'Annual':
            rec_compounding_frequency = ql.Annual
        elif rec_compounding_frequency_val == 'Semiannual':
            rec_compounding_frequency = ql.Semiannual
        elif rec_compounding_frequency_val == 'Quarterly':
            rec_compounding_frequency = ql.Quarterly
        elif rec_compounding_frequency_val == 'Monthly':
            rec_compounding_frequency = ql.Monthly
        elif rec_compounding_frequency_val == 'Daily':
            rec_compounding_frequency = ql.Daily

        # Map pay_tenor_val to QuantLib Tenor
        if pay_tenor_val == 'Annual':
            pay_tenor = ql.Period(ql.Annual)
        elif pay_tenor_val == 'Semiannual':
            pay_tenor = ql.Period(ql.Semiannual)
        elif pay_tenor_val == 'Quarterly':
            pay_tenor = ql.Period(ql.Quarterly)
        elif pay_tenor_val == 'Monthly':
            pay_tenor = ql.Period(ql.Monthly)
        
        # Map pay_day_count_val to a QuantLib DayCount
        if pay_day_count_val == 'ActualActual':
            pay_day_count = ql.ActualActual(ql.ActualActual.Bond)
        elif pay_day_count_val == 'Thirty360':
            pay_day_count = ql.Thirty360(ql.Thirty360.BondBasis)
        elif pay_day_count_val == 'Actual360':
            pay_day_count = ql.Actual360()
        elif pay_day_count_val == 'Actual365Fixed':
            pay_day_count = ql.Actual365Fixed()
        
        # Block for rec_ variables
        
        # Map rec_tenor_val to QuantLib Tenor
        if rec_tenor_val == 'Annual':
            rec_tenor = ql.Period(ql.Annual)
        elif rec_tenor_val == 'Semiannual':
            rec_tenor = ql.Period(ql.Semiannual)
        elif rec_tenor_val == 'Quarterly':
            rec_tenor = ql.Period(ql.Quarterly)
        elif rec_tenor_val == 'Monthly':
            rec_tenor = ql.Period(ql.Monthly)
        
        # Map rec_day_count_val to a QuantLib DayCount
        if rec_day_count_val == 'ActualActual':
            rec_day_count = ql.ActualActual(ql.ActualActual.Bond)
        elif rec_day_count_val == 'Thirty360':
            rec_day_count = ql.Thirty360(ql.Thirty360.BondBasis)
        elif rec_day_count_val == 'Actual360':
            rec_day_count = ql.Actual360()
        elif rec_day_count_val == 'Actual365Fixed':
            rec_day_count = ql.Actual365Fixed()
        
        if pay_flag == 'Fixed':    
            pay_leg = NCFixedBonds(value_date, pay_spot_dates, pay_spot_rates,
                                   shocks, pay_day_count, calendar, interpolation,
                                   compounding, pay_compounding_frequency).fixed_rate(
                                       pay_issue_date, pay_maturity_date, pay_tenor,
                                       pay_coupon_rate, pay_notional)
        if pay_flag == 'Floating':
            pay_leg = NCFloatingBonds(value_date, pay_spot_dates, pay_spot_rates,
                                      pay_index_dates, pay_index_rates, calendar,
                                      currency, interpolation, compounding,
                                      pay_compounding_frequency).price_floating(
                                          shocks, pay_issue_date, pay_maturity_date, pay_tenor,
                                          pay_spread, pay_notional, pay_day_count)
        if rec_flag == 'Floating':
             rec_leg = NCFloatingBonds(value_date, rec_spot_dates, rec_spot_rates,
                                      rec_index_dates, rec_index_rates, calendar,
                                      currency, interpolation, compounding,
                                      rec_compounding_frequency).price_floating(
                                          shocks, rec_issue_date, rec_maturity_date, rec_tenor,
                                          rec_spread, rec_notional, rec_day_count)
        if rec_flag == 'Fixed':
            rec_leg = NCFixedBonds(value_date, rec_spot_dates, rec_spot_rates,
                                   shocks, rec_day_count, calendar, interpolation,
                                   compounding, rec_compounding_frequency).fixed_rate(
                                       rec_issue_date, rec_maturity_date, rec_tenor,
                                       rec_coupon_rate, rec_notional)
        
               
    return render_template('swaps.html', form_data=form_data, rec_results=rec_leg,
                           pay_results=pay_leg, md_content=md_content)




