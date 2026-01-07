import QuantLib as ql
import uuid
import os
import matplotlib.pyplot as plt
from flask import session, current_app
from flask import Blueprint, render_template, request, jsonify, session, redirect, url_for
import markdown
import numpy as np
from ..models.yieldterm_market_data import TreasuryRateProvider, SOFRRateProvider, FREDSwapRatesProvider


extract_market_data_bp = Blueprint('extract_market_data', __name__)

API_KEY = 'a7a1a9c282ee0093003008999c337857'

@extract_market_data_bp.route('/extract-market-data', methods=['GET', 'POST'])
def extract_treasury_data():  # Make sure this matches
    readme_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'extract_market_data.md')
    with open(readme_path, 'r') as readme_file:
        content = readme_file.read()
    md_content = markdown.markdown(content)

    form_data = {}
    treasury_rates = None
    available_tenors = ['1M', '3M', '6M', '1Y', '2Y', '5Y', '7Y', '10Y', '30Y']
    
    if request.method == 'POST':
        form_data = {
            'start_date': request.form['start_date'],
            'selected_tenors': request.form.getlist('selected_tenors'),  # Get list of selected tenors
        }
        start_date = form_data['start_date']
        selected_tenors = form_data['selected_tenors']
        
        treasury_provider = TreasuryRateProvider(API_KEY)
        all_rates = treasury_provider.get_market_rates(start_date=start_date)
        
        # Filter rates based on selected tenors
        if selected_tenors:
            # Convert selected tenor strings to QuantLib Periods for comparison
            selected_periods = []
            for tenor_str in selected_tenors:
                if 'M' in tenor_str:
                    months = int(tenor_str.replace('M', ''))
                    selected_periods.append(ql.Period(months, ql.Months))
                elif 'Y' in tenor_str:
                    years = int(tenor_str.replace('Y', ''))
                    selected_periods.append(ql.Period(years, ql.Years))
            
            # Filter rates to only include selected tenors
            treasury_rates = [
                (period, rate) for period, rate in all_rates 
                if period in selected_periods
            ]
            
            # Sort by converting periods to total months for proper chronological order
            def period_to_months(period):
                """Convert QuantLib Period to total months for sorting"""
                if period.units() == ql.Months:
                    return period.length()
                elif period.units() == ql.Years:
                    return period.length() * 12
                else:
                    return 0
            
            treasury_rates.sort(key=lambda x: period_to_months(x[0]))
        else:
            # If no tenors selected, return all rates (already sorted by provider)
            treasury_rates = all_rates

    return render_template('extract_market_data.html', 
                         md_content=md_content, 
                         treasury_rates=treasury_rates,
                         available_tenors=available_tenors,
                         form_data=form_data)