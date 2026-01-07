import QuantLib as ql
import uuid
import os
import matplotlib.pyplot as plt
from flask import session, current_app
from flask import Blueprint, render_template, request, jsonify, session, redirect, url_for
import markdown
import numpy as np
import datetime
from ..models.yieldterm_market_data import TreasuryRateProvider, SOFRRateProvider, FREDSwapRatesProvider, SOFRCompoundedRateCalculator


extract_market_data_bp = Blueprint('extract_market_data', __name__)

API_KEY = 'a7a1a9c282ee0093003008999c337857'

# @extract_market_data_bp.route('/extract-market-data', methods=['GET', 'POST'])
# def extract_treasury_data():  # Make sure this matches
#     readme_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'extract_market_data.md')
#     with open(readme_path, 'r') as readme_file:
#         content = readme_file.read()
#     md_content = markdown.markdown(content)

#     form_data = {}
#     treasury_rates = None
#     available_tenors = ['1M', '3M', '6M', '1Y', '2Y', '5Y', '7Y', '10Y', '30Y']
#     compounding_methods = ['simple', 'compounded', 'continuous']  # Add this line

#     if request.method == 'POST':
#         form_data = {
#             'start_date': request.form['start_date'],
#             'selected_tenors': request.form.getlist('selected_tenors'),  # Get list of selected tenors
#         }
#         start_date = form_data['start_date']
#         selected_tenors = form_data['selected_tenors']
        
#         treasury_provider = TreasuryRateProvider(API_KEY)
#         all_rates = treasury_provider.get_market_rates(start_date=start_date)
        
#         # Filter rates based on selected tenors
#         if selected_tenors:
#             # Convert selected tenor strings to QuantLib Periods for comparison
#             selected_periods = []
#             for tenor_str in selected_tenors:
#                 if 'M' in tenor_str:
#                     months = int(tenor_str.replace('M', ''))
#                     selected_periods.append(ql.Period(months, ql.Months))
#                 elif 'Y' in tenor_str:
#                     years = int(tenor_str.replace('Y', ''))
#                     selected_periods.append(ql.Period(years, ql.Years))
            
#             # Filter rates to only include selected tenors
#             treasury_rates = [
#                 (period, rate) for period, rate in all_rates 
#                 if period in selected_periods
#             ]
            
#             # Sort by converting periods to total months for proper chronological order
#             def period_to_months(period):
#                 """Convert QuantLib Period to total months for sorting"""
#                 if period.units() == ql.Months:
#                     return period.length()
#                 elif period.units() == ql.Years:
#                     return period.length() * 12
#                 else:
#                     return 0
            
#             treasury_rates.sort(key=lambda x: period_to_months(x[0]))
#         else:
#             # If no tenors selected, return all rates (already sorted by provider)
#             treasury_rates = all_rates

#     return render_template('extract_market_data.html', 
#                          md_content=md_content, 
#                          treasury_rates=treasury_rates,
#                          available_tenors=available_tenors,
#                          compounding_methods=compounding_methods,  # Add this parameter
#                          form_data=form_data)


# @extract_market_data_bp.route('/extract-sofr-data', methods=['GET', 'POST'])
# def extract_sofr_data():
#     readme_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'extract_market_data.md')
#     with open(readme_path, 'r') as readme_file:
#         content = readme_file.read()
#     md_content = markdown.markdown(content)

#     form_data = {}
#     sofr_rates_compounded = None
#     available_tenors = ['1M', '3M', '6M', '12M', '24M', '36M']  # keep consistent with template
#     compounding_methods = ['simple', 'compounded', 'continuous']  # <-- make sure this is defined

#     if request.method == 'POST':
#         form_data = {
#             'start_date': request.form['start_date'],
#             'selected_tenors': request.form.getlist('selected_tenors'),
#             'compounding_method': request.form.get('compounding_method', 'compounded')
#         }
#         start_date = form_data['start_date']
#         selected_tenors = form_data['selected_tenors']
#         comp_method = form_data['compounding_method']

#         # Fetch SOFR data and compound rates as before...
#         sofr_provider = SOFRRateProvider()
#         data = sofr_provider.get_sofr_data(startDate=start_date)

#         if data and "refRates" in data:
#             rates = [
#                 (ql.DateParser.parseISO(entry["effectiveDate"]), entry["percentRate"] / 100.0)
#                 for entry in data["refRates"]
#             ]
#             calculator = SOFRCompoundedRateCalculator(rates)

#             sofr_rates_compounded = []
#             for tenor_str in selected_tenors:
#                 if 'M' in tenor_str:
#                     period = ql.Period(int(tenor_str.replace('M', '')), ql.Months)
#                 elif 'Y' in tenor_str:
#                     period = ql.Period(int(tenor_str.replace('Y', '')), ql.Years)
#                 else:
#                     continue

#                 try:
#                     rate = calculator.compound(ql.Date.todaysDate(), period, method=comp_method)
#                     sofr_rates_compounded.append((tenor_str, rate))
#                 except Exception as e:
#                     sofr_rates_compounded.append((tenor_str, f"Error: {str(e)}"))

#             sofr_rates_compounded.sort(key=lambda x: int(x[0].replace('M','').replace('Y','')) * (12 if 'Y' in x[0] else 1))

#     return render_template(
#         'extract_market_data.html',
#         md_content=md_content,
#         sofr_rates_compounded=sofr_rates_compounded,
#         available_tenors=available_tenors,
#         compounding_methods=compounding_methods,  # <-- must pass this!
#         form_data=form_data
#     )

@extract_market_data_bp.route('/', methods=['GET', 'POST'])
def extract_market_data():
    readme_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'extract_market_data.md')
    with open(readme_path, 'r') as readme_file:
        content = readme_file.read()
    md_content = markdown.markdown(content)

    treasury_form_data = {}
    sofr_form_data = {}
    treasury_rates = None
    sofr_rates_compounded = None
    sofr_spot_rate = None  # Add this for spot rate
    treasury_available_tenors = ['1M', '3M', '6M', '1Y', '2Y', '5Y', '7Y', '10Y', '30Y']
    sofr_available_tenors = ['24M', '36M', '60M', '120M', '240M', '360M']

    compounding_methods = ['Simple', 'Compounded', 'Continuous']
    compounding_frequencies = ['Annual', 'Semiannual', 'Quarterly', 'Monthly', 'Weekly', 'Daily']


    if request.method == 'POST':
        # --- Treasury extraction ---
        if 'treasury_submit' in request.form:
            treasury_form_data = {
                'start_date': request.form.get('treasury_start_date', ''),
                'selected_tenors': request.form.getlist('treasury_selected_tenors'),
            }
            start_date = treasury_form_data['start_date']
            selected_tenors = treasury_form_data['selected_tenors']
            
            if start_date and selected_tenors:
                treasury_provider = TreasuryRateProvider(API_KEY)
                all_rates = treasury_provider.get_market_rates(start_date=start_date)
                
                # Convert selected tenors to QuantLib Periods
                selected_periods = []
                for tenor_str in selected_tenors:
                    if 'M' in tenor_str:
                        months = int(tenor_str.replace('M', ''))
                        selected_periods.append(ql.Period(months, ql.Months))
                    elif 'Y' in tenor_str:
                        years = int(tenor_str.replace('Y', ''))
                        selected_periods.append(ql.Period(years, ql.Years))
                
                treasury_rates = [
                    (period, rate) for period, rate in all_rates if period in selected_periods
                ]
                
                # Sort by total months
                def period_to_months(period):
                    if period.units() == ql.Months:
                        return period.length()
                    elif period.units() == ql.Years:
                        return period.length() * 12
                    return 0
                
                treasury_rates.sort(key=lambda x: period_to_months(x[0]))

        # --- SOFR extraction ---
        # --- SOFR Spot Rate extraction (separate button) ---
        elif 'spot_rate_submit' in request.form:
            sofr_form_data = {
                'start_date': request.form.get('sofr_start_date', ''),
            }
            start_date = sofr_form_data['start_date']

            if start_date:
                try:
                    as_of_date = datetime.datetime.strptime(start_date, "%Y-%m-%d").date()
                except Exception as e:
                    return f"Invalid SOFR start date: {str(e)}"

                sofr_provider = SOFRRateProvider()
                spot_rate_data = sofr_provider.get_sofr_data(startDate=start_date)
                
                if spot_rate_data and "refRates" in spot_rate_data:
                    # Find the rate for the exact as-of date (or closest available)
                    as_of_ql_date = ql.DateParser.parseISO(start_date)
                    as_of_py_date = as_of_ql_date.to_date()
                    
                    # Look for exact date match first
                    spot_rate = None
                    for entry in spot_rate_data["refRates"]:
                        entry_date = datetime.datetime.strptime(entry["effectiveDate"], "%Y-%m-%d").date()
                        if entry_date == as_of_py_date:
                            spot_rate = entry["percentRate"] / 100.0
                            break
                    
                    # If no exact match, find the closest date (most recent before or on as-of date)
                    if spot_rate is None:
                        closest_entry = None
                        closest_date = None
                        for entry in spot_rate_data["refRates"]:
                            entry_date = datetime.datetime.strptime(entry["effectiveDate"], "%Y-%m-%d").date()
                            if entry_date <= as_of_py_date:
                                if closest_date is None or entry_date > closest_date:
                                    closest_date = entry_date
                                    closest_entry = entry
                        if closest_entry:
                            spot_rate = closest_entry["percentRate"] / 100.0
                            sofr_spot_rate = spot_rate
                        else:
                            sofr_spot_rate = "No rate data available for this date"
                    else:
                        sofr_spot_rate = spot_rate
                else:
                    sofr_spot_rate = "Failed to fetch SOFR data"

        # --- SOFR Compounded Rates extraction ---
        elif 'sofr_submit' in request.form:
            sofr_form_data = {
                'start_date': request.form.get('sofr_start_date', ''),
                'selected_tenors': request.form.getlist('sofr_selected_tenors'),
                'compounding_method': request.form.get('compounding_method', 'Compounded'),
                'compounding_frequency': request.form.get('compounding_frequency', 'Daily')
            }

            start_date = sofr_form_data['start_date']
            selected_tenors = sofr_form_data['selected_tenors']
            method_input = sofr_form_data['compounding_method']
            freq_input = sofr_form_data['compounding_frequency']

            if start_date and selected_tenors:
                # Parse as-of date from user input
                try:
                    as_of_date = datetime.datetime.strptime(start_date, "%Y-%m-%d").date()
                except Exception as e:
                    return f"Invalid SOFR start date: {str(e)}"

                sofr_provider = SOFRRateProvider()

                # Determine earliest historical date needed for longest tenor
                max_months = max([
                    int(t.replace('M','').replace('Y','')) * (12 if 'Y' in t else 1)
                    for t in selected_tenors
                ])
                history_start_date = as_of_date - datetime.timedelta(days=30*max_months)
                history_start_str = history_start_date.isoformat()

                # Fetch historical SOFR data starting from history_start_date
                data = sofr_provider.get_sofr_data(startDate=history_start_str)

                if data and "refRates" in data:
                    rates = [
                        (ql.DateParser.parseISO(entry["effectiveDate"]), entry["percentRate"] / 100.0)
                        for entry in data["refRates"]
                    ]

                    # Map compounding method
                    method_map = {
                        'compounded': ql.Compounded,
                        'simple': ql.Simple,
                        'continuous': ql.Continuous
                    }
                    compounding = method_map.get(method_input.lower(), ql.Compounded)

                    # Map compounding frequency
                    freq_map = {
                        'Annual': ql.Annual,
                        'Semiannual': ql.Semiannual,
                        'Quarterly': ql.Quarterly,
                        'Monthly': ql.Monthly,
                        'Weekly': ql.Weekly,
                        'Daily': ql.Daily
                    }
                    compounding_frequency = freq_map.get(freq_input, ql.Daily)

                    day_count = ql.Actual360()  # Standard for SOFR

                    # Initialize SOFR calculator
                    calculator = SOFRCompoundedRateCalculator(
                        rates=rates,
                        day_count=day_count,
                        compounding=compounding,
                        compounding_frequency=compounding_frequency
                    )

                    # Compute rates for selected tenors as of user-provided date
                    sofr_rates_compounded = []
                    as_of_ql_date = ql.DateParser.parseISO(start_date)
                    for tenor_str in selected_tenors:
                        if 'M' in tenor_str:
                            period = ql.Period(int(tenor_str.replace('M','')), ql.Months)
                        elif 'Y' in tenor_str:
                            period = ql.Period(int(tenor_str.replace('Y','')), ql.Years)
                        else:
                            continue
                        try:
                            rate = calculator.compound(as_of_ql_date, period)
                            sofr_rates_compounded.append((tenor_str, rate))
                        except Exception as e:
                            sofr_rates_compounded.append((tenor_str, f"Error: {str(e)}"))

                    # Sort by total months
                    def period_to_months(tenor_str):
                        if 'M' in tenor_str:
                            return int(tenor_str.replace('M',''))
                        elif 'Y' in tenor_str:
                            return int(tenor_str.replace('Y','')) * 12
                        return 0

                    sofr_rates_compounded.sort(key=lambda x: period_to_months(x[0]))

    return render_template(
        'extract_market_data.html',
        md_content=md_content,
        treasury_rates=treasury_rates,
        sofr_rates_compounded=sofr_rates_compounded,
        sofr_spot_rate=sofr_spot_rate,  # Add this
        treasury_available_tenors=treasury_available_tenors,
        sofr_available_tenors=sofr_available_tenors,
        compounding_methods=compounding_methods,
        compounding_frequencies=compounding_frequencies,
        treasury_form_data=treasury_form_data,
        sofr_form_data=sofr_form_data
    )