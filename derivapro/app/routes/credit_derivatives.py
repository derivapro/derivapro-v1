# -*- coding: utf-8 -*-
"""
Created on Sun Jun  9 00:47:33 2024

@author: minwuu01
"""

from flask import Blueprint, render_template, request, session
from derivapro.app.models.mdls_credit import CreditDefaultSwap, SyntheticCDO, CLNPricingFixed, CLNPricingFloat, CLNSensitivityAnalysis
from derivapro.app.models.mdls_bonds import NCFixedBonds, NCFloatingBonds
import QuantLib as ql
import os
import markdown
import matplotlib.pyplot as plt

credit_derivatives_bp = Blueprint('credit_derivatives', __name__)

@credit_derivatives_bp.route('/', methods=['GET', 'POST'])
def credit_derivatives():
    return render_template('credit_derivatives.html')

@credit_derivatives_bp.route('/credit_default_swap', methods=['GET','POST'])
def creditDefaultSwaps():
    readme_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'credit_default_swaps.md')
    with open(readme_path, 'r') as readme_file:
        content = readme_file.read()
    md_content = markdown.markdown(content)
    
    cds_results = cds_analysis_results = baseline_EL = stressed_EL = None
    form_data = {}
    
    if request.method == 'POST':
        action = request.form.get('analysis_type')
     
        form_data = {
            'nominal': request.form['nominal'],
            'spread': request.form['spread'],
            'recovery_rate': request.form['recovery_rate'],
            'risk_free': request.form['risk_free'],
            'calendar_type': request.form['calendar_type'],
            'side_type': request.form['side_type'],
            'selected_tenor': request.form['selected_tenor'],
            'entry_date': request.form['entry_date'],
            'end_date': request.form['end_date']
        }

        nominal = float(form_data['nominal'])
        spread = float(form_data['spread'])
        recovery_rate = float(form_data['recovery_rate'])
        risk_free = float(form_data['risk_free'])
        calendar_type = form_data['calendar_type']
        side_type = form_data['side_type']
        selected_tenor = form_data['selected_tenor']
        entry_date = form_data['entry_date']
        end_date = form_data['end_date']

        
        # Define Calendar Type
        if calendar_type == "American":
            calendar = ql.UnitedStates(ql.UnitedStates.NYSE)
        elif calendar_type == 'TARGET':
            calendar = ql.TARGET()
        elif calendar_type == 'UnitedKingdom':
            calendar = ql.UnitedKingdom()
        elif calendar_type == 'China':
            calendar = ql.China()

        # Define Side (Seller/Buyer)
        if side_type == "Seller":
            side = ql.Protection.Seller
        elif side_type == "Buyer":
            side = ql.Protection.Buyer
        else:
            raise ValueError("Invalid side type. It should be 'Buyer' or 'Seller'.")
            
        # Define the corresponding tenor and map to QuantLib frequency
        if selected_tenor == "Annual":
            selected_period = ql.Period(1, ql.Years)
            payment_frequency_ql = ql.Annual
            period_fraction = 1  # Payment once per year
        elif selected_tenor == "Semiannual":
            selected_period = ql.Period(6, ql.Months)
            payment_frequency_ql = ql.Semiannual
            period_fraction = 0.5  # Payment twice per year
        elif selected_tenor == "Quarterly":
            selected_period = ql.Period(3, ql.Months)
            payment_frequency_ql = ql.Quarterly
            period_fraction = 0.25  # Payment four times per year
        elif selected_tenor == "Monthly":
            selected_period = ql.Period(1, ql.Months)
            payment_frequency_ql = ql.Monthly
            period_fraction = 1 / 12  # Payment twelve times per year
        else:
            raise ValueError("Invalid tenor selected")

        variable = step_range = num_steps = None
        cds = CreditDefaultSwap(nominal, spread, recovery_rate, risk_free,
                     payment_frequency_ql, calendar, side, selected_period, period_fraction, entry_date, end_date, num_steps, step_range, variable)

        cds_results = cds.cds_results()
        
        if action == 'sensitivity':
            try:
                form_data['num_steps'] = int(request.form['num_steps'])
                form_data['range_span'] = float(request.form['range_span'])
                form_data['variable'] = str(request.form['variable'])
                
                num_steps = form_data['num_steps']
                range_span = form_data['range_span']
                variable = form_data['variable']
                
                cds = CreditDefaultSwap(nominal, spread, recovery_rate, risk_free,
                         payment_frequency_ql, calendar, side, selected_period, period_fraction, entry_date, end_date, variable, range_span, num_steps)
                
                
                if variable == 'risk_free':
                    cds_analysis_results = cds.analyze_variable_sensitivity(variable, range_span, num_steps)
                    cds.plot_sensitivity_analysis(variable, range_span, num_steps)              
                elif variable == 'recovery_rate':
                    cds_analysis_results = cds.analyze_variable_sensitivity(variable, range_span, num_steps)
                    cds.plot_sensitivity_analysis(variable, range_span, num_steps)
                elif variable == 'spread':
                    cds_analysis_results = cds.analyze_variable_sensitivity(variable, range_span, num_steps)
                    cds.plot_sensitivity_analysis(variable, range_span, num_steps)
                else:
                    pass
                
                plot_filename = f'Credit Default Swap_{variable}_{range_span}_sensitivity_plot.png'
                plot_path = os.path.join('app', 'static', plot_filename)
                plt.savefig(plot_path)   
                session['cds_sensitivity_analysis_results'] = {'plot_filename': plot_filename,
                                                                   'range_span': range_span,
                                                                   'num_steps': num_steps}
                cds_analysis_results = True
                
                # plt.close()  # Close the plot after saving 
            except Exception as e:
                cds_analysis_results = f"Error in sensitivity analysis: {str(e)}"
                session['cds_sensitivity_analysis_results'] = None
        
        elif action == 'scenario':
            try:
                rate_change = float(request.form.get('rate_scenario', 0))
                spread_change = float(request.form.get('spread_scenario',0))
                recovery_change = float(request.form.get('recovery_scenario',0))
                
                variable = step_range = num_steps = None
                cds_class = CreditDefaultSwap(nominal, spread, recovery_rate, risk_free,
                             payment_frequency_ql, calendar, side, selected_period, period_fraction, entry_date, end_date, num_steps, step_range, variable)
                
                baseline_EL = cds_class.cds_results()                
                
                stressed_rate = risk_free + rate_change
                stressed_rec = recovery_rate + recovery_change
                stressed_spread = spread * (1+spread_change)
                
                cds_stressed = CreditDefaultSwap(nominal, stressed_spread, stressed_rec, stressed_rate,
                             payment_frequency_ql, calendar, side, selected_period, period_fraction, entry_date, end_date, num_steps, step_range, variable)
                
                stressed_EL = cds_stressed.cds_results()

            except Exception as e:
                print(f"An error occurred during scenario analysis: {e}")                 
        
    else:
        pass
    
    return render_template('creditdefaultswap.html',cds_results=cds_results, cds_analysis_results=cds_analysis_results,baseline_EL=baseline_EL, stressed_EL=stressed_EL, 
                            form_data=form_data, md_content=md_content)


@credit_derivatives_bp.route('/synthetic_collateralized_debt_obligation', methods=['GET', 'POST'])
def syntheticCDO():
    readme_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'synthetic_CDO.md')
    with open(readme_path, 'r') as readme_file:
        content = readme_file.read()
    md_content = markdown.markdown(content)

    cds_results = cdo_analysis_results = baseline_CDO = stressed_CDO = None
    form_data = {}
    cds_list = []
    num_contracts = 0

    if request.method == 'POST':
        
        action = request.form.get('analysis_type')
        form_data = {
            'tranche_upper_1': request.form.get('tranche_upper_1'),
            'tranche_lower_1': request.form.get('tranche_lower_1'),
            'tranche_upper_2': request.form.get('tranche_upper_2'),
            'tranche_lower_2': request.form.get('tranche_lower_2'),
            'tranche_upper_3': request.form.get('tranche_upper_3'),
            'tranche_lower_3': request.form.get('tranche_lower_3'),
        }
        
        try:
            tranche_lower_1 = float(form_data['tranche_lower_1'])
            tranche_upper_1 = float(form_data['tranche_upper_1'])
            tranche_lower_2 = float(form_data['tranche_lower_2'])
            tranche_upper_2 = float(form_data['tranche_upper_2'])
            tranche_lower_3 = float(form_data['tranche_lower_3'])
            tranche_upper_3 = float(form_data['tranche_upper_3'])
        except ValueError as e:
            print(f"Error converting to float: {e}")
            # Handle the error, e.g., set default values or return an error message

        # Retrieve the tranche inputs dynamically and create a list of tranches
        tranches = [
            ("Equity", tranche_lower_1, tranche_upper_1),  # Swap the order
            ("Mezzanine", tranche_lower_2, tranche_upper_2 ),
            ("Senior", tranche_lower_3,tranche_upper_3)
        ]
        
        if 'remove_ids' in request.form:
            # Remove selected contracts
            remove_ids = list(map(int, request.form.getlist('remove_ids')))
            cds_list = [cds for cds in cds_list if cds['cds_id'] not in remove_ids]

        # Dynamically determine the number of contracts
        while f'nominal_{num_contracts + 1}' in request.form:
            num_contracts += 1

        for i in range(num_contracts):
            form_data[f'nominal_{i+1}'] = request.form[f'nominal_{i+1}']
            form_data[f'spread_{i+1}'] = request.form[f'spread_{i+1}']
            form_data[f'recovery_rate_{i+1}'] = request.form[f'recovery_rate_{i+1}']
            form_data[f'risk_free_{i+1}'] = request.form[f'risk_free_{i+1}']
            form_data[f'calendar_type_{i+1}'] = request.form[f'calendar_type_{i+1}']
            form_data[f'side_type_{i+1}'] = request.form[f'side_type_{i+1}']
            form_data[f'selected_tenor_{i+1}'] = request.form[f'selected_tenor_{i+1}']
            form_data[f'entry_date_{i+1}'] = request.form[f'entry_date_{i+1}']
            form_data[f'end_date_{i+1}'] = request.form[f'end_date_{i+1}']

            nominal = float(form_data[f'nominal_{i+1}'])
            spread = float(form_data[f'spread_{i+1}'])
            recovery_rate = float(form_data[f'recovery_rate_{i+1}'])
            risk_free = float(form_data[f'risk_free_{i+1}'])
            calendar_type = form_data[f'calendar_type_{i+1}']
            side_type = form_data[f'side_type_{i+1}']
            selected_tenor = form_data[f'selected_tenor_{i+1}']
            entry_date = form_data[f'entry_date_{i+1}']
            end_date = form_data[f'end_date_{i+1}']
            variable = step_range = num_steps = None
            # Define Calendar Type
            calendar = ql.UnitedStates(ql.UnitedStates.NYSE) if calendar_type == "American" else (
                ql.TARGET() if calendar_type == 'TARGET' else (
                    ql.UnitedKingdom() if calendar_type == 'UnitedKingdom' else ql.China()))

            # Define Side (Seller/Buyer)
            side = ql.Protection.Seller if side_type == "Seller" else ql.Protection.Buyer

            # Define Tenor
            tenor_mapping = {
                "Annual": (ql.Period(1, ql.Years), ql.Annual, 1),
                "Semiannual": (ql.Period(6, ql.Months), ql.Semiannual, 0.5),
                "Quarterly": (ql.Period(3, ql.Months), ql.Quarterly, 0.25),
                "Monthly": (ql.Period(1, ql.Months), ql.Monthly, 1 / 12)
            }
            selected_period, payment_frequency_ql, period_fraction = tenor_mapping[selected_tenor]

            # Create the CDS instance
            cds = {
                "cds_id": len(cds_list) + 1,
                "instance": CreditDefaultSwap(nominal, spread, recovery_rate, risk_free,
                             payment_frequency_ql, calendar, side, selected_period, period_fraction, entry_date, end_date, num_steps, step_range, variable)
            }
            cds_list.append(cds)

        synthetic_cdo = SyntheticCDO(cds_list=[cds['instance'] for cds in cds_list], tranches=tranches)

        cds_results = synthetic_cdo.calculate_synthetic_cdo()
        if action == 'sensitivity':
            try:
                
                form_data['num_steps'] = int(request.form['num_steps'])
                form_data['range_span'] = float(request.form['range_span'])
                form_data['variable'] = str(request.form['variable'])
                
                num_steps = form_data['num_steps']
                range_span = form_data['range_span']
                variable = form_data['variable']
                
                # Create the CDS instance
                if variable == 'risk_free':
                    synthetic_cdo.plot_sensitivity_analysis(variable, range_span, num_steps)
                elif variable == 'recovery_rate':
                    synthetic_cdo.plot_sensitivity_analysis(variable, range_span, num_steps)
                elif variable == 'spread':
                    synthetic_cdo.plot_sensitivity_analysis(variable, range_span, num_steps)
                else:
                    pass
                
                plot_filename = f'Synthetic CDO_{variable}_{range_span}_sensitivity_plot.png'
                plot_path = os.path.join('app', 'static', plot_filename)
                plt.savefig(plot_path)   
                session['cdo_analysis_results'] = {'plot_filename': plot_filename,
                                                                    'range_span': range_span,
                                                                    'num_steps': num_steps}
                cdo_analysis_results = True
                
             #   plt.close()  # Close the plot after saving 
            except Exception as e:
                print(f"An error occurred during Sensitivity analysis: {e}") 
                
        elif action == 'scenario':
            try:
                # Get the scenario values from the form (default to 0 if not provided)
                rate_change = float(request.form.get('rate_scenario', 0))
                spread_change = float(request.form.get('spread_scenario', 0))
                recovery_change = float(request.form.get('recovery_scenario', 0))
                
                # Initialize the synthetic CDO class and calculate baseline CDO
                baseline_CDO = synthetic_cdo.calculate_synthetic_cdo()  # This should calculate the baseline CDO
                
                # Apply the stressed values for rate, recovery rate, and spread
                stressed_rate = risk_free + rate_change
                stressed_rec = recovery_rate + recovery_change
                stressed_spread = spread * (1 + spread_change)
                for cds_entry in cds_list:
                    cds_instance = cds["instance"]
                    cds_instance.risk_free = stressed_rate
                    cds_instance.recovery_rate = stressed_rec
                    cds_instance.spread = stressed_spread
                   
                stressed_CDO = synthetic_cdo.calculate_synthetic_cdo()

                print(stressed_CDO)
            except Exception as e:
                print(f"An error occurred during scenario analysis: {e}") 
    else:
        pass
    return render_template('synthetic_CDO.html', cds_results=cds_results, form_data=form_data, cdo_analysis_results=cdo_analysis_results,
                           cds_list=cds_list, md_content=md_content, num_contracts=num_contracts, baseline_CDO=baseline_CDO, stressed_CDO=stressed_CDO)

@credit_derivatives_bp.route('/credit_linked_notes', methods=['GET','POST'])
def creditLinkedNotes():
    readme_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'credit_linked_notes.md')
    with open(readme_path, 'r') as readme_file:
        content = readme_file.read()
    md_content = markdown.markdown(content)
    
    fr_bond_results = cds_results_fixed = cds_results_float = cln_pricing_fixed = cln_pricing_float = fram_bond_results = fl_bond_results = flam_bond_results = None
    cln_sensitivity_analysis_results_fixed = cln_sensitivity_analysis_results_float = baseline_CLN = stressed_CLN = baseline_CLN_float = stressed_CLN_float = None
    form_data = {}

    if request.method == 'POST':     
        tab_selected = request.form.get('tab_selected')
        action = request.form.get('analysis_type')
        if tab_selected == 'fixed-tab':
            # Retrieve form data from URL parameters
            form_data = {
                'value_date': request.form['value_date'], 
                'spot_dates': request.form['spot_dates'], 
                'spot_rates': request.form['spot_rates'], 
                'shocks': request.form['shocks'], 
                'day_count': request.form['day_count'],
                'day_count_val': request.form['day_count'],
                'calendar': request.form['calendar'],
                'calendar_val': request.form['calendar'],
                'interpolation': request.form['interpolation'],
                'interpolation_val': request.form['interpolation'],
                'compounding': request.form['compounding'],
                'compounding_val': request.form['compounding'],
                'compounding_frequency': request.form['compounding_frequency'],
                'compounding_frequency_val': request.form['compounding_frequency'],
                'tenor': request.form['tenor'],
                'tenor_val': request.form['tenor'],
                'issue_date': request.form['issue_date'], 
                'maturity_date': request.form['maturity_date'],
                'coupon_rate': request.form['coupon_rate'],
                'notional': request.form['notional'],
                'notionals': request.form['notionals'],
                'amort_selection': request.form['amort_selection'],
                'spread': request.form['spread'],
                'recovery_rate': request.form['recovery_rate'],
                'risk_free': request.form['risk_free'],
                'side_type': request.form['side_type']
            }
            
            notional_cds = float(form_data['notional'])
            value_date = form_data['value_date']
            spot_dates = form_data['spot_dates']
            spot_rates = form_data['spot_rates']
            shocks = form_data['shocks']
            issue_date = form_data['issue_date']
            maturity_date = form_data['maturity_date']
            coupon_rate = form_data['coupon_rate']
            notional = form_data['notional']
            
            calendar_val = form_data['calendar_val']
            interpolation_val = form_data['interpolation_val']
            compounding_val = form_data['compounding_val']
            compounding_frequency_val = form_data['compounding_frequency_val']
            tenor_val = form_data['tenor_val']
            day_count_val = form_data['day_count_val']
            
            spread = float(form_data['spread'])
            recovery_rate = float(form_data['recovery_rate'])
            risk_free = float(form_data['risk_free'])
            side_type = form_data['side_type']
            
            notionals = form_data['notionals']
            amort_selection = form_data['amort_selection']
            
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
            
            # Map compounding frequency value to QuantLib Frequency
            if compounding_frequency_val == 'Annual':
                compounding_frequency = ql.Annual
            elif compounding_frequency_val == 'Semiannual':
                compounding_frequency = ql.Semiannual
            elif compounding_frequency_val == 'Quarterly':
                compounding_frequency = ql.Quarterly
            elif compounding_frequency_val == 'Monthly':
                compounding_frequency = ql.Monthly
            elif compounding_frequency_val == 'Daily':
                compounding_frequency = ql.Daily
    
            # Map tenor value to QuantLib Tenor
            if tenor_val == 'Annual':
                tenor = ql.Period(ql.Annual)
                selected_period = ql.Period(1, ql.Years)
                payment_frequency_ql = ql.Annual
                period_fraction = 1  # Payment once per year
            elif tenor_val == 'Semiannual':
                tenor = ql.Period(ql.Semiannual)
                selected_period = ql.Period(6, ql.Months)
                payment_frequency_ql = ql.Semiannual
                period_fraction = 0.5  # Payment twice per year
            elif tenor_val == 'Quarterly':
                tenor = ql.Period(ql.Quarterly)
                selected_period = ql.Period(3, ql.Months)
                payment_frequency_ql = ql.Quarterly
                period_fraction = 0.25  # Payment four times per year
            elif tenor_val == 'Monthly':
                tenor = ql.Period(ql.Monthly)
                selected_period = ql.Period(1, ql.Months)
                payment_frequency_ql = ql.Monthly
                period_fraction = 1 / 12  # Payment twelve times per year
            else:
                raise ValueError("Invalid tenor selected")
                
            # Map the day count value to a QuantLib DayCount
            if day_count_val == 'ActualActual':
                day_count = ql.ActualActual(ql.ActualActual.Bond)
            elif day_count_val == 'Thirty360':
                day_count = ql.Thirty360(ql.Thirty360.BondBasis)
            elif day_count_val == 'Actual360':
                day_count = ql.Actual360()
            elif day_count_val == 'Actual365Fixed':
                day_count = ql.Actual365Fixed()
                
            # Define Side (Seller/Buyer)
            if side_type == "Seller":
                side = ql.Protection.Seller
            elif side_type == "Buyer":
                side = ql.Protection.Buyer
            else:
                raise ValueError("Invalid side type. It should be 'Buyer' or 'Seller'.")
            
            fixed_bond = NCFixedBonds(value_date, spot_dates, spot_rates, shocks, day_count, calendar,
                                          interpolation, compounding, compounding_frequency)         
            variable = range_span = num_steps = None
            cds = CreditDefaultSwap(notional_cds, spread, recovery_rate, risk_free,
                         payment_frequency_ql, calendar, side, selected_period, period_fraction, issue_date, maturity_date,
                         variable, range_span, num_steps)
            
            cds_results_fixed = cds.cds_results()
            
            # Collect form data for Fixed Bond parameters
            bond_params = {
                'value_date': form_data['value_date'],
                'spot_dates': form_data['spot_dates'],
                'spot_rates': form_data['spot_rates'],
                'shocks': form_data['shocks'],
                'day_count': day_count,
                'calendar': calendar,
                'interpolation': interpolation,
                'compounding': compounding,
                'compounding_frequency': compounding_frequency
            }
            
            # Map bond type value
            if amort_selection == 'No':
                amortization = 'No'
                fr_bond_results = fixed_bond.fixed_rate(issue_date, maturity_date, tenor, coupon_rate, notional)
            elif amort_selection == 'Yes':
                amortization = 'Yes'
                fram_bond_results = fixed_bond.fixed_rate_amortizing(issue_date, maturity_date, tenor, coupon_rate, notionals) 
            
            cln_pricing = CLNPricingFixed(bond_params=bond_params,cds_instance=cds)
            
            cln_pricing_fixed = cln_pricing.calculate_cln_price_fixed(issue_date, maturity_date, tenor, coupon_rate, notional, notionals, amortization)
            
            if action == 'fixed-sensitivity':                
                try:
                    # Get sensitivity-specific parameters
                    form_data['num_steps'] = int(request.form['num_steps'])
                    form_data['range_span'] = float(request.form['range_span'])
                    form_data['variable'] = str(request.form['variable'])
                    
                    # Get values from form data
                    num_steps = form_data['num_steps']
                    range_span = form_data['range_span']
                    variable = form_data['variable']
                    
                    # Get dropdown values
                    calendar_val = form_data['calendar_val']
                    interpolation_val = form_data['interpolation_val']
                    compounding_val = form_data['compounding_val']
                    compounding_frequency_val = form_data['compounding_frequency_val']
                    tenor_val = form_data['tenor_val']
                    day_count_val = form_data['day_count_val']

                    fixed_bond = NCFixedBonds(value_date, spot_dates, spot_rates, shocks, day_count, calendar,
                                          interpolation, compounding, compounding_frequency)   
                    
                    cds_class = CreditDefaultSwap(notional_cds, spread, recovery_rate, risk_free,
                                 payment_frequency_ql, calendar, side, selected_period, period_fraction, issue_date, maturity_date,
                                 variable, range_span, num_steps)

                    if amort_selection == 'No':
                        if variable == 'risk_free':
                            fr_bond_results = fixed_bond.fixed_rate(issue_date, maturity_date, tenor, coupon_rate, notional)
                            sensitivity_analysis = CLNSensitivityAnalysis(bond_params=bond_params, cds_instance=cds_class, bond_results=fr_bond_results)
                            # results = sensitivity_analysis.analyze_variable_sensitivity(variable, range_span, num_steps)
                            sensitivity_analysis.plot_sensitivity_analysis(variable, range_span, num_steps)

                        elif variable == 'recovery_rate':
                            fr_bond_results = fixed_bond.fixed_rate(issue_date, maturity_date, tenor, coupon_rate, notional)
                            sensitivity_analysis = CLNSensitivityAnalysis(bond_params=bond_params, cds_instance=cds_class, bond_results=fr_bond_results)
                            # results = sensitivity_analysis.analyze_variable_sensitivity(variable, range_span, num_steps)
                            sensitivity_analysis.plot_sensitivity_analysis(variable, range_span, num_steps)
                        elif variable == 'spread':
                            fr_bond_results = fixed_bond.fixed_rate(issue_date, maturity_date, tenor, coupon_rate, notional)
                            sensitivity_analysis = CLNSensitivityAnalysis(bond_params=bond_params, cds_instance=cds_class, bond_results=fr_bond_results)
                            # results = sensitivity_analysis.analyze_variable_sensitivity(variable, range_span, num_steps)
                            sensitivity_analysis.plot_sensitivity_analysis(variable, range_span, num_steps)
                    elif amort_selection == 'Yes':
                        if variable == 'risk_free':
                            fram_bond_results = fixed_bond.fixed_rate_amortizing(issue_date, maturity_date, tenor, coupon_rate, notionals)
                            sensitivity_analysis = CLNSensitivityAnalysis(bond_params=bond_params, cds_instance=cds_class, bond_results=fram_bond_results)
                            # results = sensitivity_analysis.analyze_variable_sensitivity(variable, range_span, num_steps)
                            sensitivity_analysis.plot_sensitivity_analysis(variable, range_span, num_steps)
                        elif variable == 'recovery_rate':
                            fram_bond_results = fixed_bond.fixed_rate_amortizing(issue_date, maturity_date, tenor, coupon_rate, notionals)
                            sensitivity_analysis = CLNSensitivityAnalysis(bond_params=bond_params, cds_instance=cds_class, bond_results=fram_bond_results)
                            # results = sensitivity_analysis.analyze_variable_sensitivity(variable, range_span, num_steps)
                            sensitivity_analysis.plot_sensitivity_analysis(variable, range_span, num_steps)
                        elif variable == 'spread':
                            fram_bond_results = fixed_bond.fixed_rate_amortizing(issue_date, maturity_date, tenor, coupon_rate, notionals)
                            sensitivity_analysis = CLNSensitivityAnalysis(bond_params=bond_params, cds_instance=cds_class, bond_results=fram_bond_results)
                            # results = sensitivity_analysis.analyze_variable_sensitivity(variable, range_span, num_steps)
                            sensitivity_analysis.plot_sensitivity_analysis(variable, range_span, num_steps)

                    
                    plot_filename = f'Credit Linked Notes_{variable}_{range_span}_sensitivity_plot.png'
                    plot_path = os.path.join('app', 'static', plot_filename)
                    plt.savefig(plot_path)   
                    session['cln_sensitivity_analysis_results_fixed'] = {'plot_filename': plot_filename,
                                                                        'range_span': range_span,
                                                                        'num_steps': num_steps}
                    cln_sensitivity_analysis_results_fixed = True
                    
                    # plt.close()  # Close the plot after saving 
    
                except Exception as e:
                    print(f"An error occurred during Sensitivity analysis: {e}") 
                
            elif action == 'fixed-scenario':
                try:
                    rate_change = float(request.form.get('rate_scenario', 0))
                    spread_change = float(request.form.get('spread_scenario',0))
                    recovery_change = float(request.form.get('recovery_scenario',0))
                    
                    variable = step_range = num_steps = None
                    cds_class = CreditDefaultSwap(notional_cds, spread, recovery_rate, risk_free,
                                 payment_frequency_ql, calendar, side, selected_period, period_fraction, issue_date, maturity_date,
                                 variable, range_span, num_steps)
                                        
                    stressed_rate = risk_free + rate_change
                    stressed_rec = recovery_rate + recovery_change
                    stressed_spread = spread * (1+spread_change)
                    cds_stressed = CreditDefaultSwap(notional_cds, stressed_spread, stressed_rec, stressed_rate,
                                 payment_frequency_ql, calendar, side, selected_period, period_fraction, issue_date, maturity_date,
                                 variable, range_span, num_steps)
                
                    # Map bond type value
                    if amort_selection == 'No':
                        amortization = 'No'
                        fr_bond_results = fixed_bond.fixed_rate(issue_date, maturity_date, tenor, coupon_rate, notional)
                        cln_pricing_baseline = CLNPricingFixed(bond_params=bond_params,cds_instance=cds)
                        baseline_CLN = cln_pricing_baseline.calculate_cln_price_fixed(issue_date, maturity_date, tenor, coupon_rate, notional, notionals, amortization)
                        cln_pricing_stressed = CLNPricingFixed(bond_params=bond_params,cds_instance=cds_stressed)
                        stressed_CLN = cln_pricing_stressed.calculate_cln_price_fixed(issue_date, maturity_date, tenor, coupon_rate, notional, notionals, amortization)
                        print("Baseline", baseline_CLN)
                        print("Stressed", stressed_CLN)
                    elif amort_selection == 'Yes':
                        amortization = 'Yes'
                        fram_bond_results = fixed_bond.fixed_rate_amortizing(issue_date, maturity_date, tenor, coupon_rate, notionals) 
                        cln_pricing_baseline = CLNPricingFixed(bond_params=bond_params,cds_instance=cds)
                        baseline_CLN = cln_pricing_baseline.calculate_cln_price_fixed(issue_date, maturity_date, tenor, coupon_rate, notional, notionals, amortization)
                        cln_pricing_stressed = CLNPricingFixed(bond_params=bond_params,cds_instance=cds_stressed)
                        stressed_CLN = cln_pricing_stressed.calculate_cln_price_fixed(issue_date, maturity_date, tenor, coupon_rate, notional, notionals, amortization)
                        print("Baseline", baseline_CLN)
                        print("Stressed", stressed_CLN)
                except Exception as e:
                    print(f"An error occurred during scenario analysis: {e}")    
           
        elif tab_selected == 'float-tab':
            # Retrieve form data from URL parameters
            form_data = {
                'value_date': request.form['value_date'],
                'spot_dates': request.form['spot_dates'],
                'spot_rates': request.form['spot_rates'],
                'index_dates': request.form['index_dates'],
                'index_rates': request.form['index_rates'],
                'calendar_val': request.form['calendar'],
                'currency_val': request.form['currency'],
                'interpolation_val': request.form['interpolation'],
                'compounding_val': request.form['compounding'],
                'compounding_frequency_val': request.form['compounding_frequency'],
                'shocks': request.form['shocks'],
                'issue_date': request.form['issue_date'],
                'maturity_date': request.form['maturity_date'],
                'tenor_val': request.form['tenor'],
                'spread': request.form['spread'],
                'notional': request.form['notional'],
                'notionals': request.form['notionals'],
                'day_count_val': request.form['day_count'],
                'recovery_rate': request.form['recovery_rate'],
                'risk_free': request.form['risk_free'],
                'side_type': request.form['side_type'],
                'coupon_rate': request.form['coupon_rate'],
                'amort_selection_float': request.form['amort_selection_float']
            }
            
            value_date = form_data['value_date']
            spot_dates = form_data['spot_dates']
            spot_rates = form_data['spot_rates']
            index_dates = form_data['index_dates']
            index_rates = form_data['index_rates']
            coupon_rate = form_data['coupon_rate']
            shocks = form_data['shocks']
            issue_date = form_data['issue_date']
            maturity_date = form_data['maturity_date']

            amort_selection_float = form_data['amort_selection_float']
            notional = float(form_data['notional'])
            spread = float(form_data['spread'])
            notional_cds = float(form_data['notional'])
            recovery_rate = float(form_data['recovery_rate'])
            risk_free = float(form_data['risk_free'])
            side_type = form_data['side_type']
            notionals = form_data['notionals']
            calendar_val = form_data['calendar_val']
            currency_val = form_data['currency_val']
            interpolation_val = form_data['interpolation_val']
            compounding_val = form_data['compounding_val']
            compounding_frequency_val = form_data['compounding_frequency_val']
            tenor_val = form_data['tenor_val']
            day_count_val = form_data['day_count_val']
            
            # Map the calendar value to a QuantLib Calendar
            if calendar_val == 'UnitedStates':
                calendar = ql.UnitedStates(ql.UnitedStates.NYSE)
            elif calendar_val == 'TARGET':
                calendar = ql.TARGET()
            elif calendar_val == 'China':
                calendar = ql.China()
            
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
            elif compounding_val == 'Continuous':
                compounding = ql.Continuous
            elif compounding_val == 'Simple':
                compounding = ql.Simple
                    
            # Map compounding frequency value to QuantLib Frequency
            if compounding_frequency_val == 'Annual':
                compounding_frequency = ql.Annual
            elif compounding_frequency_val == 'Semiannual':
                compounding_frequency = ql.Semiannual
            elif compounding_frequency_val == 'Quarterly':
                compounding_frequency = ql.Quarterly
            elif compounding_frequency_val == 'Monthly':
                compounding_frequency = ql.Monthly

            # Map tenor value to QuantLib Tenor
            if tenor_val == 'Annual':
                tenor = ql.Period(ql.Annual)
                selected_period = ql.Period(1, ql.Years)
                payment_frequency_ql = ql.Annual
                period_fraction = 1  # Payment once per year
            elif tenor_val == 'Semiannual':
                tenor = ql.Period(ql.Semiannual)
                selected_period = ql.Period(6, ql.Months)
                payment_frequency_ql = ql.Semiannual
                period_fraction = 0.5  # Payment twice per year
            elif tenor_val == 'Quarterly':
                tenor = ql.Period(ql.Quarterly)
                selected_period = ql.Period(3, ql.Months)
                payment_frequency_ql = ql.Quarterly
                period_fraction = 0.25  # Payment four times per year
            elif tenor_val == 'Monthly':
                tenor = ql.Period(ql.Monthly)
                selected_period = ql.Period(1, ql.Months)
                payment_frequency_ql = ql.Monthly
                period_fraction = 1 / 12  # Payment twelve times per year
            else:
                raise ValueError("Invalid tenor selected")
                
            # Map the day count value to a QuantLib DayCount
            if day_count_val == 'ActualActual':
                day_count = ql.ActualActual(ql.ActualActual.Bond)
            elif day_count_val == 'Thirty360':
                day_count = ql.Thirty360(ql.Thirty360.BondBasis)
            elif day_count_val == 'Actual360':
                day_count = ql.Actual360()
            elif day_count_val == 'Actual365Fixed':
                day_count = ql.Actual365Fixed()
            
            # Define Side (Seller/Buyer)
            if side_type == "Seller":
                side = ql.Protection.Seller
            elif side_type == "Buyer":
                side = ql.Protection.Buyer
            else:
                raise ValueError("Invalid side type. It should be 'Buyer' or 'Seller'.")
                    
            floating_bond = NCFloatingBonds(value_date, spot_dates, spot_rates, index_dates, index_rates,
                                            calendar, currency, interpolation, compounding,
                                            compounding_frequency, epsilon=0.001)
            variable = range_span = num_steps = None
            cds = CreditDefaultSwap(notional_cds, spread, recovery_rate, risk_free,
                         payment_frequency_ql, calendar, side, selected_period, period_fraction, issue_date, maturity_date,
                         variable, range_span, num_steps)         
            cds_results_float = cds.cds_results()

            bond_params_float = {
                'value_date': form_data['value_date'],
                'spotDates': form_data['spot_dates'],
                'spotRates': form_data['spot_rates'],
                'indexRates': form_data['index_rates'],
                'indexDates': form_data['index_dates'],
                'currency': currency,
                'calendar': calendar,
                'interpolation': interpolation,
                'compounding': compounding,
                'compoundingFrequency': compounding_frequency
            }

            cln_pricing = CLNPricingFloat(cds_instance=cds, bond_params_float=bond_params_float)
            notional_dates = "2024-11-24, 2027-11-24, 2034-12-03"
            holding = 1

            if amort_selection_float == 'No':
                amortization = 'No'
                fl_bond_results = floating_bond.price_floating(shocks, issue_date, maturity_date,
                                                            tenor, spread, notional, day_count)
            elif amort_selection_float == 'Yes':
                amortization = 'Yes'
                flam_bond_results = floating_bond.price_amortizing_floating(shocks, issue_date, maturity_date,
                                                               tenor, spread, notional, notional_dates, day_count)
            cln_pricing_float = cln_pricing.calculate_cln_price_float(shocks, issue_date, maturity_date, tenor, spread, holding, day_count, notional, notional_dates, notionals, amortization)
            if action == 'float-sensitivity':                
                try:
                    # Get sensitivity-specific parameters
                    form_data['num_steps'] = int(request.form['num_steps'])
                    form_data['range_span'] = float(request.form['range_span'])
                    form_data['variable'] = str(request.form['variable'])
                    
                    # Get values from form data
                    num_steps = form_data['num_steps']
                    range_span = form_data['range_span']
                    variable = form_data['variable']
                    
                    # Get dropdown values
                    currency_val = form_data['currency_val']
                    calendar_val = form_data['calendar_val']
                    interpolation_val = form_data['interpolation_val']
                    compounding_val = form_data['compounding_val']
                    compounding_frequency_val = form_data['compounding_frequency_val']
                    tenor_val = form_data['tenor_val']
                    day_count_val = form_data['day_count_val']

                    floating_bond = NCFloatingBonds(value_date, spot_dates, spot_rates, index_dates, index_rates,
                                            calendar, currency, interpolation, compounding,
                                            compounding_frequency, epsilon=0.001)
                    
                    cds_class = CreditDefaultSwap(notional_cds, spread, recovery_rate, risk_free,
                                 payment_frequency_ql, calendar, side, selected_period, period_fraction, issue_date, maturity_date,
                                 variable, range_span, num_steps)
# shocks, issueDate, maturityDate, tenor, spread, notional, dayCount
                    if amort_selection_float == 'No':
                        if variable == 'risk_free':
                            fl_bond_results = floating_bond.price_floating(shocks, issue_date, maturity_date,
                                                            tenor, spread, notional, day_count)
                            sensitivity_analysis = CLNSensitivityAnalysis(bond_params=bond_params_float, cds_instance=cds_class, bond_results=fl_bond_results)
                            sensitivity_analysis.plot_sensitivity_analysis(variable, range_span, num_steps)
                        elif variable == 'recovery_rate':
                            fl_bond_results = floating_bond.price_floating(shocks, issue_date, maturity_date,
                                                            tenor, spread, notional, day_count)
                            sensitivity_analysis = CLNSensitivityAnalysis(bond_params=bond_params_float, cds_instance=cds_class, bond_results=fl_bond_results)
                            sensitivity_analysis.plot_sensitivity_analysis(variable, range_span, num_steps)
                        elif variable == 'spread':
                            fl_bond_results = floating_bond.price_floating(shocks, issue_date, maturity_date,
                                                            tenor, spread, notional, day_count)
                            sensitivity_analysis = CLNSensitivityAnalysis(bond_params=bond_params_float, cds_instance=cds_class, bond_results=fl_bond_results)
                            sensitivity_analysis.plot_sensitivity_analysis(variable, range_span, num_steps)
                    elif amort_selection_float == 'Yes':
                        if variable == 'risk_free':
                            flam_bond_results = floating_bond.price_amortizing_floating(shocks, issue_date, maturity_date,
                                                               tenor, spread, notional, notional_dates, day_count)
                            sensitivity_analysis = CLNSensitivityAnalysis(bond_params=bond_params_float, cds_instance=cds_class, bond_results=flam_bond_results)
                            sensitivity_analysis.plot_sensitivity_analysis(variable, range_span, num_steps)
                        elif variable == 'recovery_rate':
                            flam_bond_results = floating_bond.price_amortizing_floating(shocks, issue_date, maturity_date,
                                                               tenor, spread, notional, notional_dates, day_count)
                            sensitivity_analysis = CLNSensitivityAnalysis(bond_params=bond_params_float, cds_instance=cds_class, bond_results=flam_bond_results)
                            sensitivity_analysis.plot_sensitivity_analysis(variable, range_span, num_steps)
                        elif variable == 'spread':
                            flam_bond_results = floating_bond.price_amortizing_floating(shocks, issue_date, maturity_date,
                                                               tenor, spread, notional, notional_dates, day_count)
                            sensitivity_analysis = CLNSensitivityAnalysis(bond_params=bond_params_float, cds_instance=cds_class, bond_results=flam_bond_results)
                            sensitivity_analysis.plot_sensitivity_analysis(variable, range_span, num_steps)

                    
                    plot_filename = f'Credit Linked Notes Floating_{variable}_{range_span}_sensitivity_plot.png'
                    plot_path = os.path.join('app', 'static', plot_filename)
                    plt.savefig(plot_path)   
                    session['cln_sensitivity_analysis_results_float'] = {'plot_filename': plot_filename,
                                                                        'range_span': range_span,
                                                                        'num_steps': num_steps}
                    cln_sensitivity_analysis_results_float = True
                   
                    # plt.close()  # Close the plot after saving 
    
                except Exception as e:
                    print(f"An error occurred during Sensitivity analysis: {e}")
            
            elif action == 'float-scenario':
                try:
                    rate_change = float(request.form.get('rate_scenario_float', 0))
                    spread_change = float(request.form.get('spread_scenario_float',0))
                    recovery_change = float(request.form.get('recovery_scenario_float',0))

                    variable = step_range = num_steps = None
                    cds_class = CreditDefaultSwap(notional_cds, spread, recovery_rate, risk_free,
                                 payment_frequency_ql, calendar, side, selected_period, period_fraction, issue_date, maturity_date,
                                 variable, step_range, num_steps)
                 
                    stressed_rate = risk_free + rate_change
                    stressed_rec = recovery_rate + recovery_change
                    stressed_spread = spread * (1+spread_change)

                    cds_stressed = CreditDefaultSwap(notional_cds, stressed_spread, stressed_rec, stressed_rate,
                                 payment_frequency_ql, calendar, side, selected_period, period_fraction, issue_date, maturity_date,
                                 variable, step_range, num_steps)

                    # Map bond type value
                    if amort_selection_float == 'No':
                        amortization = 'No'
                        fl_bond_results = floating_bond.price_floating(shocks, issue_date, maturity_date,
                                                               tenor, spread, holding, day_count,notional)
                        cln_pricing_baseline = CLNPricingFloat(bond_params_float=bond_params_float,cds_instance=cds_class)
                        baseline_CLN_float = cln_pricing_baseline.calculate_cln_price_float(shocks, issue_date, maturity_date, tenor, spread, holding, 
                                                                                      day_count, notional, notional_dates, notionals, amortization)
                        cln_pricing_stressed = CLNPricingFloat(bond_params_float=bond_params_float,cds_instance=cds_stressed)   
                        stressed_CLN_float = cln_pricing_stressed.calculate_cln_price_float(shocks, issue_date, maturity_date, tenor, spread, holding, 
                                                                                      day_count, notional, notional_dates, notionals, amortization)
                    elif amort_selection_float == 'Yes':
                        amortization = 'Yes'
                        flam_bond_results = floating_bond.price_amortizing_floating(shocks, issue_date, maturity_date,
                                                               tenor, spread, notional, notional_dates, day_count)
                        cln_pricing_baseline = CLNPricingFloat(bond_params_float=bond_params_float,cds_instance=cds)
                        baseline_CLN_float = cln_pricing_baseline.calculate_cln_price_float(shocks, issue_date, maturity_date, tenor, spread, holding, 
                                                                                      day_count, notional, notional_dates, notionals, amortization)
                        cln_pricing_stressed = CLNPricingFloat(bond_params_float=bond_params_float,cds_instance=cds_stressed)
                        stressed_CLN_float = cln_pricing_stressed.calculate_cln_price_float(shocks, issue_date, maturity_date, tenor, spread, holding, 
                                                                                      day_count, notional, notional_dates, notionals, amortization)

                except Exception as e:
                    print(f"An error occurred during scenario analysis: {e}")
            
            
    return render_template('credit_linked_notes.html', md_content=md_content, form_data=form_data, fr_bond_results=fr_bond_results, fram_bond_results=fram_bond_results, 
                           fl_bond_results=fl_bond_results, flam_bond_results=flam_bond_results, cds_results_fixed=cds_results_fixed, cds_results_float=cds_results_float, cln_pricing_fixed=cln_pricing_fixed, cln_pricing_float=cln_pricing_float,

                           cln_sensitivity_analysis_results_fixed=cln_sensitivity_analysis_results_fixed, cln_sensitivity_analysis_results_float=cln_sensitivity_analysis_results_float,baseline_CLN=baseline_CLN, stressed_CLN=stressed_CLN, baseline_CLN_float=baseline_CLN_float, stressed_CLN_float=stressed_CLN_float)

