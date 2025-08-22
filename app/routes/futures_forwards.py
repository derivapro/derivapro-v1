# -*- coding: utf-8 -*-
"""
Created on Sun Jun  9 00:46:08 2024

@author: minwuu01
"""

from flask import Blueprint, render_template, request, session
from ..models.mdls_futures_forwards import Forwards, Futures
from ..models.mdls_futures_forwards import ForwardsAnalysis
from ..models.mdls_futures_forwards import FuturesAnalysis
import os
import markdown
import matplotlib.pyplot as plt
from datetime import datetime


futures_forwards_bp = Blueprint('futures_forwards', __name__)

@futures_forwards_bp.route('/', methods=['GET', 'POST'])
def futures_forwards():
    return render_template('futures_forwards.html')

@futures_forwards_bp.route('/forwards', methods=['GET', 'POST'])
def forwards():
    readme_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'forwards.md')
    with open(readme_path, 'r') as readme_file:
        content = readme_file.read()
    md_content = markdown.markdown(content)
    
    forward_price_results = forward_pL = margin_requirement = forward_sensitivity_analysis_results = forward_scenario_results = forward_risk_pl = None
    form_data = {}
    
    if request.method == 'POST':
        action = request.form.get('analysis_type')
        
        form_data = {
            'ticker': request.form['ticker'],
            'risk_free_rate': request.form['risk_free_rate'],
            'dividend_yield': request.form['dividend_yield'],
            'convenience_yield': float(request.form['convenience_yield']) if 'convenience_yield' in request.form else 0,
            'settlement_price': request.form['settlement_price'],
            'num_contracts': request.form['num_contracts'],
            'multiplier': request.form['multiplier'],
            'position': request.form['position'],
            'contract_fee': request.form['contract_fee'],
            'entry_date': request.form[('entry_date')],
            'settlement_date': request.form['settlement_date'],
            'model_selection': request.form['model_selection'],
            'storage_cost': request.form['storage_cost']
        }
        
        ticker = form_data['ticker']
        risk_free_rate = float(form_data['risk_free_rate'])
        dividend_yield = float(form_data['dividend_yield'])
        convenience_yield = float(form_data['convenience_yield'])
        contract_fee = float(form_data['contract_fee'])
        settlement_price = float(form_data['settlement_price'])
        entry_date = form_data['entry_date']
        settlement_date = form_data['settlement_date']
        num_contracts = int(form_data['num_contracts'])
        multiplier = int(form_data['multiplier'])
        position = str(form_data['position'].lower())
        model_selection = str(form_data['model_selection'].lower())
        storage_cost = int(form_data['storage_cost'])
        if position not in ["long", "short"]:
            return "Invalid position! Must be 'long' or 'short'.", 400
        
        
        forward_model = Forwards(ticker, risk_free_rate, dividend_yield, convenience_yield, entry_date,
                           settlement_date, settlement_price, num_contracts, multiplier, 
                           position, contract_fee, model_selection, storage_cost)
        
        forward_price_results = forward_model.forward_price()
        forward_price_results = "${:,.4f}".format(forward_price_results)
        forward_pL = forward_model.calculate_profit_loss()
        forward_pL = "${:,.4f}".format(forward_pL)
  
        if action == 'sensitivity':
            try:
                form_data['number_of_steps'] = int(request.form['number_of_steps'])
                form_data['sens_step_range'] = float(request.form['sens_step_range'])
                form_data['variable'] = str(request.form['variable'])
                num_steps = form_data['number_of_steps']
                step_range = form_data['sens_step_range']
                variable = form_data['variable']
                
                entry_date = datetime.strptime(form_data['entry_date'], '%Y-%m-%d')
                settlement_date = datetime.strptime(form_data['settlement_date'], '%Y-%m-%d')
                forwards_analysis = ForwardsAnalysis(ticker, risk_free_rate, dividend_yield, convenience_yield, entry_date,
                                         settlement_date, settlement_price, num_contracts, multiplier,
                                         position, contract_fee, num_steps, step_range, model_selection, storage_cost)
                
                if variable == 'risk_free_rate':
                    forward_sensitivity_analysis_results = forwards_analysis.analyze_variable_sensitivity(variable, step_range, num_steps)
                    forwards_analysis.plot_sensitivity_analysis(variable, step_range, num_steps)
                elif variable == 'dividend_yield':
                    forward_sensitivity_analysis_results = forwards_analysis.analyze_variable_sensitivity(variable, step_range, num_steps)
                    forwards_analysis.plot_sensitivity_analysis(variable, step_range, num_steps)
                elif model_selection == 'cost_carry_model' and variable == 'convenience_yield':
                    forward_sensitivity_analysis_results = forwards_analysis.analyze_variable_sensitivity(variable, step_range, num_steps)
                    forwards_analysis.plot_sensitivity_analysis(variable, step_range, num_steps)
                else:
                    pass
                
                plot_filename = f'forwards_{variable}_{step_range}_sensitivity_plot.png'
                plot_path = os.path.join('app', 'static', plot_filename)
                plt.savefig(plot_path)   
                session['forward_sensitivity_analysis_results'] = {'plot_filename': plot_filename,
                                                                   'step_range': step_range,
                                                                   'num_steps': num_steps}
                forward_sensitivity_analysis_results = True
                
                plt.close()  # Close the plot after saving 
                
            except Exception as e:
                forward_sensitivity_analysis_results = f"Error in sensitivity analysis: {str(e)}"
                session['sensitivity_analysis_results'] = None
        
        elif action == 'scenario':
            try:
                rate_change = float(request.form.get('rate_scenario', 0))
                div_change = float(request.form.get('div_scenario',0))
                cc_change = float(request.form.get('cc_scenario',0))
                
                forward_model = Forwards(ticker, risk_free_rate, dividend_yield, convenience_yield, entry_date,
                                   settlement_date, settlement_price, num_contracts, multiplier, 
                                   position, contract_fee, model_selection, storage_cost)
                
                baseline_forwardPrice = forward_model.forward_price()
                baseline_forwardPL = forward_model.calculate_profit_loss()
                
                
                stressed_rate = risk_free_rate + rate_change
                stressed_div = dividend_yield + div_change
                stressed_cc = convenience_yield * (1+cc_change)
                stressed_forward = Forwards(ticker, stressed_rate, stressed_div, stressed_cc, entry_date,
                                   settlement_date, settlement_price, num_contracts, multiplier, 
                                   position, contract_fee, model_selection, storage_cost)
                
                stressed_forwardPrice = stressed_forward.forward_price()
                stressed_forwardPL = stressed_forward.calculate_profit_loss()
                
                stressed_forwardPrice = "{:.4f}".format(stressed_forwardPrice)                    
                
                baseline_scenario_table = {
                    'scenario': 'Baseline',
                    'baseline_forwardPrice': baseline_forwardPrice,
                    'baseline_forwardPL': baseline_forwardPL
                }
                
                stressed_scenario_table = {
                    'scenario': 'Stressed',
                    'stressed_forwardPrice': stressed_forwardPrice,
                    'stressed_forwardPL': stressed_forwardPL
                }
                
                session['forward_scenario_results'] = {
                    'baseline_scenario_table': baseline_scenario_table,
                    'stressed_scenario_table': stressed_scenario_table
                }
                
                forward_scenario_results = True
            except Exception as e:
                print(f"An error occurred during scenario analysis: {e}")
                forward_scenario_results = None   
                
        elif action == 'risk_pl':
            try:
                form_data['price_change'] = float(request.form['price_change'])
                price_change = form_data['price_change']                
                
                forward_model = Forwards(ticker, risk_free_rate, dividend_yield, convenience_yield, entry_date,
                                   settlement_date, settlement_price, num_contracts, multiplier, 
                                   position, contract_fee, model_selection, storage_cost)
                forward_risk_pl = forward_model.risk_pl_analysis(price_change)
                
                print(forward_risk_pl)
            except Exception as e:
                print(f'An error occurred during Risk-Based P&L analysis: {e}')
                forward_risk_pl = None


    return render_template('forwards.html', form_data=form_data, forward_price_results=forward_price_results, 
                           forward_pL=forward_pL, margin_requirement=margin_requirement, 
                           forward_sensitivity_analysis_results=forward_sensitivity_analysis_results, forward_scenario_results= forward_scenario_results, forward_risk_pl=forward_risk_pl,
                           md_content=md_content
                           )

@futures_forwards_bp.route('/futures', methods=['GET', 'POST'])
def futures():
    readme_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'futures.md')
    with open(readme_path, 'r') as readme_file:
        content = readme_file.read()
    md_content = markdown.markdown(content)
    
    futures_price_results = futures_pL = margin_dict = mark_to_market_results = future_sensitivity_analysis_results = future_scenario_results= future_risk_pl = None
    form_data = {}
    
    if request.method == 'POST':
        action = request.form.get('analysis_type')
        
        try:
            storage_cost = int(request.form.get('storage_cost', 0.0))
        except ValueError:
            storage_cost = 0.0  # Default to 0.0 if the value is invalid
        
        form_data = {
            'ticker': request.form['ticker'],
            'risk_free_rate': request.form['risk_free_rate'],
            'dividend_yield': request.form['dividend_yield'],
            'storage_cost': storage_cost, 
            'convenience_yield': request.form['convenience_yield'],
            'settlement_price': request.form['settlement_price'],
            'num_contracts': request.form['num_contracts'],
            'multiplier': request.form['multiplier'],
            'position': request.form['position'],
            'contract_fee': request.form['contract_fee'],
            'entry_date': request.form['entry_date'],
            'settlement_date': request.form['settlement_date'],
            'initial_margin_pct': request.form['initial_margin_pct'],
            'maintenance_margin_pct': request.form['maintenance_margin_pct'],
            'model_selection': request.form['model_selection']
        }
        
        ticker = form_data['ticker']
        risk_free_rate = float(form_data['risk_free_rate'])
        dividend_yield = float(form_data['dividend_yield'])
        storage_cost = int(form_data['storage_cost'])
        convenience_yield = float(form_data['convenience_yield'])
        settlement_price = float(form_data['settlement_price'])
        num_contracts = int(form_data['num_contracts'])
        multiplier = int(form_data['multiplier'])
        contract_fee = float(form_data['contract_fee'])
        entry_date = form_data['entry_date']
        settlement_date = form_data['settlement_date']
        position = str(form_data['position'].lower())
        initial_margin_pct = float(form_data['initial_margin_pct'])
        maintenance_margin_pct = float(form_data['maintenance_margin_pct'])
        model_selection = str(form_data['model_selection'].lower())

        if position not in ["long", "short"]:
            return "Invalid position! Must be 'long' or 'short'.", 400

        # Create an instance of Futures
        future_model = Futures(ticker, risk_free_rate, dividend_yield, entry_date,
                      settlement_date, settlement_price, num_contracts, multiplier, 
                      position, storage_cost, contract_fee, model_selection, initial_margin_pct, maintenance_margin_pct, convenience_yield)
        future_model.generate_daily_prices()
        
        futures_price_results = future_model.futures_price()
        futures_price_results = "${:,.4f}".format(futures_price_results)
        futures_pL = future_model.calculate_profit_loss()
        futures_pL = "${:,.4f}".format(futures_pL)
        mark_to_market_results = future_model.mark_to_market()
        margin_dict = future_model.calculate_margin_requirement(initial_margin_pct, maintenance_margin_pct)
        
        if action == 'sensitivity':
            try:
                form_data['number_of_steps'] = int(request.form['number_of_steps'])
                form_data['sens_step_range'] = float(request.form['sens_step_range'])
                form_data['variable'] = str(request.form['variable'])
                
                num_steps = form_data['number_of_steps']
                step_range = form_data['sens_step_range']
                variable = form_data['variable']
                
                entry_date = datetime.strptime(form_data['entry_date'], '%Y-%m-%d')
                settlement_date = datetime.strptime(form_data['settlement_date'], '%Y-%m-%d')

                futures_analysis = FuturesAnalysis(ticker, risk_free_rate, dividend_yield, entry_date, 
                             settlement_date, settlement_price, num_contracts, multiplier, 
                             position, contract_fee, num_steps, step_range, storage_cost, 
                             initial_margin_pct, maintenance_margin_pct, model_selection,convenience_yield)
                                                        
                
                if variable == 'risk_free_rate':
                    future_sensitivity_analysis_results = futures_analysis.analyze_variable_sensitivity(variable, step_range, num_steps)
                    futures_analysis.plot_sensitivity_analysis(variable,step_range,num_steps)
                elif variable == 'dividend_yield':
                    future_sensitivity_analysis_results = futures_analysis.analyze_variable_sensitivity(variable, step_range, num_steps)
                    futures_analysis.plot_sensitivity_analysis(variable,step_range,num_steps)
                elif variable == 'convenience_yield':
                    future_sensitivity_analysis_results = futures_analysis.analyze_variable_sensitivity(variable, step_range, num_steps) 
                    futures_analysis.plot_sensitivity_analysis(variable,step_range,num_steps)
                else:
                    pass
                
                plot_filename = f'futures-{variable}-{step_range}_sensitivity_plot.png'
                plot_path = os.path.join('app', 'static', plot_filename)
                plt.savefig(plot_path)   
                
                session['future_sensitivity_analysis_results'] = {'plot_filename': plot_filename,
                                                                   'step_range': step_range,
                                                                   'num_steps': num_steps,
                                                                   'model_selection': model_selection}
                
                future_sensitivity_analysis_results = True
                
                plt.close()
            except Exception as e:
                future_sensitivity_analysis_results = f"Error in sensitivity analysis: {str(e)}"
                session['future_sensitivity_analysis_results'] = None
        elif action == 'scenario':
            try:
                rate_change = float(request.form.get('rate_scenario', 0))
                div_change = float(request.form.get('div_scenario',0))
                cc_change = float(request.form.get('cc_scenario',0))
                
                future_model = Futures(ticker, risk_free_rate, dividend_yield, entry_date,
                              settlement_date, settlement_price, num_contracts, multiplier, 
                              position, storage_cost, contract_fee, model_selection, initial_margin_pct, maintenance_margin_pct,convenience_yield)
                
                baseline_futPrice = future_model.futures_price()
                baseline_futPL = future_model.calculate_profit_loss()
                baseline_futPrice = "${:.4f}".format(baseline_futPrice)
                baseline_futPL = "${:.4f}".format(baseline_futPL)
                stressed_rate = risk_free_rate + rate_change
                stressed_div = dividend_yield + div_change
                stressed_cc = convenience_yield * (1+cc_change) 
                
                stressed_future = Futures(ticker, stressed_rate, stressed_div, entry_date,
                              settlement_date, settlement_price, num_contracts, multiplier, 
                              position, storage_cost, contract_fee, model_selection, initial_margin_pct, maintenance_margin_pct,stressed_cc)
                
                stressed_futPrice = stressed_future.futures_price()
                stressed_futPL = stressed_future.calculate_profit_loss()
                
                stressed_futPrice = "${:.4f}".format(stressed_futPrice)                    
                stressed_futPL = "${:.4f}".format(stressed_futPL)
                baseline_scenario_table = {
                    'scenario': 'Baseline',
                    'baseline_futPrice': baseline_futPrice,
                    'baseline_futPL': baseline_futPL
                }
                
                stressed_scenario_table = {
                    'scenario': 'Stressed',
                    'stressed_futPrice': stressed_futPrice,
                    'stressed_futPL': stressed_futPL
                }
                
                session['future_scenario_results'] = {
                    'baseline_scenario_table': baseline_scenario_table,
                    'stressed_scenario_table': stressed_scenario_table
                }
                
                future_scenario_results = True
            except Exception as e:
                print(f"An error occurred during scenario analysis: {e}")
                future_scenario_results = None  
        elif action == 'risk_pl':
            try:
                form_data['price_change'] = float(request.form['price_change'])
                price_change = form_data['price_change']
                
                future_model = Futures(ticker, risk_free_rate, dividend_yield, entry_date,
                      settlement_date, settlement_price, num_contracts, multiplier, 
                      position, storage_cost, contract_fee, model_selection, initial_margin_pct, maintenance_margin_pct,convenience_yield)
                future_risk_pl = future_model.risk_pl_analysis(price_change)
                
                print(future_risk_pl)
            except Exception as e:
                print(f'An error occurred during Risk-Based P&L analysis: {e}')
                future_risk_pl = None
        else:
            pass
                
    return render_template('futures.html', form_data=form_data, futures_price_results=futures_price_results, mark_to_market_results=mark_to_market_results,
                           futures_pL=futures_pL, margin_requirement=margin_dict,future_sensitivity_analysis_results=future_sensitivity_analysis_results,
                           future_scenario_results=future_scenario_results, future_risk_pl=future_risk_pl,md_content=md_content)

