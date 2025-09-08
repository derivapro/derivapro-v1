# Note: last updated on Aug 06

from datetime import datetime
from flask import Blueprint, render_template, request, session
from ..models.mdls_monte_carlo import MonteCarlo
from ..models.mdls_monte_carlo import convergence_test
from ..models.mdls_monte_carlo import plot_convergence
from ..models.mdls_monte_carlo import MonteCarloSmoothnessTest
from ..models.mdls_asian_options import AsianOption, AsianOptionSmoothnessTest, lattice_convergence_test
from ..models.mdls_autocallables import AutoMonteCarlo, AutocallableSmoothnessTest, auto_convergence_test
import importlib.util
import sys
import os

# Import the Monte Carlo PKIC module
monte_carlo_pkic_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'mdls_monte_carlo_PKIC.py')
spec = importlib.util.spec_from_file_location("monte_carlo_pkic_module", monte_carlo_pkic_path)
if spec is not None:
    monte_carlo_pkic_module = importlib.util.module_from_spec(spec)
    if spec.loader is not None:
        spec.loader.exec_module(monte_carlo_pkic_module)
else:
    raise ImportError(f"Could not load Monte Carlo PKIC module from {monte_carlo_pkic_path}")
import matplotlib.pyplot as plt
import os
import markdown
from openai import AzureOpenAI
import logging

exotic_options_bp = Blueprint('exotic_options', __name__)

# Get the values from the environment variables
api_key = "687ac7173dfd4a45a45573435a4daac9"
base_url = "https://atlas.protiviti.com/experiment20240821"
api_version = "2025-03-01-preview"
model = "gpt-4o-mini-20240718-gs"  # Internal company GenAI model

# Add auth header
auth_headers = {
    "Experiment20240821-Subscription-Key": api_key
}

# Instantiate the Azure OpenAI client
client = AzureOpenAI(
    api_key=api_key,  
    api_version=api_version,
    default_headers=auth_headers,
    azure_endpoint=base_url
)

def ask_gpt(question):
    """
    Sends a request to Azure OpenAI's GPT-4 API with the given question.
    
    Args:
        question (str): The input question or prompt to GPT.
        
    Returns:
        str: The generated response from GPT, or an error message in case of failure.
    """
    try:
        print(api_key)
    
        # Send the request to Azure OpenAI API
        response = client.chat.completions.create(
            model=model, 
            messages=[
                {"role": "system", "content": "Assistant is a large language model hosted in Azure OpenAI."},
                {"role": "user", "content": f"{question}"}
            ]
        )

        # Extract the content of the response
        return response.choices[0].message.content

    except Exception as e:
        logging.error(f"Error occurred while calling OpenAI API: {e}")
        return f"An error occurred: {e}"

@exotic_options_bp.route('/', methods=['GET', 'POST'])
def exotic_options():
    return render_template('exotic_options.html')

@exotic_options_bp.route('/autocallable', methods=['GET', 'POST'])
def autocallable_options():
    readme_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'autocallable_options.md')
    with open(readme_path, 'r') as readme_file:
        content = readme_file.read()
    md_content = markdown.markdown(content)
    
    option_price = sensitivity_results = scenario_results = convergence_results = risk_pl_results = None
    form_data = {}

    if request.method == 'POST':
        action = request.form.get('analysis_type')
        form_data = {
            'ticker': request.form['ticker'],
            'K': request.form['K'],
            'r': request.form['r'],
            'sigma': request.form['sigma'],
            'T': request.form['T'],
            'q': request.form['q'],
            'N': request.form['N'],
            'M': request.form['M'],
            'barrier_levels': request.form['barrier_levels'],
            'coupon_rates': request.form['coupon_rates'],
            'discretization': request.form['discretization'],
            'simulation_engine': request.form.get('simulation_engine', 'original'),
            'option_type': request.form.get('option_type', 'call')
        }

        ticker = form_data['ticker']
        K = float(form_data['K']) if form_data['K'] not in [None, ''] else 0.0
        r = float(form_data['r']) if form_data['r'] not in [None, ''] else 0.0
        sigma = float(form_data['sigma']) if form_data['sigma'] not in [None, ''] else 0.0
        T = float(form_data['T']) if form_data['T'] not in [None, ''] else 0.0
        q = float(form_data['q']) if form_data['q'] not in [None, ''] else 0.0
        N = int(form_data['N']) if form_data['N'] not in [None, ''] else 0
        M = int(form_data['M']) if form_data['M'] not in [None, ''] else 0
        barrier_levels = float(form_data['barrier_levels']) if form_data['barrier_levels'] not in [None, ''] else 0.0
        coupon_rates = float(form_data['coupon_rates']) if form_data['coupon_rates'] not in [None, ''] else 0.0
        discretization = form_data['discretization']
        simulation_engine = form_data.get('simulation_engine', 'original')
        option_type = form_data.get('option_type', 'call')

        if simulation_engine == 'pkic':
            try:
                from ..models.market_data import StockData
                stock_data = StockData(ticker)
                S0 = float(stock_data.get_current_price())
                mc_engine = monte_carlo_pkic_module.create_monte_carlo_engine(
                    S0=S0,
                    r=r,
                    sigma=sigma,
                    T=T,
                    num_paths=M,
                    num_steps=N,
                    random_type="sobol"
                )
                option_price = mc_engine.price_autocallable_option(
                    strike_price=K,
                    barrier_levels=barrier_levels,
                    coupon_rates=coupon_rates,
                    T=T,
                    option_type=option_type,
                    discretization=discretization,
                    dividend_yield=q
                )
                option_price = "${:,.4f}".format(option_price)
            except Exception as e:
                print(f"Error using PKIC engine: {e}")
                option = AutoMonteCarlo(ticker, K, r, sigma, T, q, N, M)
                option_price = option.price_autocallable_option(discretization=discretization, barrier_levels=barrier_levels, coupon_rates=coupon_rates)
                option_price = "${:,.4f}".format(option_price)
        else:
            option = AutoMonteCarlo(ticker, K, r, sigma, T, q, N, M)
            option_price = option.price_autocallable_option(discretization=discretization, barrier_levels=barrier_levels, coupon_rates=coupon_rates)
            option_price = "${:,.4f}".format(option_price)

    return render_template('autocallables.html', option_price=option_price, form_data=form_data,
                           sensitivity_results=sensitivity_results, convergence_results=convergence_results,
                           scenario_results=scenario_results, risk_pl_results=risk_pl_results, md_content=md_content)

@exotic_options_bp.route('/asian', methods=['GET', 'POST'])
def asian_options():
    readme_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'asian_options.md')
    with open(readme_path, 'r') as readme_file:
        content = readme_file.read()
    md_content = markdown.markdown(content)
    
    option_price = sensitivity_results = scenario_results = convergence_results = risk_pl_results = None
    form_data = {}

    if request.method == 'POST':
        action = request.form.get('analysis_type')
        form_data = {
            'ticker': request.form['ticker'],
            'K': request.form['K'],
            'r': request.form['r'],
            'sigma': request.form['sigma'],
            'T': request.form['T'],
            'q': request.form['q'],
            'averaging_dates': request.form['averaging_dates'],
            'num_paths': request.form['num_paths'],
            'option_type': request.form['option_type'],
            'simulation_engine': request.form.get('simulation_engine', 'original')
        }

        ticker = form_data['ticker']
        K = float(form_data['K']) if form_data['K'] not in [None, ''] else 0.0
        r = float(form_data['r']) if form_data['r'] not in [None, ''] else 0.0
        sigma = float(form_data['sigma']) if form_data['sigma'] not in [None, ''] else 0.0
        T_str = form_data['T']
        T = datetime.strptime(T_str.strip(), '%Y-%m-%d')
        q = float(form_data['q']) if form_data['q'] not in [None, ''] else 0.0
        averaging_dates_str = form_data['averaging_dates']
        averaging_dates = [datetime.strptime(date.strip(), '%Y-%m-%d') for date in averaging_dates_str.split(',')]
        num_paths = int(form_data['num_paths']) if form_data['num_paths'] not in [None, ''] else 0
        option_type = form_data['option_type']
        simulation_engine = form_data.get('simulation_engine', 'original')

        if simulation_engine == 'pkic':
            try:
                from ..models.market_data import StockData
                stock_data = StockData(ticker)
                S0 = float(stock_data.get_current_price())
                averaging_dates_sorted = sorted(averaging_dates)
                num_steps = len(averaging_dates_sorted) - 1  # one fewer than the number of dates
                T_years = (averaging_dates_sorted[-1] - averaging_dates_sorted[0]).days / 365.25

                mc_engine = monte_carlo_pkic_module.create_monte_carlo_engine(
                    S0=S0,
                    r=r,
                    sigma=sigma,
                    T=T_years,
                    num_paths=num_paths,
                    num_steps=num_steps,     # <<- Now exactly matches intervals between averaging dates
                    random_type="sobol"
                )

                option_price = mc_engine.price_asian_option(
                    strike_price=K,
                    averaging_dates=averaging_dates_sorted,
                    option_type=option_type,
                    dividend_yield=q
                )
                option_price = "${:,.4f}".format(option_price)
            except Exception as e:
                print(f"Error using PKIC engine: {e}")
                option_price = f"New MC error: {e}"
        else:
            option = AsianOption(ticker, K, sigma, r, q, T, averaging_dates, option_type, num_paths)
            option_price = option.price()
            option_price = "${:,.4f}".format(option_price)

    return render_template('asian_options.html', option_price=option_price,
                           sensitivity_results=sensitivity_results, convergence_results=convergence_results,
                           scenario_results=scenario_results, risk_pl_results=risk_pl_results, form_data=form_data, md_content=md_content)

@exotic_options_bp.route('/barrier', methods=['GET', 'POST'])
def barrier_options():
    readme_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'barrier_options.md')
    with open(readme_path, 'r') as readme_file:
        content = readme_file.read()
    md_content = markdown.markdown(content)
    
    option_price = sensitivity_results = scenario_results = convergence_results = risk_pl_results = None
    form_data = {}

    if request.method == 'POST':
        action = request.form.get('analysis_type')

        form_data = {
            'ticker': request.form['ticker'],
            'K': float(request.form['K']) if request.form['K'] not in [None, ''] else 0.0,
            'r': float(request.form['r']) if request.form['r'] not in [None, ''] else 0.0,
            'sigma': float(request.form['sigma']) if request.form['sigma'] not in [None, ''] else 0.0,
            'start_date': str(request.form['start_date']),
            'end_date': str(request.form['end_date']),
            'q': float(request.form['q']) if request.form['q'] not in [None, ''] else 0.0,
            'N': int(request.form['N']) if request.form['N'] not in [None, ''] else 0,
            'M': int(request.form['M']) if request.form['M'] not in [None, ''] else 0,
            'barrier': float(request.form['barrier']) if request.form['barrier'] not in [None, ''] else 0.0,
            'option_type': request.form['option_type'],
            'barrier_type': request.form['barrier_type'],
            'discretization': request.form['discretization'],
            'simulation_engine': request.form.get('simulation_engine', 'original')
        }

        if action == 'sensitivity':
            try:
                # Sensitivity analysis logic
                form_data['num_sensitivity_steps'] = int(request.form['num_sensitivity_steps'])
                form_data['step_range'] = int(request.form['step_range'])
                form_data['variable'] = request.form['variable']
                form_data['target_variable'] = request.form['target_variable']

                num_steps = form_data['num_sensitivity_steps']
                step_range = form_data['step_range']
                variable = form_data['variable']
                target_variable = form_data['target_variable']

                tester = MonteCarloSmoothnessTest(**form_data)

                values, greek_values = tester.calculate_greeks_over_range(variable, num_steps, step_range,
                                                                          target_variable)
                tester.plot_single_greek(values, greek_values, target_variable, variable)

                # Save plot to static directory
                print('start plotting')
                plot_filename = f'barrier_{target_variable}-{variable}_sensitivity_plot.png'
                plot_path = os.path.join('app', 'static', plot_filename)

                # Print the plot path to ensure it's correct
                print(f"Saving plot to: {plot_path}")

                # Save plot as a base64 encoded string
                # img_buffer = BytesIO()
                # plt.savefig(img_buffer, format='png')  # Save plot to BytesIO buffer
                plt.savefig(plot_path)

                # img_buffer.seek(0)
                # plot_base64 = base64.b64encode(img_buffer.read()).decode('utf-8')

                # Store the results and plot path in the session
                session['sensitivity_results'] = {
                    'variable': variable,
                    'values': values.tolist(),
                    'greek_values': greek_values,
                    'target_variable': target_variable,
                    'plot_filename': plot_filename,
                    # 'plot_base64': plot_base64  # Store the plot as base64
                }

                sensitivity_results = True

                plt.close()  # Close the plot to release memory

            except Exception as e:
                print(f"An error occurred during sensitivity analysis: {e}")
                sensitivity_results = None

        elif action == 'convergence':
            # Get necessary form data for convergence analysis

            form_data['max_steps'] = int(request.form['max_steps'])
            form_data['max_sims'] = int(request.form['max_sims'])
            form_data['obs'] = int(request.form['obs'])
            form_data['mode'] = request.form['mode']

            mode = form_data['mode']
            max_steps = form_data['max_steps']
            max_sims = form_data['max_sims']
            obs = form_data['obs']

            barrier_step_results = convergence_test(max_steps, max_sims, obs,
                                                    MonteCarlo, form_data, mode)

            plot_convergence(barrier_step_results, mode)

            # Save plot to static directory
            plt.savefig('app/static/barrier_convergence_plot.png')
            convergence_results = True
        elif action == 'scenario':
            try:
                # Scenario analysis logic...
                spot_change = float(request.form.get('spot_scenario', 0))
                vol_change = float(request.form.get('vol_scenario', 0))
                rate_change = float(request.form.get('rate_scenario', 0))
                strike_price_raw = form_data.get('K')
                strike_price = float(str(strike_price_raw).strip()) if strike_price_raw not in [None, ''] and str(strike_price_raw).strip() != '' else 0.0
                risk_free_rate_raw = form_data.get('r')
                risk_free_rate = float(str(risk_free_rate_raw).strip()) if risk_free_rate_raw not in [None, ''] and str(risk_free_rate_raw).strip() != '' else 0.0
                volatility_raw = form_data.get('sigma')
                volatility = float(str(volatility_raw).strip()) if volatility_raw not in [None, ''] and str(volatility_raw).strip() != '' else 0.0

                # Baseline calculation
                option = MonteCarlo(**form_data)
                baseline_price = option.price_barrier_option()
                baseline_greeks = option.calculate_greeks()
                baseline_price = "{:.4f}".format(baseline_price)
                baseline_delta = "{:.4f}".format(baseline_greeks['Delta'])
                baseline_gamma = "{:.4f}".format(baseline_greeks['Gamma'])
                baseline_vega = "{:.4f}".format(baseline_greeks['Vega'])
                baseline_theta = "{:.4f}".format(baseline_greeks['Theta'])
                baseline_rho = "{:.4f}".format(baseline_greeks['Rho'])

                # Stressed scenario calculation
                stressed_spot = strike_price * (1 + spot_change)
                stressed_vol = volatility + vol_change
                stressed_rate = risk_free_rate + rate_change

                stressed_input = {
                    'ticker': form_data['ticker'], 'start_date': form_data['start_date'], 'end_date': form_data['end_date'],
                    'r': stressed_rate, 'sigma': stressed_vol, 'N': form_data['N'], 'M': form_data['M'],
                    'K': stressed_spot, 'q': form_data['q'], 'barrier': form_data['barrier'],
                    'option_type': form_data['option_type'], 'barrier_type': form_data['barrier_type']
                }

                stressed_option = MonteCarlo(**stressed_input)
                stressed_price = stressed_option.price_barrier_option()
                stressed_greeks = stressed_option.calculate_greeks()
                stressed_price = "{:.4f}".format(stressed_price)
                stressed_delta = "{:.4f}".format(stressed_greeks['Delta'])
                stressed_gamma = "{:.4f}".format(stressed_greeks['Gamma'])
                stressed_vega = "{:.4f}".format(stressed_greeks['Vega'])
                stressed_theta = "{:.4f}".format(stressed_greeks['Theta'])
                stressed_rho = "{:.4f}".format(stressed_greeks['Rho'])

                '''
                scenario_table = [
                    {
                        "scenario": "Baseline",
                        "option_price": baseline_price,
                        "delta": baseline_delta,
                        "gamma": baseline_gamma,
                        "vega": baseline_vega,
                        "theta": baseline_theta,
                        "rho": baseline_rho,
                    },
                    {
                        "scenario": "Stressed Scenario",
                        "option_price": stressed_price,
                        "delta": stressed_delta,
                        "gamma": stressed_gamma,
                        "vega": stressed_vega,
                        "theta": stressed_theta,
                        "rho": stressed_rho,
                    },
                ]
                '''

                baseline_scenario_table = {
                        "scenario": "Baseline",
                        "baseline_price": baseline_price,
                        "baseline_delta": baseline_delta,
                        "baseline_gamma": baseline_gamma,
                        "baseline_vega": baseline_vega,
                        "baseline_theta": baseline_theta,
                        "baseline_rho": baseline_rho,
                    }

                stressed_scenario_table = {
                    "scenario": "Stressed",
                    "stressed_price": stressed_price,
                    "stressed_delta": stressed_delta,
                    "stressed_gamma": stressed_gamma,
                    "stressed_vega": stressed_vega,
                    "stressed_theta": stressed_theta,
                    "stressed_rho": stressed_rho,
                }

                print(baseline_scenario_table)
                print(stressed_scenario_table)

                # Save both the scenario table and the assessment to session
                session['scenario_results'] = {
                    'baseline_scenario_table': baseline_scenario_table,
                    'stressed_scenario_table': stressed_scenario_table,
                    'gpt_scenario_assessment': "No assessment yet."  # Placeholder for GPT assessment
                }

                scenario_results = True

            except Exception as e:
                print(f"An error occurred during scenario analysis: {e}")
                scenario_results = None

        elif action == 'ai_scenario_assessment':
            scenario_results = session.get('scenario_results', {})
            baseline_table = scenario_results.get('baseline_scenario_table', {})
            stressed_table = scenario_results.get('stressed_scenario_table', {})

            if baseline_table and stressed_table:
                # Format the scenario results for the assessment
                table_text = f"""
Baseline Scenario:
Option Price={baseline_table['baseline_price']}, Delta={baseline_table['baseline_delta']}, 
Gamma={baseline_table['baseline_gamma']}, Vega={baseline_table['baseline_vega']}, 
Theta={baseline_table['baseline_theta']}, Rho={baseline_table['baseline_rho']}

Stressed Scenario:
Option Price={stressed_table['stressed_price']}, Delta={stressed_table['stressed_delta']}, 
Gamma={stressed_table['stressed_gamma']}, Vega={stressed_table['stressed_vega']}, 
Theta={stressed_table['stressed_theta']}, Rho={stressed_table['stressed_rho']}
"""
                assessment_input = f"Please assess the scenario analysis of the option price and Greeks based on the following results: {table_text}. Please limit the assessment to be less than 100 words."
                gpt_scenario_assessment = ask_gpt(assessment_input)

                # Save the GPT assessment to the session
                scenario_results['gpt_scenario_assessment'] = gpt_scenario_assessment
                session['scenario_results'] = scenario_results  # Update session with the assessment
            else:
                scenario_results['gpt_scenario_assessment'] = "No scenario analysis results available for assessment."
                session['scenario_results'] = scenario_results

        elif action == 'risk_pl':
            try:
                # RBPL analysis logic
                form_data['price_change'] = float(request.form['price_change'])
                form_data['vol_change'] = float(request.form['vol_change'])

                price_change = form_data['price_change']
                vol_change = form_data['vol_change']

                option = MonteCarlo(**form_data)
                risk_pl_results = option.risk_pl_analysis(price_change, vol_change, discretization=form_data['discretization'])

                print(risk_pl_results)

            except Exception as e:
                print(f'An error occurred during Risk-Based P&L analysis: {e}')
                risk_pl_results = None

        else:
            # Get necessary form data for option pricing
            simulation_engine = request.form.get('simulation_engine', 'original')
            
            if simulation_engine == 'pkic':
                # Use PKIC Monte Carlo engine
                try:
                    # Get stock data
                    from ..models.market_data import StockData
                    stock_data = StockData(form_data['ticker'], form_data['start_date'], form_data['end_date'])
                    S0 = float(stock_data.get_closing_price())
                    T = stock_data.get_years_difference()
                    
                    # Create PKIC Monte Carlo engine
                    mc_engine = monte_carlo_pkic_module.create_monte_carlo_engine(
                        S0=S0,
                        r=form_data['r'],
                        sigma=form_data['sigma'],
                        T=T,
                        num_paths=form_data['M'],
                        num_steps=form_data['N'],
                        random_type="sobol"
                    )
                    
                    # Price barrier option using PKIC engine
                    option_price = mc_engine.price_barrier_option(
                        strike_price=form_data['K'],
                        barrier_level=form_data['barrier'],
                        option_type=form_data['option_type'],
                        barrier_type=form_data['barrier_type'],
                        dividend_yield=form_data['q']
                    )
                    
                    option_price = "${:,.4f}".format(option_price)
                    
                except Exception as e:
                    print(f"Error using PKIC engine: {e}")
                    # Fallback to original engine
                    option = MonteCarlo(**form_data)
                    option_price = option.price_barrier_option()
                    option_price = "${:,.4f}".format(option_price)
            else:
                # Use original Monte Carlo engine
                option = MonteCarlo(**form_data)
                option_price = option.price_barrier_option()
                option_price = "${:,.4f}".format(option_price)

    return render_template('barrier_options.html', option_price=option_price, form_data=form_data,
                           sensitivity_results=sensitivity_results, convergence_results=convergence_results,
                           scenario_results=scenario_results, risk_pl_results=risk_pl_results, md_content=md_content)
