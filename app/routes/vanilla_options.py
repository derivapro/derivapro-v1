# Note: last updated on Aug 06

from ..models.mdls_lattice_trees import LatticeModel, AmericanOptionSmoothnessTest, lattice_convergence_test
from flask import Blueprint, render_template, request, jsonify, session, redirect, url_for
from flask import send_file
#import pdfkit  # For PDF export (if needed)
from ..models.mdls_vanilla_options import BlackScholes, SmoothnessTest
from ..models.market_data import StockData
import matplotlib.pyplot as plt
import os
import markdown
from random import random
from datetime import datetime
from openai import OpenAI
#import json
#from weasyprint import HTML
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
#import base64
#from io import BytesIO
from ..models.mdls_monte_carlo import convergence_test, MonteCarlo, plot_convergence
from openai import AzureOpenAI
import os
from dotenv import load_dotenv, find_dotenv
import logging
import numpy as np
import importlib.util

# Import the Monte Carlo module with space in filename
monte_carlo_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models', 'mdls_monte_carlo_PKIC.py')
spec = importlib.util.spec_from_file_location("monte_carlo_module", monte_carlo_path)
if spec is not None:
    monte_carlo_module = importlib.util.module_from_spec(spec)
    if spec.loader is not None:
        spec.loader.exec_module(monte_carlo_module)
else:
    raise ImportError(f"Could not load Monte Carlo module from {monte_carlo_path}")

vanilla_options_bp = Blueprint('vanilla_options', __name__)


# Initialize OpenAI API
#api_key = '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'


# Load the environment variables from the .env file
load_dotenv(find_dotenv())

# Get the values from the environment variables
api_key =  "687ac7173dfd4a45a45573435a4daac9"
base_url = "https://atlas.protiviti.com/experiment20240821"
api_version = "2025-03-01-preview"
model = "gpt-4o-mini-20240718-gs"

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
        max_tokens (int): The maximum number of tokens in the response.
        
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
    
@vanilla_options_bp.route('/save-assessment', methods=['POST'])
def save_assessment():
    assessment_data = request.json.get('assessment')
    
    try:
        # Save the assessment to a file or database for later use
        with open('app/static/assessment.txt', 'w') as f:
            f.write(assessment_data)
        return jsonify({"status": "success"})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})
    
    

@vanilla_options_bp.route('/', methods=['GET', 'POST'])
def vanilla_options():   
    return render_template('vanilla_options.html')


@vanilla_options_bp.route('/european', methods=['GET', 'POST'])
def european_options():
    readme_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'european_options.md')
    with open(readme_path, 'r') as readme_file:
        content = readme_file.read()
    md_content = markdown.markdown(content)
    
    # Retrieve form data from URL parameters
    form_data = {
        'ticker': request.args.get('ticker', ''),
        'strike_price': request.args.get('strike_price', ''),
        'start_date': request.args.get('start_date', ''),
        'end_date': request.args.get('end_date', ''),
        'risk_free_rate': request.args.get('risk_free_rate', ''),
        'volatility': request.args.get('volatility', ''),
        'option_type': request.args.get('option_type', ''),
        'model_type': request.args.get('model_type', '')
    }
    
    # Retrieve results if present
    option_price = request.args.get('option_price')
    delta = request.args.get('delta')
    gamma = request.args.get('gamma')
    vega = request.args.get('vega')
    theta = request.args.get('theta')
    rho = request.args.get('rho')
    
    sensitivity_results = None
    gpt_assessment = None

    if request.method == 'POST':
        print("POST request received")
        action = request.form.get('analysis_type')
        print(f"Action: {action}")  # Add this line to see the action being processed
        
        form_data = {
            'ticker': request.form.get('ticker', ''),
            'strike_price': request.form.get('strike_price', type=float),
            'start_date': request.form.get('start_date', ''),
            'end_date': request.form.get('end_date', ''),
            'risk_free_rate': request.form.get('risk_free_rate', type=float),
            'volatility': request.form.get('volatility', type=float),
            'option_type': request.form.get('option_type', ''),
            'model_type': request.form.get('model_type', 'black_scholes'),
            'num_paths': request.form.get('num_paths', type=int, default=10000),
            'num_steps': request.form.get('num_steps', type=int, default=252)
        }
        
        # Save form data to session
        session['form_data'] = form_data


        ticker = form_data['ticker']
        strike_price = form_data['strike_price']
        start_date = form_data['start_date']
        end_date = form_data['end_date']
        risk_free_rate = form_data['risk_free_rate']
        volatility = form_data['volatility']
        option_type = form_data['option_type']
        
        # Print the raw start_date and end_date for debugging
        print(f"Raw start_date: {start_date}, Raw end_date: {end_date}")
        
        try:
            # Ensure the dates are in the correct format
            start_date = datetime.strptime(start_date, '%Y-%m-%d').date()
            end_date = datetime.strptime(end_date, '%Y-%m-%d').date()
        except ValueError as e:
            print('error')
            return render_template('european_options.html', form_data=form_data, option_price=option_price, delta=delta,
                                   gamma=gamma, vega=vega, theta=theta, rho=rho, error=f"Date format error: {e}")

        if action == 'sensitivity':
            try: 
                print("Sensitivity analysis triggered")
                # Sensitivity analysis logic
                form_data['num_steps'] = int(request.form['num_steps'])
                form_data['step_range'] = float(request.form['step_range'])
                form_data['variable'] = request.form['variable']
                form_data['target_variable'] = request.form['target_variable']
    
                num_steps = form_data['num_steps']
                step_range = form_data['step_range']
                variable = form_data['variable']
                target_variable = form_data['target_variable']
                
                print(f"Parameters: num_steps={num_steps}, step_range={step_range}, variable={variable}")  # Debugging line

    
                tester = SmoothnessTest(ticker, strike_price, start_date, end_date, risk_free_rate, volatility, option_type)
                
                print("Running sensitivity analysis...")
                values, delta, gamma, vega, theta, rho = tester.calculate_greeks_over_range(variable, num_steps, step_range, target_variable)
                
                print("Plotting Greeks...")
                tester.plot_greeks(values, delta, gamma, vega, theta, rho, variable)
                
                # Show plot for debugging
                #plt.show()

                # Save the plot to the static directory
                plot_filename = f'{target_variable}-{variable}_sensitivity_plot.png'
                plot_path = os.path.join('app', 'static', plot_filename)
                
                plt.savefig(plot_path)
                plt.close()
                
                # Print the plot path to ensure it's correct
                print(f"Plot saved to {plot_path}")
                
                # Store the results and plot path in the session
                session['sensitivity_results'] = {
                    'variable': variable,
                    'values': values.tolist(),
                    #'greek_values': greek_values,
                    'target_variable': target_variable,
                    'plot_filename': plot_filename  # Save the plot filename to session
                }
                
                
                sensitivity_results = True
                
            except Exception as e:
                print(f"An error occurred during sensitivity analysis: {e}")
                sensitivity_results = None

        elif action == 'ai_assessment':
            # AI Assessment logic
            sensitivity_results = request.form.get('sensitivity_results')
            if sensitivity_results:
                # Prepare the input for AI
                assessment_input = f"Please assess the sensitivity analysis based on the outputs: {sensitivity_results}."
                gpt_assessment = ask_gpt(assessment_input)
            else:
                gpt_assessment = "No sensitivity analysis results available for assessment."

        else:
            # Option pricing logic
            model_type = form_data.get('model_type', 'black_scholes')
            
            if model_type == 'black_scholes':
                from ..models.mdls_vanilla_options import BlackScholes
                option = BlackScholes(ticker, strike_price, start_date, end_date, risk_free_rate, volatility, option_type)
                
                if option_type == 'call':
                    option_price = option.call_price()
                else:
                    option_price = option.put_price()

                # Calculate Greeks for Black-Scholes
                delta = "{:.4f}".format(option.delta())
                gamma = "{:.4f}".format(option.gamma())
                vega = "{:.4f}".format(option.vega())
                theta = "{:.4f}".format(option.theta())
                rho = "{:.4f}".format(option.rho())
                
            elif model_type == 'monte_carlo':
                # Use unified Monte Carlo engine
                num_paths = form_data.get('num_paths', 10000)
                num_steps = form_data.get('num_steps', 252)
                
                # Create Monte Carlo engine
                mc_engine = monte_carlo_module.create_monte_carlo_engine(
                    S0=float(StockData(ticker, start_date, end_date).get_closing_price()),
                    r=risk_free_rate,
                    sigma=volatility,
                    T=StockData(ticker, start_date, end_date).get_years_difference(),
                    num_paths=num_paths,
                    num_steps=num_steps,
                    random_type="sobol"
                )
                
                # Price the option
                if option_type == 'call':
                    option_price = mc_engine.price_european_option(strike_price, 'call')
                else:
                    option_price = mc_engine.price_european_option(strike_price, 'put')
                
                # Calculate Greeks
                greeks = mc_engine.calculate_greeks_finite_difference(strike_price, option_type, 'european')
                delta = "{:.4f}".format(greeks['Delta'])
                gamma = "{:.4f}".format(greeks['Gamma'])
                vega = "{:.4f}".format(greeks['Vega'])
                theta = "{:.4f}".format(greeks['Theta'])
                rho = "{:.4f}".format(greeks['Rho'])
            else:
                # Use existing lattice models
                option = LatticeModel(ticker, strike_price, start_date, end_date, risk_free_rate, volatility)
                
                if model_type == 'Cox Ross Rubinstein Tree':
                    option_price = option.Cox_Ross_Rubinstein_Tree(option_type, num_steps)
                elif model_type == 'Jarrow Rudd Tree':
                    option_price = option.Jarrow_Rudd_Tree(option_type, num_steps)
                elif model_type == 'Trinomial Asset Pricing':
                    option_price = option.Trinomial_Asset_Pricing(option_type, num_steps)
                else:
                    option_price = option.Cox_Ross_Rubinstein_Tree(option_type, num_steps)  # Default
                
                # Calculate Greeks for lattice models
                if model_type == 'Cox Ross Rubinstein Tree':
                    greeks = option.CRRGreeks(option_type, num_steps)
                elif model_type == 'Jarrow Rudd Tree':
                    greeks = option.JRTGreeks(option_type, num_steps)
                elif model_type == 'Trinomial Asset Pricing':
                    greeks = option.TAPGreeks(option_type, num_steps)
                else:
                    greeks = option.CRRGreeks(option_type, num_steps)  # Default
                
                delta = "{:.4f}".format(greeks['Delta'])
                gamma = "{:.4f}".format(greeks['Gamma'])
                vega = "{:.4f}".format(greeks['Vega'])
                theta = "{:.4f}".format(greeks['Theta'])
                rho = "{:.4f}".format(greeks['Rho'])

            option_price = "${:,.4f}".format(option_price)
            
            # Save form data and results to session
            session['option_price'] = option_price
            session['delta'] = delta
            session['gamma'] = gamma
            session['vega'] = vega
            session['theta'] = theta
            session['rho'] = rho

        return render_template('european_options.html', form_data=form_data, option_price=option_price, delta=delta,
                               gamma=gamma, vega=vega, theta=theta, rho=rho, sensitivity_results=sensitivity_results, gpt_assessment=gpt_assessment, md_content=md_content)
    
    return render_template('european_options.html', form_data=form_data, option_price=option_price, delta=delta,
                           gamma=gamma, vega=vega, theta=theta, rho=rho, md_content=md_content)



@vanilla_options_bp.route('/model-performance', methods=['GET', 'POST'])
def model_performance():
    # Retrieve form data from session
    convergence_results = None
    form_data = session.get('form_data', {})
    sensitivity_results = session.get('sensitivity_results', None)
    scenario_table = session.get('scenario_table', None)
    sensitivity_results = None
    scenario_results = None
    gpt_assessment = None
    gpt_scenario_assessment = None
    baseline_price = stressed_price = None
    baseline_delta = baseline_gamma = baseline_vega = baseline_theta = baseline_rho = None
    stressed_delta = stressed_gamma = stressed_vega = stressed_theta = stressed_rho = None


    if request.method == 'POST':
        action = request.form.get('analysis_type')

        if action == 'sensitivity':
            try:
                 # Sensitivity analysis logic
                form_data['num_steps'] = int(request.form['num_steps'])
                form_data['step_range'] = float(request.form['step_range'])
                form_data['variable'] = request.form['variable']
                form_data['target_variable'] = request.form['target_variable']

                num_steps = form_data['num_steps']
                step_range = form_data['step_range']
                variable = form_data['variable']
                target_variable = form_data['target_variable']

                tester = SmoothnessTest(form_data['ticker'], form_data['strike_price'], form_data['start_date'],
                                        form_data['end_date'], form_data['risk_free_rate'], form_data['volatility'],
                                        form_data['option_type'])

                values, greek_values = tester.calculate_greeks_over_range(variable, num_steps, step_range, target_variable)
                tester.plot_single_greek(values, greek_values, target_variable, variable)

                # Save plot to static directory
                print('start plotting')
                plot_filename = f'european_{target_variable}-{variable}_sensitivity_plot.png'
                plot_path = os.path.join('app', 'static', plot_filename)

                # Print the plot path to ensure it's correct
                print(f"Saving plot to: {plot_path}")


                # Save plot as a base64 encoded string
                #img_buffer = BytesIO()
                #plt.savefig(img_buffer, format='png')  # Save plot to BytesIO buffer
                plt.savefig(plot_path)

                #img_buffer.seek(0)
                #plot_base64 = base64.b64encode(img_buffer.read()).decode('utf-8')

                # Store the results and plot path in the session
                session['sensitivity_results'] = {
                    'variable': variable,
                    'values': values.tolist(),
                    'greek_values': greek_values,
                    'target_variable': target_variable,
                    'plot_filename': plot_filename,
                    #'plot_base64': plot_base64  # Store the plot as base64
                }

                sensitivity_results = True

                plt.close() # Close the plot to release memory

            except Exception as e:
                print(f"An error occurred during sensitivity analysis: {e}")
                sensitivity_results = None
                
        elif action == 'ai_sensitivity_assessment':
            # Generate AI assessment for sensitivity analysis
            sensitivity_data = session.get('sensitivity_results')
            
            if sensitivity_data:
                variable = sensitivity_data.get('variable')
                values = sensitivity_data.get('values')
                target_variable = sensitivity_data.get('target_variable')
        
                if values and variable and target_variable:
                    print(values)
                        
                    sensitivity_results_text = f"Sensitivity Analysis Results for {target_variable} with respect to {variable}:\n" + \
                        "\n".join([f"{variable}={v}" for v in values])
        
                    assessment_input = f"Please assess the sensitivity analysis based on the outputs: {sensitivity_results_text}."
                    gpt_assessment = ask_gpt(assessment_input)
                    
                    # Save the AI assessment to session['sensitivity_results']
                    sensitivity_data['gpt_sensitivity_assessment'] = gpt_assessment
                    session['sensitivity_results'] = sensitivity_data
                else:
                    gpt_assessment = "Incomplete sensitivity analysis data available for assessment."
                    sensitivity_data['gpt_sensitivity_assessment'] = gpt_assessment
                    session['sensitivity_results'] = sensitivity_data
            else:
                gpt_assessment = "No sensitivity analysis results available for assessment."
                if sensitivity_data:
                    sensitivity_data['gpt_sensitivity_assessment'] = gpt_assessment
                    session['sensitivity_results'] = sensitivity_data
                else:
                    session['sensitivity_results'] = {'gpt_sensitivity_assessment': gpt_assessment}
            

        elif action == 'scenario':
            try:
                # Scenario analysis logic...
                spot_change = float(request.form.get('spot_scenario', 0))
                vol_change = float(request.form.get('vol_scenario', 0))
                rate_change = float(request.form.get('rate_scenario', 0))

                ticker = form_data.get('ticker')
                strike_price = float(form_data.get('strike_price'))
                start_date = datetime.strptime(form_data.get('start_date'), '%Y-%m-%d').date()
                end_date = datetime.strptime(form_data.get('end_date'), '%Y-%m-%d').date()
                risk_free_rate = float(form_data.get('risk_free_rate'))
                volatility = float(form_data.get('volatility'))
                option_type = form_data.get('option_type')
                model = form_data.get('model')

                # Baseline calculation
                if model == 'Monte_Carlo':
                    # from ..models.mdls_vanilla_options import AmericanMonteCarloOption
                    num_paths = form_data.get('num_paths', 10000)
                    mc_steps = form_data.get('mc_steps', 252)
                    # option = AmericanMonteCarloOption(ticker, strike_price, start_date, end_date, risk_free_rate, volatility, option_type, num_paths, mc_steps)
                    # baseline_price = option.call_price() if option_type == 'call' else option.put_price()
                    # baseline_greeks = option.get_greeks()
                    baseline_price = None
                    baseline_greeks = None
                else:
                    # Use existing lattice models
                    option = LatticeModel(ticker, strike_price, start_date, end_date, risk_free_rate, volatility)
                    if model == 'Cox Ross Rubinstein Tree':
                        baseline_price = option.Cox_Ross_Rubinstein_Tree(option_type, num_steps)
                        baseline_greeks = option.CRRGreeks(option_type, num_steps)
                    elif model == 'Jarrow Rudd Tree':
                        baseline_price = option.Jarrow_Rudd_Tree(option_type, num_steps)
                        baseline_greeks = option.JRTGreeks(option_type, num_steps)
                    else:
                        baseline_price = option.Trinomial_Asset_Pricing(option_type, num_steps)
                        baseline_greeks = option.TAPGreeks(option_type, num_steps)
                
                baseline_price = "{:.4f}".format(baseline_price)
                baseline_delta = "{:.4f}".format(baseline_greeks['delta'])
                baseline_gamma = "{:.4f}".format(baseline_greeks['gamma'])
                baseline_vega = "{:.4f}".format(baseline_greeks['vega'])
                baseline_theta = "{:.4f}".format(baseline_greeks['theta'])
                baseline_rho = "{:.4f}".format(baseline_greeks['rho'])

                # Stressed scenario calculation
                stressed_spot = strike_price * (1 + spot_change)
                stressed_vol = volatility + vol_change
                stressed_rate = risk_free_rate + rate_change

                if model == 'Monte_Carlo':
                    # stressed_option = AmericanMonteCarloOption(ticker, stressed_spot, start_date, end_date, stressed_rate, stressed_vol, option_type, num_paths, mc_steps)
                    # stressed_price = stressed_option.call_price() if option_type == 'call' else stressed_option.put_price()
                    # stressed_greeks = stressed_option.get_greeks()
                    stressed_price = None
                    stressed_greeks = None
                else:
                    stressed_option = LatticeModel(ticker, stressed_spot, start_date, end_date, stressed_rate, stressed_vol)
                    if model == 'Cox Ross Rubinstein Tree':
                        stressed_price = stressed_option.Cox_Ross_Rubinstein_Tree(option_type, num_steps, greeks=False)
                        stressed_greeks = stressed_option.CRRGreeks(option_type, num_steps)
                    elif model == 'Jarrow Rudd Tree':
                        stressed_price = stressed_option.Jarrow_Rudd_Tree(option_type, num_steps, greeks=True)
                        stressed_greeks = stressed_option.JRTGreeks(option_type, num_steps)
                    else:
                        stressed_price = stressed_option.Trinomial_Asset_Pricing(option_type, num_steps)
                        stressed_greeks = stressed_option.TAPGreeks(option_type, num_steps)
                
                stressed_price = "{:.4f}".format(stressed_price)
                stressed_delta = "{:.4f}".format(stressed_greeks['delta'])
                stressed_gamma = "{:.4f}".format(stressed_greeks['gamma'])
                stressed_vega = "{:.4f}".format(stressed_greeks['vega'])
                stressed_theta = "{:.4f}".format(stressed_greeks['theta'])
                stressed_rho = "{:.4f}".format(stressed_greeks['rho'])

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
    
        elif action == 'convergence':
            try:
                # Get convergence form data
                model_type = request.form.get('model_type', 'monte_carlo')  # Should be 'monte_carlo'
                num_mc_paths = int(request.form.get('num_mc_paths', 10000))
                num_mc_steps = int(request.form.get('num_mc_steps', 252))
                obs = int(request.form.get('obs', 10))

                # Retrieve option params (from form_data or session)
                ticker = form_data.get('ticker', '')
                strike_price = float(form_data.get('strike_price', 100))
                start_date = form_data.get('start_date', '')
                end_date = form_data.get('end_date', '')
                r = float(form_data.get('risk_free_rate', 0.01))
                sigma = float(form_data.get('volatility', 0.2))
                option_type = form_data.get('option_type', 'call')

                # Get underlying info
                S0 = float(StockData(ticker, start_date, end_date).get_closing_price())
                T = StockData(ticker, start_date, end_date).get_years_difference()

                # Run MC convergence over number of paths
                mc_results = []
                paths_range = np.linspace(800, num_mc_paths, obs).round().astype(int)
                for n_paths in paths_range:
                    mc_engine = monte_carlo_module.create_monte_carlo_engine(
                        S0=S0, r=r, sigma=sigma, T=T,
                        num_paths=int(n_paths), num_steps=num_mc_steps,
                        random_type="sobol"
                    )
                    if option_type == "call":
                        price = mc_engine.price_european_option(strike_price, "call")
                    else:
                        price = mc_engine.price_european_option(strike_price, "put")
                    mc_results.append((int(n_paths), float(price)))

                # Plot and save
                plot_convergence(mc_results, mode="simulations")
                plt.savefig('app/static/vanilla_convergence_plot.png')
                plt.close()
                print("Plot file exists after save?", os.path.exists('app/static/vanilla_convergence_plot.png'))

                session['convergence_results'] = {
                    'results': mc_results,
                    'mode': "simulations",
                    'plot_filename': 'vanilla_convergence_plot.png'
                }
                convergence_results = True

            except Exception as e:
                print(f"An error occurred during convergence analysis: {e}")
                convergence_results = None

    return render_template(
        'model_performance.html', 
        form_data=form_data, 
        sensitivity_results=sensitivity_results, 
        scenario_results=scenario_results,
        baseline_price=baseline_price, 
        baseline_delta=baseline_delta, 
        baseline_gamma=baseline_gamma, 
        baseline_vega=baseline_vega, 
        baseline_theta=baseline_theta, 
        baseline_rho=baseline_rho,
        stressed_price=stressed_price, 
        stressed_delta=stressed_delta, 
        stressed_gamma=stressed_gamma, 
        stressed_vega=stressed_vega, 
        stressed_theta=stressed_theta, 
        stressed_rho=stressed_rho,
        gpt_assessment=gpt_assessment,
        convergence_results = convergence_results,
        gpt_scenario_assessment=gpt_scenario_assessment,
        random=random
    )

@vanilla_options_bp.route('/go-back', methods=['GET'])
def go_back():
    # Retrieve form data and results from session
    form_data = session.get('form_data', {})
    option_price = session.get('option_price')
    delta = session.get('delta')
    gamma = session.get('gamma')
    vega = session.get('vega')
    theta = session.get('theta')
    rho = session.get('rho')

    # Construct the URL with query parameters
    params = {
        'ticker': form_data.get('ticker', ''),
        'strike_price': form_data.get('strike_price', ''),
        'start_date': form_data.get('start_date', ''),
        'end_date': form_data.get('end_date', ''),
        'risk_free_rate': form_data.get('risk_free_rate', ''),
        'volatility': form_data.get('volatility', ''),
        'option_type': form_data.get('option_type', ''),
        'model_type': form_data.get('model_type', ''),
        'option_price': option_price,
        'delta': delta,
        'gamma': gamma,
        'vega': vega,
        'theta': theta,
        'rho': rho
    }

    return redirect(url_for('vanilla_options.european_options', **params))


@vanilla_options_bp.route('/model-governance', methods=['GET', 'POST'])
def model_governance():
    md_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'model_governance.md')
    with open(md_path, 'r', encoding='utf-8') as file:
        md_content = file.read()
        html_content = markdown.markdown(md_content)

    gpt_assessment = None

    if request.method == 'POST':
        action = request.form.get('analysis_type')

        if action == 'ai_assessment':
            # Prepare the prompt for AI assessment
            prompt = f"Please assess the reasonableness and comprehensiveness of the model governance for the vanilla option model as shown in the following text:\n\n{md_content}"
            # Generate the assessment
            gpt_assessment = ask_gpt(prompt)  # Assuming `ask_gpt` is a function that generates ChatGPT responses

    return render_template('model_governance.html', md_content=html_content, gpt_assessment=gpt_assessment)


@vanilla_options_bp.route('/ongoing-monitoring', methods=['GET'])
def ongoing_monitoring():
    # Path to the markdown file
    md_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'ongoing_monitoring.md')
    
    # Read and render the markdown content
    with open(md_file_path, 'r', encoding='utf-8') as md_file:
        md_content = markdown.markdown(md_file.read())
    
    return render_template('ongoing_monitoring.html', md_content=md_content)


  

@vanilla_options_bp.route('/american', methods=['GET', 'POST'])
def american_options():
    readme_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'american_options.md')
    with open(readme_path, 'r') as readme_file:
        content = readme_file.read()
    md_content = markdown.markdown(content)
    
    option_price = convergence_results = sensitivity_results = scenario_results = risk_pl_results = None
    form_data = {}
    
    if request.method == 'POST':
        action = request.form.get('analysis_type')
        
        def safe_int(val, default):
            try:
                val = str(val)
                if val.strip() == '':
                    return default
                return int(val)
            except (ValueError, TypeError):
                return default
        
        form_data = {
            'ticker': request.form['ticker'],
            'strike_price': float(request.form['strike_price']),
            'start_date': str(request.form['start_date']),
            'end_date': str(request.form['end_date']),
            'r': float(request.form['r']),
            'sigma': float(request.form['sigma']),
            'option_type': request.form['option_type'],
            'num_steps': safe_int(request.form.get('num_steps', request.form.get('mc_steps', 252)), 252),
            'pricing_model': request.form.get('pricing_model', 'Cox Ross Rubinstein Tree'),  # NEW NAME for pricing
            'model': request.form.get('model'),  # for convergence analysis
            'num_paths': safe_int(request.form.get('num_paths'), 10000),
            'mc_steps': safe_int(request.form.get('mc_steps'), 252)
        }
                
        ticker = form_data['ticker']
        strike_price = form_data['strike_price']
        start_date = form_data['start_date']
        end_date = form_data['end_date']
        risk_free_rate = form_data['r']
        volatility = form_data['sigma']
        option_type = form_data['option_type']
        num_steps = form_data['num_steps']
        model = form_data['model']
        pricing_model = form_data['pricing_model']
        
        # Handle different model types
        if pricing_model == 'Monte Carlo':
            # Use unified Monte Carlo engine for American options
            num_paths = form_data.get('num_paths', 10000)
            mc_steps = form_data.get('mc_steps', 252)
            
            # Create Monte Carlo engine
            mc_engine = monte_carlo_module.create_monte_carlo_engine(
                S0=float(StockData(ticker, start_date, end_date).get_closing_price()),
                r=risk_free_rate,
                sigma=volatility,
                T=StockData(ticker, start_date, end_date).get_years_difference(),
                num_paths=num_paths,
                num_steps=mc_steps,
                random_type="sobol"
            )
            
            # Price the option
            if option_type == 'call':
                option_price = mc_engine.price_american_option(strike_price, 'call')
            else:
                option_price = mc_engine.price_american_option(strike_price, 'put')
            
            # Calculate Greeks
            greeks = mc_engine.calculate_greeks_finite_difference(strike_price, option_type, 'american')
            delta = "{:.4f}".format(greeks['Delta'])
            gamma = "{:.4f}".format(greeks['Gamma'])
            vega = "{:.4f}".format(greeks['Vega'])
            theta = "{:.4f}".format(greeks['Theta'])
            rho = "{:.4f}".format(greeks['Rho'])
            
        else:
            # Use existing lattice models
            option = LatticeModel(ticker, strike_price, start_date, end_date, risk_free_rate, volatility)
            
            if pricing_model == 'Cox Ross Rubinstein Tree':
                option_price = option.Cox_Ross_Rubinstein_Tree(option_type, num_steps)
            elif pricing_model == 'Jarrow Rudd Tree':
                option_price = option.Jarrow_Rudd_Tree(option_type, num_steps)
            elif pricing_model == 'Trinomial Asset Pricing':
                option_price = option.Trinomial_Asset_Pricing(option_type, num_steps)
            else:
                option_price = option.Cox_Ross_Rubinstein_Tree(option_type, num_steps)  # Default
            
            # Calculate Greeks for lattice models
            if pricing_model == 'Cox Ross Rubinstein Tree':
                greeks = option.CRRGreeks(option_type, num_steps)
            elif pricing_model == 'Jarrow Rudd Tree':
                greeks = option.JRTGreeks(option_type, num_steps)
            elif pricing_model == 'Trinomial Asset Pricing':
                greeks = option.TAPGreeks(option_type, num_steps)
            else:
                greeks = option.CRRGreeks(option_type, num_steps)  # Default
            
            delta = "{:.4f}".format(greeks['Delta'])
            gamma = "{:.4f}".format(greeks['Gamma'])
            vega = "{:.4f}".format(greeks['Vega'])
            theta = "{:.4f}".format(greeks['Theta'])
            rho = "{:.4f}".format(greeks['Rho'])

        option_price = "${:,.4f}".format(option_price)

        if action == 'sensitivity':
            try:
                # Sensitivity analysis logic
                form_data['num_sensitivity_steps'] = int(request.form['num_sensitivity_steps'])
                form_data['step_range'] = float(request.form['step_range'])
                form_data['variable'] = request.form['variable']
                form_data['target_variable'] = request.form['target_variable']

                num_steps = form_data['num_sensitivity_steps']
                step_range = form_data['step_range']
                variable = form_data['variable']
                target_variable = form_data['target_variable']

                if form_data['model'] == 'Cox Ross Rubinstein Tree':
                    smoothness_model = 'CRR'
                elif form_data['model'] == 'Jarrow Rudd Tree':
                    smoothness_model = 'JRT'
                elif form_data['model'] == 'Trinomial Asset Pricing':
                    smoothness_model = 'TAP'
                elif form_data['model'] == 'Monte Carlo':
                    # Handle Monte Carlo sensitivity analysis
                    num_paths = form_data.get('num_paths', 10000)
                    mc_steps = form_data.get('mc_steps', 252)
                    
                    # Generate variable range
                    if variable == 'strike_price':
                        base_value = form_data['strike_price']
                    elif variable == 'risk_free_rate':
                        base_value = form_data['r']
                    elif variable == 'volatility':
                        base_value = form_data['sigma']
                    else:
                        base_value = 0
                    
                    # Create variable range
                    variable_range = np.linspace(base_value * (1 - step_range), base_value * (1 + step_range), num_steps)
                    greek_values = []
                    
                    for val in variable_range:
                        # Create Monte Carlo engine for this value
                        if variable == 'strike_price':
                            mc_engine = monte_carlo_module.create_monte_carlo_engine(
                                S0=float(StockData(ticker, start_date, end_date).get_closing_price()),
                                r=risk_free_rate,
                                sigma=volatility,
                                T=StockData(ticker, start_date, end_date).get_years_difference(),
                                num_paths=num_paths,
                                num_steps=mc_steps,
                                random_type="sobol"
                            )
                            if option_type == 'call':
                                price = mc_engine.price_american_option(val, 'call')
                            else:
                                price = mc_engine.price_american_option(val, 'put')
                        elif variable == 'risk_free_rate':
                            mc_engine = monte_carlo_module.create_monte_carlo_engine(
                                S0=float(StockData(ticker, start_date, end_date).get_closing_price()),
                                r=val,
                                sigma=volatility,
                                T=StockData(ticker, start_date, end_date).get_years_difference(),
                                num_paths=num_paths,
                                num_steps=mc_steps,
                                random_type="sobol"
                            )
                            if option_type == 'call':
                                price = mc_engine.price_american_option(strike_price, 'call')
                            else:
                                price = mc_engine.price_american_option(strike_price, 'put')
                        elif variable == 'volatility':
                            mc_engine = monte_carlo_module.create_monte_carlo_engine(
                                S0=float(StockData(ticker, start_date, end_date).get_closing_price()),
                                r=risk_free_rate,
                                sigma=val,
                                T=StockData(ticker, start_date, end_date).get_years_difference(),
                                num_paths=num_paths,
                                num_steps=mc_steps,
                                random_type="sobol"
                            )
                            if option_type == 'call':
                                price = mc_engine.price_american_option(strike_price, 'call')
                            else:
                                price = mc_engine.price_american_option(strike_price, 'put')
                        
                        greek_values.append(price)
                    
                    # Create plot
                    plt.figure(figsize=(10, 6))
                    plt.plot(variable_range, greek_values, 'b-', linewidth=2)
                    plt.xlabel(variable.replace('_', ' ').title())
                    plt.ylabel(target_variable.replace('_', ' ').title())
                    plt.title(f'{target_variable.replace("_", " ").title()} vs {variable.replace("_", " ").title()} (Monte Carlo)')
                    plt.grid(True)
                    plt.tight_layout()
                    
                    # Save plot
                    plot_filename = f'american_{target_variable}-{variable}_sensitivity_plot.png'
                    plot_path = os.path.join('app', 'static', plot_filename)
                    plt.savefig(plot_path)
                    plt.close()
                    
                    sensitivity_results = {
                        'plot_path': plot_filename,
                        'variable_range': variable_range.tolist(),
                        'greek_values': greek_values
                    }
                else:
                    # Use existing lattice models for sensitivity analysis
                    tester = AmericanOptionSmoothnessTest(form_data['ticker'], form_data['strike_price'], form_data['start_date'],
                                                          form_data['end_date'], form_data['r'], form_data['sigma'],
                                                          smoothness_model, form_data['option_type'],
                                                          form_data['num_sensitivity_steps'])

                    values, greek_values = tester.calculate_greeks_over_range(variable, num_steps, step_range,
                                                                              target_variable)
                    tester.plot_single_greek(values, greek_values, target_variable, variable)

                    # Save plot to static directory
                    print('start plotting')
                    plot_filename = f'american_{target_variable}-{variable}_sensitivity_plot.png'
                    plot_path = os.path.join('app', 'static', plot_filename)

                    plt.savefig(plot_path)

                    sensitivity_results = {
                        'plot_path': plot_filename,
                        'values': values.tolist(),
                        'greek_values': greek_values
                    }

            except Exception as e:
                print(f"An error occurred during sensitivity analysis: {e}")
                sensitivity_results = None

        elif action == 'ai_sensitivity_assessment':
            sensitivity_data = session.get('sensitivity_results', {})
            
            if sensitivity_data:
                variable = sensitivity_data.get('variable')
                values = sensitivity_data.get('values')
                target_variable = sensitivity_data.get('target_variable')
                greek_values = sensitivity_data.get('greek_values')
                
                if values and variable and target_variable and greek_values:
                    # Format the sensitivity results for the assessment
                    sensitivity_text = f"""
                    Sensitivity Analysis Results for {target_variable} with respect to {variable}:
                    Input Values: {values}
                    Output Values: {greek_values}
                    """
                    assessment_input = f"Please assess the sensitivity analysis results based on the following data: {sensitivity_text}. Focus on the relationship between {variable} and {target_variable}, and any notable patterns or risks. Please limit the assessment to be less than 100 words."
                    gpt_sensitivity_assessment = ask_gpt(assessment_input)
                    
                    # Save the assessment to session
                    sensitivity_data['gpt_sensitivity_assessment'] = gpt_sensitivity_assessment
                    session['sensitivity_results'] = sensitivity_data
                else:
                    sensitivity_data['gpt_sensitivity_assessment'] = "Incomplete sensitivity analysis data available for assessment."
                    session['sensitivity_results'] = sensitivity_data
            else:
                session['sensitivity_results'] = {'gpt_sensitivity_assessment': "No sensitivity analysis results available for assessment."}

        elif action == 'risk_pl':
            try:
                # RBPL analysis logic
                form_data['price_change'] = float(request.form['price_change'])
                form_data['vol_change'] = float(request.form['vol_change'])

                price_change = form_data['price_change']
                vol_change = form_data['vol_change']
                
                if form_data['model'] == 'Cox Ross Rubinstein Tree':
                    rbpl_model = 'CRR'
                elif form_data['model'] == 'Jarrow Rudd Tree':
                    rbpl_model = 'JRT'
                else:
                    rbpl_model = 'TAP'
                    
                risk_pl_results = option.risk_pl_analysis(option_type=form_data['option_type'],
                                                           steps=form_data['num_steps'],
                                                           price_change=price_change,
                                                           vol_change=vol_change,
                                                           model=rbpl_model)

                # Save only the RPBL results to session, without any assessment
                session['risk_pl_results'] = {
                    'results': risk_pl_results
                }

                print(risk_pl_results)

            except Exception as e:
                print(f"An error occurred during Risk-Based P&L analysis: {e}")
                risk_pl_results = None

        elif action == 'ai_rpbl_assessment':
            risk_pl_data = session.get('risk_pl_results', {})
            rpbl_results = risk_pl_data.get('results', {})

            if rpbl_results:
                # Format the RPBL results for the assessment
                rpbl_text = f"""
                    Risk-Based P&L Analysis Results:
                    Price Change Impact: {rpbl_results.get('price_change_impact', 'N/A')}
                    Volatility Change Impact: {rpbl_results.get('vol_change_impact', 'N/A')}
                    Total P&L Impact: {rpbl_results.get('total_pl_impact', 'N/A')}
                    Delta Contribution: {rpbl_results.get('delta_contribution', 'N/A')}
                    Gamma Contribution: {rpbl_results.get('gamma_contribution', 'N/A')}
                    Vega Contribution: {rpbl_results.get('vega_contribution', 'N/A')}
                    Theta Contribution: {rpbl_results.get('theta_contribution', 'N/A')}
                    Rho Contribution: {rpbl_results.get('rho_contribution', 'N/A')}
                    """
                assessment_input = f"Please assess the Risk-Based P&L analysis results based on the following data: {rpbl_text}. Focus on the key drivers of P&L and potential risks. Please limit the assessment to be less than 100 words."
                gpt_rpbl_assessment = ask_gpt(assessment_input)

                # Save the GPT assessment to the session
                risk_pl_data['gpt_rpbl_assessment'] = gpt_rpbl_assessment
                session['risk_pl_results'] = risk_pl_data
            else:
                risk_pl_data['gpt_rpbl_assessment'] = "No Risk-Based P&L analysis results available for assessment."
                session['risk_pl_results'] = risk_pl_data
            
        elif action == 'convergence':
            try:
                required_fields = ['ticker', 'strike_price', 'start_date', 'end_date', 'r', 'sigma', 'option_type']
                for field in required_fields:
                    if not request.form.get(field):
                        raise ValueError(f"Missing required field: {field}")

                form_data['mode'] = request.form['mode']
                form_data['model'] = request.form['model']
                form_data['option_type'] = str(request.form['option_type'])
                form_data['obs'] = safe_int(request.form.get('obs'), 10)
                mode = form_data['mode']
                model = form_data['model']
                obs = form_data['obs']
                option_type = form_data['option_type']


                # Defensive int conversion for MC-only fields
                def safe_int(val, default):
                    try:
                        return int(val)
                    except (ValueError, TypeError):
                        return default


                if model == "Monte Carlo" and mode == "simulations":
                    # --- Monte Carlo convergence analysis over number of paths ---                    
                    num_paths = safe_int(request.form.get('num_paths'), 10000)
                    mc_steps = safe_int(request.form.get('mc_steps'), 252)

                    ticker = form_data['ticker']
                    start_date = form_data['start_date']
                    end_date = form_data['end_date']
                    S0 = float(StockData(ticker, start_date, end_date).get_closing_price())
                    T = StockData(ticker, start_date, end_date).get_years_difference()
                    r = form_data['r']
                    sigma = form_data['sigma']
                    strike_price = form_data['strike_price']
                    # Sweep num_paths values
                    mc_results = []
                    paths_range = np.linspace(800, num_paths, obs).round().astype(int)
                    for n_paths in paths_range:
                        mc_engine = monte_carlo_module.create_monte_carlo_engine(
                            S0=S0, r=r, sigma=sigma, T=T, num_paths=int(n_paths), num_steps=mc_steps, random_type="sobol"
                        )
                        if option_type == "call":
                            price = mc_engine.price_american_option(strike_price, "call")
                        else:
                            price = mc_engine.price_american_option(strike_price, "put")
                        mc_results.append((int(n_paths), float(price)))
                    plot_convergence(mc_results, mode="simulations")
                    plt.savefig('app/static/monte_carlo_convergence_plot.png')  # <-- CHANGE HERE
                    session['convergence_results'] = {
                        'results': mc_results,
                        'mode': "simulations",
                        'plot_filename': 'monte_carlo_convergence_plot.png'  # <-- CHANGE HERE
                    }
                    convergence_results = True
                    plt.close()

                else:
                    # --- Lattice convergence analysis (existing code) ---
                    form_data['max_steps'] = safe_int(request.form.get('max_steps'), 100)
                    max_steps = form_data['max_steps']
                    max_sims = 0
                    convergence_params = form_data.copy()
                    if 'pricing_model' in convergence_params:
                        del convergence_params['pricing_model']
                    american_step_results = lattice_convergence_test(max_steps, max_sims, obs, LatticeModel, convergence_params, model, option_type)

                    plot_convergence(american_step_results, mode)
                    plt.savefig('app/static/lattice_convergence_plot.png')   # <-- CHANGE HERE
                    session['convergence_results'] = {
                        'results': american_step_results,
                        'mode': mode,
                        'plot_filename': 'lattice_convergence_plot.png'      # <-- CHANGE HERE
                    }
                    convergence_results = True
                    plt.close()

            except Exception as e:
                print(f"An error occurred during convergence analysis: {e}")
                convergence_results = None

        elif action == 'ai_convergence_assessment':
            convergence_data = session.get('convergence_results', {})
            
            if convergence_data:
                results = convergence_data.get('results')
                mode = convergence_data.get('mode')
                
                if results:
                    # Format the convergence results for the assessment
                    convergence_text = f"""
                    Convergence Analysis Results:
                    Mode: {mode}
                    Results: {results}
                    """
                    assessment_input = f"Please assess the convergence analysis results based on the following data: {convergence_text}. Focus on the convergence behavior and any potential issues or recommendations. Please limit the assessment to be less than 100 words."
                    gpt_convergence_assessment = ask_gpt(assessment_input)
                    
                    # Save the assessment to session
                    convergence_data['gpt_convergence_assessment'] = gpt_convergence_assessment
                    session['convergence_results'] = convergence_data
                else:
                    convergence_data['gpt_convergence_assessment'] = "Incomplete convergence analysis data available for assessment."
                    session['convergence_results'] = convergence_data
            else:
                session['convergence_results'] = {'gpt_convergence_assessment': "No convergence analysis results available for assessment."}

        elif action == 'scenario':
            try:
                # Scenario analysis logic...
                spot_change = float(request.form.get('spot_scenario', 0))
                vol_change = float(request.form.get('vol_scenario', 0))
                rate_change = float(request.form.get('rate_scenario', 0))
                ticker = form_data.get('ticker')
                strike_price = float(form_data.get('strike_price'))
                start_date = datetime.strptime(form_data.get('start_date'), '%Y-%m-%d').date()
                end_date = datetime.strptime(form_data.get('end_date'), '%Y-%m-%d').date()
                risk_free_rate = float(form_data.get('r'))
                volatility = float(form_data.get('sigma'))
                option_type = form_data.get('option_type')
                model = form_data.get('model')

                # Baseline calculation
                if model == 'Monte Carlo':
                    # from ..models.mdls_vanilla_options import AmericanMonteCarloOption
                    num_paths = form_data.get('num_paths', 10000)
                    mc_steps = form_data.get('mc_steps', 252)
                    # option = AmericanMonteCarloOption(ticker, strike_price, start_date, end_date, risk_free_rate, volatility, option_type, num_paths, mc_steps)
                    # baseline_price = option.call_price() if option_type == 'call' else option.put_price()
                    # baseline_greeks = option.get_greeks()
                    baseline_price = None
                    baseline_greeks = None
                else:
                    # Use existing lattice models
                    option = LatticeModel(ticker, strike_price, start_date, end_date, risk_free_rate, volatility)
                    if model == 'Cox Ross Rubinstein Tree':
                        baseline_price = option.Cox_Ross_Rubinstein_Tree(option_type, num_steps)
                        baseline_greeks = option.CRRGreeks(option_type, num_steps)
                    elif model == 'Jarrow Rudd Tree':
                        baseline_price = option.Jarrow_Rudd_Tree(option_type, num_steps)
                        baseline_greeks = option.JRTGreeks(option_type, num_steps)
                    else:
                        baseline_price = option.Trinomial_Asset_Pricing(option_type, num_steps)
                        baseline_greeks = option.TAPGreeks(option_type, num_steps)
                
                baseline_price = "{:.4f}".format(baseline_price)
                baseline_delta = "{:.4f}".format(baseline_greeks['delta'])
                baseline_gamma = "{:.4f}".format(baseline_greeks['gamma'])
                baseline_vega = "{:.4f}".format(baseline_greeks['vega'])
                baseline_theta = "{:.4f}".format(baseline_greeks['theta'])
                baseline_rho = "{:.4f}".format(baseline_greeks['rho'])

                # Stressed scenario calculation
                stressed_spot = strike_price * (1 + spot_change)
                stressed_vol = volatility + vol_change
                stressed_rate = risk_free_rate + rate_change

                if model == 'Monte Carlo':
                    # stressed_option = AmericanMonteCarloOption(ticker, stressed_spot, start_date, end_date, stressed_rate, stressed_vol, option_type, num_paths, mc_steps)
                    # stressed_price = stressed_option.call_price() if option_type == 'call' else stressed_option.put_price()
                    # stressed_greeks = stressed_option.get_greeks()
                    stressed_price = None
                    stressed_greeks = None
                else:
                    stressed_option = LatticeModel(ticker, stressed_spot, start_date, end_date, stressed_rate, stressed_vol)
                    if model == 'Cox Ross Rubinstein Tree':
                        stressed_price = stressed_option.Cox_Ross_Rubinstein_Tree(option_type, num_steps, greeks=False)
                        stressed_greeks = stressed_option.CRRGreeks(option_type, num_steps)
                    elif model == 'Jarrow Rudd Tree':
                        stressed_price = stressed_option.Jarrow_Rudd_Tree(option_type, num_steps, greeks=True)
                        stressed_greeks = stressed_option.JRTGreeks(option_type, num_steps)
                    else:
                        stressed_price = stressed_option.Trinomial_Asset_Pricing(option_type, num_steps)
                        stressed_greeks = stressed_option.TAPGreeks(option_type, num_steps)
                
                stressed_price = "{:.4f}".format(stressed_price)
                stressed_delta = "{:.4f}".format(stressed_greeks['delta'])
                stressed_gamma = "{:.4f}".format(stressed_greeks['gamma'])
                stressed_vega = "{:.4f}".format(stressed_greeks['vega'])
                stressed_theta = "{:.4f}".format(stressed_greeks['theta'])
                stressed_rho = "{:.4f}".format(stressed_greeks['rho'])

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

    form_data.setdefault('num_paths', 10000)
    form_data.setdefault('mc_steps', 252)

    return render_template('american_options.html', option_price=option_price, form_data=form_data,
                           sensitivity_results=sensitivity_results, risk_pl_results=risk_pl_results,
                           convergence_results=convergence_results,
                           scenario_results=scenario_results, md_content=md_content,
                           gpt_rpbl_assessment=session.get('risk_pl_results', {}).get('gpt_rpbl_assessment'),
                           gpt_sensitivity_assessment=session.get('sensitivity_results', {}).get('gpt_sensitivity_assessment'),
                           gpt_convergence_assessment=session.get('convergence_results', {}).get('gpt_convergence_assessment'))


@vanilla_options_bp.route('/reporting', methods=['GET'])
def reporting():
    # Retrieve the stored results from session
    sensitivity_results = session.get('sensitivity_results', {})
    scenario_results = session.get('scenario_results', {})

    # Check if sensitivity_results exist before processing
    sensitivity_combined = []
    sensitivity_plot = None
    sensitivity_assessment = 'No assessment available.'

    if sensitivity_results:
        sensitivity_combined = list(zip(sensitivity_results.get('values', []), sensitivity_results.get('greek_values', [])))
        sensitivity_plot = sensitivity_results.get('plot_path', None)
        sensitivity_assessment = sensitivity_results.get('gpt_sensitivity_assessment', 'No assessment available.')  # Fetch the assessment from sensitivity_results

        if sensitivity_plot:
            print("Sensitivity plot is present.")
        else:
            print("No sensitivity plot found.")
    
    # Get the scenario table and assessment
    scenario_table = scenario_results.get('table', [])
    scenario_assessment = scenario_results.get('gpt_scenario_assessment', 'No assessment available.')


    # Prepare the sensitivity description
    target_variable = sensitivity_results.get('target_variable', 'target variable')
    variable = sensitivity_results.get('variable', 'input variable')
    sensitivity_description = f"Sensitivity analysis was conducted on {target_variable} with respect to changes in the {variable}."

    # Prepare other report fields
    report_data = {
        'model_name': "Vanilla Option Model",
        'validation_type': "Baseline Validation",
        'validation_date': "2024-09-01",
        'validation_overview': "This validation provides an overview of the model performance and governance...",
        'model_purpose': "The model is designed to price European Vanilla Options...",
        'model_overview': "This section provides an overview of the model's pricing mechanisms...",
        'key_limitations': "Key limitations include volatility assumptions and model calibration...",
        'validation_scope': "The validation scope includes conceptual soundness, model performance...",
        'data_quality': "The data quality was assessed based on completeness, accuracy, and reliability...",
        'conceptual_soundness': "The model is conceptually sound with a well-structured pricing mechanism...",
        'sensitivity_description': sensitivity_description,  # Updated to include dynamic description
        'scenario_description': "Scenario analysis was conducted with stress on spot price, volatility, and interest rates...",
        'benchmarking_description': "Benchmarking analysis compares the model outputs with standard benchmarks...",
        'rpbl_analysis': "RPBL analysis includes...",
        'model_governance': "Governance details for the model...",
        'appendix_content': "Appendix content...",
        'sensitivity_combined': sensitivity_combined,  # Pass the combined sensitivity data as tuples
        'sensitivity_plot': sensitivity_plot,  # Pass the plot
        'scenario_results': scenario_table  # Pass the scenario table
    }

    # AI Assessments
    report_data['sensitivity_assessment'] = sensitivity_assessment
    report_data['scenario_assessment'] = scenario_assessment  # Use the assessment from scenario_results

    # Render the report using the report template
    rendered_report = render_template('report_template.html', **report_data)

    return rendered_report

@vanilla_options_bp.route('/download-report', methods=['GET'])
def download_report():
    # Generate the report as PDF
    base_dir = os.path.dirname(os.path.abspath(__file__))
    pdf_output_path = os.path.join(base_dir, '..', 'static', 'model_validation_report.pdf')

    # Create the PDF using ReportLab
    c = canvas.Canvas(pdf_output_path, pagesize=letter)
    width, height = letter

    # Add title, meta information, and dynamic data like results and assessments
    c.setFont("Helvetica-Bold", 24)
    c.drawCentredString(width / 2.0, height - 100, "Model Validation Report")

    c.setFont("Helvetica", 14)
    c.drawCentredString(width / 2.0, height - 130, "Vanilla Option Model")
    c.drawCentredString(width / 2.0, height - 150, "Baseline Validation")
    c.drawCentredString(width / 2.0, height - 170, "Date: 2024-09-01")

    # Add sections and results to the PDF (e.g., Sensitivity Analysis, Scenario Analysis)
    # Add dynamic content like results from sensitivity and scenario assessments
    # ... you can continue populating the PDF here.

    c.showPage()
    c.save()

    # Serve the PDF for download
    return send_file(pdf_output_path, as_attachment=True, download_name='Model_Validation_Report.pdf')

@vanilla_options_bp.route('/conceptual-soundness',  methods=['GET'])
def conceptual_soundness():
    try:
        # Use absolute path or relative path from project root
        file_path = os.path.join(os.path.dirname(__file__), 'black_scholes.md')
        with open(file_path, 'r') as file:
            content = file.read()
            print(f"Content read: {content[:100]}")  # Debug print
    except FileNotFoundError:
        content = "File not found. Please check if black_scholes.txt exists in the routes folder."
        print("File not found error")  # Debug print
    except Exception as e:
        content = f"Error reading file: {str(e)}"
        print(f"Error: {str(e)}")  # Debug print
    
    return render_template('conceptual_soundness.html', content=content)