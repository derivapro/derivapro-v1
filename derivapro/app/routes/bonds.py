from derivapro.app.models.mdls_bonds import NCFixedBonds, NCFloatingBonds
from flask import Blueprint, render_template, request, json
import QuantLib as ql
import os
import markdown
from openai import AzureOpenAI
import logging
# Initialize Flask app
nc_bonds_bp = Blueprint('nc_bonds', __name__)


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
        error_msg = str(e)
        logging.error(f"Error occurred while calling OpenAI API: {error_msg}")
        
        if "403" in error_msg:
            logging.error(f"Authentication failed. API Key: {api_key[:4]}...{api_key[-4:]}, Base URL: {base_url}")
            return "Error: Access to the AI service is currently restricted. Please verify the API configuration or contact support."
        elif "401" in error_msg:
            logging.error("Unauthorized access attempt. Please check API credentials.")
            return "Error: Authentication failed. Please verify the API configuration or contact support."
        elif "429" in error_msg:
            logging.error("Rate limit exceeded for Azure OpenAI API")
            return "Error: Too many requests. Please wait a moment and try again."
        else:
            logging.error(f"Unexpected error in Azure OpenAI API call: {error_msg}")
            return f"An error occurred while generating the assessment. Please try again. Error details: {error_msg}"



# Route to initialize bond classes with common parameters
@nc_bonds_bp.route('/', methods=['GET', 'POST'])
def nc_bonds():   
    return render_template('nc_bonds.html')

@nc_bonds_bp.route('/fixed_bonds', methods=['GET', 'POST'])
def nc_fixed_bonds():

    readme_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'nc_fixed_bonds.md')
    with open(readme_path, 'r') as readme_file:
        content = readme_file.read()
    md_content = markdown.markdown(content)

    gpt_assessment = None
    fr_bond_results = None
    form_data = {}
    
    if request.method == 'POST':     
        action = request.form.get('analysis_type')
        # Retrieve form data from URL parameters
        form_data = {
            'value_date': request.form['value_date'],
            'spot_dates': request.form['spot_dates'],
            'spot_rates': request.form['spot_rates'],
            'shocks': request.form['shocks'],
            'day_count_val': request.form['day_count'],
            'calendar_val': request.form['calendar'],
            'interpolation_val': request.form['interpolation'],
            'compounding_val': request.form['compounding'],
            'compounding_frequency_val': request.form['compounding_frequency'],
            'issue_date': request.form['issue_date'],
            'maturity_date': request.form['maturity_date'],
            'tenor_val': request.form['tenor'],
            'coupon_rate': request.form['coupon_rate'],
            'notional': request.form['notional']
        }
        
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
        elif tenor_val == 'Semiannual':
            tenor = ql.Period(ql.Semiannual)
        elif tenor_val == 'Quarterly':
            tenor = ql.Period(ql.Quarterly)
        elif tenor_val == 'Monthly':
            tenor = ql.Period(ql.Monthly)
            
        # Map the day count value to a QuantLib DayCount
        if day_count_val == 'ActualActual':
            day_count = ql.ActualActual(ql.ActualActual.Bond)
        elif day_count_val == 'Thirty360':
            day_count = ql.Thirty360(ql.Thirty360.BondBasis)
        elif day_count_val == 'Actual360':
            day_count = ql.Actual360()
        elif day_count_val == 'Actual365Fixed':
            day_count = ql.Actual365Fixed()
        
        fixed_bond = NCFixedBonds(value_date, spot_dates, spot_rates, shocks, day_count, calendar,
                                      interpolation, compounding, compounding_frequency)
        
        fr_bond_results = fixed_bond.fixed_rate(issue_date, maturity_date, tenor,
                                                coupon_rate, notional)
        if action == 'ai_assessment':
            # AI Assessment logic
            if fr_bond_results:
                print(fr_bond_results)
                # Prepare the input for AI using the actual bond results
                assessment_input = f"Please assess the bond pricing results based on the following outputs: {fr_bond_results}. Focus on the price changes across different shocks and any notable patterns."
                gpt_assessment = ask_gpt(assessment_input)
            else:
                gpt_assessment = "No bond pricing results available for assessment."
    return render_template('ncfixedbonds.html', form_data=form_data,
                           fr_bond_results=fr_bond_results, md_content=md_content, gpt_assessment=gpt_assessment)

@nc_bonds_bp.route('/fixed_amort_bonds', methods=['GET', 'POST'])
def nc_fixed_amort_bonds():
    readme_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'nc_fixed_amort_bonds.md')
    with open(readme_path, 'r') as readme_file:
        content = readme_file.read()
    md_content = markdown.markdown(content)
    
    fram_bond_results = None
    gpt_assessment = None
    form_data = {}
    
    if request.method == 'POST':     
        action = request.form.get('analysis_type')
        # Retrieve form data from URL parameters
        form_data = {
            'value_date': request.form['value_date'],
            'spot_dates': request.form['spot_dates'],
            'spot_rates': request.form['spot_rates'],
            'shocks': request.form['shocks'],
            'day_count_val': request.form['day_count'],
            'calendar_val': request.form['calendar'],
            'interpolation_val': request.form['interpolation'],
            'compounding_val': request.form['compounding'],
            'compounding_frequency_val': request.form['compounding_frequency'],
            'issue_date': request.form['issue_date'],
            'maturity_date': request.form['maturity_date'],
            'tenor_val': request.form['tenor'],
            'coupon_rate': request.form['coupon_rate'],
            'notional': request.form['notional']
        }
        
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
        elif tenor_val == 'Semiannual':
            tenor = ql.Period(ql.Semiannual)
        elif tenor_val == 'Quarterly':
            tenor = ql.Period(ql.Quarterly)
        elif tenor_val == 'Monthly':
            tenor = ql.Period(ql.Monthly)
            
        # Map the day count value to a QuantLib DayCount
        if day_count_val == 'ActualActual':
            day_count = ql.ActualActual(ql.ActualActual.Bond)
        elif day_count_val == 'Thirty360':
            day_count = ql.Thirty360(ql.Thirty360.BondBasis)
        elif day_count_val == 'Actual360':
            day_count = ql.Actual360()
        elif day_count_val == 'Actual365Fixed':
            day_count = ql.Actual365Fixed()
        
        fixed_bond = NCFixedBonds(value_date, spot_dates, spot_rates, shocks, day_count, calendar,
                                      interpolation, compounding, compounding_frequency)
        
        fram_bond_results = fixed_bond.fixed_rate_amortizing(issue_date, maturity_date,
                                                             tenor, coupon_rate, notional)
        if action == 'ai_assessment':
            # AI Assessment logic
            if fram_bond_results:
                # Prepare the input for AI using the actual bond results
                assessment_input = f"Please assess the amortizing bond pricing results based on the following outputs: {fram_bond_results}. Focus on the price changes across different shocks and any notable patterns."
                gpt_assessment = ask_gpt(assessment_input)
            else:
                gpt_assessment = "No bond pricing results available for assessment."
    return render_template('ncfixedamortbonds.html', form_data=form_data,
                           fram_bond_results=fram_bond_results, md_content=md_content, gpt_assessment=gpt_assessment)

@nc_bonds_bp.route('/floating_bonds', methods=['GET', 'POST'])
def nc_floating_bonds():

    readme_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'nc_floating_bonds.md')
    with open(readme_path, 'r') as readme_file:
        content = readme_file.read()
    md_content = markdown.markdown(content)

    
    fl_bond_results = None
    gpt_assessment = None
    form_data = {}        
    
    if request.method == 'POST':     
        action = request.form.get('analysis_type')
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
            'day_count_val': request.form['day_count']
        }
        
        value_date = form_data['value_date']
        spot_dates = form_data['spot_dates']
        spot_rates = form_data['spot_rates']
        index_dates = form_data['index_dates']
        index_rates = form_data['index_rates']
        shocks = form_data['shocks']
        issue_date = form_data['issue_date']
        maturity_date = form_data['maturity_date']
        spread = form_data['spread']
        notional = form_data['notional']
        
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
        
        # Map tenor value to QuantLib Period
        if tenor_val == 'Annual':
            tenor = ql.Period(ql.Annual)
        elif tenor_val == 'Semiannual':
            tenor = ql.Period(ql.Semiannual)
        elif tenor_val == 'Quarterly':
            tenor = ql.Period(ql.Quarterly)
        elif tenor_val == 'Monthly':
            tenor = ql.Period(ql.Monthly)
        else:
            tenor = ql.Period(ql.Semiannual)  # Default to Semiannual

        # Map compounding value to QuantLib Compounding
        if compounding_val == 'Compounded':
            compounding = ql.Compounded
        elif compounding_val == 'Continuous':
            compounding = ql.Continuous
        elif compounding_val == 'Simple':
            compounding = ql.Simple
        else:
            compounding = ql.Compounded  # Default to Compounded

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
        elif tenor_val == 'Semiannual':
            tenor = ql.Period(ql.Semiannual)
        elif tenor_val == 'Quarterly':
            tenor = ql.Period(ql.Quarterly)
        elif tenor_val == 'Monthly':
            tenor = ql.Period(ql.Monthly)
            
        # Map the day count value to a QuantLib DayCount
        if day_count_val == 'ActualActual':
            day_count = ql.ActualActual(ql.ActualActual.Bond)
        elif day_count_val == 'Thirty360':
            day_count = ql.Thirty360(ql.Thirty360.BondBasis)
        elif day_count_val == 'Actual360':
            day_count = ql.Actual360()
        elif day_count_val == 'Actual365Fixed':
            day_count = ql.Actual365Fixed()
                
        floating_bond = NCFloatingBonds(value_date, spot_dates, spot_rates, index_dates, index_rates,
                                        calendar, currency, interpolation, compounding,
                                        compounding_frequency, epsilon=0.001)
        
        fl_bond_results = floating_bond.price_floating(shocks, issue_date, maturity_date,
                                                       tenor, spread, notional, day_count)
        if action == 'ai_assessment':
            # AI Assessment logic
            if fl_bond_results:
                # Prepare the input for AI using the actual bond results
                assessment_input = f"Please assess the floating rate bond pricing results based on the following outputs: {fl_bond_results}. Focus on the price changes across different shocks and any notable patterns."
                gpt_assessment = ask_gpt(assessment_input)
            else:
                gpt_assessment = "No bond pricing results available for assessment."

    return render_template('ncfloatingbonds.html', form_data=form_data,
                           fl_bond_results=fl_bond_results, md_content=md_content, gpt_assessment=gpt_assessment)

@nc_bonds_bp.route('/floating_amortizing_bonds', methods=['GET', 'POST'])
def nc_floating_amort_bonds():

    readme_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'nc_floating_amort_bonds.md')
    with open(readme_path, 'r') as readme_file:
        content = readme_file.read()
    md_content = markdown.markdown(content)

    
    flam_bond_results = None
    gpt_assessment = None
    form_data = {}        
    
    if request.method == 'POST':     
        action = request.form.get('analysis_type')
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
            'notional_dates': request.form['notional_dates'],
            'day_count_val': request.form['day_count']
        }
        
        value_date = form_data['value_date']
        spot_dates = form_data['spot_dates']
        spot_rates = form_data['spot_rates']
        index_dates = form_data['index_dates']
        index_rates = form_data['index_rates']
        shocks = form_data['shocks']
        issue_date = form_data['issue_date']
        maturity_date = form_data['maturity_date']
        spread = form_data['spread']
        notional = form_data['notional']
        notional_dates = form_data['notional_dates']
        
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
        
        # Map interpolation value to QuantLib Interpolation
        if interpolation_val == 'Linear':
            interpolation = ql.Linear()
        elif interpolation_val == 'LogLinear':
            interpolation = ql.LogLinear()
        elif interpolation_val == 'Cubic':
            interpolation = ql.Cubic()
        else:
            interpolation = ql.Linear()  # Default to Linear

        # Map compounding value to QuantLib Compounding
        if compounding_val == 'Compounded':
            compounding = ql.Compounded
        elif compounding_val == 'Continuous':
            compounding = ql.Continuous
        elif compounding_val == 'Simple':
            compounding = ql.Simple
        else:
            compounding = ql.Compounded  # Default to Compounded

        # Map compounding frequency value to QuantLib Frequency
        if compounding_frequency_val == 'Annual':
            compounding_frequency = ql.Annual
        elif compounding_frequency_val == 'Semiannual':
            compounding_frequency = ql.Semiannual
        elif compounding_frequency_val == 'Quarterly':
            compounding_frequency = ql.Quarterly
        elif compounding_frequency_val == 'Monthly':
            compounding_frequency = ql.Monthly
        else:
            compounding_frequency = ql.Annual  # Default to Annual

        # Map tenor value to QuantLib Period
        if tenor_val == 'Annual':
            tenor = ql.Period(ql.Annual)
        elif tenor_val == 'Semiannual':
            tenor = ql.Period(ql.Semiannual)
        elif tenor_val == 'Quarterly':
            tenor = ql.Period(ql.Quarterly)
        elif tenor_val == 'Monthly':
            tenor = ql.Period(ql.Monthly)
        else:
            tenor = ql.Period(ql.Semiannual)  # Default to Semiannual

        # Map the day count value to a QuantLib DayCount
        if day_count_val == 'ActualActual':
            day_count = ql.ActualActual(ql.ActualActual.Bond)
        elif day_count_val == 'Thirty360':
            day_count = ql.Thirty360(ql.Thirty360.BondBasis)
        elif day_count_val == 'Actual360':
            day_count = ql.Actual360()
        elif day_count_val == 'Actual365Fixed':
            day_count = ql.Actual365Fixed()
        else:
            day_count = ql.Actual360()  # Default to Actual/360

        # Initialize floating bond object
        floating_bond = NCFloatingBonds(
            value_date, spot_dates, spot_rates, index_dates, index_rates,
            calendar, currency, interpolation, compounding,
            compounding_frequency
        )
        
        flam_bond_results = floating_bond.price_amortizing_floating(shocks, issue_date, maturity_date, tenor, spread, notional, notional_dates, day_count)
        if action == 'ai_assessment':
            # AI Assessment logic
            if flam_bond_results:
                # Prepare the input for AI using the actual bond results
                assessment_input = f"Please assess the floating rate amortizing bond pricing results based on the following outputs: {flam_bond_results}. Focus on the price changes across different shocks and any notable patterns."
                gpt_assessment = ask_gpt(assessment_input)
            else:
                gpt_assessment = "No bond pricing results available for assessment."
        
    return render_template('ncfloatingamortbonds.html', form_data=form_data,
                           flam_bond_results=flam_bond_results, md_content=md_content, gpt_assessment=gpt_assessment)




