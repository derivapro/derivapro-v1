from ..models.mdls_bonds import NCFixedBonds, NCFloatingBonds
from flask import Blueprint, abort, render_template, request, json
import QuantLib as ql
import os
import markdown
from dotenv import load_dotenv
import logging
from copy import deepcopy

from ..utils.lazy_imports import LazyAttribute
from ..models.rates_fixed_income import (
    CallableBondTerms,
    CapFloorTerms,
    FraTerms,
    parse_curve,
    parse_date,
    price_callable_putable_bond,
    price_cap_floor,
    price_fra,
)

logger = logging.getLogger(__name__)

llm_client = LazyAttribute("derivapro.llm", "llm_client")

# Initialize Flask app
nc_bonds_bp = Blueprint("nc_bonds", __name__)

load_dotenv()

# Get the values from the environment variables
model = os.getenv("LLM_MODEL", os.getenv("Model"))


FIXED_INCOME_EXTENSION_CONFIGS = {
    "fra": {
        "title": "Forward Rate Agreement",
        "subtitle": "Price OTC forward-rate exposure using discount and forward curves.",
        "asset_class": "Fixed Income / Rates",
        "methodology_doc": "forward_rate_agreement",
        "description_title": "Single-period interest-rate forward contract with curve-based PV.",
        "description_body": (
            "A forward rate agreement locks a fixed rate for a future accrual period. "
            "The workflow projects the forward rate from the supplied forward curve, compares it with the contract rate, "
            "and discounts the payoff using the supplied discount curve."
        ),
        "chips": ["Forward curve", "Discount curve", "Day count", "Scenario shocks"],
        "fields": [
            {"name": "valuation_date", "label": "Valuation Date", "type": "date", "value": "2026-08-13"},
            {"name": "start_date", "label": "FRA Start Date", "type": "date", "value": "2027-02-13"},
            {"name": "end_date", "label": "FRA End Date", "type": "date", "value": "2027-08-13"},
            {"name": "notional", "label": "Notional", "type": "number", "step": "1000", "value": "1000000"},
            {"name": "strike_rate", "label": "Contract Rate", "type": "number", "step": "0.0001", "value": "0.0425"},
            {
                "name": "position",
                "label": "Position",
                "type": "select",
                "value": "pay_fixed",
                "options": [("pay_fixed", "Pay Fixed / Receive Floating"), ("receive_fixed", "Receive Fixed / Pay Floating")],
            },
            {
                "name": "day_count",
                "label": "Accrual Day Count",
                "type": "select",
                "value": "ACT/360",
                "options": [("ACT/360", "ACT/360"), ("ACT/365", "ACT/365"), ("30/360", "30/360")],
            },
            {"name": "scenario_shock_bp", "label": "Scenario Shock (bp)", "type": "number", "step": "1", "value": "25"},
        ],
    },
    "cap-floor": {
        "title": "Interest Rate Cap / Floor",
        "subtitle": "Value caplet or floorlet strips using Black-style rate optionality.",
        "asset_class": "Fixed Income / Rates",
        "methodology_doc": "cap_floor",
        "description_title": "A portfolio of rate options on forward reset periods.",
        "description_body": (
            "Caps and floors are strips of caplets or floorlets that reference forward rates over scheduled reset periods. "
            "This first-wave workflow values the strip with a Black rate-option approximation, supplied forward/discount curves, "
            "and user-entered volatility."
        ),
        "chips": ["Caps", "Floors", "Black caplets", "Rate/vol scenarios"],
        "fields": [
            {"name": "valuation_date", "label": "Valuation Date", "type": "date", "value": "2026-08-13"},
            {"name": "start_date", "label": "Start Date", "type": "date", "value": "2026-08-13"},
            {"name": "maturity_date", "label": "Maturity Date", "type": "date", "value": "2029-08-13"},
            {"name": "notional", "label": "Notional", "type": "number", "step": "1000", "value": "1000000"},
            {"name": "strike_rate", "label": "Strike Rate", "type": "number", "step": "0.0001", "value": "0.0450"},
            {"name": "volatility", "label": "Forward Rate Volatility", "type": "number", "step": "0.001", "value": "0.25"},
            {
                "name": "option_type",
                "label": "Product Type",
                "type": "select",
                "value": "cap",
                "options": [("cap", "Cap"), ("floor", "Floor")],
            },
            {
                "name": "payments_per_year",
                "label": "Payments / Year",
                "type": "select",
                "value": "4",
                "options": [("1", "Annual"), ("2", "Semiannual"), ("4", "Quarterly")],
            },
            {
                "name": "day_count",
                "label": "Accrual Day Count",
                "type": "select",
                "value": "ACT/360",
                "options": [("ACT/360", "ACT/360"), ("ACT/365", "ACT/365"), ("30/360", "30/360")],
            },
            {"name": "scenario_rate_shock_bp", "label": "Rate Shock (bp)", "type": "number", "step": "1", "value": "25"},
            {"name": "scenario_vol_shock", "label": "Vol Shock", "type": "number", "step": "0.001", "value": "0.05"},
        ],
    },
    "callable-putable-bond": {
        "title": "Callable / Putable Bond",
        "subtitle": "Evaluate fixed-rate bonds with embedded issuer call or investor put rights.",
        "asset_class": "Fixed Income",
        "methodology_doc": "callable_putable_bond",
        "description_title": "Fixed-rate bond plus embedded interest-rate optionality.",
        "description_body": (
            "Callable and putable bonds extend the existing fixed-rate bond workflow by adding exercise optionality. "
            "This page compares straight-bond PV with an option-adjusted value from a transparent short-rate lattice approximation."
        ),
        "chips": ["Callable bonds", "Putable bonds", "Short-rate lattice", "Effective duration"],
        "fields": [
            {"name": "valuation_date", "label": "Valuation Date", "type": "date", "value": "2026-08-13"},
            {"name": "maturity_date", "label": "Maturity Date", "type": "date", "value": "2031-08-13"},
            {"name": "notional", "label": "Face Value", "type": "number", "step": "1000", "value": "1000000"},
            {"name": "coupon_rate", "label": "Coupon Rate", "type": "number", "step": "0.0001", "value": "0.0550"},
            {
                "name": "payments_per_year",
                "label": "Coupon Frequency",
                "type": "select",
                "value": "2",
                "options": [("1", "Annual"), ("2", "Semiannual"), ("4", "Quarterly")],
            },
            {
                "name": "option_type",
                "label": "Embedded Option",
                "type": "select",
                "value": "callable",
                "options": [("callable", "Callable"), ("putable", "Putable")],
            },
            {"name": "call_or_put_price", "label": "Exercise Price (% of Par)", "type": "number", "step": "0.01", "value": "100.00"},
            {"name": "first_exercise_year", "label": "First Exercise Year", "type": "number", "step": "0.25", "value": "2.0"},
            {"name": "short_rate_volatility", "label": "Short-Rate Volatility", "type": "number", "step": "0.001", "value": "0.015"},
            {
                "name": "day_count",
                "label": "Coupon Day Count",
                "type": "select",
                "value": "30/360",
                "options": [("30/360", "30/360"), ("ACT/360", "ACT/360"), ("ACT/365", "ACT/365")],
            },
            {"name": "scenario_shock_bp", "label": "Scenario Shock (bp)", "type": "number", "step": "1", "value": "25"},
        ],
    },
}


CURVE_FIELD_DEFAULTS = {
    "discount_curve_tenors": "0.25,0.5,1,2,3,5,7,10",
    "discount_curve_rates": "0.0400,0.0410,0.0420,0.0430,0.0440,0.0450,0.0460,0.0470",
    "forward_curve_tenors": "0.25,0.5,1,2,3,5,7,10",
    "forward_curve_rates": "0.0410,0.0420,0.0430,0.0440,0.0450,0.0460,0.0470,0.0480",
}


def ask_gpt(question):
    """Send a request to the configured LLM provider and return text."""
    try:
        return llm_client.generate_response(prompt=question, model=model)
    except Exception as e:
        error_msg = str(e)
        logger.exception("Error occurred while calling LLM provider")

        if "403" in error_msg:
            logger.error(
                "Authentication failed or access is restricted for the LLM provider."
            )
            return "Error: Access to the AI service is currently restricted. Please verify the API configuration or contact support."
        elif "401" in error_msg:
            logger.error("Unauthorized access attempt. Please check API credentials.")
            return "Error: Authentication failed. Please verify the API configuration or contact support."
        elif "429" in error_msg:
            logger.error("Rate limit exceeded for LLM provider")
            return "Error: Too many requests. Please wait a moment and try again."
        else:
            logger.error("Unexpected error in LLM provider call")
            return f"An error occurred while generating the assessment. Please try again. Error details: {error_msg}"


def _default_form_data(config):
    data = {field["name"]: field["value"] for field in config["fields"]}
    data.update(CURVE_FIELD_DEFAULTS)
    return data


def _extension_form_data(config):
    data = _default_form_data(config)
    if request.method == "POST":
        for key in data:
            data[key] = request.form.get(key, data[key])
    return data


def _float_value(data, key):
    return float(data[key])


def _int_value(data, key):
    return int(float(data[key]))


def _price_fixed_income_extension(product_slug, form_data):
    discount_curve = parse_curve(
        form_data["discount_curve_tenors"],
        form_data["discount_curve_rates"],
    )
    forward_curve = parse_curve(
        form_data["forward_curve_tenors"],
        form_data["forward_curve_rates"],
    )

    if product_slug == "fra":
        return price_fra(
            FraTerms(
                valuation_date=parse_date(form_data["valuation_date"]),
                start_date=parse_date(form_data["start_date"]),
                end_date=parse_date(form_data["end_date"]),
                notional=_float_value(form_data, "notional"),
                strike_rate=_float_value(form_data, "strike_rate"),
                position=form_data["position"],
                day_count=form_data["day_count"],
                discount_curve=discount_curve,
                forward_curve=forward_curve,
                scenario_shock_bp=_float_value(form_data, "scenario_shock_bp"),
            )
        )

    if product_slug == "cap-floor":
        return price_cap_floor(
            CapFloorTerms(
                valuation_date=parse_date(form_data["valuation_date"]),
                start_date=parse_date(form_data["start_date"]),
                maturity_date=parse_date(form_data["maturity_date"]),
                notional=_float_value(form_data, "notional"),
                strike_rate=_float_value(form_data, "strike_rate"),
                option_type=form_data["option_type"],
                volatility=_float_value(form_data, "volatility"),
                payments_per_year=_int_value(form_data, "payments_per_year"),
                day_count=form_data["day_count"],
                discount_curve=discount_curve,
                forward_curve=forward_curve,
                scenario_rate_shock_bp=_float_value(form_data, "scenario_rate_shock_bp"),
                scenario_vol_shock=_float_value(form_data, "scenario_vol_shock"),
            )
        )

    if product_slug == "callable-putable-bond":
        return price_callable_putable_bond(
            CallableBondTerms(
                valuation_date=parse_date(form_data["valuation_date"]),
                maturity_date=parse_date(form_data["maturity_date"]),
                notional=_float_value(form_data, "notional"),
                coupon_rate=_float_value(form_data, "coupon_rate"),
                payments_per_year=_int_value(form_data, "payments_per_year"),
                day_count=form_data["day_count"],
                discount_curve=discount_curve,
                option_type=form_data["option_type"],
                call_or_put_price=_float_value(form_data, "call_or_put_price"),
                first_exercise_year=_float_value(form_data, "first_exercise_year"),
                short_rate_volatility=_float_value(form_data, "short_rate_volatility"),
                scenario_shock_bp=_float_value(form_data, "scenario_shock_bp"),
            )
        )

    raise ValueError("Unsupported fixed-income extension product.")


# Route to initialize bond classes with common parameters
@nc_bonds_bp.route("/", methods=["GET", "POST"])
def nc_bonds():
    return render_template("nc_bonds.html")


@nc_bonds_bp.route("/fixed_bonds", methods=["GET", "POST"])
def nc_fixed_bonds():

    readme_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "nc_fixed_bonds.md"
    )
    with open(readme_path, "r") as readme_file:
        content = readme_file.read()
    md_content = markdown.markdown(content)

    gpt_assessment = None
    fr_bond_results = None
    form_data = {}

    if request.method == "POST":
        action = request.form.get("analysis_type")
        # Retrieve form data from URL parameters
        form_data = {
            "value_date": request.form["value_date"],
            "spot_dates": request.form["spot_dates"],
            "spot_rates": request.form["spot_rates"],
            "shocks": request.form["shocks"],
            "day_count_val": request.form["day_count"],
            "calendar_val": request.form["calendar"],
            "interpolation_val": request.form["interpolation"],
            "compounding_val": request.form["compounding"],
            "compounding_frequency_val": request.form["compounding_frequency"],
            "issue_date": request.form["issue_date"],
            "maturity_date": request.form["maturity_date"],
            "tenor_val": request.form["tenor"],
            "coupon_rate": request.form["coupon_rate"],
            "notional": request.form["notional"],
        }

        value_date = form_data["value_date"]
        spot_dates = form_data["spot_dates"]
        spot_rates = form_data["spot_rates"]
        shocks = form_data["shocks"]
        issue_date = form_data["issue_date"]
        maturity_date = form_data["maturity_date"]
        coupon_rate = form_data["coupon_rate"]
        notional = form_data["notional"]

        calendar_val = form_data["calendar_val"]
        interpolation_val = form_data["interpolation_val"]
        compounding_val = form_data["compounding_val"]
        compounding_frequency_val = form_data["compounding_frequency_val"]
        tenor_val = form_data["tenor_val"]
        day_count_val = form_data["day_count_val"]

        # Map the calendar value to a QuantLib Calendar
        if calendar_val == "UnitedStates":
            calendar = ql.UnitedStates(ql.UnitedStates.NYSE)
        elif calendar_val == "TARGET":
            calendar = ql.TARGET()
        elif calendar_val == "UnitedKingdom":
            calendar = ql.UnitedKingdom()
        elif calendar_val == "China":
            calendar = ql.China()

        # Map interpolation value to a QuantLib interpolation type
        if interpolation_val == "Linear":
            interpolation = ql.Linear()
        elif interpolation_val == "LogLinear":
            interpolation = ql.LogLinear()
        elif interpolation_val == "Cubic":
            interpolation = ql.Cubic()

        # Map compounding value to a QuantLib Compounding type
        if compounding_val == "Compounded":
            compounding = ql.Compounded
        elif compounding_val == "Simple":
            compounding = ql.Simple
        elif compounding_val == "Continuous":
            compounding = ql.Continuous

        # Map compounding frequency value to QuantLib Frequency
        if compounding_frequency_val == "Annual":
            compounding_frequency = ql.Annual
        elif compounding_frequency_val == "Semiannual":
            compounding_frequency = ql.Semiannual
        elif compounding_frequency_val == "Quarterly":
            compounding_frequency = ql.Quarterly
        elif compounding_frequency_val == "Monthly":
            compounding_frequency = ql.Monthly
        elif compounding_frequency_val == "Daily":
            compounding_frequency = ql.Daily

        # Map tenor value to QuantLib Tenor
        if tenor_val == "Annual":
            tenor = ql.Period(ql.Annual)
        elif tenor_val == "Semiannual":
            tenor = ql.Period(ql.Semiannual)
        elif tenor_val == "Quarterly":
            tenor = ql.Period(ql.Quarterly)
        elif tenor_val == "Monthly":
            tenor = ql.Period(ql.Monthly)

        # Map the day count value to a QuantLib DayCount
        if day_count_val == "ActualActual":
            day_count = ql.ActualActual(ql.ActualActual.Bond)
        elif day_count_val == "Thirty360":
            day_count = ql.Thirty360(ql.Thirty360.BondBasis)
        elif day_count_val == "Actual360":
            day_count = ql.Actual360()
        elif day_count_val == "Actual365Fixed":
            day_count = ql.Actual365Fixed()

        fixed_bond = NCFixedBonds(
            value_date,
            spot_dates,
            spot_rates,
            shocks,
            day_count,
            calendar,
            interpolation,
            compounding,
            compounding_frequency,
        )

        fr_bond_results = fixed_bond.fixed_rate(
            issue_date, maturity_date, tenor, coupon_rate, notional
        )
        if action == "ai_assessment":
            # AI Assessment logic
            if fr_bond_results:
                logger.debug("Fixed bond results available for AI assessment.")
                # Prepare the input for AI using the actual bond results
                assessment_input = f"Please assess the bond pricing results based on the following outputs: {fr_bond_results}. Focus on the price changes across different shocks and any notable patterns."
                gpt_assessment = ask_gpt(assessment_input)
            else:
                gpt_assessment = "No bond pricing results available for assessment."
    return render_template(
        "ncfixedbonds.html",
        form_data=form_data,
        fr_bond_results=fr_bond_results,
        md_content=md_content,
        gpt_assessment=gpt_assessment,
    )


@nc_bonds_bp.route("/fixed_amort_bonds", methods=["GET", "POST"])
def nc_fixed_amort_bonds():
    readme_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "nc_fixed_amort_bonds.md"
    )
    with open(readme_path, "r") as readme_file:
        content = readme_file.read()
    md_content = markdown.markdown(content)

    fram_bond_results = None
    gpt_assessment = None
    form_data = {}

    if request.method == "POST":
        action = request.form.get("analysis_type")
        # Retrieve form data from URL parameters
        form_data = {
            "value_date": request.form["value_date"],
            "spot_dates": request.form["spot_dates"],
            "spot_rates": request.form["spot_rates"],
            "shocks": request.form["shocks"],
            "day_count_val": request.form["day_count"],
            "calendar_val": request.form["calendar"],
            "interpolation_val": request.form["interpolation"],
            "compounding_val": request.form["compounding"],
            "compounding_frequency_val": request.form["compounding_frequency"],
            "issue_date": request.form["issue_date"],
            "maturity_date": request.form["maturity_date"],
            "tenor_val": request.form["tenor"],
            "coupon_rate": request.form["coupon_rate"],
            "notional": request.form["notional"],
        }

        value_date = form_data["value_date"]
        spot_dates = form_data["spot_dates"]
        spot_rates = form_data["spot_rates"]
        shocks = form_data["shocks"]
        issue_date = form_data["issue_date"]
        maturity_date = form_data["maturity_date"]
        coupon_rate = form_data["coupon_rate"]
        notional = form_data["notional"]

        calendar_val = form_data["calendar_val"]
        interpolation_val = form_data["interpolation_val"]
        compounding_val = form_data["compounding_val"]
        compounding_frequency_val = form_data["compounding_frequency_val"]
        tenor_val = form_data["tenor_val"]
        day_count_val = form_data["day_count_val"]

        # Map the calendar value to a QuantLib Calendar
        if calendar_val == "UnitedStates":
            calendar = ql.UnitedStates(ql.UnitedStates.NYSE)
        elif calendar_val == "TARGET":
            calendar = ql.TARGET()
        elif calendar_val == "UnitedKingdom":
            calendar = ql.UnitedKingdom()
        elif calendar_val == "China":
            calendar = ql.China()

        # Map interpolation value to a QuantLib interpolation type
        if interpolation_val == "Linear":
            interpolation = ql.Linear()
        elif interpolation_val == "LogLinear":
            interpolation = ql.LogLinear()
        elif interpolation_val == "Cubic":
            interpolation = ql.Cubic()

        # Map compounding value to a QuantLib Compounding type
        if compounding_val == "Compounded":
            compounding = ql.Compounded
        elif compounding_val == "Simple":
            compounding = ql.Simple
        elif compounding_val == "Continuous":
            compounding = ql.Continuous

        # Map compounding frequency value to QuantLib Frequency
        if compounding_frequency_val == "Annual":
            compounding_frequency = ql.Annual
        elif compounding_frequency_val == "Semiannual":
            compounding_frequency = ql.Semiannual
        elif compounding_frequency_val == "Quarterly":
            compounding_frequency = ql.Quarterly
        elif compounding_frequency_val == "Monthly":
            compounding_frequency = ql.Monthly
        elif compounding_frequency_val == "Daily":
            compounding_frequency = ql.Daily

        # Map tenor value to QuantLib Tenor
        if tenor_val == "Annual":
            tenor = ql.Period(ql.Annual)
        elif tenor_val == "Semiannual":
            tenor = ql.Period(ql.Semiannual)
        elif tenor_val == "Quarterly":
            tenor = ql.Period(ql.Quarterly)
        elif tenor_val == "Monthly":
            tenor = ql.Period(ql.Monthly)

        # Map the day count value to a QuantLib DayCount
        if day_count_val == "ActualActual":
            day_count = ql.ActualActual(ql.ActualActual.Bond)
        elif day_count_val == "Thirty360":
            day_count = ql.Thirty360(ql.Thirty360.BondBasis)
        elif day_count_val == "Actual360":
            day_count = ql.Actual360()
        elif day_count_val == "Actual365Fixed":
            day_count = ql.Actual365Fixed()

        fixed_bond = NCFixedBonds(
            value_date,
            spot_dates,
            spot_rates,
            shocks,
            day_count,
            calendar,
            interpolation,
            compounding,
            compounding_frequency,
        )

        fram_bond_results = fixed_bond.fixed_rate_amortizing(
            issue_date, maturity_date, tenor, coupon_rate, notional
        )
        if action == "ai_assessment":
            # AI Assessment logic
            if fram_bond_results:
                # Prepare the input for AI using the actual bond results
                assessment_input = f"Please assess the amortizing bond pricing results based on the following outputs: {fram_bond_results}. Focus on the price changes across different shocks and any notable patterns."
                gpt_assessment = ask_gpt(assessment_input)
            else:
                gpt_assessment = "No bond pricing results available for assessment."
    return render_template(
        "ncfixedamortbonds.html",
        form_data=form_data,
        fram_bond_results=fram_bond_results,
        md_content=md_content,
        gpt_assessment=gpt_assessment,
    )


@nc_bonds_bp.route("/floating_bonds", methods=["GET", "POST"])
def nc_floating_bonds():

    readme_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "nc_floating_bonds.md"
    )
    with open(readme_path, "r") as readme_file:
        content = readme_file.read()
    md_content = markdown.markdown(content)

    fl_bond_results = None
    gpt_assessment = None
    form_data = {}

    if request.method == "POST":
        action = request.form.get("analysis_type")
        # Retrieve form data from URL parameters
        form_data = {
            "value_date": request.form["value_date"],
            "spot_dates": request.form["spot_dates"],
            "spot_rates": request.form["spot_rates"],
            "index_dates": request.form["index_dates"],
            "index_rates": request.form["index_rates"],
            "calendar_val": request.form["calendar"],
            "currency_val": request.form["currency"],
            "interpolation_val": request.form["interpolation"],
            "compounding_val": request.form["compounding"],
            "compounding_frequency_val": request.form["compounding_frequency"],
            "shocks": request.form["shocks"],
            "issue_date": request.form["issue_date"],
            "maturity_date": request.form["maturity_date"],
            "tenor_val": request.form["tenor"],
            "spread": request.form["spread"],
            "notional": request.form["notional"],
            "day_count_val": request.form["day_count"],
        }

        value_date = form_data["value_date"]
        spot_dates = form_data["spot_dates"]
        spot_rates = form_data["spot_rates"]
        index_dates = form_data["index_dates"]
        index_rates = form_data["index_rates"]
        shocks = form_data["shocks"]
        issue_date = form_data["issue_date"]
        maturity_date = form_data["maturity_date"]
        spread = form_data["spread"]
        notional = form_data["notional"]

        calendar_val = form_data["calendar_val"]
        currency_val = form_data["currency_val"]
        interpolation_val = form_data["interpolation_val"]
        compounding_val = form_data["compounding_val"]
        compounding_frequency_val = form_data["compounding_frequency_val"]
        tenor_val = form_data["tenor_val"]
        day_count_val = form_data["day_count_val"]

        # Map the calendar value to a QuantLib Calendar
        if calendar_val == "UnitedStates":
            calendar = ql.UnitedStates(ql.UnitedStates.NYSE)
        elif calendar_val == "TARGET":
            calendar = ql.TARGET()
        elif calendar_val == "China":
            calendar = ql.China()

        # Map the currency value to a QuantLib Currency
        if currency_val == "USD":
            currency = ql.USDCurrency()
        elif currency_val == "EUR":
            currency = ql.EURCurrency()
        elif currency_val == "CNY":
            currency = ql.CNYCurrency()
        elif currency_val == "GBP":
            currency = ql.GBPCurrency()
        elif currency_val == "JPY":
            currency = ql.JPYCurrency()

        # Map interpolation value to a QuantLib interpolation type
        if interpolation_val == "Linear":
            interpolation = ql.Linear()
        elif interpolation_val == "LogLinear":
            interpolation = ql.LogLinear()
        elif interpolation_val == "Cubic":
            interpolation = ql.Cubic()

        # Map tenor value to QuantLib Period
        if tenor_val == "Annual":
            tenor = ql.Period(ql.Annual)
        elif tenor_val == "Semiannual":
            tenor = ql.Period(ql.Semiannual)
        elif tenor_val == "Quarterly":
            tenor = ql.Period(ql.Quarterly)
        elif tenor_val == "Monthly":
            tenor = ql.Period(ql.Monthly)
        else:
            tenor = ql.Period(ql.Semiannual)  # Default to Semiannual

        # Map compounding value to QuantLib Compounding
        if compounding_val == "Compounded":
            compounding = ql.Compounded
        elif compounding_val == "Continuous":
            compounding = ql.Continuous
        elif compounding_val == "Simple":
            compounding = ql.Simple
        else:
            compounding = ql.Compounded  # Default to Compounded

        # Map compounding frequency value to QuantLib Frequency
        if compounding_frequency_val == "Annual":
            compounding_frequency = ql.Annual
        elif compounding_frequency_val == "Semiannual":
            compounding_frequency = ql.Semiannual
        elif compounding_frequency_val == "Quarterly":
            compounding_frequency = ql.Quarterly
        elif compounding_frequency_val == "Monthly":
            compounding_frequency = ql.Monthly

        # Map tenor value to QuantLib Tenor
        if tenor_val == "Annual":
            tenor = ql.Period(ql.Annual)
        elif tenor_val == "Semiannual":
            tenor = ql.Period(ql.Semiannual)
        elif tenor_val == "Quarterly":
            tenor = ql.Period(ql.Quarterly)
        elif tenor_val == "Monthly":
            tenor = ql.Period(ql.Monthly)

        # Map the day count value to a QuantLib DayCount
        if day_count_val == "ActualActual":
            day_count = ql.ActualActual(ql.ActualActual.Bond)
        elif day_count_val == "Thirty360":
            day_count = ql.Thirty360(ql.Thirty360.BondBasis)
        elif day_count_val == "Actual360":
            day_count = ql.Actual360()
        elif day_count_val == "Actual365Fixed":
            day_count = ql.Actual365Fixed()

        floating_bond = NCFloatingBonds(
            value_date,
            spot_dates,
            spot_rates,
            index_dates,
            index_rates,
            calendar,
            currency,
            interpolation,
            compounding,
            compounding_frequency,
            epsilon=0.001,
        )

        fl_bond_results = floating_bond.price_floating(
            shocks, issue_date, maturity_date, tenor, spread, notional, day_count
        )
        if action == "ai_assessment":
            # AI Assessment logic
            if fl_bond_results:
                # Prepare the input for AI using the actual bond results
                assessment_input = f"Please assess the floating rate bond pricing results based on the following outputs: {fl_bond_results}. Focus on the price changes across different shocks and any notable patterns."
                gpt_assessment = ask_gpt(assessment_input)
            else:
                gpt_assessment = "No bond pricing results available for assessment."

    return render_template(
        "ncfloatingbonds.html",
        form_data=form_data,
        fl_bond_results=fl_bond_results,
        md_content=md_content,
        gpt_assessment=gpt_assessment,
    )


@nc_bonds_bp.route("/floating_amortizing_bonds", methods=["GET", "POST"])
def nc_floating_amort_bonds():

    readme_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "nc_floating_amort_bonds.md"
    )
    with open(readme_path, "r") as readme_file:
        content = readme_file.read()
    md_content = markdown.markdown(content)

    flam_bond_results = None
    gpt_assessment = None
    form_data = {}

    if request.method == "POST":
        action = request.form.get("analysis_type")
        # Retrieve form data from URL parameters
        form_data = {
            "value_date": request.form["value_date"],
            "spot_dates": request.form["spot_dates"],
            "spot_rates": request.form["spot_rates"],
            "index_dates": request.form["index_dates"],
            "index_rates": request.form["index_rates"],
            "calendar_val": request.form["calendar"],
            "currency_val": request.form["currency"],
            "interpolation_val": request.form["interpolation"],
            "compounding_val": request.form["compounding"],
            "compounding_frequency_val": request.form["compounding_frequency"],
            "shocks": request.form["shocks"],
            "issue_date": request.form["issue_date"],
            "maturity_date": request.form["maturity_date"],
            "tenor_val": request.form["tenor"],
            "spread": request.form["spread"],
            "notional": request.form["notional"],
            "notional_dates": request.form["notional_dates"],
            "day_count_val": request.form["day_count"],
        }

        value_date = form_data["value_date"]
        spot_dates = form_data["spot_dates"]
        spot_rates = form_data["spot_rates"]
        index_dates = form_data["index_dates"]
        index_rates = form_data["index_rates"]
        shocks = form_data["shocks"]
        issue_date = form_data["issue_date"]
        maturity_date = form_data["maturity_date"]
        spread = form_data["spread"]
        notional = form_data["notional"]
        notional_dates = form_data["notional_dates"]

        calendar_val = form_data["calendar_val"]
        currency_val = form_data["currency_val"]
        interpolation_val = form_data["interpolation_val"]
        compounding_val = form_data["compounding_val"]
        compounding_frequency_val = form_data["compounding_frequency_val"]
        tenor_val = form_data["tenor_val"]
        day_count_val = form_data["day_count_val"]

        # Map the calendar value to a QuantLib Calendar
        if calendar_val == "UnitedStates":
            calendar = ql.UnitedStates(ql.UnitedStates.NYSE)
        elif calendar_val == "TARGET":
            calendar = ql.TARGET()
        elif calendar_val == "China":
            calendar = ql.China()

        # Map the currency value to a QuantLib Currency
        if currency_val == "USD":
            currency = ql.USDCurrency()
        elif currency_val == "EUR":
            currency = ql.EURCurrency()
        elif currency_val == "CNY":
            currency = ql.CNYCurrency()
        elif currency_val == "GBP":
            currency = ql.GBPCurrency()
        elif currency_val == "JPY":
            currency = ql.JPYCurrency()

        # Map interpolation value to QuantLib Interpolation
        if interpolation_val == "Linear":
            interpolation = ql.Linear()
        elif interpolation_val == "LogLinear":
            interpolation = ql.LogLinear()
        elif interpolation_val == "Cubic":
            interpolation = ql.Cubic()
        else:
            interpolation = ql.Linear()  # Default to Linear

        # Map compounding value to QuantLib Compounding
        if compounding_val == "Compounded":
            compounding = ql.Compounded
        elif compounding_val == "Continuous":
            compounding = ql.Continuous
        elif compounding_val == "Simple":
            compounding = ql.Simple
        else:
            compounding = ql.Compounded  # Default to Compounded

        # Map compounding frequency value to QuantLib Frequency
        if compounding_frequency_val == "Annual":
            compounding_frequency = ql.Annual
        elif compounding_frequency_val == "Semiannual":
            compounding_frequency = ql.Semiannual
        elif compounding_frequency_val == "Quarterly":
            compounding_frequency = ql.Quarterly
        elif compounding_frequency_val == "Monthly":
            compounding_frequency = ql.Monthly
        else:
            compounding_frequency = ql.Annual  # Default to Annual

        # Map tenor value to QuantLib Period
        if tenor_val == "Annual":
            tenor = ql.Period(ql.Annual)
        elif tenor_val == "Semiannual":
            tenor = ql.Period(ql.Semiannual)
        elif tenor_val == "Quarterly":
            tenor = ql.Period(ql.Quarterly)
        elif tenor_val == "Monthly":
            tenor = ql.Period(ql.Monthly)
        else:
            tenor = ql.Period(ql.Semiannual)  # Default to Semiannual

        # Map the day count value to a QuantLib DayCount
        if day_count_val == "ActualActual":
            day_count = ql.ActualActual(ql.ActualActual.Bond)
        elif day_count_val == "Thirty360":
            day_count = ql.Thirty360(ql.Thirty360.BondBasis)
        elif day_count_val == "Actual360":
            day_count = ql.Actual360()
        elif day_count_val == "Actual365Fixed":
            day_count = ql.Actual365Fixed()
        else:
            day_count = ql.Actual360()  # Default to Actual/360

        # Initialize floating bond object
        floating_bond = NCFloatingBonds(
            value_date,
            spot_dates,
            spot_rates,
            index_dates,
            index_rates,
            calendar,
            currency,
            interpolation,
            compounding,
            compounding_frequency,
        )

        flam_bond_results = floating_bond.price_amortizing_floating(
            shocks,
            issue_date,
            maturity_date,
            tenor,
            spread,
            notional,
            notional_dates,
            day_count,
        )
        if action == "ai_assessment":
            # AI Assessment logic
            if flam_bond_results:
                # Prepare the input for AI using the actual bond results
                assessment_input = f"Please assess the floating rate amortizing bond pricing results based on the following outputs: {flam_bond_results}. Focus on the price changes across different shocks and any notable patterns."
                gpt_assessment = ask_gpt(assessment_input)
            else:
                gpt_assessment = "No bond pricing results available for assessment."

    return render_template(
        "ncfloatingamortbonds.html",
        form_data=form_data,
        flam_bond_results=flam_bond_results,
        md_content=md_content,
        gpt_assessment=gpt_assessment,
    )


@nc_bonds_bp.route("/rates-fixed-income", methods=["GET"])
def rates_fixed_income_home():
    return render_template(
        "fixed_income_extensions_home.html",
        products=FIXED_INCOME_EXTENSION_CONFIGS,
    )


@nc_bonds_bp.route("/rates-fixed-income/<product_slug>", methods=["GET", "POST"])
def rates_fixed_income_product(product_slug):
    config = FIXED_INCOME_EXTENSION_CONFIGS.get(product_slug)
    if not config:
        abort(404)

    form_data = _extension_form_data(config)
    results = None
    pricing_error = None

    if request.method == "POST":
        try:
            results = _price_fixed_income_extension(product_slug, form_data)
        except Exception as exc:
            pricing_error = str(exc)

    field_sections = [
        {"title": "Trade Terms", "fields": deepcopy(config["fields"])},
        {
            "title": "Curve Assumptions",
            "fields": [
                {
                    "name": "discount_curve_tenors",
                    "label": "Discount Curve Tenors (Years)",
                    "type": "text",
                    "value": form_data["discount_curve_tenors"],
                    "hint": "Comma-separated year tenors.",
                },
                {
                    "name": "discount_curve_rates",
                    "label": "Discount Curve Zero Rates",
                    "type": "text",
                    "value": form_data["discount_curve_rates"],
                    "hint": "Comma-separated continuously compounded zero rates.",
                },
                {
                    "name": "forward_curve_tenors",
                    "label": "Forward Curve Tenors (Years)",
                    "type": "text",
                    "value": form_data["forward_curve_tenors"],
                    "hint": "Used for FRA and cap/floor projected rates.",
                },
                {
                    "name": "forward_curve_rates",
                    "label": "Forward Curve Zero Rates",
                    "type": "text",
                    "value": form_data["forward_curve_rates"],
                    "hint": "Used for FRA and cap/floor projected rates.",
                },
            ],
        },
    ]

    for section in field_sections:
        for field in section["fields"]:
            field["value"] = form_data.get(field["name"], field.get("value", ""))
            field.setdefault("hint", "")
            field.setdefault("step", "any")

    return render_template(
        "fixed_income_extension_product.html",
        config=config,
        product_slug=product_slug,
        field_sections=field_sections,
        form_data=form_data,
        results=results,
        pricing_error=pricing_error,
    )
