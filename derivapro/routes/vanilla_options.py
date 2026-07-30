# Note: last updated on Aug 06

from ..models.mdls_lattice_trees import (
    LatticeModel,
    AmericanOptionSmoothnessTest,
    lattice_convergence_test,
    plot_convergence,
)

from flask import (
    Blueprint,
    render_template,
    request,
    jsonify,
    session,
    redirect,
    url_for,
    abort,
)
from flask_login import current_user, login_required
from flask import send_file

from ..models.mdls_vanilla_options import BlackScholes, SmoothnessTest
from ..models.market_data import StockData
import matplotlib.pyplot as plt
import os
import markdown
from random import random
from datetime import datetime
from ..extensions import db
from ..models.db_models import AnalysisResult, Instrument, PricingResult, Report
from ..models.mdls_binomial_tree import BinomialTreeEngineCRR
from ..models import mdls_monte_carlo_v2 as monte_carlo_module
from ..llm import llm_client
from ..services.report_builder import ReportTemplate, render_report_pdf

from dotenv import load_dotenv, find_dotenv
import io
import logging
import numpy as np
import uuid
from dataclasses import asdict

logger = logging.getLogger(__name__)

vanilla_options_bp = Blueprint("vanilla_options", __name__)

# Load the environment variables from the .env file
load_dotenv(find_dotenv())

# Get the values from the environment variables
model = os.getenv("LLM_MODEL", os.getenv("Model"))


def ask_gpt(question):
    """Send a request to the configured LLM provider and return text."""
    try:
        return llm_client.generate_response(prompt=question, model=model)
    except Exception as e:
        logger.exception("Error occurred while calling LLM provider")
        return f"An error occurred: {e}"


def _get_latest_pricing_result_for_user(product_type):
    if not current_user.is_authenticated:
        return None

    return (
        PricingResult.query
        .join(Instrument, PricingResult.instrument_id == Instrument.id)
        .filter(
            PricingResult.user_id == current_user.id,
            Instrument.user_id == current_user.id,
            Instrument.product_type == product_type,
        )
        .order_by(PricingResult.created_at.desc())
        .first()
    )


def _resolve_analysis_instrument(product_type, standalone_instrument):
    linked_pricing_result = _get_latest_pricing_result_for_user(product_type)

    if linked_pricing_result is not None:
        return linked_pricing_result.instrument, linked_pricing_result

    db.session.add(standalone_instrument)
    db.session.flush()
    return standalone_instrument, None


def _get_latest_analysis_by_type_for_user(product_type, analysis_type):
    if not current_user.is_authenticated:
        return None

    return (
        AnalysisResult.query
        .join(Instrument, AnalysisResult.instrument_id == Instrument.id)
        .filter(
            AnalysisResult.user_id == current_user.id,
            Instrument.user_id == current_user.id,
            Instrument.product_type == product_type,
            AnalysisResult.analysis_type == analysis_type,
        )
        .order_by(AnalysisResult.created_at.desc())
        .first()
    )


def _get_latest_pricing_result_for_product_types(product_types):
    if not current_user.is_authenticated:
        return None

    return (
        PricingResult.query
        .join(Instrument, PricingResult.instrument_id == Instrument.id)
        .filter(
            PricingResult.user_id == current_user.id,
            Instrument.user_id == current_user.id,
            Instrument.product_type.in_(product_types),
        )
        .order_by(PricingResult.created_at.desc())
        .first()
    )


def _get_latest_analysis_by_types_for_product_types(product_types, analysis_types):
    if not current_user.is_authenticated:
        return None

    return (
        AnalysisResult.query
        .join(Instrument, AnalysisResult.instrument_id == Instrument.id)
        .filter(
            AnalysisResult.user_id == current_user.id,
            Instrument.user_id == current_user.id,
            Instrument.product_type.in_(product_types),
            AnalysisResult.analysis_type.in_(analysis_types),
        )
        .order_by(AnalysisResult.created_at.desc())
        .first()
    )


def _get_latest_analysis_result_for_product_types(product_types):
    if not current_user.is_authenticated:
        return None

    return (
        AnalysisResult.query
        .join(Instrument, AnalysisResult.instrument_id == Instrument.id)
        .filter(
            AnalysisResult.user_id == current_user.id,
            Instrument.user_id == current_user.id,
            Instrument.product_type.in_(product_types),
        )
        .order_by(AnalysisResult.created_at.desc())
        .first()
    )


def _build_european_form_data_from_instrument(instrument):
    if not instrument:
        return {}

    params = instrument.params_json or {}
    return {
        "ticker": instrument.ticker or "",
        "strike_price": params.get("strike_price"),
        "start_date": instrument.start_date or "",
        "end_date": instrument.end_date or "",
        "risk_free_rate": params.get("risk_free_rate"),
        "volatility": params.get("volatility"),
        "option_type": params.get("option_type", ""),
        "model_type": params.get(
            "model_type", instrument.model_name or "black_scholes"
        ),
        "num_paths": params.get("num_paths", 10000),
        "num_steps": params.get("num_steps", 252),
    }


def _build_american_form_data_from_instrument(instrument):
    if not instrument:
        return {}

    params = instrument.params_json or {}
    return {
        "ticker": instrument.ticker or "",
        "strike_price": params.get("strike_price"),
        "start_date": instrument.start_date or "",
        "end_date": instrument.end_date or "",
        "r": params.get("risk_free_rate"),
        "sigma": params.get("volatility"),
        "option_type": params.get("option_type", ""),
        "num_steps": params.get("num_steps", 252),
        "pricing_model": params.get(
            "pricing_model",
            instrument.model_name or "Cox Ross Rubinstein Tree",
        ),
        "model": instrument.model_name or params.get("pricing_model"),
        "num_paths": params.get("num_paths", 10000),
        "mc_steps": params.get("mc_steps", 252),
        "dividends": params.get("dividends"),
    }


@vanilla_options_bp.route("/save-assessment", methods=["POST"])
def save_assessment():
    assessment_data = request.json.get("assessment")

    try:
        with open("derivapro/static/assessment.txt", "w") as f:
            f.write(assessment_data)
        return jsonify({"status": "success"})
    except Exception as e:
        logger.exception("Error saving assessment")
        return jsonify({"status": "error", "message": str(e)})


@vanilla_options_bp.route("/", methods=["GET", "POST"])
def vanilla_options():
    return render_template("vanilla_options.html")


@vanilla_options_bp.route("/european", methods=["GET", "POST"])
def european_options():
    readme_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "european_options.md"
    )
    with open(readme_path, "r") as readme_file:
        content = readme_file.read()
    md_content = markdown.markdown(content)

    form_data = {}

    option_price = None
    delta = None
    gamma = None
    vega = None
    theta = None
    rho = None

    sensitivity_results = None
    gpt_assessment = None
    latest_analysis = None
    latest_pricing_result = None

    if current_user.is_authenticated:
        latest_pricing_result = _get_latest_pricing_result_for_user("european_option")
        if latest_pricing_result:
            option_price = "${:,.4f}".format(float(latest_pricing_result.price))
            delta = "{:.4f}".format(float(latest_pricing_result.delta))
            gamma = "{:.4f}".format(float(latest_pricing_result.gamma))
            vega = "{:.4f}".format(float(latest_pricing_result.vega))
            theta = "{:.4f}".format(float(latest_pricing_result.theta))
            rho = "{:.4f}".format(float(latest_pricing_result.rho))

            if latest_pricing_result.instrument:
                form_data = _build_european_form_data_from_instrument(
                    latest_pricing_result.instrument
                )

        latest_analysis = _get_latest_analysis_by_type_for_user(
            "european_option",
            "sensitivity",
        )
        if latest_analysis and latest_analysis.result_json:
            sensitivity_results = latest_analysis.result_json
            gpt_assessment = latest_analysis.result_json.get(
                "gpt_sensitivity_assessment"
            )
    else:
        sensitivity_results = None
        if sensitivity_results:
            gpt_assessment = sensitivity_results.get("gpt_sensitivity_assessment")

    if request.method == "POST":
        logger.debug("POST request received for european options")
        action = request.form.get("analysis_type")
        logger.debug("European options action: %s", action)

        form_data = {
            "ticker": request.form.get("ticker", ""),
            "strike_price": request.form.get("strike_price", type=float),
            "start_date": request.form.get("start_date", ""),
            "end_date": request.form.get("end_date", ""),
            "risk_free_rate": request.form.get("risk_free_rate", type=float),
            "volatility": request.form.get("volatility", type=float),
            "option_type": request.form.get("option_type", ""),
            "model_type": request.form.get("model_type", "black_scholes"),
            "num_paths": request.form.get("num_paths", type=int, default=10000),
            "num_steps": request.form.get("num_steps", type=int, default=252),
        }

        session["european_form_data"] = {
            "ticker": form_data.get("ticker", ""),
            "strike_price": form_data.get("strike_price"),
            "start_date": form_data.get("start_date", ""),
            "end_date": form_data.get("end_date", ""),
            "risk_free_rate": form_data.get("risk_free_rate"),
            "volatility": form_data.get("volatility"),
            "option_type": form_data.get("option_type", "call"),
            "model_type": form_data.get("model_type", "black_scholes"),
            "num_paths": form_data.get("num_paths", 10000),
            "num_steps": form_data.get("num_steps", 252),
        }

        ticker = form_data["ticker"]
        strike_price = form_data["strike_price"]
        start_date = form_data["start_date"]
        end_date = form_data["end_date"]
        risk_free_rate = form_data["risk_free_rate"]
        volatility = form_data["volatility"]
        option_type = form_data["option_type"]

        logger.debug(
            "European options raw dates received: start_date=%s, end_date=%s",
            start_date,
            end_date,
        )

        try:
            start_date = datetime.strptime(start_date, "%Y-%m-%d").date()
            end_date = datetime.strptime(end_date, "%Y-%m-%d").date()
        except ValueError as e:
            logger.warning("European options date format error: %s", e)
            return render_template(
                "european_options.html",
                form_data=form_data,
                option_price=option_price,
                delta=delta,
                gamma=gamma,
                vega=vega,
                theta=theta,
                rho=rho,
                error=f"Date format error: {e}",
            )

        if action == "sensitivity":
            try:
                logger.debug("European options sensitivity analysis triggered")

                form_data["num_steps"] = int(request.form["num_steps"])
                form_data["step_range"] = float(request.form["step_range"])
                form_data["variable"] = request.form["variable"]
                form_data["target_variable"] = request.form["target_variable"]

                num_steps = form_data["num_steps"]
                step_range = form_data["step_range"]

                if step_range > 1:
                    step_range = step_range / 100.0
                    form_data["step_range"] = step_range

                variable = form_data["variable"]
                target_variable = form_data["target_variable"]

                logger.debug(
                    "European sensitivity parameters: num_steps=%s, step_range=%s, variable=%s",
                    num_steps,
                    step_range,
                    variable,
                )

                tester = SmoothnessTest(
                    ticker,
                    strike_price,
                    start_date,
                    end_date,
                    risk_free_rate,
                    volatility,
                    option_type,
                )

                logger.debug("Running European sensitivity analysis")
                values, delta, gamma, vega, theta, rho = (
                    tester.calculate_greeks_over_range(
                        variable, num_steps, step_range, target_variable
                    )
                )

                logger.debug("Plotting European sensitivity Greeks")
                tester.plot_greeks(values, delta, gamma, vega, theta, rho, variable)

                base_dir = os.path.dirname(os.path.abspath(__file__))
                static_dir = os.path.join(base_dir, "..", "static")
                os.makedirs(static_dir, exist_ok=True)

                plot_filename = f"{target_variable}-{variable}_sensitivity_plot_{uuid.uuid4().hex}.png"
                plot_path = os.path.join(static_dir, plot_filename)

                plt.savefig(plot_path)
                plt.close()

                logger.debug("European sensitivity plot saved to %s", plot_path)

                serialized_values = (
                    values.tolist() if hasattr(values, "tolist") else list(values)
                )

                sensitivity_results_data = {
                    "variable": variable,
                    "values": serialized_values,
                    "target_variable": target_variable,
                    "plot_filename": plot_filename,
                }

                if current_user.is_authenticated:
                    instrument = Instrument(
                        user_id=current_user.id,
                        product_type="european_option",
                        ticker=form_data["ticker"],
                        model_name=form_data.get("model_type", "black_scholes"),
                        start_date=str(form_data["start_date"]),
                        end_date=str(form_data["end_date"]),
                        params_json={
                            "strike_price": form_data["strike_price"],
                            "risk_free_rate": form_data["risk_free_rate"],
                            "volatility": form_data["volatility"],
                            "option_type": form_data["option_type"],
                        },
                    )
                    instrument, linked_pricing_result = (
                        _resolve_analysis_instrument(
                            "european_option",
                            instrument,
                        )
                    )

                    analysis_result = AnalysisResult(
                        user_id=current_user.id,
                        instrument_id=instrument.id,
                        pricing_result_id=(
                            linked_pricing_result.id if linked_pricing_result else None
                        ),
                        analysis_type="sensitivity",
                        result_json=sensitivity_results_data,
                    )
                    db.session.add(analysis_result)
                    db.session.commit()
                else:
                    sensitivity_results = sensitivity_results_data

                sensitivity_results = sensitivity_results_data

            except Exception:
                logger.exception(
                    "An error occurred during European sensitivity analysis"
                )
                sensitivity_results = None

        elif action == "ai_assessment":
            sensitivity_data = None

            if current_user.is_authenticated:
                latest_analysis = _get_latest_analysis_by_type_for_user(
                    "european_option",
                    "sensitivity",
                )

                if (
                    latest_analysis
                    and latest_analysis.analysis_type == "sensitivity"
                    and latest_analysis.result_json
                ):
                    sensitivity_data = latest_analysis.result_json
            else:
                sensitivity_data = sensitivity_results or {}

            if sensitivity_data:
                assessment_input = (
                    "Please assess the sensitivity analysis based on the outputs: "
                    f"{sensitivity_data}."
                )
                gpt_assessment = ask_gpt(assessment_input)

                if current_user.is_authenticated:
                    sensitivity_data["gpt_sensitivity_assessment"] = gpt_assessment
                    latest_analysis.result_json = sensitivity_data
                    db.session.commit()
                else:
                    sensitivity_data["gpt_sensitivity_assessment"] = gpt_assessment

                sensitivity_results = sensitivity_data
            else:
                gpt_assessment = (
                    "No sensitivity analysis results available for assessment."
                )

        else:
            model_type = form_data.get("model_type", "black_scholes")

            if model_type == "black_scholes":
                option = BlackScholes(
                    ticker,
                    strike_price,
                    start_date,
                    end_date,
                    risk_free_rate,
                    volatility,
                    option_type,
                )

                option_price = (
                    option.call_price() if option_type == "call" else option.put_price()
                )

                delta = "{:.4f}".format(option.delta())
                gamma = "{:.4f}".format(option.gamma())
                vega = "{:.4f}".format(option.vega())
                theta = "{:.4f}".format(option.theta())
                rho = "{:.4f}".format(option.rho())

            elif model_type == "monte_carlo":
                num_paths = form_data.get("num_paths", 10000)
                num_steps = form_data.get("num_steps", 252)

                mc_engine = monte_carlo_module.create_monte_carlo_engine(
                    S0=float(
                        StockData(ticker, start_date, end_date).get_closing_price()
                    ),
                    r=risk_free_rate,
                    sigma=volatility,
                    T=StockData(ticker, start_date, end_date).get_years_difference(),
                    num_paths=num_paths,
                    num_steps=num_steps,
                    random_type="sobol",
                )

                option_price = (
                    mc_engine.price_european_option(strike_price, "call")
                    if option_type == "call"
                    else mc_engine.price_european_option(strike_price, "put")
                )

                greeks = mc_engine.calculate_greeks_finite_difference(
                    strike_price, option_type, "european"
                )
                delta = "{:.4f}".format(greeks["Delta"])
                gamma = "{:.4f}".format(greeks["Gamma"])
                vega = "{:.4f}".format(greeks["Vega"])
                theta = "{:.4f}".format(greeks["Theta"])
                rho = "{:.4f}".format(greeks["Rho"])

            else:
                option = LatticeModel(
                    ticker,
                    strike_price,
                    start_date,
                    end_date,
                    risk_free_rate,
                    volatility,
                )

                if model_type == "Cox Ross Rubinstein Tree":
                    option_price = option.Cox_Ross_Rubinstein_Tree(
                        option_type, num_steps
                    )
                elif model_type == "Jarrow Rudd Tree":
                    option_price = option.Jarrow_Rudd_Tree(option_type, num_steps)
                elif model_type == "Trinomial Asset Pricing":
                    option_price = option.Trinomial_Asset_Pricing(
                        option_type, num_steps
                    )
                else:
                    option_price = option.Cox_Ross_Rubinstein_Tree(
                        option_type, num_steps
                    )

                if model_type == "Cox Ross Rubinstein Tree":
                    greeks = option.CRRGreeks(option_type, num_steps)
                elif model_type == "Jarrow Rudd Tree":
                    greeks = option.JRTGreeks(option_type, num_steps)
                elif model_type == "Trinomial Asset Pricing":
                    greeks = option.TAPGreeks(option_type, num_steps)
                else:
                    greeks = option.CRRGreeks(option_type, num_steps)

                delta = "{:.4f}".format(greeks["Delta"])
                gamma = "{:.4f}".format(greeks["Gamma"])
                vega = "{:.4f}".format(greeks["Vega"])
                theta = "{:.4f}".format(greeks["Theta"])
                rho = "{:.4f}".format(greeks["Rho"])

            raw_option_price = float(option_price)
            raw_delta = float(delta)
            raw_gamma = float(gamma)
            raw_vega = float(vega)
            raw_theta = float(theta)
            raw_rho = float(rho)

            option_price = "${:,.4f}".format(raw_option_price)
            delta = "{:.4f}".format(raw_delta)
            gamma = "{:.4f}".format(raw_gamma)
            vega = "{:.4f}".format(raw_vega)
            theta = "{:.4f}".format(raw_theta)
            rho = "{:.4f}".format(raw_rho)

            if current_user.is_authenticated:
                instrument = Instrument(
                    user_id=current_user.id,
                    product_type="european_option",
                    ticker=ticker,
                    model_name=model_type,
                    start_date=str(start_date),
                    end_date=str(end_date),
                    params_json={
                        "strike_price": strike_price,
                        "risk_free_rate": risk_free_rate,
                        "volatility": volatility,
                        "option_type": option_type,
                        "model_type": model_type,
                        "num_paths": form_data.get("num_paths"),
                        "num_steps": form_data.get("num_steps"),
                    },
                )
                db.session.add(instrument)
                db.session.flush()

                pricing_result = PricingResult(
                    user_id=current_user.id,
                    instrument_id=instrument.id,
                    price=raw_option_price,
                    delta=raw_delta,
                    gamma=raw_gamma,
                    vega=raw_vega,
                    theta=raw_theta,
                    rho=raw_rho,
                    result_json={
                        "option_price": raw_option_price,
                        "delta": raw_delta,
                        "gamma": raw_gamma,
                        "vega": raw_vega,
                        "theta": raw_theta,
                        "rho": raw_rho,
                    },
                )
                db.session.add(pricing_result)
                db.session.commit()

        return render_template(
            "european_options.html",
            form_data=form_data,
            option_price=option_price,
            delta=delta,
            gamma=gamma,
            vega=vega,
            theta=theta,
            rho=rho,
            sensitivity_results=sensitivity_results,
            gpt_assessment=gpt_assessment,
            md_content=md_content,
        )

    return render_template(
        "european_options.html",
        form_data=form_data,
        option_price=option_price,
        delta=delta,
        gamma=gamma,
        vega=vega,
        theta=theta,
        rho=rho,
        sensitivity_results=sensitivity_results,
        gpt_assessment=gpt_assessment,
        md_content=md_content,
    )


@vanilla_options_bp.route("/model-performance", methods=["GET", "POST"])
def model_performance():

    form_data = {}
    session_form_data = session.get("european_form_data", {})
    if isinstance(session_form_data, dict):
        form_data.update(session_form_data)

    sensitivity_results = None
    scenario_results = None
    convergence_results = None
    gpt_assessment = None
    gpt_scenario_assessment = None
    gpt_convergence_assessment = None
    error_message = None
    latest_analysis = None

    baseline_price = stressed_price = None
    baseline_delta = baseline_gamma = baseline_vega = baseline_theta = baseline_rho = (
        None
    )
    stressed_delta = stressed_gamma = stressed_vega = stressed_theta = stressed_rho = (
        None
    )

    if current_user.is_authenticated:
        latest_pricing_result = _get_latest_pricing_result_for_user("european_option")
        if latest_pricing_result and latest_pricing_result.instrument:
            form_data = _build_european_form_data_from_instrument(
                latest_pricing_result.instrument
            )

        latest_sensitivity_analysis = _get_latest_analysis_by_type_for_user(
            "european_option",
            "sensitivity",
        )
        latest_scenario_analysis = _get_latest_analysis_by_type_for_user(
            "european_option",
            "scenario",
        )
        latest_convergence_analysis = _get_latest_analysis_by_type_for_user(
            "european_option",
            "convergence",
        )

        if latest_sensitivity_analysis:
            sensitivity_results = latest_sensitivity_analysis.result_json or None

        if latest_scenario_analysis:
            scenario_results = latest_scenario_analysis.result_json or None

        if latest_convergence_analysis:
            convergence_results = latest_convergence_analysis.result_json or None

        latest_analysis = (
            latest_convergence_analysis
            or latest_scenario_analysis
            or latest_sensitivity_analysis
        )
    else:
        sensitivity_results = None
        scenario_results = None
        convergence_results = None

    if sensitivity_results:
        gpt_assessment = sensitivity_results.get("gpt_sensitivity_assessment")

    if scenario_results:
        gpt_scenario_assessment = scenario_results.get("gpt_scenario_assessment")
        baseline_table = scenario_results.get("baseline_scenario_table", {})
        stressed_table = scenario_results.get("stressed_scenario_table", {})
        baseline_price = baseline_table.get("baseline_price")
        baseline_delta = baseline_table.get("baseline_delta")
        baseline_gamma = baseline_table.get("baseline_gamma")
        baseline_vega = baseline_table.get("baseline_vega")
        baseline_theta = baseline_table.get("baseline_theta")
        baseline_rho = baseline_table.get("baseline_rho")
        stressed_price = stressed_table.get("stressed_price")
        stressed_delta = stressed_table.get("stressed_delta")
        stressed_gamma = stressed_table.get("stressed_gamma")
        stressed_vega = stressed_table.get("stressed_vega")
        stressed_theta = stressed_table.get("stressed_theta")
        stressed_rho = stressed_table.get("stressed_rho")

    if convergence_results:
        gpt_convergence_assessment = convergence_results.get(
            "gpt_convergence_assessment"
        )

    if request.method == "POST":
        action = request.form.get("analysis_type")

        # Merge hidden baseline fields from the posted form into form_data.
        baseline_keys = [
            "ticker",
            "strike_price",
            "start_date",
            "end_date",
            "risk_free_rate",
            "volatility",
            "option_type",
            "model",
            "model_type",
            "num_steps",
        ]
        for key in baseline_keys:
            raw_value = request.form.get(key)
            if raw_value not in (None, ""):
                form_data[key] = raw_value

        # Normalize numeric baseline inputs used by all analysis tabs.
        for float_key in ["strike_price", "risk_free_rate", "volatility"]:
            if float_key in form_data and isinstance(form_data[float_key], str):
                try:
                    form_data[float_key] = float(form_data[float_key])
                except ValueError:
                    pass

        if "num_steps" in form_data and isinstance(form_data["num_steps"], str):
            try:
                form_data["num_steps"] = int(form_data["num_steps"])
            except ValueError:
                pass

        # Persist the latest usable baseline for anonymous users.
        session["european_form_data"] = {
            "ticker": form_data.get("ticker", ""),
            "strike_price": form_data.get("strike_price"),
            "start_date": form_data.get("start_date", ""),
            "end_date": form_data.get("end_date", ""),
            "risk_free_rate": form_data.get("risk_free_rate"),
            "volatility": form_data.get("volatility"),
            "option_type": form_data.get("option_type", "call"),
            "model_type": form_data.get("model_type", "black_scholes"),
            "num_steps": form_data.get("num_steps", 252),
        }

        if action == "sensitivity":
            required_fields = [
                "ticker",
                "strike_price",
                "start_date",
                "end_date",
                "risk_free_rate",
                "volatility",
                "option_type",
            ]
            missing = [
                field for field in required_fields if form_data.get(field) in (None, "")
            ]
            if missing:
                error_message = (
                    "Run option pricing first (European page) so baseline inputs are available "
                    f"for sensitivity analysis. Missing: {', '.join(missing)}"
                )
                logger.warning(
                    "Model performance sensitivity skipped due to missing fields: %s",
                    ", ".join(missing),
                )
            else:
                try:
                    form_data["num_steps"] = int(request.form["num_steps"])
                    form_data["step_range"] = float(request.form["step_range"])
                    form_data["variable"] = request.form["variable"]
                    form_data["target_variable"] = request.form["target_variable"]

                    num_steps = form_data["num_steps"]
                    step_range = form_data["step_range"]

                    if step_range > 1:
                        step_range = step_range / 100.0
                        form_data["step_range"] = step_range

                    variable = form_data["variable"]
                    target_variable = form_data["target_variable"]

                    tester = SmoothnessTest(
                        form_data["ticker"],
                        float(form_data["strike_price"]),
                        form_data["start_date"],
                        form_data["end_date"],
                        float(form_data["risk_free_rate"]),
                        float(form_data["volatility"]),
                        form_data["option_type"],
                    )

                    values, greek_values = tester.calculate_greeks_over_range(
                        variable, num_steps, step_range, target_variable
                    )
                    tester.plot_single_greek(
                        values, greek_values, target_variable, variable
                    )

                    base_dir = os.path.dirname(os.path.abspath(__file__))
                    static_dir = os.path.join(base_dir, "..", "static")
                    os.makedirs(static_dir, exist_ok=True)

                    plot_filename = f"european_{target_variable}-{variable}_sensitivity_plot_{uuid.uuid4().hex}.png"
                    plot_path = os.path.join(static_dir, plot_filename)
                    plt.savefig(plot_path)
                    plt.close()

                    serialized_values = (
                        values.tolist() if hasattr(values, "tolist") else list(values)
                    )
                    serialized_greek_values = [
                        float(v) if isinstance(v, (np.floating, np.integer)) else v
                        for v in greek_values
                    ]

                    sensitivity_results_data = {
                        "variable": variable,
                        "values": serialized_values,
                        "greek_values": serialized_greek_values,
                        "target_variable": target_variable,
                        "plot_filename": plot_filename,
                    }

                    if current_user.is_authenticated:
                        instrument = Instrument(
                            user_id=current_user.id,
                            product_type="european_option",
                            ticker=form_data["ticker"],
                            model_name=form_data.get("model_type", "black_scholes"),
                            start_date=str(form_data["start_date"]),
                            end_date=str(form_data["end_date"]),
                            params_json={
                                "strike_price": form_data["strike_price"],
                                "risk_free_rate": form_data["risk_free_rate"],
                                "volatility": form_data["volatility"],
                                "option_type": form_data["option_type"],
                            },
                        )
                        instrument, linked_pricing_result = (
                            _resolve_analysis_instrument(
                                "european_option",
                                instrument,
                            )
                        )

                        analysis_result = AnalysisResult(
                            user_id=current_user.id,
                            instrument_id=instrument.id,
                            pricing_result_id=(
                                linked_pricing_result.id
                                if linked_pricing_result
                                else None
                            ),
                            analysis_type="sensitivity",
                            result_json=sensitivity_results_data,
                        )
                        db.session.add(analysis_result)
                        db.session.commit()
                        latest_analysis = analysis_result

                    sensitivity_results = sensitivity_results_data
                    gpt_assessment = sensitivity_results_data.get(
                        "gpt_sensitivity_assessment"
                    )

                except Exception:
                    logger.exception(
                        "An error occurred during model performance sensitivity analysis"
                    )
                    sensitivity_results = None

        elif action == "ai_sensitivity_assessment":
            sensitivity_data = sensitivity_results

            if current_user.is_authenticated and latest_analysis:
                if latest_analysis.analysis_type == "sensitivity":
                    sensitivity_data = latest_analysis.result_json or {}
                else:
                    sensitivity_data = {}
            elif sensitivity_data is None:
                sensitivity_data = sensitivity_results or {} or {}

            if sensitivity_data:
                variable = sensitivity_data.get("variable")
                values = sensitivity_data.get("values")
                target_variable = sensitivity_data.get("target_variable")

                if values and variable and target_variable:
                    sensitivity_results_text = (
                        f"Sensitivity Analysis Results for {target_variable} with respect to {variable}:\n"
                        + "\n".join([f"{variable}={v}" for v in values])
                    )
                    gpt_assessment = ask_gpt(
                        f"Please assess the sensitivity analysis based on the outputs: {sensitivity_results_text}."
                    )
                else:
                    gpt_assessment = (
                        "Incomplete sensitivity analysis data available for assessment."
                    )

                sensitivity_data["gpt_sensitivity_assessment"] = gpt_assessment

                if current_user.is_authenticated and latest_analysis:
                    latest_analysis.result_json = sensitivity_data
                    db.session.commit()
                else:
                    sensitivity_results = sensitivity_data
            else:
                gpt_assessment = (
                    "No sensitivity analysis results available for assessment."
                )
                sensitivity_results = {"gpt_sensitivity_assessment": gpt_assessment}
                if current_user.is_authenticated and latest_analysis:
                    latest_analysis.result_json = sensitivity_results
                    db.session.commit()
        elif action == "scenario":
            required_fields = [
                "ticker",
                "strike_price",
                "start_date",
                "end_date",
                "risk_free_rate",
                "volatility",
                "option_type",
            ]
            missing = [
                field for field in required_fields if form_data.get(field) in (None, "")
            ]
            if missing:
                error_message = (
                    "Run option pricing first (European page) so baseline inputs are available "
                    f"for scenario analysis. Missing: {', '.join(missing)}"
                )
                logger.warning(
                    "Model performance scenario skipped due to missing fields: %s",
                    ", ".join(missing),
                )
            else:
                try:
                    spot_change = float(request.form.get("spot_scenario", 0))
                    vol_change = float(request.form.get("vol_scenario", 0))
                    rate_change = float(request.form.get("rate_scenario", 0))

                    ticker = form_data.get("ticker")
                    strike_price = float(form_data.get("strike_price"))
                    start_date = datetime.strptime(
                        form_data.get("start_date"), "%Y-%m-%d"
                    ).date()
                    end_date = datetime.strptime(
                        form_data.get("end_date"), "%Y-%m-%d"
                    ).date()
                    risk_free_rate = float(form_data.get("risk_free_rate"))
                    volatility = float(form_data.get("volatility"))
                    option_type = form_data.get("option_type")
                    model_name = (
                        form_data.get("model")
                        or form_data.get("model_type")
                        or "black_scholes"
                    )
                    num_steps = int(form_data.get("num_steps", 252))

                    if model_name in {
                        "black_scholes",
                        "monte_carlo",
                        "Monte_Carlo",
                        "Monte Carlo",
                    }:
                        baseline_option = BlackScholes(
                            ticker,
                            strike_price,
                            start_date,
                            end_date,
                            risk_free_rate,
                            volatility,
                            option_type,
                        )
                        baseline_price = (
                            baseline_option.call_price()
                            if option_type.lower() == "call"
                            else baseline_option.put_price()
                        )
                        baseline_greeks = {
                            "Delta": baseline_option.delta(),
                            "Gamma": baseline_option.gamma(),
                            "Vega": baseline_option.vega(),
                            "Theta": baseline_option.theta(),
                            "Rho": baseline_option.rho(),
                        }
                    else:
                        option = LatticeModel(
                            ticker,
                            strike_price,
                            start_date,
                            end_date,
                            risk_free_rate,
                            volatility,
                        )
                        if model_name == "Cox Ross Rubinstein Tree":
                            baseline_price = option.Cox_Ross_Rubinstein_Tree(
                                option_type, num_steps
                            )
                            baseline_greeks = option.CRRGreeks(option_type, num_steps)
                        elif model_name == "Jarrow Rudd Tree":
                            baseline_price = option.Jarrow_Rudd_Tree(
                                option_type, num_steps
                            )
                            baseline_greeks = option.JRTGreeks(option_type, num_steps)
                        else:
                            baseline_price = option.Trinomial_Asset_Pricing(
                                option_type, num_steps
                            )
                            baseline_greeks = option.TAPGreeks(option_type, num_steps)

                    baseline_price = "{:.4f}".format(baseline_price)
                    baseline_delta = "{:.4f}".format(baseline_greeks["Delta"])
                    baseline_gamma = "{:.4f}".format(baseline_greeks["Gamma"])
                    baseline_vega = "{:.4f}".format(baseline_greeks["Vega"])
                    baseline_theta = "{:.4f}".format(baseline_greeks["Theta"])
                    baseline_rho = "{:.4f}".format(baseline_greeks["Rho"])

                    stressed_spot = strike_price * (1 + spot_change)
                    stressed_vol = volatility + vol_change
                    stressed_rate = risk_free_rate + rate_change

                    if model_name in {
                        "black_scholes",
                        "monte_carlo",
                        "Monte_Carlo",
                        "Monte Carlo",
                    }:
                        stressed_bs = BlackScholes(
                            ticker,
                            stressed_spot,
                            start_date,
                            end_date,
                            stressed_rate,
                            stressed_vol,
                            option_type,
                        )
                        stressed_price = (
                            stressed_bs.call_price()
                            if option_type.lower() == "call"
                            else stressed_bs.put_price()
                        )
                        stressed_greeks = {
                            "Delta": stressed_bs.delta(),
                            "Gamma": stressed_bs.gamma(),
                            "Vega": stressed_bs.vega(),
                            "Theta": stressed_bs.theta(),
                            "Rho": stressed_bs.rho(),
                        }
                    else:
                        stressed_option = LatticeModel(
                            ticker,
                            stressed_spot,
                            start_date,
                            end_date,
                            stressed_rate,
                            stressed_vol,
                        )
                        if model_name == "Cox Ross Rubinstein Tree":
                            stressed_price = stressed_option.Cox_Ross_Rubinstein_Tree(
                                option_type, num_steps, greeks=False
                            )
                            stressed_greeks = stressed_option.CRRGreeks(
                                option_type, num_steps
                            )
                        elif model_name == "Jarrow Rudd Tree":
                            stressed_price = stressed_option.Jarrow_Rudd_Tree(
                                option_type, num_steps, greeks=False
                            )
                            stressed_greeks = stressed_option.JRTGreeks(
                                option_type, num_steps
                            )
                        else:
                            stressed_price = stressed_option.Trinomial_Asset_Pricing(
                                option_type, num_steps
                            )
                            stressed_greeks = stressed_option.TAPGreeks(
                                option_type, num_steps
                            )

                    stressed_price = "{:.4f}".format(stressed_price)
                    stressed_delta = "{:.4f}".format(stressed_greeks["Delta"])
                    stressed_gamma = "{:.4f}".format(stressed_greeks["Gamma"])
                    stressed_vega = "{:.4f}".format(stressed_greeks["Vega"])
                    stressed_theta = "{:.4f}".format(stressed_greeks["Theta"])
                    stressed_rho = "{:.4f}".format(stressed_greeks["Rho"])

                    scenario_results_data = {
                        "baseline_scenario_table": {
                            "scenario": "Baseline",
                            "baseline_price": baseline_price,
                            "baseline_delta": baseline_delta,
                            "baseline_gamma": baseline_gamma,
                            "baseline_vega": baseline_vega,
                            "baseline_theta": baseline_theta,
                            "baseline_rho": baseline_rho,
                        },
                        "stressed_scenario_table": {
                            "scenario": "Stressed",
                            "stressed_price": stressed_price,
                            "stressed_delta": stressed_delta,
                            "stressed_gamma": stressed_gamma,
                            "stressed_vega": stressed_vega,
                            "stressed_theta": stressed_theta,
                            "stressed_rho": stressed_rho,
                        },
                        "scenario_inputs": {
                            "spot_change": spot_change,
                            "vol_change": vol_change,
                            "rate_change": rate_change,
                        },
                        "gpt_scenario_assessment": "No assessment yet.",
                    }

                    if current_user.is_authenticated:
                        instrument = Instrument(
                            user_id=current_user.id,
                            product_type="european_option",
                            ticker=ticker,
                            model_name=model_name,
                            start_date=str(form_data.get("start_date")),
                            end_date=str(form_data.get("end_date")),
                            params_json={
                                "strike_price": strike_price,
                                "risk_free_rate": risk_free_rate,
                                "volatility": volatility,
                                "option_type": option_type,
                                "num_steps": num_steps,
                            },
                        )
                        instrument, linked_pricing_result = (
                            _resolve_analysis_instrument(
                                "european_option",
                                instrument,
                            )
                        )

                        analysis_result = AnalysisResult(
                            user_id=current_user.id,
                            instrument_id=instrument.id,
                            pricing_result_id=(
                                linked_pricing_result.id
                                if linked_pricing_result
                                else None
                            ),
                            analysis_type="scenario",
                            result_json=scenario_results_data,
                        )
                        db.session.add(analysis_result)
                        db.session.commit()
                        latest_analysis = analysis_result

                    scenario_results = scenario_results_data
                    gpt_scenario_assessment = scenario_results_data.get(
                        "gpt_scenario_assessment"
                    )
                    baseline_price = scenario_results_data["baseline_scenario_table"][
                        "baseline_price"
                    ]
                    baseline_delta = scenario_results_data["baseline_scenario_table"][
                        "baseline_delta"
                    ]
                    baseline_gamma = scenario_results_data["baseline_scenario_table"][
                        "baseline_gamma"
                    ]
                    baseline_vega = scenario_results_data["baseline_scenario_table"][
                        "baseline_vega"
                    ]
                    baseline_theta = scenario_results_data["baseline_scenario_table"][
                        "baseline_theta"
                    ]
                    baseline_rho = scenario_results_data["baseline_scenario_table"][
                        "baseline_rho"
                    ]
                    stressed_price = scenario_results_data["stressed_scenario_table"][
                        "stressed_price"
                    ]
                    stressed_delta = scenario_results_data["stressed_scenario_table"][
                        "stressed_delta"
                    ]
                    stressed_gamma = scenario_results_data["stressed_scenario_table"][
                        "stressed_gamma"
                    ]
                    stressed_vega = scenario_results_data["stressed_scenario_table"][
                        "stressed_vega"
                    ]
                    stressed_theta = scenario_results_data["stressed_scenario_table"][
                        "stressed_theta"
                    ]
                    stressed_rho = scenario_results_data["stressed_scenario_table"][
                        "stressed_rho"
                    ]

                except Exception:
                    logger.exception("An error occurred during scenario analysis")
                    scenario_results = None

        elif action == "ai_scenario_assessment":
            scenario_data = scenario_results

            if current_user.is_authenticated and latest_analysis:
                if latest_analysis.analysis_type == "scenario":
                    scenario_data = latest_analysis.result_json or {}
                else:
                    scenario_data = {}
            elif scenario_data is None:
                scenario_data = scenario_results or {}

            baseline_table = scenario_data.get("baseline_scenario_table", {})
            stressed_table = scenario_data.get("stressed_scenario_table", {})

            if baseline_table and stressed_table:
                table_text = f"""
                Baseline Scenario:
                Option Price={baseline_table["baseline_price"]}, Delta={baseline_table["baseline_delta"]},
                Gamma={baseline_table["baseline_gamma"]}, Vega={baseline_table["baseline_vega"]},
                Theta={baseline_table["baseline_theta"]}, Rho={baseline_table["baseline_rho"]}

                Stressed Scenario:
                Option Price={stressed_table["stressed_price"]}, Delta={stressed_table["stressed_delta"]},
                Gamma={stressed_table["stressed_gamma"]}, Vega={stressed_table["stressed_vega"]},
                Theta={stressed_table["stressed_theta"]}, Rho={stressed_table["stressed_rho"]}
                """
                gpt_scenario_assessment = ask_gpt(
                    "Please assess the scenario analysis of the option price and Greeks based on the following results: "
                    f"{table_text}. Please limit the assessment to be less than 100 words."
                )
            else:
                gpt_scenario_assessment = (
                    "No scenario analysis results available for assessment."
                )

            scenario_data["gpt_scenario_assessment"] = gpt_scenario_assessment

            if current_user.is_authenticated and latest_analysis:
                latest_analysis.result_json = scenario_data
                db.session.commit()
            else:
                pass

            scenario_results = scenario_data

        elif action == "convergence":
            required_fields = [
                "ticker",
                "strike_price",
                "start_date",
                "end_date",
                "risk_free_rate",
                "volatility",
                "option_type",
            ]
            missing = [
                field for field in required_fields if form_data.get(field) in (None, "")
            ]
            if missing:
                error_message = (
                    "Run option pricing first (European page) so baseline inputs are available "
                    f"for convergence analysis. Missing: {', '.join(missing)}"
                )
                logger.warning(
                    "Model performance convergence skipped due to missing fields: %s",
                    ", ".join(missing),
                )
            else:
                try:
                    model_type = request.form.get("model_type", "monte_carlo")
                    num_mc_paths = int(request.form.get("num_mc_paths", 10000))
                    num_mc_steps = int(request.form.get("num_mc_steps", 252))
                    obs = int(request.form.get("obs", 10))

                    ticker = form_data.get("ticker", "")
                    strike_price = float(form_data.get("strike_price", 100))
                    start_date = form_data.get("start_date", "")
                    end_date = form_data.get("end_date", "")
                    r = float(form_data.get("risk_free_rate", 0.01))
                    sigma = float(form_data.get("volatility", 0.2))
                    option_type = form_data.get("option_type", "call")

                    # Validate dates before using StockData to avoid NaT/strftime failures.
                    datetime.strptime(start_date, "%Y-%m-%d")
                    datetime.strptime(end_date, "%Y-%m-%d")

                    S0 = float(
                        StockData(ticker, start_date, end_date).get_closing_price()
                    )
                    T = StockData(ticker, start_date, end_date).get_years_difference()

                    mc_results = []
                    paths_range = (
                        np.linspace(800, num_mc_paths, obs).round().astype(int)
                    )
                    for n_paths in paths_range:
                        mc_engine = monte_carlo_module.create_monte_carlo_engine(
                            S0=S0,
                            r=r,
                            sigma=sigma,
                            T=T,
                            num_paths=int(n_paths),
                            num_steps=num_mc_steps,
                            random_type="sobol",
                        )
                        if option_type == "call":
                            price = mc_engine.price_european_option(
                                strike_price, "call"
                            )
                        else:
                            price = mc_engine.price_european_option(strike_price, "put")
                        mc_results.append((int(n_paths), float(price)))

                    plot_convergence(mc_results, mode="simulations")
                    plot_filename = f"vanilla_convergence_plot_{uuid.uuid4().hex}.png"
                    plot_path = os.path.join("derivapro", "static", plot_filename)
                    plt.savefig(plot_path)
                    plt.close()

                    convergence_results_data = {
                        "results": mc_results,
                        "mode": "simulations",
                        "plot_filename": plot_filename,
                        "model_type": model_type,
                        "num_mc_paths": num_mc_paths,
                        "num_mc_steps": num_mc_steps,
                        "obs": obs,
                    }

                    if current_user.is_authenticated:
                        instrument = Instrument(
                            user_id=current_user.id,
                            product_type="european_option",
                            ticker=ticker,
                            model_name=model_type,
                            start_date=str(start_date),
                            end_date=str(end_date),
                            params_json={
                                "strike_price": strike_price,
                                "risk_free_rate": r,
                                "volatility": sigma,
                                "option_type": option_type,
                            },
                        )
                        instrument, linked_pricing_result = (
                            _resolve_analysis_instrument(
                                "european_option",
                                instrument,
                            )
                        )

                        analysis_result = AnalysisResult(
                            user_id=current_user.id,
                            instrument_id=instrument.id,
                            pricing_result_id=(
                                linked_pricing_result.id
                                if linked_pricing_result
                                else None
                            ),
                            analysis_type="convergence",
                            result_json=convergence_results_data,
                        )
                        db.session.add(analysis_result)
                        db.session.commit()
                        session.pop("convergence_results", None)
                        latest_analysis = analysis_result

                    convergence_results = convergence_results_data
                    gpt_convergence_assessment = convergence_results_data.get(
                        "gpt_convergence_assessment"
                    )

                except Exception:
                    logger.exception(
                        "An error occurred during model performance convergence analysis"
                    )
                    convergence_results = None

        elif action == "ai_convergence_assessment":
            convergence_data = convergence_results

            if current_user.is_authenticated and latest_analysis:
                if latest_analysis.analysis_type == "convergence":
                    convergence_data = latest_analysis.result_json or {}
                else:
                    convergence_data = {}
            elif convergence_data is None:
                convergence_data = convergence_results or {}

            results = convergence_data.get("results")
            mode = convergence_data.get("mode")

            if results:
                convergence_text = f"""
                Convergence Analysis Results:
                Mode: {mode}
                Results: {results}
                """
                gpt_convergence_assessment = ask_gpt(
                    "Please assess the convergence analysis results based on the following data: "
                    f"{convergence_text}. Focus on the convergence behavior and any potential issues or recommendations. "
                    "Please limit the assessment to be less than 100 words."
                )
            else:
                gpt_convergence_assessment = (
                    "No convergence analysis results available for assessment."
                )

            convergence_data["gpt_convergence_assessment"] = gpt_convergence_assessment

            if current_user.is_authenticated and latest_analysis:
                latest_analysis.result_json = convergence_data
                db.session.commit()
            else:
                pass

            convergence_results = convergence_data

    return render_template(
        "model_performance.html",
        form_data=form_data,
        error_message=error_message,
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
        convergence_results=convergence_results,
        gpt_scenario_assessment=gpt_scenario_assessment,
        gpt_convergence_assessment=gpt_convergence_assessment,
        random=random,
    )


@vanilla_options_bp.route("/go-back", methods=["GET"])
def go_back():
    return redirect(url_for("vanilla_options.european_options"))


@vanilla_options_bp.route("/model-governance", methods=["GET", "POST"])
def model_governance():
    md_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "model_governance.md"
    )
    with open(md_path, "r", encoding="utf-8") as file:
        md_content = file.read()
        html_content = markdown.markdown(md_content)

    gpt_assessment = None

    if request.method == "POST":
        action = request.form.get("analysis_type")

        if action == "ai_assessment":
            prompt = (
                "Please assess the reasonableness and comprehensiveness of the "
                f"model governance for the vanilla option model as shown in the following text:\n\n{md_content}"
            )
            gpt_assessment = ask_gpt(prompt)

    return render_template(
        "model_governance.html", md_content=html_content, gpt_assessment=gpt_assessment
    )


@vanilla_options_bp.route("/ongoing-monitoring", methods=["GET"])
def ongoing_monitoring():
    md_file_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "ongoing_monitoring.md"
    )

    with open(md_file_path, "r", encoding="utf-8") as md_file:
        md_content = markdown.markdown(md_file.read())

    return render_template("ongoing_monitoring.html", md_content=md_content)


@vanilla_options_bp.route("/american", methods=["GET", "POST"])
def american_options():
    action = None
    readme_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "american_options.md"
    )
    with open(readme_path, "r") as readme_file:
        content = readme_file.read()
    md_content = markdown.markdown(content)

    delta = None
    gamma = None
    vega = None
    theta = None
    rho = None

    option_price = convergence_results = sensitivity_results = scenario_results = (
        risk_pl_results
    ) = None
    gpt_sensitivity_assessment = None
    gpt_scenario_assessment = None
    gpt_convergence_assessment = None
    gpt_rpbl_assessment = None
    sensitivity_error = None
    latest_analysis = None

    form_data = {}

    if current_user.is_authenticated:

        latest_pricing_result = _get_latest_pricing_result_for_user("american_option")
        if latest_pricing_result:
            option_price = "${:,.4f}".format(float(latest_pricing_result.price))
            delta = "{:.4f}".format(float(latest_pricing_result.delta))
            gamma = "{:.4f}".format(float(latest_pricing_result.gamma))
            vega = "{:.4f}".format(float(latest_pricing_result.vega))
            theta = "{:.4f}".format(float(latest_pricing_result.theta))
            rho = "{:.4f}".format(float(latest_pricing_result.rho))

            if latest_pricing_result.instrument:
                form_data = _build_american_form_data_from_instrument(
                    latest_pricing_result.instrument
                )

        latest_sensitivity_analysis = _get_latest_analysis_by_type_for_user(
            "american_option",
            "sensitivity",
        )
        latest_scenario_analysis = _get_latest_analysis_by_type_for_user(
            "american_option",
            "scenario",
        )
        latest_convergence_analysis = _get_latest_analysis_by_type_for_user(
            "american_option",
            "convergence",
        )
        latest_risk_pl_analysis = _get_latest_analysis_by_type_for_user(
            "american_option",
            "risk_pl",
        )

        if latest_sensitivity_analysis and latest_sensitivity_analysis.result_json:
            sensitivity_results = latest_sensitivity_analysis.result_json

        if latest_scenario_analysis and latest_scenario_analysis.result_json:
            scenario_results = latest_scenario_analysis.result_json

        if latest_convergence_analysis and latest_convergence_analysis.result_json:
            convergence_results = latest_convergence_analysis.result_json

        if latest_risk_pl_analysis and latest_risk_pl_analysis.result_json:
            risk_pl_results = latest_risk_pl_analysis.result_json

        latest_analysis = (
            latest_risk_pl_analysis
            or latest_convergence_analysis
            or latest_scenario_analysis
            or latest_sensitivity_analysis
        )

    else:
        sensitivity_results = None
        scenario_results = None
        convergence_results = None
        risk_pl_results = None

    if sensitivity_results:
        gpt_sensitivity_assessment = sensitivity_results.get(
            "gpt_sensitivity_assessment"
        )

        if scenario_results:
            gpt_scenario_assessment = scenario_results.get("gpt_scenario_assessment")

    if convergence_results:
        gpt_convergence_assessment = convergence_results.get(
            "gpt_convergence_assessment"
        )

    if risk_pl_results:
        gpt_rpbl_assessment = risk_pl_results.get("gpt_rpbl_assessment")

    if request.method == "POST":
        action = request.form.get("analysis_type")

        def safe_int(val, default):
            try:
                val = str(val)
                if val.strip() == "":
                    return default
                return int(val)
            except (ValueError, TypeError):
                return default

        selected_model = (
            request.form.get("pricing_model")
            or request.form.get("model")
            or "Cox Ross Rubinstein Tree"
        )


        form_data = {
            "ticker": request.form["ticker"],
            "strike_price": float(request.form["strike_price"]),
            "start_date": str(request.form["start_date"]),
            "end_date": str(request.form["end_date"]),
            "r": float(request.form["r"]),
            "sigma": float(request.form["sigma"]),
            "option_type": request.form["option_type"],
            "num_steps": safe_int(
                request.form.get("num_steps", request.form.get("mc_steps", 252)), 252
            ),
            "pricing_model": selected_model,
            "model": selected_model,
            "num_paths": safe_int(request.form.get("num_paths"), 10000),
            "mc_steps": safe_int(request.form.get("mc_steps"), 252),
        }


        ticker = form_data["ticker"]
        strike_price = form_data["strike_price"]
        start_date = form_data["start_date"]
        end_date = form_data["end_date"]
        risk_free_rate = form_data["r"]
        volatility = form_data["sigma"]
        option_type = form_data["option_type"]
        num_steps = form_data["num_steps"]
        model_name = form_data["model"]
        pricing_model = form_data["pricing_model"]

        if pricing_model == "Monte Carlo":
            num_paths = form_data.get("num_paths", 10000)
            mc_steps = form_data.get("mc_steps", 252)

            mc_engine = monte_carlo_module.create_monte_carlo_engine(
                S0=float(StockData(ticker, start_date, end_date).get_closing_price()),
                r=risk_free_rate,
                sigma=volatility,
                T=StockData(ticker, start_date, end_date).get_years_difference(),
                num_paths=num_paths,
                num_steps=mc_steps,
                random_type="sobol",
            )

            payoff_func = (
                (lambda S: np.maximum(S - strike_price, 0))
                if option_type == "call"
                else (lambda S: np.maximum(strike_price - S, 0))
            )
            lsmc_engine = monte_carlo_module.LSMCEngine(mc_engine)
            option_price = lsmc_engine.price_option(payoff_func, option_type)

            greeks = mc_engine.calculate_greeks_finite_difference(
                strike_price, option_type, "american"
            )
            delta = "{:.4f}".format(greeks["Delta"])
            gamma = "{:.4f}".format(greeks["Gamma"])
            vega = "{:.4f}".format(greeks["Vega"])
            theta = "{:.4f}".format(greeks["Theta"])
            rho = "{:.4f}".format(greeks["Rho"])

        elif pricing_model == "Binomial Tree":
            raw_dividends = request.form.get("dividends", "").strip()
            parsed_dividends = []
            if raw_dividends:
                for entry in raw_dividends.split(","):
                    parts = entry.strip().split(":")
                    if len(parts) == 2:
                        parsed_dividends.append((parts[0], float(parts[1])))
                    elif len(parts) == 3:
                        parsed_dividends.append((
                            parts[0],
                            float(parts[1]),
                            float(parts[2]),
                        ))

            engine = BinomialTreeEngineCRR(
                ticker=ticker,
                strike_price=strike_price,
                start_date=start_date,
                end_date=end_date,
                risk_free_rate=risk_free_rate,
                volatility=volatility,
                num_steps=num_steps,
                option_type=option_type,
                dividends=parsed_dividends,
            )
            option_price = engine.price_american_option()
            greeks = engine.get_greeks()

            delta = "{:.4f}".format(greeks["delta"])
            gamma = "{:.4f}".format(greeks["gamma"])
            vega = "{:.4f}".format(greeks["vega"])
            theta = "{:.4f}".format(greeks["theta"])
            rho = "{:.4f}".format(greeks["rho"])

            form_data["dividends"] = raw_dividends

        else:
            option = LatticeModel(
                ticker, strike_price, start_date, end_date, risk_free_rate, volatility
            )

            if pricing_model == "Cox Ross Rubinstein Tree":
                option_price = option.Cox_Ross_Rubinstein_Tree(option_type, num_steps)
            elif pricing_model == "Jarrow Rudd Tree":
                option_price = option.Jarrow_Rudd_Tree(option_type, num_steps)
            elif pricing_model == "Trinomial Asset Pricing":
                option_price = option.Trinomial_Asset_Pricing(option_type, num_steps)
            else:
                option_price = option.Cox_Ross_Rubinstein_Tree(option_type, num_steps)

            if pricing_model == "Cox Ross Rubinstein Tree":
                greeks = option.CRRGreeks(option_type, num_steps)
            elif pricing_model == "Jarrow Rudd Tree":
                greeks = option.JRTGreeks(option_type, num_steps)
            elif pricing_model == "Trinomial Asset Pricing":
                greeks = option.TAPGreeks(option_type, num_steps)
            else:
                greeks = option.CRRGreeks(option_type, num_steps)

            delta = "{:.4f}".format(greeks["Delta"])
            gamma = "{:.4f}".format(greeks["Gamma"])
            vega = "{:.4f}".format(greeks["Vega"])
            theta = "{:.4f}".format(greeks["Theta"])
            rho = "{:.4f}".format(greeks["Rho"])

        option_price = "${:,.4f}".format(option_price)
        raw_option_price = float(option_price.replace("$", "").replace(",", ""))
        raw_delta = float(delta)
        raw_gamma = float(gamma)
        raw_vega = float(vega)
        raw_theta = float(theta)
        raw_rho = float(rho)

        if current_user.is_authenticated:
            instrument = Instrument(
                user_id=current_user.id,
                product_type="american_option",
                ticker=ticker,
                model_name=pricing_model,
                start_date=str(start_date),
                end_date=str(end_date),
                params_json={
                    "strike_price": strike_price,
                    "risk_free_rate": risk_free_rate,
                    "volatility": volatility,
                    "option_type": option_type,
                    "pricing_model": pricing_model,
                    "num_steps": form_data.get("num_steps"),
                    "num_paths": form_data.get("num_paths"),
                    "mc_steps": form_data.get("mc_steps"),
                    "dividends": form_data.get("dividends"),
                },
            )
            db.session.add(instrument)
            db.session.flush()

            pricing_result = PricingResult(
                user_id=current_user.id,
                instrument_id=instrument.id,
                price=raw_option_price,
                delta=raw_delta,
                gamma=raw_gamma,
                vega=raw_vega,
                theta=raw_theta,
                rho=raw_rho,
                result_json={
                    "option_price": raw_option_price,
                    "delta": raw_delta,
                    "gamma": raw_gamma,
                    "vega": raw_vega,
                    "theta": raw_theta,
                    "rho": raw_rho,
                },
            )
            db.session.add(pricing_result)
            db.session.commit()

        if action == "sensitivity":
            try:
                form_data["num_sensitivity_steps"] = int(
                    request.form["num_sensitivity_steps"]
                )
                form_data["step_range"] = float(request.form["step_range"])
                form_data["variable"] = request.form["variable"]
                form_data["target_variable"] = request.form["target_variable"]

                num_steps = form_data["num_sensitivity_steps"]
                step_range = form_data["step_range"]
                variable = form_data["variable"]
                target_variable = form_data["target_variable"]

                if num_steps < 2:
                    raise ValueError("Number of sensitivity steps must be at least 2.")
                if step_range == 0:
                    raise ValueError("Step range must be greater than zero.")

                                                                                # Accept UI input as either ratio (0.2) or percentage (20 for 20%).
                if step_range > 1:
                    step_range = step_range / 100.0
                if step_range < 0:
                    step_range = abs(step_range)
                if step_range >= 1:

                    step_range = 0.99
                form_data["step_range"] = step_range

                if form_data["model"] in {
                    "Cox Ross Rubinstein Tree",
                    "Binomial Tree",
                }:
                    smoothness_model = "CRR"
                elif form_data["model"] == "Jarrow Rudd Tree":
                    smoothness_model = "JRT"
                elif form_data["model"] == "Trinomial Asset Pricing":
                    smoothness_model = "TAP"

                elif "monte" in str(form_data["model"]).lower():
                    num_paths = form_data.get("num_paths", 10000)
                    mc_steps = form_data.get("mc_steps", 252)

                    if variable == "strike_price":
                        base_value = form_data["strike_price"]
                    elif variable == "risk_free_rate":
                        base_value = form_data["r"]
                    elif variable == "volatility":
                        base_value = form_data["sigma"]
                    else:
                        base_value = 0

                    variable_range = np.linspace(
                        base_value * (1 - step_range),
                        base_value * (1 + step_range),
                        num_steps,
                    )
                    greek_values = []

                    for val in variable_range:
                        if variable == "strike_price":
                            mc_engine = monte_carlo_module.create_monte_carlo_engine(
                                S0=float(
                                    StockData(
                                        ticker, start_date, end_date
                                    ).get_closing_price()
                                ),
                                r=risk_free_rate,
                                sigma=volatility,
                                T=StockData(
                                    ticker, start_date, end_date
                                ).get_years_difference(),
                                num_paths=num_paths,
                                num_steps=mc_steps,
                                random_type="sobol",
                            )
                            payoff_func = (
                                (lambda S: np.maximum(S - val, 0))
                                if option_type == "call"
                                else (lambda S: np.maximum(val - S, 0))
                            )
                            lsmc_engine = monte_carlo_module.LSMCEngine(mc_engine)
                            price = lsmc_engine.price_option(payoff_func, option_type)

                        elif variable == "risk_free_rate":
                            mc_engine = monte_carlo_module.create_monte_carlo_engine(
                                S0=float(
                                    StockData(
                                        ticker, start_date, end_date
                                    ).get_closing_price()
                                ),
                                r=val,
                                sigma=volatility,
                                T=StockData(
                                    ticker, start_date, end_date
                                ).get_years_difference(),
                                num_paths=num_paths,
                                num_steps=mc_steps,
                                random_type="sobol",
                            )
                            payoff_func = (
                                (lambda S: np.maximum(S - strike_price, 0))
                                if option_type == "call"
                                else (lambda S: np.maximum(strike_price - S, 0))
                            )
                            lsmc_engine = monte_carlo_module.LSMCEngine(mc_engine)
                            price = lsmc_engine.price_option(payoff_func, option_type)

                        elif variable == "volatility":
                            mc_engine = monte_carlo_module.create_monte_carlo_engine(
                                S0=float(
                                    StockData(
                                        ticker, start_date, end_date
                                    ).get_closing_price()
                                ),
                                r=risk_free_rate,
                                sigma=val,
                                T=StockData(
                                    ticker, start_date, end_date
                                ).get_years_difference(),
                                num_paths=num_paths,
                                num_steps=mc_steps,
                                random_type="sobol",
                            )
                            payoff_func = (
                                (lambda S: np.maximum(S - strike_price, 0))
                                if option_type == "call"
                                else (lambda S: np.maximum(strike_price - S, 0))
                            )
                            lsmc_engine = monte_carlo_module.LSMCEngine(mc_engine)
                            price = lsmc_engine.price_option(payoff_func, option_type)

                        greek_values.append(price)

                    plt.figure(figsize=(10, 6))
                    plt.plot(variable_range, greek_values, "b-", linewidth=2)
                    plt.xlabel(variable.replace("_", " ").title())
                    plt.ylabel(target_variable.replace("_", " ").title())
                    plt.title(
                        f"{target_variable.replace('_', ' ').title()} vs {variable.replace('_', ' ').title()} (Monte Carlo)"
                    )
                    plt.grid(True)
                    plt.tight_layout()

                    plot_filename = f"american_{target_variable}-{variable}_sensitivity_plot_{uuid.uuid4().hex}.png"
                    plot_path = os.path.join("derivapro", "static", plot_filename)
                    plt.savefig(plot_path)
                    plt.close()

                    sensitivity_results = {
                        "plot_filename": plot_filename,
                        "values": [
                            float(value) for value in variable_range
                        ],
                        "greek_values": [
                            float(value) for value in greek_values
                        ],
                        "variable": variable,
                        "target_variable": target_variable,
                    }

                    logger.info(
                        "American sensitivity analysis completed: "
                        "model=%s variable=%s target=%s points=%s plot=%s",
                        form_data.get("pricing_model"),
                        variable,
                        target_variable,
                        len(sensitivity_results["values"]),
                        plot_filename,
                    )

                    if current_user.is_authenticated:

                        instrument = Instrument(
                            user_id=current_user.id,
                            product_type="american_option",
                            ticker=form_data["ticker"],
                            model_name=form_data.get("pricing_model"),
                            start_date=str(form_data["start_date"]),
                            end_date=str(form_data["end_date"]),
                            params_json={
                                "strike_price": form_data["strike_price"],
                                "risk_free_rate": form_data["r"],
                                "volatility": form_data["sigma"],
                                "option_type": form_data["option_type"],
                                "num_paths": form_data.get("num_paths"),
                                "mc_steps": form_data.get("mc_steps"),
                            },
                        )
                        instrument, linked_pricing_result = (
                            _resolve_analysis_instrument(
                                "american_option",
                                instrument,
                            )
                        )

                        analysis_result = AnalysisResult(
                            user_id=current_user.id,
                            instrument_id=instrument.id,
                            pricing_result_id=(
                                linked_pricing_result.id
                                if linked_pricing_result
                                else None
                            ),
                            analysis_type="sensitivity",
                            result_json=sensitivity_results,
                        )
                        db.session.add(analysis_result)
                        db.session.commit()
                        session.pop("sensitivity_results", None)
                        latest_analysis = analysis_result
                    else:
                        pass

                if "monte" not in str(form_data["model"]).lower():
                    tester = AmericanOptionSmoothnessTest(
                        form_data["ticker"],
                        form_data["strike_price"],
                        form_data["start_date"],
                        form_data["end_date"],
                        form_data["r"],
                        form_data["sigma"],
                        smoothness_model,
                        form_data["option_type"],
                        form_data["num_sensitivity_steps"],
                    )

                    values, greek_values = tester.calculate_greeks_over_range(
                        variable, num_steps, step_range, target_variable
                    )
                    tester.plot_single_greek(
                        values, greek_values, target_variable, variable
                    )

                    static_dir = os.path.join(
                        os.path.dirname(os.path.abspath(__file__)),
                        "..",
                        "static",
                    )
                    os.makedirs(static_dir, exist_ok=True)

                    plot_filename = (
                        f"american_{target_variable}-{variable}_"
                        f"sensitivity_plot_{uuid.uuid4().hex}.png"
                    )
                    plot_path = os.path.join(static_dir, plot_filename)

                    logger.debug(
                        "Saving American sensitivity plot to %s",
                        plot_path,
                    )
                    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
                    plt.close()

                                        
                    sensitivity_results = {
                        "plot_filename": plot_filename,
                        "values": [
                            float(value) for value in values
                        ],
                        "greek_values": [
                            float(value) for value in greek_values
                        ],
                        "variable": variable,
                        "target_variable": target_variable,
                    }


                    logger.info(
                        "American sensitivity analysis completed: "
                        "model=%s variable=%s target=%s points=%s plot=%s",
                        form_data.get("pricing_model"),
                        variable,
                        target_variable,
                        len(sensitivity_results["values"]),
                        plot_filename,
                    )

                    if current_user.is_authenticated:

                        instrument = Instrument(
                            user_id=current_user.id,
                            product_type="american_option",
                            ticker=form_data["ticker"],
                            model_name=form_data.get("pricing_model"),
                            start_date=str(form_data["start_date"]),
                            end_date=str(form_data["end_date"]),
                            params_json={
                                "strike_price": form_data["strike_price"],
                                "risk_free_rate": form_data["r"],
                                "volatility": form_data["sigma"],
                                "option_type": form_data["option_type"],
                                "num_steps": form_data.get("num_steps"),
                            },
                        )
                        instrument, linked_pricing_result = (
                            _resolve_analysis_instrument(
                                "american_option",
                                instrument,
                            )
                        )

                        analysis_result = AnalysisResult(
                            user_id=current_user.id,
                            instrument_id=instrument.id,
                            pricing_result_id=(
                                linked_pricing_result.id
                                if linked_pricing_result
                                else None
                            ),
                            analysis_type="sensitivity",
                            result_json=sensitivity_results,
                        )
                        db.session.add(analysis_result)
                        db.session.commit()
                        session.pop("sensitivity_results", None)
                        latest_analysis = analysis_result
                    else:
                        pass

            except Exception as exc:
                session.pop("sensitivity_results", None)

                logger.exception(
                    "An error occurred during American sensitivity analysis"
                )
                sensitivity_error = f"Sensitivity analysis failed: {exc}"
                sensitivity_results = None

        elif action == "ai_sensitivity_assessment":
            sensitivity_data = sensitivity_results or {}

            if current_user.is_authenticated and latest_analysis:

                if latest_analysis.analysis_type == "sensitivity":
                    sensitivity_data = latest_analysis.result_json or {}
                else:
                    sensitivity_data = {}
            elif not sensitivity_data:
                sensitivity_data = sensitivity_results or {}

            if sensitivity_data:
                variable = sensitivity_data.get("variable")
                values = sensitivity_data.get("values")
                target_variable = sensitivity_data.get("target_variable")
                greek_values = sensitivity_data.get("greek_values")

                if values and variable and target_variable and greek_values:
                    sensitivity_text = f"""
                    Sensitivity Analysis Results for {target_variable} with respect to {variable}:
                    Input Values: {values}
                    Output Values: {greek_values}
                    """
                    assessment_input = (
                        f"Please assess the sensitivity analysis results based on the following data: {sensitivity_text}. "
                        f"Focus on the relationship between {variable} and {target_variable}, and any notable patterns or risks. "
                        "Please limit the assessment to be less than 100 words."
                    )
                    gpt_sensitivity_assessment = ask_gpt(assessment_input)

                    sensitivity_data["gpt_sensitivity_assessment"] = (
                        gpt_sensitivity_assessment
                    )
                    gpt_sensitivity_assessment = sensitivity_data[
                        "gpt_sensitivity_assessment"
                    ]
                    if current_user.is_authenticated and latest_analysis:
                        latest_analysis.result_json = sensitivity_data
                        db.session.commit()
                    else:
                        pass
                else:
                    sensitivity_data["gpt_sensitivity_assessment"] = (
                        "Incomplete sensitivity analysis data available for assessment."
                    )
                    gpt_sensitivity_assessment = sensitivity_data[
                        "gpt_sensitivity_assessment"
                    ]
                    if current_user.is_authenticated and latest_analysis:
                        latest_analysis.result_json = sensitivity_data
                        db.session.commit()
                    else:
                        pass
            else:
                sensitivity_results = {
                    "gpt_sensitivity_assessment": "No sensitivity analysis results available for assessment."
                }
                gpt_sensitivity_assessment = sensitivity_results[
                    "gpt_sensitivity_assessment"
                ]

                if current_user.is_authenticated and latest_analysis:
                    latest_analysis.result_json = sensitivity_results
                    db.session.commit()
                else:
                    pass

        elif action == "risk_pl":
            try:
                form_data["price_change"] = float(request.form["price_change"])
                form_data["vol_change"] = float(request.form["vol_change"])

                price_change = form_data["price_change"]
                vol_change = form_data["vol_change"]

                if form_data["model"] == "Cox Ross Rubinstein Tree":
                    rbpl_model = "CRR"
                elif form_data["model"] == "Jarrow Rudd Tree":
                    rbpl_model = "JRT"
                else:
                    rbpl_model = "TAP"

                option = LatticeModel(
                    form_data["ticker"],
                    form_data["strike_price"],
                    form_data["start_date"],
                    form_data["end_date"],
                    form_data["r"],
                    form_data["sigma"],
                )

                risk_pl_results = option.risk_pl_analysis(
                    option_type=form_data["option_type"],
                    steps=form_data["num_steps"],
                    price_change=price_change,
                    vol_change=vol_change,
                    model=rbpl_model,
                )

                risk_pl_results = {"results": risk_pl_results}

                if current_user.is_authenticated:
                    instrument = Instrument(
                        user_id=current_user.id,
                        product_type="american_option",
                        ticker=form_data["ticker"],
                        model_name=form_data.get("pricing_model"),
                        start_date=str(form_data["start_date"]),
                        end_date=str(form_data["end_date"]),
                        params_json={
                            "strike_price": form_data["strike_price"],
                            "risk_free_rate": form_data["r"],
                            "volatility": form_data["sigma"],
                            "option_type": form_data["option_type"],
                            "num_steps": form_data.get("num_steps"),
                        },
                    )
                    instrument, linked_pricing_result = (
                        _resolve_analysis_instrument(
                            "american_option",
                            instrument,
                        )
                    )

                    analysis_result = AnalysisResult(
                        user_id=current_user.id,
                        instrument_id=instrument.id,
                        pricing_result_id=(
                            linked_pricing_result.id if linked_pricing_result else None
                        ),
                        analysis_type="risk_pl",
                        result_json=risk_pl_results,
                    )
                    db.session.add(analysis_result)
                    db.session.commit()
                    session.pop("risk_pl_results", None)
                    latest_analysis = analysis_result
                else:
                    pass

                logger.debug("American risk-based P&L results: %s", risk_pl_results)

            except Exception:
                logger.exception("An error occurred during Risk-Based P&L analysis")
                risk_pl_results = None

        elif action == "ai_rpbl_assessment":
            risk_pl_data = risk_pl_results or {}

            if current_user.is_authenticated and latest_analysis:
                if latest_analysis.analysis_type == "risk_pl":
                    risk_pl_data = latest_analysis.result_json or {}
                else:
                    risk_pl_data = {}
            elif not risk_pl_data:
                risk_pl_data = risk_pl_results or {}
            rpbl_results = risk_pl_data.get("results", {})

            if rpbl_results:
                rpbl_text = f"""
                    Risk-Based P&L Analysis Results:
                    Price Change Impact: {rpbl_results.get("price_change_impact", "N/A")}
                    Volatility Change Impact: {rpbl_results.get("vol_change_impact", "N/A")}
                    Total P&L Impact: {rpbl_results.get("total_pl_impact", "N/A")}
                    Delta Contribution: {rpbl_results.get("delta_contribution", "N/A")}
                    Gamma Contribution: {rpbl_results.get("gamma_contribution", "N/A")}
                    Vega Contribution: {rpbl_results.get("vega_contribution", "N/A")}
                    Theta Contribution: {rpbl_results.get("theta_contribution", "N/A")}
                    Rho Contribution: {rpbl_results.get("rho_contribution", "N/A")}
                    """
                assessment_input = (
                    f"Please assess the Risk-Based P&L analysis results based on the following data: {rpbl_text}. "
                    "Focus on the key drivers of P&L and potential risks. Please limit the assessment to be less than 100 words."
                )
                gpt_rpbl_assessment = ask_gpt(assessment_input)

                risk_pl_data["gpt_rpbl_assessment"] = gpt_rpbl_assessment
                gpt_rpbl_assessment = risk_pl_data["gpt_rpbl_assessment"]
                if current_user.is_authenticated and latest_analysis:
                    latest_analysis.result_json = risk_pl_data
                    db.session.commit()
                else:
                    pass

                risk_pl_results = risk_pl_data
            else:
                risk_pl_data["gpt_rpbl_assessment"] = (
                    "No Risk-Based P&L analysis results available for assessment."
                )
                gpt_rpbl_assessment = risk_pl_data["gpt_rpbl_assessment"]
                if current_user.is_authenticated and latest_analysis:
                    latest_analysis.result_json = risk_pl_data
                    db.session.commit()
                else:
                    pass

                risk_pl_results = risk_pl_data

        elif action == "convergence":
            try:
                required_fields = [
                    "ticker",
                    "strike_price",
                    "start_date",
                    "end_date",
                    "r",
                    "sigma",
                    "option_type",
                ]
                for field in required_fields:
                    if not request.form.get(field):
                        raise ValueError(f"Missing required field: {field}")

                form_data["mode"] = request.form["mode"]
                form_data["model"] = request.form["model"]

                if "pricing_model" not in form_data or not form_data["pricing_model"]:
                    last_session_data = {}
                    if (
                        "pricing_model" in last_session_data
                        and last_session_data["pricing_model"]
                    ):
                        form_data["pricing_model"] = last_session_data["pricing_model"]

                form_data["option_type"] = str(request.form["option_type"])
                form_data["obs"] = safe_int(request.form.get("obs"), 10)
                mode = form_data["mode"]
                model_name = form_data["model"]
                obs = form_data["obs"]
                option_type = form_data["option_type"]

                def safe_int_local(val, default):
                    try:
                        return int(val)
                    except (ValueError, TypeError):
                        return default

                if model_name == "Monte Carlo" and mode == "simulations":
                    try:
                        logger.debug(
                            "Starting Monte Carlo convergence analysis block (simulations mode)"
                        )
                        num_paths = safe_int_local(request.form.get("num_paths"), 10000)
                        mc_steps = safe_int_local(request.form.get("mc_steps"), 252)
                        ticker = form_data["ticker"]
                        start_date = form_data["start_date"]
                        end_date = form_data["end_date"]
                        S0 = float(
                            StockData(ticker, start_date, end_date).get_closing_price()
                        )
                        T = StockData(
                            ticker, start_date, end_date
                        ).get_years_difference()
                        r = form_data["r"]
                        sigma = form_data["sigma"]
                        strike_price = form_data["strike_price"]

                        mc_results = []
                        paths_range = (
                            np.linspace(800, num_paths, obs).round().astype(int)
                        )
                        logger.debug(
                            "Monte Carlo convergence paths_range: %s", paths_range
                        )

                        for n_paths in paths_range:
                            logger.debug(
                                "Running Monte Carlo convergence pricing with n_paths=%s",
                                n_paths,
                            )
                            mc_engine = monte_carlo_module.create_monte_carlo_engine(
                                S0=S0,
                                r=r,
                                sigma=sigma,
                                T=T,
                                num_paths=int(n_paths),
                                num_steps=mc_steps,
                                random_type="sobol",
                            )
                            payoff_func = (
                                (lambda S: np.maximum(S - strike_price, 0))
                                if option_type == "call"
                                else (lambda S: np.maximum(strike_price - S, 0))
                            )
                            lsmc_engine = monte_carlo_module.LSMCEngine(mc_engine)
                            price = lsmc_engine.price_option(payoff_func, option_type)
                            logger.debug(
                                "Monte Carlo convergence price for %s paths: %s",
                                n_paths,
                                price,
                            )
                            mc_results.append((int(n_paths), float(price)))

                        plot_convergence(mc_results, mode="simulations")
                        plot_filename = (
                            f"monte_carlo_convergence_plot_{uuid.uuid4().hex}.png"
                        )
                        plot_path = os.path.join("derivapro", "static", plot_filename)
                        plt.savefig(plot_path)
                        plt.close()
                        file_exists = os.path.exists(plot_path)

                        logger.debug(
                            "Monte Carlo convergence plot %s exists after save? %s",
                            plot_path,
                            file_exists,
                        )

                        convergence_results = {
                            "results": mc_results,
                            "mode": "simulations",
                            "plot_filename": plot_filename,
                        }

                        if current_user.is_authenticated:
                            instrument = Instrument(
                                user_id=current_user.id,
                                product_type="american_option",
                                ticker=form_data["ticker"],
                                model_name=model_name,
                                start_date=str(form_data["start_date"]),
                                end_date=str(form_data["end_date"]),
                                params_json={
                                    "strike_price": form_data["strike_price"],
                                    "risk_free_rate": form_data["r"],
                                    "volatility": form_data["sigma"],
                                    "option_type": form_data["option_type"],
                                    "num_paths": num_paths,
                                    "mc_steps": mc_steps,
                                },
                            )
                            instrument, linked_pricing_result = (
                                _resolve_analysis_instrument(
                                    "american_option",
                                    instrument,
                                )
                            )

                            analysis_result = AnalysisResult(
                                user_id=current_user.id,
                                instrument_id=instrument.id,
                                pricing_result_id=(
                                    linked_pricing_result.id
                                    if linked_pricing_result
                                    else None
                                ),
                                analysis_type="convergence",
                                result_json=convergence_results,
                            )
                            db.session.add(analysis_result)
                            db.session.commit()
                            session.pop("convergence_results", None)
                            latest_analysis = analysis_result
                        else:
                            pass

                        logger.debug(
                            "Saved convergence results: %s",
                            convergence_results,
                        )

                    except Exception:
                        logger.exception("Exception in Monte Carlo convergence block")
                        convergence_results = None

                elif model_name == "Binomial Tree" and mode == "steps":
                    max_steps = safe_int_local(request.form.get("max_steps"), 100)
                    obs = safe_int_local(request.form.get("obs"), 10)
                    ticker = form_data["ticker"]
                    strike_price = form_data["strike_price"]
                    start_date = form_data["start_date"]
                    end_date = form_data["end_date"]
                    r = form_data["r"]
                    sigma = form_data["sigma"]
                    option_type = form_data["option_type"]

                    raw_dividends = request.form.get("dividends", "").strip()
                    parsed_dividends = []
                    if raw_dividends:
                        for entry in raw_dividends.split(","):
                            parts = entry.strip().split(":")
                            if len(parts) == 2:
                                parsed_dividends.append((parts[0], float(parts[1])))
                            elif len(parts) == 3:
                                parsed_dividends.append((
                                    parts[0],
                                    float(parts[1]),
                                    float(parts[2]),
                                ))

                    steps_range = np.linspace(2, max_steps, obs).astype(int)
                    results = []
                    for nsteps in steps_range:
                        engine = BinomialTreeEngineCRR(
                            ticker=ticker,
                            strike_price=strike_price,
                            start_date=start_date,
                            end_date=end_date,
                            risk_free_rate=r,
                            volatility=sigma,
                            num_steps=int(nsteps),
                            option_type=option_type,
                            dividends=parsed_dividends,
                        )
                        price = engine.price_american_option()
                        results.append((int(nsteps), float(price)))

                    plot_convergence(results, mode)
                    plot_filename = (
                        f"binomial_tree_convergence_plot_{uuid.uuid4().hex}.png"
                    )
                    plot_path = os.path.join("derivapro", "static", plot_filename)
                    plt.savefig(plot_path)
                    convergence_results = {
                        "results": results,
                        "mode": mode,
                        "plot_filename": plot_filename,
                    }

                    if current_user.is_authenticated:
                        instrument = Instrument(
                            user_id=current_user.id,
                            product_type="american_option",
                            ticker=form_data["ticker"],
                            model_name=model_name,
                            start_date=str(form_data["start_date"]),
                            end_date=str(form_data["end_date"]),
                            params_json={
                                "strike_price": form_data["strike_price"],
                                "risk_free_rate": form_data["r"],
                                "volatility": form_data["sigma"],
                                "option_type": form_data["option_type"],
                                "max_steps": max_steps,
                                "obs": obs,
                            },
                        )
                        instrument, linked_pricing_result = (
                            _resolve_analysis_instrument(
                                "american_option",
                                instrument,
                            )
                        )

                        analysis_result = AnalysisResult(
                            user_id=current_user.id,
                            instrument_id=instrument.id,
                            pricing_result_id=(
                                linked_pricing_result.id
                                if linked_pricing_result
                                else None
                            ),
                            analysis_type="convergence",
                            result_json=convergence_results,
                        )
                        db.session.add(analysis_result)
                        db.session.commit()
                        session.pop("convergence_results", None)
                        latest_analysis = analysis_result
                    else:
                        pass

                    plt.close()

                else:
                    form_data["max_steps"] = safe_int_local(
                        request.form.get("max_steps"), 100
                    )
                    max_steps = form_data["max_steps"]
                    max_sims = 0
                    convergence_params = form_data.copy()
                    if "pricing_model" in convergence_params:
                        del convergence_params["pricing_model"]

                    american_step_results = lattice_convergence_test(
                        max_steps,
                        max_sims,
                        obs,
                        LatticeModel,
                        convergence_params,
                        model_name,
                        option_type,
                    )
                    logger.debug(
                        "American step convergence results: %s",
                        american_step_results,
                    )

                    plot_convergence(american_step_results, mode)
                    plot_filename = f"lattice_convergence_plot_{uuid.uuid4().hex}.png"
                    plot_path = os.path.join("derivapro", "static", plot_filename)
                    plt.savefig(plot_path)
                    convergence_results = {
                        "results": american_step_results,
                        "mode": mode,
                        "plot_filename": plot_filename,
                    }

                    if current_user.is_authenticated:
                        instrument = Instrument(
                            user_id=current_user.id,
                            product_type="american_option",
                            ticker=form_data["ticker"],
                            model_name=model_name,
                            start_date=str(form_data["start_date"]),
                            end_date=str(form_data["end_date"]),
                            params_json={
                                "strike_price": form_data["strike_price"],
                                "risk_free_rate": form_data["r"],
                                "volatility": form_data["sigma"],
                                "option_type": form_data["option_type"],
                                "max_steps": max_steps,
                                "obs": obs,
                            },
                        )
                        instrument, linked_pricing_result = (
                            _resolve_analysis_instrument(
                                "american_option",
                                instrument,
                            )
                        )

                        analysis_result = AnalysisResult(
                            user_id=current_user.id,
                            instrument_id=instrument.id,
                            pricing_result_id=(
                                linked_pricing_result.id
                                if linked_pricing_result
                                else None
                            ),
                            analysis_type="convergence",
                            result_json=convergence_results,
                        )
                        db.session.add(analysis_result)
                        db.session.commit()
                        session.pop("convergence_results", None)
                        latest_analysis = analysis_result
                    else:
                        pass

                    plt.close()

            except Exception:
                logger.exception(
                    "An error occurred during American convergence analysis"
                )
                convergence_results = None

        elif action == "ai_convergence_assessment":
            convergence_data = convergence_results or {}

            if current_user.is_authenticated and latest_analysis:
                if latest_analysis.analysis_type == "convergence":
                    convergence_data = latest_analysis.result_json or {}
                else:
                    convergence_data = {}
            elif not convergence_data:
                convergence_data = convergence_results or {}

            if convergence_data:
                results = convergence_data.get("results")
                mode = convergence_data.get("mode")

                if results:
                    convergence_text = f"""
                    Convergence Analysis Results:
                    Mode: {mode}
                    Results: {results}
                    """
                    assessment_input = (
                        f"Please assess the convergence analysis results based on the following data: {convergence_text}. "
                        "Focus on the convergence behavior and any potential issues or recommendations. "
                        "Please limit the assessment to be less than 100 words."
                    )
                    gpt_convergence_assessment = ask_gpt(assessment_input)
                    convergence_data["gpt_convergence_assessment"] = (
                        gpt_convergence_assessment
                    )
                    gpt_convergence_assessment = convergence_data[
                        "gpt_convergence_assessment"
                    ]
                    if current_user.is_authenticated and latest_analysis:
                        latest_analysis.result_json = convergence_data
                        db.session.commit()
                    else:
                        pass

                    convergence_results = convergence_data
                else:
                    convergence_data["gpt_convergence_assessment"] = (
                        "Incomplete convergence analysis data available for assessment."
                    )
                    gpt_convergence_assessment = convergence_data[
                        "gpt_convergence_assessment"
                    ]
                    if current_user.is_authenticated and latest_analysis:
                        latest_analysis.result_json = convergence_data
                        db.session.commit()
                    else:
                        pass

                    convergence_results = convergence_data
            else:
                convergence_results = {
                    "gpt_convergence_assessment": "No convergence analysis results available for assessment."
                }
                gpt_convergence_assessment = convergence_results[
                    "gpt_convergence_assessment"
                ]

                if current_user.is_authenticated and latest_analysis:
                    latest_analysis.result_json = convergence_results
                    db.session.commit()
                else:
                    pass

        elif action == "scenario":
            try:
                spot_change = float(request.form.get("spot_scenario", 0))
                vol_change = float(request.form.get("vol_scenario", 0))
                rate_change = float(request.form.get("rate_scenario", 0))
                ticker = form_data.get("ticker")
                strike_price = float(form_data.get("strike_price"))
                start_date = datetime.strptime(
                    form_data.get("start_date"), "%Y-%m-%d"
                ).date()
                end_date = datetime.strptime(
                    form_data.get("end_date"), "%Y-%m-%d"
                ).date()
                risk_free_rate = float(form_data.get("r"))
                volatility = float(form_data.get("sigma"))
                option_type = form_data.get("option_type")
                model_name = form_data.get("model")

                if model_name == "Monte Carlo":
                    baseline_price = None
                    baseline_greeks = None
                else:
                    option = LatticeModel(
                        ticker,
                        strike_price,
                        start_date,
                        end_date,
                        risk_free_rate,
                        volatility,
                    )
                    if model_name == "Cox Ross Rubinstein Tree":
                        baseline_price = option.Cox_Ross_Rubinstein_Tree(
                            option_type, num_steps
                        )
                        baseline_greeks = option.CRRGreeks(option_type, num_steps)
                    elif model_name == "Jarrow Rudd Tree":
                        baseline_price = option.Jarrow_Rudd_Tree(option_type, num_steps)
                        baseline_greeks = option.JRTGreeks(option_type, num_steps)
                    else:
                        baseline_price = option.Trinomial_Asset_Pricing(
                            option_type, num_steps
                        )
                        baseline_greeks = option.TAPGreeks(option_type, num_steps)

                baseline_price = "{:.4f}".format(baseline_price)
                baseline_delta = "{:.4f}".format(baseline_greeks["Delta"])
                baseline_gamma = "{:.4f}".format(baseline_greeks["Gamma"])
                baseline_vega = "{:.4f}".format(baseline_greeks["Vega"])
                baseline_theta = "{:.4f}".format(baseline_greeks["Theta"])
                baseline_rho = "{:.4f}".format(baseline_greeks["Rho"])

                stressed_spot = strike_price * (1 + spot_change)
                stressed_vol = volatility + vol_change
                stressed_rate = risk_free_rate + rate_change

                if model_name == "Monte Carlo":
                    stressed_price = None
                    stressed_greeks = None
                else:
                    stressed_option = LatticeModel(
                        ticker,
                        stressed_spot,
                        start_date,
                        end_date,
                        stressed_rate,
                        stressed_vol,
                    )
                    if model_name == "Cox Ross Rubinstein Tree":
                        stressed_price = stressed_option.Cox_Ross_Rubinstein_Tree(
                            option_type, num_steps, greeks=False
                        )
                        stressed_greeks = stressed_option.CRRGreeks(
                            option_type, num_steps
                        )
                    elif model_name == "Jarrow Rudd Tree":
                        stressed_price = stressed_option.Jarrow_Rudd_Tree(
                            option_type, num_steps, greeks=False
                        )
                        stressed_greeks = stressed_option.JRTGreeks(
                            option_type, num_steps
                        )
                    else:
                        stressed_price = stressed_option.Trinomial_Asset_Pricing(
                            option_type, num_steps
                        )
                        stressed_greeks = stressed_option.TAPGreeks(
                            option_type, num_steps
                        )

                stressed_price = "{:.4f}".format(stressed_price)
                stressed_delta = "{:.4f}".format(stressed_greeks["Delta"])
                stressed_gamma = "{:.4f}".format(stressed_greeks["Gamma"])
                stressed_vega = "{:.4f}".format(stressed_greeks["Vega"])
                stressed_theta = "{:.4f}".format(stressed_greeks["Theta"])
                stressed_rho = "{:.4f}".format(stressed_greeks["Rho"])

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

                logger.debug("Scenario baseline table: %s", baseline_scenario_table)
                logger.debug("Scenario stressed table: %s", stressed_scenario_table)

                scenario_results = {
                    "baseline_scenario_table": baseline_scenario_table,
                    "stressed_scenario_table": stressed_scenario_table,
                    "gpt_scenario_assessment": "No assessment yet.",
                }

                if current_user.is_authenticated:
                    instrument = Instrument(
                        user_id=current_user.id,
                        product_type="american_option",
                        ticker=form_data["ticker"],
                        model_name=form_data.get("pricing_model"),
                        start_date=str(form_data["start_date"]),
                        end_date=str(form_data["end_date"]),
                        params_json={
                            "strike_price": form_data["strike_price"],
                            "risk_free_rate": form_data["r"],
                            "volatility": form_data["sigma"],
                            "option_type": form_data["option_type"],
                            "num_steps": form_data.get("num_steps"),
                        },
                    )
                    instrument, linked_pricing_result = (
                        _resolve_analysis_instrument(
                            "american_option",
                            instrument,
                        )
                    )

                    analysis_result = AnalysisResult(
                        user_id=current_user.id,
                        instrument_id=instrument.id,
                        pricing_result_id=(
                            linked_pricing_result.id if linked_pricing_result else None
                        ),
                        analysis_type="scenario",
                        result_json=scenario_results,
                    )
                    db.session.add(analysis_result)
                    db.session.commit()
                    session.pop("scenario_results", None)
                    latest_analysis = analysis_result
                else:
                    pass

            except Exception:
                logger.exception("An error occurred during scenario analysis")
                scenario_results = None

        elif action == "ai_scenario_assessment":
            scenario_data = scenario_results or {}

            if current_user.is_authenticated and latest_analysis:
                if latest_analysis.analysis_type == "scenario":
                    scenario_data = latest_analysis.result_json or {}
                else:
                    scenario_data = {}
            elif not scenario_data:
                scenario_data = scenario_results or {}

            baseline_table = scenario_data.get("baseline_scenario_table", {})
            stressed_table = scenario_data.get("stressed_scenario_table", {})

            if baseline_table and stressed_table:
                table_text = f"""
                Baseline Scenario:
                Option Price={baseline_table["baseline_price"]}, Delta={baseline_table["baseline_delta"]}, 
                Gamma={baseline_table["baseline_gamma"]}, Vega={baseline_table["baseline_vega"]}, 
                Theta={baseline_table["baseline_theta"]}, Rho={baseline_table["baseline_rho"]}

                Stressed Scenario:
                Option Price={stressed_table["stressed_price"]}, Delta={stressed_table["stressed_delta"]}, 
                Gamma={stressed_table["stressed_gamma"]}, Vega={stressed_table["stressed_vega"]}, 
                Theta={stressed_table["stressed_theta"]}, Rho={stressed_table["stressed_rho"]}
                """
                assessment_input = (
                    f"Please assess the scenario analysis of the option price and Greeks based on the following results: {table_text}. "
                    "Please limit the assessment to be less than 100 words."
                )
                gpt_scenario_assessment = ask_gpt(assessment_input)
                scenario_data["gpt_scenario_assessment"] = gpt_scenario_assessment

                if current_user.is_authenticated and latest_analysis:
                    latest_analysis.result_json = scenario_data
                    db.session.commit()
                else:
                    pass

                scenario_results = scenario_data
            else:
                scenario_data["gpt_scenario_assessment"] = (
                    "No scenario analysis results available for assessment."
                )

                if current_user.is_authenticated and latest_analysis:
                    latest_analysis.result_json = scenario_data
                    db.session.commit()
                else:
                    pass

                scenario_results = scenario_data

    form_data.setdefault("num_paths", 10000)
    form_data.setdefault("mc_steps", 252)

    logger.debug(
        "Pricing model used for main pricing: %s",
        form_data.get("pricing_model"),
    )

    return render_template(
        "american_options.html",
        option_price=option_price,
        form_data=form_data,

        sensitivity_results=sensitivity_results,
        sensitivity_error=sensitivity_error,
                risk_pl_results=risk_pl_results,
        convergence_results=convergence_results,
        scenario_results=scenario_results,

        md_content=md_content,
        gpt_rpbl_assessment=gpt_rpbl_assessment,
        gpt_sensitivity_assessment=gpt_sensitivity_assessment,
        gpt_convergence_assessment=gpt_convergence_assessment,
        gpt_scenario_assessment=gpt_scenario_assessment,
        action=action,
    )


def _build_report_template():
    """Collect pricing results, analysis output, and AI assessments into a
    ``ReportTemplate`` used to drive both the HTML preview and the PDF export.

    Returns a tuple of ``(ReportTemplate, latest_pricing_result, latest_analysis)``
    since the two SQLAlchemy rows are also needed by callers for persistence.
    """
    sensitivity_results = {}
    scenario_results = {}
    convergence_results = {}
    latest_analysis = None
    latest_pricing_result = None

    if current_user.is_authenticated:
        product_types = ["european_option", "american_option"]

        latest_pricing_result = _get_latest_pricing_result_for_product_types(
            product_types
        )

        latest_sensitivity_analysis = _get_latest_analysis_by_types_for_product_types(
            product_types,
            [
                "sensitivity",
                "barrier_sensitivity",
                "asian_sensitivity",
                "autocallable_sensitivity",
            ],
        )
        latest_scenario_analysis = _get_latest_analysis_by_types_for_product_types(
            product_types,
            [
                "scenario",
                "barrier_scenario",
                "asian_scenario",
                "autocallable_scenario",
            ],
        )
        latest_convergence_analysis = _get_latest_analysis_by_types_for_product_types(
            product_types,
            [
                "convergence",
                "barrier_convergence",
                "asian_convergence",
                "autocallable_convergence",
            ],
        )

        if latest_sensitivity_analysis and latest_sensitivity_analysis.result_json:
            sensitivity_results = latest_sensitivity_analysis.result_json

        if latest_scenario_analysis and latest_scenario_analysis.result_json:
            scenario_results = latest_scenario_analysis.result_json

        if latest_convergence_analysis and latest_convergence_analysis.result_json:
            convergence_results = latest_convergence_analysis.result_json

        latest_analysis = (
            latest_convergence_analysis
            or latest_scenario_analysis
            or latest_sensitivity_analysis
        )

    sensitivity_results = sensitivity_results or {}
    scenario_results = scenario_results or {}
    convergence_results = convergence_results or {}

    sensitivity_combined = []
    sensitivity_plot = None
    sensitivity_assessment = "No assessment available."

    if sensitivity_results:
        sensitivity_combined = list(
            zip(
                sensitivity_results.get("values", []),
                sensitivity_results.get("greek_values", []),
            )
        )
        sensitivity_plot = sensitivity_results.get(
            "plot_filename",
            sensitivity_results.get("plot_path"),
        )
        sensitivity_assessment = sensitivity_results.get(
            "gpt_sensitivity_assessment", "No assessment available."
        )

    scenario_table = []
    if scenario_results:
        baseline = scenario_results.get("baseline_scenario_table")
        stressed = scenario_results.get("stressed_scenario_table")

        if baseline:
            scenario_table.append({
                "scenario": baseline.get("scenario", "Baseline"),
                "option_price": baseline.get("baseline_price"),
                "delta": baseline.get("baseline_delta"),
                "gamma": baseline.get("baseline_gamma"),
                "vega": baseline.get("baseline_vega"),
                "theta": baseline.get("baseline_theta"),
                "rho": baseline.get("baseline_rho"),
            })

        if stressed:
            scenario_table.append({
                "scenario": stressed.get("scenario", "Stressed"),
                "option_price": stressed.get("stressed_price"),
                "delta": stressed.get("stressed_delta"),
                "gamma": stressed.get("stressed_gamma"),
                "vega": stressed.get("stressed_vega"),
                "theta": stressed.get("stressed_theta"),
                "rho": stressed.get("stressed_rho"),
            })

    scenario_assessment = scenario_results.get(
        "gpt_scenario_assessment", "No assessment available."
    )

    convergence_plot = convergence_results.get("plot_filename")
    convergence_summary = convergence_results.get("results", [])
    convergence_assessment = convergence_results.get(
        "gpt_convergence_assessment", "No assessment available."
    )

    target_variable = sensitivity_results.get("target_variable", "target variable")
    variable = sensitivity_results.get("variable", "input variable")
    sensitivity_description = (
        f"Sensitivity analysis was conducted on {target_variable} "
        f"with respect to changes in the {variable}."
    )

    report = ReportTemplate(
        sensitivity_description=sensitivity_description,
        sensitivity_combined=sensitivity_combined,
        sensitivity_plot=sensitivity_plot,
        sensitivity_assessment=sensitivity_assessment,
        scenario_results=scenario_table,
        scenario_assessment=scenario_assessment,
        convergence_plot=convergence_plot,
        convergence_summary=convergence_summary,
        convergence_assessment=convergence_assessment,
        latest_pricing_result=latest_pricing_result,
        latest_analysis=latest_analysis,
    )

    return report, latest_pricing_result, latest_analysis


@vanilla_options_bp.route("/reporting", methods=["GET"])
def reporting():
    report, _latest_pricing_result, _latest_analysis = _build_report_template()
    return render_template("report_template.html", **asdict(report))


@vanilla_options_bp.route("/download-report", methods=["GET"])
def download_report():
    report, latest_pricing_result, latest_analysis = _build_report_template()

    base_dir = os.path.dirname(os.path.abspath(__file__))
    static_dir = os.path.join(base_dir, "..", "static")
    reports_dir = os.path.join(static_dir, "reports")
    os.makedirs(reports_dir, exist_ok=True)

    pdf_bytes = render_report_pdf(report, static_dir)

    report_filename = f"model_validation_report_{uuid.uuid4().hex}.pdf"
    pdf_output_path = os.path.join(reports_dir, report_filename)
    with open(pdf_output_path, "wb") as pdf_file:
        pdf_file.write(pdf_bytes)

    if current_user.is_authenticated:
        report_row = Report(
            user_id=current_user.id,
            instrument_id=(
                latest_pricing_result.instrument_id
                if latest_pricing_result
                else (latest_analysis.instrument_id if latest_analysis else None)
            ),
            pricing_result_id=(
                latest_pricing_result.id if latest_pricing_result else None
            ),
            analysis_result_id=(latest_analysis.id if latest_analysis else None),
            report_type="model_validation_report",
            filename=report_filename,
            filepath=os.path.join("derivapro", "static", "reports", report_filename),
            pdf_data=pdf_bytes,
        )
        db.session.add(report_row)
        db.session.commit()

    return send_file(
        io.BytesIO(pdf_bytes),
        as_attachment=True,
        download_name="Model_Validation_Report.pdf",
        mimetype="application/pdf",
    )


@vanilla_options_bp.route("/reports/<int:report_id>/download", methods=["GET"])
@login_required
def download_saved_report(report_id):
    report_row = Report.query.filter_by(
        id=report_id, user_id=current_user.id
    ).first_or_404()

    if report_row.pdf_data:
        return send_file(
            io.BytesIO(report_row.pdf_data),
            as_attachment=True,
            download_name=report_row.filename,
            mimetype="application/pdf",
        )

    base_dir = os.path.dirname(os.path.abspath(__file__))
    disk_path = os.path.join(base_dir, "..", "..", report_row.filepath)
    if not os.path.isfile(disk_path):
        abort(404)

    return send_file(
        disk_path,
        as_attachment=True,
        download_name=report_row.filename,
        mimetype="application/pdf",
    )


@vanilla_options_bp.route("/conceptual-soundness", methods=["GET"])
def conceptual_soundness():
    try:
        file_path = os.path.join(os.path.dirname(__file__), "black_scholes.md")
        with open(file_path, "r") as file:
            raw_content = file.read()
            content = markdown.markdown(raw_content, extensions=["extra", "nl2br"])
            logger.debug("Conceptual soundness content read successfully")
    except FileNotFoundError:
        content = "File not found. Please check if black_scholes.txt exists in the routes folder."
        logger.warning("Conceptual soundness file not found")
    except Exception as e:
        content = f"Error reading file: {str(e)}"
        logger.exception("Error reading conceptual soundness content")

    return render_template("conceptual_soundness.html", content=content)
