# Note: last updated on Aug 06

from datetime import datetime
from flask import Blueprint, render_template, request, session
from ..models.mdls_monte_carlo import MonteCarlo
from ..models.mdls_monte_carlo import convergence_test
from ..models.mdls_monte_carlo import plot_convergence as mc_plot_convergence
from ..models import mdls_monte_carlo_v2 as monte_carlo_New_module
from ..models.mdls_asian_options import (
    AsianOption,
    AsianOptionSmoothnessTest,
    lattice_convergence_test,
    plot_convergence as asian_plot_convergence,
)
from ..models.mdls_monte_carlo import MonteCarloSmoothnessTest
from ..models.mdls_autocallables import (
    AutoMonteCarlo,
    AutocallableSmoothnessTest,
    auto_convergence_test,
)
from ..models.mdls_structured_products import (
    AutocallableNoteTerms,
    price_autocallable_note,
)


import sys
import os
import numpy as np
import uuid
import matplotlib.pyplot as plt
import markdown
from openai import AzureOpenAI
from dotenv import load_dotenv
import logging

logger = logging.getLogger(__name__)

# Import the New Monte Carlo module
# monte_carlo_newMC_path = os.path.join(
#     os.path.dirname(__file__), "..", "models", "mdls_monte_carlo_NEW.py"
# )
# spec = importlib.util.spec_from_file_location(
#     "monte_carlo_New_module", monte_carlo_newMC_path
# )
# if spec is not None:
#     monte_carlo_New_module = importlib.util.module_from_spec(spec)
#     if spec.loader is not None:
#         spec.loader.exec_module(monte_carlo_New_module)
# else:
#     raise ImportError(
#         f"Could not load NEw Monte Carlo module from {monte_carlo_newMC_path}"
#     )

exotic_options_bp = Blueprint("exotic_options", __name__)

load_dotenv()

# Get the values from the environment variables
api_key = os.getenv("OpenAI_API_Key")
base_url = os.getenv("Base_URL")
api_version = os.getenv("API_Version")
model = os.getenv("Model")
Auth_headers = os.getenv("Auth_headers")

# Add auth header
auth_headers = {Auth_headers: api_key}

# Instantiate the Azure OpenAI client
client = AzureOpenAI(
    api_key=api_key,
    api_version=api_version,
    default_headers=auth_headers,
    azure_endpoint=base_url,
)


def ask_gpt(question):
    """
    Sends a request to Azure OpenAI's GPT-4 API with the given question.
    """
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": "Assistant is a large language model hosted in Azure OpenAI.",
                },
                {"role": "user", "content": f"{question}"},
            ],
        )
        return response.choices[0].message.content
    except Exception as e:
        logger.exception("Error occurred while calling OpenAI API")
        return f"An error occurred: {e}"


def _parse_float_list(raw_value, default_values):
    if raw_value in [None, ""]:
        return list(default_values)
    return [float(item.strip()) for item in raw_value.split(",") if item.strip()]


def _format_currency(value):
    return "${:,.4f}".format(float(value))


def _format_percent(value):
    return "{:.2f}%".format(float(value) * 100)


@exotic_options_bp.route("/", methods=["GET", "POST"])
def exotic_options():
    return render_template("exotic_options.html")


@exotic_options_bp.route("/autocallable", methods=["GET", "POST"])
def autocallable_options():
    readme_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "autocallable_options.md"
    )
    with open(readme_path, "r") as readme_file:
        content = readme_file.read()
    md_content = markdown.markdown(content)

    option_price = sensitivity_results = scenario_results = convergence_results = (
        risk_pl_results
    ) = None
    structured_results = None
    form_data = {}

    if request.method == "POST":
        action = request.form.get("analysis_type")

        if action == "structured_pricing":
            structured_form_data = dict(request.form)
            try:
                spot_prices = _parse_float_list(
                    request.form.get("structured_spot_prices"),
                    [100.0],
                )
                volatilities = _parse_float_list(
                    request.form.get("structured_volatilities"),
                    [0.20] * len(spot_prices),
                )
                if len(volatilities) == 1 and len(spot_prices) > 1:
                    volatilities = volatilities * len(spot_prices)

                observation_times = _parse_float_list(
                    request.form.get("structured_observation_times"),
                    [0.25, 0.50, 0.75, 1.00],
                )

                terms = AutocallableNoteTerms(
                    spot_prices=spot_prices,
                    volatilities=volatilities,
                    risk_free_rate=float(request.form.get("structured_r", 0.045)),
                    dividend_yield=float(request.form.get("structured_q", 0.0)),
                    maturity=float(request.form.get("structured_maturity", 1.0)),
                    observation_times=observation_times,
                    notional=float(request.form.get("structured_notional", 1000000)),
                    coupon_rate=float(request.form.get("structured_coupon_rate", 0.025)),
                    coupon_barrier=float(request.form.get("structured_coupon_barrier", 0.70)),
                    autocall_barrier=float(request.form.get("structured_autocall_barrier", 1.00)),
                    protection_barrier=float(request.form.get("structured_protection_barrier", 0.60)),
                    memory_coupon=request.form.get("structured_memory_coupon") == "on",
                    correlation=float(request.form.get("structured_correlation", 0.30)),
                    num_paths=int(request.form.get("structured_num_paths", 10000)),
                    num_steps=int(request.form.get("structured_num_steps", 252)),
                    random_type=request.form.get("structured_random_type", "sobol"),
                )

                raw_results = price_autocallable_note(terms)
                structured_results = {
                    "price": _format_currency(raw_results["price"]),
                    "standard_error": _format_currency(raw_results["standard_error"]),
                    "autocall_probability": _format_percent(raw_results["autocall_probability"]),
                    "protection_breach_probability": _format_percent(
                        raw_results["protection_breach_probability"]
                    ),
                    "average_coupon_count": "{:.2f}".format(
                        raw_results["average_coupon_count"]
                    ),
                    "observation_count": raw_results["observation_count"],
                    "underlying_count": raw_results["underlying_count"],
                    "worst_final_level_mean": _format_percent(
                        raw_results["worst_final_level_mean"]
                    ),
                    "worst_final_level_p05": _format_percent(
                        raw_results["worst_final_level_p05"]
                    ),
                    "worst_final_level_p50": _format_percent(
                        raw_results["worst_final_level_p50"]
                    ),
                    "worst_final_level_p95": _format_percent(
                        raw_results["worst_final_level_p95"]
                    ),
                }
            except Exception as exc:
                logger.exception("Structured autocallable pricing failed")
                structured_results = {"error": str(exc)}

            return render_template(
                "autocallables.html",
                option_price=option_price,
                form_data=form_data,
                structured_form_data=structured_form_data,
                structured_results=structured_results,
                sensitivity_results=sensitivity_results,
                convergence_results=convergence_results,
                scenario_results=scenario_results,
                risk_pl_results=risk_pl_results,
                md_content=md_content,
            )

        form_data = {
            "ticker": request.form["ticker"],
            "K": request.form["K"],
            "r": request.form["r"],
            "sigma": request.form["sigma"],
            "T": request.form["T"],
            "q": request.form["q"],
            "N": request.form["N"],
            "M": request.form["M"],
            "barrier_levels": request.form["barrier_levels"],
            "coupon_rates": request.form["coupon_rates"],
            "discretization": request.form["discretization"],
            "simulation_engine": request.form.get("simulation_engine", "original"),
            "option_type": request.form.get("option_type", "call"),
        }

        ticker = form_data["ticker"]
        K = float(form_data["K"]) if form_data["K"] not in [None, ""] else 0.0
        r = float(form_data["r"]) if form_data["r"] not in [None, ""] else 0.0
        sigma = (
            float(form_data["sigma"]) if form_data["sigma"] not in [None, ""] else 0.0
        )
        T = float(form_data["T"]) if form_data["T"] not in [None, ""] else 0.0
        q = float(form_data["q"]) if form_data["q"] not in [None, ""] else 0.0
        N = int(form_data["N"]) if form_data["N"] not in [None, ""] else 0
        M = int(form_data["M"]) if form_data["M"] not in [None, ""] else 0
        barrier_levels = (
            float(form_data["barrier_levels"])
            if form_data["barrier_levels"] not in [None, ""]
            else 0.0
        )
        coupon_rates = (
            float(form_data["coupon_rates"])
            if form_data["coupon_rates"] not in [None, ""]
            else 0.0
        )
        discretization = form_data["discretization"]
        simulation_engine = form_data.get("simulation_engine", "original")
        option_type = form_data.get("option_type", "call")

        if action == "sensitivity":
            try:
                num_steps = int(request.form.get("num_steps", 50))
                step_range = float(request.form.get("step_range", 0.1))
                variable = request.form.get("variable", "strike_price")
                target_variable = request.form.get("target_variable", "option_price")

                tester = AutocallableSmoothnessTest(
                    ticker,
                    K,
                    r,
                    sigma,
                    T,
                    q,
                    N,
                    M,
                    discretization=discretization,
                    barrier_levels=np.full(N, barrier_levels),
                    coupon_rates=np.full(N, coupon_rates),
                )

                values, outputs = tester.calculate_greeks_over_range(
                    variable, num_steps, step_range, target_variable
                )
                plot_path = tester.plot_single_greek(
                    values, outputs, target_variable, variable
                )
                plot_filename = os.path.basename(plot_path)

                session["sensitivity_results"] = {
                    "variable": variable,
                    "values": values.tolist()
                    if hasattr(values, "tolist")
                    else list(values),
                    "greek_values": outputs,
                    "target_variable": target_variable,
                    "plot_filename": plot_filename,
                }
                sensitivity_results = True

            except Exception:
                logger.exception(
                    "An error occurred during autocallable sensitivity analysis"
                )
                sensitivity_results = None

        elif action == "scenario":
            try:
                spot_change = float(request.form.get("spot_scenario", 0))
                vol_change = float(request.form.get("vol_scenario", 0))
                rate_change = float(request.form.get("rate_scenario", 0))

                base_barriers = np.full(N, barrier_levels)
                base_coupons = np.full(N, coupon_rates)

                option = AutoMonteCarlo(ticker, K, r, sigma, T, q, N, M)
                base_spot = float(option.S0)
                base_greeks = option.calculate_greeks(
                    discretization, base_barriers, base_coupons
                )

                baseline_price = "{:.4f}".format(base_greeks["option_price"])
                baseline_delta = "{:.4f}".format(base_greeks["delta"])
                baseline_gamma = "{:.4f}".format(base_greeks["gamma"])
                baseline_vega = "{:.4f}".format(base_greeks["vega"])
                baseline_theta = "{:.4f}".format(base_greeks["theta"])
                baseline_rho = "{:.4f}".format(base_greeks["rho"])

                stressed_S = base_spot * (1 + spot_change)
                stressed_sigma = sigma + vol_change
                stressed_r = r + rate_change

                stressed_option = AutoMonteCarlo(
                    ticker, K, stressed_r, stressed_sigma, T, q, N, M, S0=stressed_S
                )
                stressed_greeks = stressed_option.calculate_greeks(
                    discretization, base_barriers, base_coupons
                )

                stressed_price = "{:.4f}".format(stressed_greeks["option_price"])
                stressed_delta = "{:.4f}".format(stressed_greeks["delta"])
                stressed_gamma = "{:.4f}".format(stressed_greeks["gamma"])
                stressed_vega = "{:.4f}".format(stressed_greeks["vega"])
                stressed_theta = "{:.4f}".format(stressed_greeks["theta"])
                stressed_rho = "{:.4f}".format(stressed_greeks["rho"])

                session["scenario_results"] = {
                    "baseline_scenario_table": {
                        "baseline_price": baseline_price,
                        "baseline_delta": baseline_delta,
                        "baseline_gamma": baseline_gamma,
                        "baseline_vega": baseline_vega,
                        "baseline_theta": baseline_theta,
                        "baseline_rho": baseline_rho,
                    },
                    "stressed_scenario_table": {
                        "stressed_price": stressed_price,
                        "stressed_delta": stressed_delta,
                        "stressed_gamma": stressed_gamma,
                        "stressed_vega": stressed_vega,
                        "stressed_theta": stressed_theta,
                        "stressed_rho": stressed_rho,
                    },
                    "gpt_scenario_assessment": "No assessment yet.",
                }
                scenario_results = True
                return render_template(
                    "autocallables.html",
                    option_price=None,
                    form_data=form_data,
                    sensitivity_results=sensitivity_results,
                    convergence_results=convergence_results,
                    scenario_results=scenario_results,
                    risk_pl_results=risk_pl_results,
                    md_content=md_content,
                )
            except Exception:
                logger.exception(
                    "An error occurred during autocallable scenario analysis"
                )
                scenario_results = None

        elif action == "convergence":
            try:
                mode = request.form.get("mode", "steps")
                num_steps_max = int(request.form.get("num_steps", N))
                max_sims = int(request.form.get("max_sims", M))
                obs = int(request.form.get("obs", 10))

                pricer_params = {
                    "ticker": ticker,
                    "K": K,
                    "r": r,
                    "sigma": sigma,
                    "T": T,
                    "q": q,
                    "N": N,
                    "M": M,
                }

                results = auto_convergence_test(
                    num_steps=num_steps_max,
                    max_sims=max_sims,
                    obs=obs,
                    pricer_class=AutoMonteCarlo,
                    mode=mode,
                    discretization=discretization,
                    barrier_levels=barrier_levels,
                    coupon_rates=coupon_rates,
                    pricer_params=pricer_params,
                )

                plot_path = mc_plot_convergence(results, mode)
                session["autocallable_convergence_plot"] = (
                    os.path.basename(plot_path)
                    if plot_path
                    else "autocallable_convergence_plot.png"
                )

                convergence_results = True

                return render_template(
                    "autocallables.html",
                    option_price=None,
                    form_data=form_data,
                    sensitivity_results=sensitivity_results,
                    convergence_results=convergence_results,
                    scenario_results=scenario_results,
                    risk_pl_results=risk_pl_results,
                    md_content=md_content,
                )

            except Exception:
                logger.exception(
                    "An error occurred during autocallable convergence analysis"
                )
                convergence_results = None

        elif action == "risk_pl":
            try:
                price_change = float(request.form.get("price_change", 0.0))
                vol_change = float(request.form.get("vol_change", 0.0))

                base_barriers = np.full(N, barrier_levels)
                base_coupons = np.full(N, coupon_rates)

                option = AutoMonteCarlo(ticker, K, r, sigma, T, q, N, M)
                risk_pl_results = option.risk_pl_analysis(
                    discretization=discretization,
                    barrier_levels=base_barriers,
                    coupon_rates=base_coupons,
                    price_change=price_change,
                    vol_change=vol_change,
                )

                session["risk_pl_results"] = risk_pl_results

                return render_template(
                    "autocallables.html",
                    option_price=None,
                    form_data=form_data,
                    sensitivity_results=sensitivity_results,
                    convergence_results=convergence_results,
                    scenario_results=scenario_results,
                    risk_pl_results=risk_pl_results,
                    md_content=md_content,
                )

            except Exception:
                logger.exception("An error occurred during autocallable RBPL analysis")
                risk_pl_results = None

                return render_template(
                    "autocallables.html",
                    option_price=None,
                    form_data=form_data,
                    sensitivity_results=sensitivity_results,
                    convergence_results=convergence_results,
                    scenario_results=scenario_results,
                    risk_pl_results=risk_pl_results,
                    md_content=md_content,
                )

        if simulation_engine == "new_MC":
            try:
                from ..models.market_data import StockData

                stock_data = StockData(ticker)
                S0 = float(stock_data.get_current_price())
                mc_engine = monte_carlo_New_module.create_monte_carlo_engine(
                    S0=S0,
                    r=r,
                    sigma=sigma,
                    T=T,
                    num_paths=M,
                    num_steps=N,
                    random_type="sobol",
                )
                option_price = mc_engine.price_autocallable_option(
                    strike_price=K,
                    barrier_levels=barrier_levels,
                    coupon_rates=coupon_rates,
                    T=T,
                    option_type=option_type,
                    discretization=discretization,
                    dividend_yield=q,
                )
                option_price = "${:,.4f}".format(option_price)
            except Exception:
                logger.exception("Error using new MC engine for autocallable pricing")
                option = AutoMonteCarlo(ticker, K, r, sigma, T, q, N, M)
                option_price = option.price_autocallable_option(
                    discretization=discretization,
                    barrier_levels=barrier_levels,
                    coupon_rates=coupon_rates,
                )
                option_price = "${:,.4f}".format(option_price)
        else:
            option = AutoMonteCarlo(ticker, K, r, sigma, T, q, N, M)
            option_price = option.price_autocallable_option(
                discretization=discretization,
                barrier_levels=barrier_levels,
                coupon_rates=coupon_rates,
            )
            option_price = "${:,.4f}".format(option_price)

    return render_template(
        "autocallables.html",
        option_price=option_price,
        form_data=form_data,
        sensitivity_results=sensitivity_results,
        convergence_results=convergence_results,
        scenario_results=scenario_results,
        risk_pl_results=risk_pl_results,
        structured_results=structured_results,
        structured_form_data={},
        md_content=md_content,
    )


@exotic_options_bp.route("/asian", methods=["GET", "POST"])
def asian_options():
    readme_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "asian_options.md"
    )
    with open(readme_path, "r") as readme_file:
        content = readme_file.read()
    md_content = markdown.markdown(content)

    option_price = sensitivity_results = scenario_results = convergence_results = (
        risk_pl_results
    ) = None
    form_data = {}

    if request.method == "POST":
        action = request.form.get("analysis_type")
        form_data = {
            "ticker": request.form["ticker"],
            "K": request.form["K"],
            "r": request.form["r"],
            "sigma": request.form["sigma"],
            "T": request.form["T"],
            "q": request.form["q"],
            "averaging_dates": request.form["averaging_dates"],
            "num_paths": request.form["num_paths"],
            "option_type": request.form["option_type"],
            "simulation_engine": request.form.get("simulation_engine", "original"),
        }

        ticker = form_data["ticker"]
        K = float(form_data["K"]) if form_data["K"] not in [None, ""] else 0.0
        r = float(form_data["r"]) if form_data["r"] not in [None, ""] else 0.0
        sigma = (
            float(form_data["sigma"]) if form_data["sigma"] not in [None, ""] else 0.0
        )
        T_str = form_data["T"]
        T = datetime.strptime(T_str.strip(), "%Y-%m-%d")
        q = float(form_data["q"]) if form_data["q"] not in [None, ""] else 0.0
        averaging_dates_str = form_data["averaging_dates"]
        averaging_dates = [
            datetime.strptime(date.strip(), "%Y-%m-%d")
            for date in averaging_dates_str.split(",")
        ]
        num_paths = (
            int(form_data["num_paths"])
            if form_data["num_paths"] not in [None, ""]
            else 0
        )
        option_type = form_data["option_type"]
        simulation_engine = form_data.get("simulation_engine", "original")

        if action == "sensitivity":
            try:
                num_steps = int(request.form.get("num_sensitivity_steps", 50))
                step_range = float(request.form.get("step_range", 0.1))

                variable = request.form.get("variable", "strike_price")
                target_variable = request.form.get("target_variable", "option_price")

                tester = AsianOptionSmoothnessTest(
                    ticker, K, sigma, r, q, T, averaging_dates, option_type, num_paths
                )

                values, outputs = tester.calculate_greeks_over_range(
                    variable, num_steps, step_range, target_variable
                )

                plot_path = tester.plot_single_greek(
                    values, outputs, target_variable, variable
                )
                plot_filename = os.path.basename(plot_path)

                session["sensitivity_results"] = {
                    "variable": variable,
                    "values": values.tolist()
                    if hasattr(values, "tolist")
                    else list(values),
                    "greek_values": outputs,
                    "target_variable": target_variable,
                    "plot_filename": plot_filename,
                }
                sensitivity_results = True

                return render_template(
                    "asian_options.html",
                    option_price=None,
                    sensitivity_results=sensitivity_results,
                    convergence_results=convergence_results,
                    scenario_results=scenario_results,
                    risk_pl_results=risk_pl_results,
                    form_data=form_data,
                    md_content=md_content,
                )

            except Exception:
                logger.exception("An error occurred during Asian sensitivity analysis")
                sensitivity_results = None

        elif action == "scenario":
            try:
                spot_change = float(request.form.get("spot_scenario", 0))
                vol_change = float(request.form.get("vol_scenario", 0))
                rate_change = float(request.form.get("rate_scenario", 0))

                base_option = AsianOption(
                    ticker, K, sigma, r, q, T, averaging_dates, option_type, num_paths
                )
                base_spot = float(base_option.S)
                base_greeks = base_option.calculate_greeks()

                baseline_price = "{:.4f}".format(base_greeks["option_price"])
                baseline_delta = "{:.4f}".format(base_greeks["Delta"])
                baseline_gamma = "{:.4f}".format(base_greeks["Gamma"])
                baseline_vega = "{:.4f}".format(base_greeks["Vega"])
                baseline_theta = "{:.4f}".format(base_greeks["Theta"])
                baseline_rho = "{:.4f}".format(base_greeks["Rho"])

                stressed_S = base_spot * (1 + spot_change)
                stressed_sigma = sigma + vol_change
                stressed_r = r + rate_change

                stressed_option = AsianOption(
                    ticker,
                    K,
                    stressed_sigma,
                    stressed_r,
                    q,
                    T,
                    averaging_dates,
                    option_type,
                    num_paths,
                    S0=stressed_S,
                )
                stressed_greeks = stressed_option.calculate_greeks()

                stressed_price = "{:.4f}".format(stressed_greeks["option_price"])
                stressed_delta = "{:.4f}".format(stressed_greeks["Delta"])
                stressed_gamma = "{:.4f}".format(stressed_greeks["Gamma"])
                stressed_vega = "{:.4f}".format(stressed_greeks["Vega"])
                stressed_theta = "{:.4f}".format(stressed_greeks["Theta"])
                stressed_rho = "{:.4f}".format(stressed_greeks["Rho"])

                session["scenario_results"] = {
                    "baseline_scenario_table": {
                        "baseline_price": baseline_price,
                        "baseline_delta": baseline_delta,
                        "baseline_gamma": baseline_gamma,
                        "baseline_vega": baseline_vega,
                        "baseline_theta": baseline_theta,
                        "baseline_rho": baseline_rho,
                    },
                    "stressed_scenario_table": {
                        "stressed_price": stressed_price,
                        "stressed_delta": stressed_delta,
                        "stressed_gamma": stressed_gamma,
                        "stressed_vega": stressed_vega,
                        "stressed_theta": stressed_theta,
                        "stressed_rho": stressed_rho,
                    },
                    "gpt_scenario_assessment": "No assessment yet.",
                }
                scenario_results = True

                return render_template(
                    "asian_options.html",
                    option_price=None,
                    sensitivity_results=sensitivity_results,
                    convergence_results=convergence_results,
                    scenario_results=scenario_results,
                    risk_pl_results=risk_pl_results,
                    form_data=form_data,
                    md_content=md_content,
                )

            except Exception:
                logger.exception("An error occurred during Asian scenario analysis")
                scenario_results = None

        elif action == "convergence":
            try:
                mode = request.form.get("mode", "steps")
                max_steps = int(request.form.get("max_steps", 100))
                obs = int(request.form.get("obs", 10))

                pricer_params = {
                    "ticker": ticker,
                    "K": K,
                    "sigma": sigma,
                    "r": r,
                    "q": q,
                    "T": T,
                    "averaging_dates": averaging_dates,
                    "option_type": option_type,
                    "num_paths": num_paths,
                }

                results = lattice_convergence_test(
                    max_steps=max_steps,
                    max_sims=0,
                    obs=obs,
                    pricer_class=AsianOption,
                    pricer_params=pricer_params,
                    mode=mode,
                )

                plot_path = asian_plot_convergence(results, mode)
                plot_filename = os.path.basename(plot_path)
                session["asian_convergence_plot"] = plot_filename
                convergence_results = True

                return render_template(
                    "asian_options.html",
                    option_price=None,
                    sensitivity_results=sensitivity_results,
                    convergence_results=convergence_results,
                    scenario_results=scenario_results,
                    risk_pl_results=risk_pl_results,
                    form_data=form_data,
                    md_content=md_content,
                )

            except Exception:
                logger.exception("An error occurred during Asian convergence analysis")
                convergence_results = None

                return render_template(
                    "asian_options.html",
                    option_price=None,
                    sensitivity_results=sensitivity_results,
                    convergence_results=convergence_results,
                    scenario_results=scenario_results,
                    risk_pl_results=risk_pl_results,
                    form_data=form_data,
                    md_content=md_content,
                )

        elif action == "risk_pl":
            try:
                price_change = float(request.form.get("price_change", 0.0))
                vol_change = float(request.form.get("vol_change", 0.0))

                option = AsianOption(
                    ticker, K, sigma, r, q, T, averaging_dates, option_type, num_paths
                )
                risk_pl_results = option.risk_pl_analysis(
                    price_change=price_change, vol_change=vol_change
                )

                return render_template(
                    "asian_options.html",
                    option_price=None,
                    sensitivity_results=sensitivity_results,
                    convergence_results=convergence_results,
                    scenario_results=scenario_results,
                    risk_pl_results=risk_pl_results,
                    form_data=form_data,
                    md_content=md_content,
                )

            except Exception:
                logger.exception("An error occurred during Asian RBPL analysis")
                risk_pl_results = None

        if simulation_engine == "new_MC":
            try:
                from ..models.market_data import StockData

                stock_data = StockData(ticker)
                S0 = float(stock_data.get_current_price())
                averaging_dates_sorted = sorted(averaging_dates)
                num_steps = len(averaging_dates_sorted) - 1
                T_years = (
                    averaging_dates_sorted[-1] - averaging_dates_sorted[0]
                ).days / 365.25

                mc_engine = monte_carlo_New_module.create_monte_carlo_engine(
                    S0=S0,
                    r=r,
                    sigma=sigma,
                    T=T_years,
                    num_paths=num_paths,
                    num_steps=num_steps,
                    random_type="sobol",
                )

                option_price = mc_engine.price_asian_option(
                    strike_price=K,
                    averaging_dates=averaging_dates_sorted,
                    option_type=option_type,
                    dividend_yield=q,
                )
                option_price = "${:,.4f}".format(option_price)
            except Exception as e:
                logger.exception("Error using new MC engine for Asian pricing")
                option_price = f"New MC error: {e}"
        else:
            option = AsianOption(
                ticker, K, sigma, r, q, T, averaging_dates, option_type, num_paths
            )
            option_price = option.price()
            option_price = "${:,.4f}".format(option_price)

    return render_template(
        "asian_options.html",
        option_price=option_price,
        sensitivity_results=sensitivity_results,
        convergence_results=convergence_results,
        scenario_results=scenario_results,
        risk_pl_results=risk_pl_results,
        form_data=form_data,
        md_content=md_content,
    )


@exotic_options_bp.route("/barrier", methods=["GET", "POST"])
def barrier_options():
    readme_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "barrier_options.md"
    )
    with open(readme_path, "r") as readme_file:
        content = readme_file.read()
    md_content = markdown.markdown(content)

    option_price = sensitivity_results = scenario_results = convergence_results = (
        risk_pl_results
    ) = None
    form_data = {}

    if request.method == "POST":
        action = request.form.get("analysis_type")

        form_data = {
            "ticker": request.form["ticker"],
            "K": float(request.form["K"])
            if request.form["K"] not in [None, ""]
            else 0.0,
            "r": float(request.form["r"])
            if request.form["r"] not in [None, ""]
            else 0.0,
            "sigma": float(request.form["sigma"])
            if request.form["sigma"] not in [None, ""]
            else 0.0,
            "start_date": str(request.form["start_date"]),
            "end_date": str(request.form["end_date"]),
            "q": float(request.form["q"])
            if request.form["q"] not in [None, ""]
            else 0.0,
            "N": int(request.form["N"]) if request.form["N"] not in [None, ""] else 0,
            "M": int(request.form["M"]) if request.form["M"] not in [None, ""] else 0,
            "barrier": float(request.form["barrier"])
            if request.form["barrier"] not in [None, ""]
            else 0.0,
            "option_type": request.form["option_type"],
            "barrier_type": request.form["barrier_type"],
            "discretization": request.form["discretization"],
            "simulation_engine": request.form.get("simulation_engine", "original"),
        }

        if action == "sensitivity":
            try:
                form_data["num_sensitivity_steps"] = int(
                    request.form["num_sensitivity_steps"]
                )
                form_data["step_range"] = int(request.form["step_range"])
                form_data["variable"] = request.form["variable"]
                form_data["target_variable"] = request.form["target_variable"]

                num_steps = form_data["num_sensitivity_steps"]
                step_range = form_data["step_range"]
                variable = form_data["variable"]
                target_variable = form_data["target_variable"]

                tester = MonteCarloSmoothnessTest(**form_data)

                values, greek_values = tester.calculate_greeks_over_range(
                    variable, num_steps, step_range, target_variable
                )
                tester.plot_single_greek(
                    values, greek_values, target_variable, variable
                )

                logger.debug("Starting barrier sensitivity plot generation")
                base_dir = os.path.dirname(os.path.abspath(__file__))
                static_dir = os.path.join(base_dir, "..", "static")
                os.makedirs(static_dir, exist_ok=True)
                plot_filename = f"barrier_{target_variable}-{variable}_sensitivity_plot_{uuid.uuid4().hex}.png"
                plot_path = os.path.join(static_dir, plot_filename)

                logger.debug("Saving barrier sensitivity plot to: %s", plot_path)
                plt.savefig(plot_path)

                session["sensitivity_results"] = {
                    "variable": variable,
                    "values": values.tolist(),
                    "greek_values": greek_values,
                    "target_variable": target_variable,
                    "plot_filename": plot_filename,
                }

                sensitivity_results = True
                plt.close()

            except Exception:
                logger.exception(
                    "An error occurred during barrier sensitivity analysis"
                )
                sensitivity_results = None

        elif action == "convergence":
            form_data["max_steps"] = int(request.form["max_steps"])
            form_data["max_sims"] = int(request.form["max_sims"])
            form_data["obs"] = int(request.form["obs"])
            form_data["mode"] = request.form["mode"]

            mode = form_data["mode"]
            max_steps = form_data["max_steps"]
            max_sims = form_data["max_sims"]
            obs = form_data["obs"]

            barrier_step_results = convergence_test(
                max_steps, max_sims, obs, MonteCarlo, form_data, mode
            )

            mc_plot_convergence(barrier_step_results, mode)

            base_dir = os.path.dirname(os.path.abspath(__file__))
            static_dir = os.path.join(base_dir, "..", "static")
            os.makedirs(static_dir, exist_ok=True)
            plot_filename = f"barrier_convergence_plot_{uuid.uuid4().hex}.png"
            plot_path = os.path.join(static_dir, plot_filename)
            plt.savefig(plot_path)
            session["barrier_convergence_plot"] = plot_filename
            convergence_results = True

        elif action == "scenario":
            try:
                spot_change = float(request.form.get("spot_scenario", 0))
                vol_change = float(request.form.get("vol_scenario", 0))
                rate_change = float(request.form.get("rate_scenario", 0))

                strike_price_raw = form_data.get("K")
                strike_price = (
                    float(str(strike_price_raw).strip())
                    if strike_price_raw not in [None, ""]
                    and str(strike_price_raw).strip() != ""
                    else 0.0
                )
                risk_free_rate_raw = form_data.get("r")
                risk_free_rate = (
                    float(str(risk_free_rate_raw).strip())
                    if risk_free_rate_raw not in [None, ""]
                    and str(risk_free_rate_raw).strip() != ""
                    else 0.0
                )
                volatility_raw = form_data.get("sigma")
                volatility = (
                    float(str(volatility_raw).strip())
                    if volatility_raw not in [None, ""]
                    and str(volatility_raw).strip() != ""
                    else 0.0
                )

                option = MonteCarlo(**form_data)
                baseline_price = option.price_barrier_option()
                baseline_greeks = option.calculate_greeks()
                baseline_price = "{:.4f}".format(baseline_price)
                baseline_delta = "{:.4f}".format(baseline_greeks["Delta"])
                baseline_gamma = "{:.4f}".format(baseline_greeks["Gamma"])
                baseline_vega = "{:.4f}".format(baseline_greeks["Vega"])
                baseline_theta = "{:.4f}".format(baseline_greeks["Theta"])
                baseline_rho = "{:.4f}".format(baseline_greeks["Rho"])

                stressed_spot = strike_price * (1 + spot_change)
                stressed_vol = volatility + vol_change
                stressed_rate = risk_free_rate + rate_change

                stressed_input = {
                    "ticker": form_data["ticker"],
                    "start_date": form_data["start_date"],
                    "end_date": form_data["end_date"],
                    "r": stressed_rate,
                    "sigma": stressed_vol,
                    "N": form_data["N"],
                    "M": form_data["M"],
                    "K": stressed_spot,
                    "q": form_data["q"],
                    "barrier": form_data["barrier"],
                    "option_type": form_data["option_type"],
                    "barrier_type": form_data["barrier_type"],
                }

                stressed_option = MonteCarlo(**stressed_input)
                stressed_price = stressed_option.price_barrier_option()
                stressed_greeks = stressed_option.calculate_greeks()
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

                logger.debug(
                    "Barrier baseline scenario table: %s", baseline_scenario_table
                )
                logger.debug(
                    "Barrier stressed scenario table: %s", stressed_scenario_table
                )

                session["scenario_results"] = {
                    "baseline_scenario_table": baseline_scenario_table,
                    "stressed_scenario_table": stressed_scenario_table,
                    "gpt_scenario_assessment": "No assessment yet.",
                }

                scenario_results = True

            except Exception:
                logger.exception("An error occurred during barrier scenario analysis")
                scenario_results = None

        elif action == "ai_scenario_assessment":
            scenario_results = session.get("scenario_results", {})
            baseline_table = scenario_results.get("baseline_scenario_table", {})
            stressed_table = scenario_results.get("stressed_scenario_table", {})

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
                assessment_input = f"Please assess the scenario analysis of the option price and Greeks based on the following results: {table_text}. Please limit the assessment to be less than 100 words."
                gpt_scenario_assessment = ask_gpt(assessment_input)

                scenario_results["gpt_scenario_assessment"] = gpt_scenario_assessment
                session["scenario_results"] = scenario_results
            else:
                scenario_results["gpt_scenario_assessment"] = (
                    "No scenario analysis results available for assessment."
                )
                session["scenario_results"] = scenario_results

        elif action == "risk_pl":
            try:
                form_data["price_change"] = float(request.form["price_change"])
                form_data["vol_change"] = float(request.form["vol_change"])

                price_change = form_data["price_change"]
                vol_change = form_data["vol_change"]

                option = MonteCarlo(**form_data)
                risk_pl_results = option.risk_pl_analysis(
                    price_change, vol_change, discretization=form_data["discretization"]
                )

                logger.debug("Barrier risk-based P&L results: %s", risk_pl_results)

            except Exception:
                logger.exception(
                    "An error occurred during barrier Risk-Based P&L analysis"
                )
                risk_pl_results = None

        else:
            simulation_engine = request.form.get("simulation_engine", "original")

            if simulation_engine == "new_MC":
                try:
                    from ..models.market_data import StockData

                    stock_data = StockData(
                        form_data["ticker"],
                        form_data["start_date"],
                        form_data["end_date"],
                    )
                    S0 = float(stock_data.get_closing_price())
                    T = stock_data.get_years_difference()

                    mc_engine = monte_carlo_New_module.create_monte_carlo_engine(
                        S0=S0,
                        r=form_data["r"],
                        sigma=form_data["sigma"],
                        T=T,
                        num_paths=form_data["M"],
                        num_steps=form_data["N"],
                        random_type="sobol",
                    )

                    option_price = mc_engine.price_barrier_option(
                        strike_price=form_data["K"],
                        barrier_level=form_data["barrier"],
                        option_type=form_data["option_type"],
                        barrier_type=form_data["barrier_type"],
                        dividend_yield=form_data["q"],
                    )
                    option_price = "${:,.4f}".format(option_price)

                except Exception:
                    logger.exception("Error using new MC engine for barrier pricing")
                    option = MonteCarlo(**form_data)
                    option_price = option.price_barrier_option()
                    option_price = "${:,.4f}".format(option_price)
            else:
                option = MonteCarlo(**form_data)
                option_price = option.price_barrier_option()
                option_price = "${:,.4f}".format(option_price)

    return render_template(
        "barrier_options.html",
        option_price=option_price,
        form_data=form_data,
        sensitivity_results=sensitivity_results,
        convergence_results=convergence_results,
        scenario_results=scenario_results,
        risk_pl_results=risk_pl_results,
        md_content=md_content,
    )
