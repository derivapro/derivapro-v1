# Note: last updated on Aug 06

from datetime import datetime, timedelta
from flask import Blueprint, render_template, request, session
from flask_login import current_user

import os
import markdown
import math
import random as random_module
from dotenv import load_dotenv
from ..extensions import db
from ..models.db_models import AnalysisResult, Instrument, Plot, PricingResult
from ..utils.lazy_imports import LazyAttribute, LazyImport
import logging

logger = logging.getLogger(__name__)

np = LazyImport("numpy")
monte_carlo_module = LazyImport("derivapro.models.mdls_monte_carlo_v2")
llm_client = LazyAttribute("derivapro.llm", "llm_client")
StockData = LazyAttribute("derivapro.models.market_data", "StockData")
build_equity_market_reference = LazyAttribute(
    "derivapro.services.market_reference",
    "build_equity_market_reference",
)
AsianOption = LazyAttribute("derivapro.models.mdls_asian_options", "AsianOption")
AsianOptionSmoothnessTest = LazyAttribute(
    "derivapro.models.mdls_asian_options", "AsianOptionSmoothnessTest"
)
lattice_convergence_test = LazyAttribute(
    "derivapro.models.mdls_asian_options", "lattice_convergence_test"
)
asian_plot_convergence = LazyAttribute(
    "derivapro.models.mdls_asian_options", "plot_convergence"
)
AutoMonteCarlo = LazyAttribute("derivapro.models.mdls_autocallables", "AutoMonteCarlo")
AutocallableSmoothnessTest = LazyAttribute(
    "derivapro.models.mdls_autocallables", "AutocallableSmoothnessTest"
)
auto_convergence_test = LazyAttribute(
    "derivapro.models.mdls_autocallables", "auto_convergence_test"
)
AutocallableNoteTerms = LazyAttribute(
    "derivapro.models.mdls_structured_products", "AutocallableNoteTerms"
)
price_autocallable_note = LazyAttribute(
    "derivapro.models.mdls_structured_products", "price_autocallable_note"
)

exotic_options_bp = Blueprint("exotic_options", __name__)

load_dotenv()

# Get the values from the environment variables
model = os.getenv("LLM_MODEL", os.getenv("Model"))


def ask_gpt(question):
    """Send a request to the configured LLM provider and return text."""
    try:
        return llm_client.generate_response(prompt=question, model=model)
    except Exception as e:
        logger.exception("Error occurred while calling LLM provider")
        return f"An error occurred: {e}"


def _parse_float_list(raw_value, default_values):
    if raw_value in [None, ""]:
        return list(default_values)
    return [float(item.strip()) for item in raw_value.split(",") if item.strip()]


def _format_currency(value):
    return "${:,.4f}".format(float(value))


def _format_percent(value):
    return "{:.2f}%".format(float(value) * 100)


def _get_latest_result_ids(user_id):
    latest_pricing = (
        PricingResult.query
        .filter_by(user_id=user_id)
        .order_by(PricingResult.created_at.desc())
        .first()
    )
    latest_analysis = (
        AnalysisResult.query
        .filter_by(user_id=user_id)
        .order_by(AnalysisResult.created_at.desc())
        .first()
    )
    return (
        latest_pricing.id if latest_pricing else None,
        latest_analysis.id if latest_analysis else None,
    )


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


def _year_fraction(start_date, end_date, day_count):
    days = (end_date - start_date).days
    if days <= 0:
        raise ValueError("Maturity date must be after valuation date.")
    if day_count == "ACT/360":
        return days / 360.0
    return days / 365.25


def _normal_cdf(value):
    return 0.5 * (1.0 + math.erf(value / math.sqrt(2.0)))


def _price_european_black_scholes(
    spot_price,
    strike_price,
    time_to_maturity,
    risk_free_rate,
    volatility,
    dividend_yield,
    option_type,
):
    if spot_price <= 0:
        raise ValueError("Spot price must be positive.")
    if strike_price <= 0:
        raise ValueError("Strike price must be positive.")
    if volatility <= 0:
        raise ValueError("Volatility must be positive.")
    if time_to_maturity <= 0:
        raise ValueError("Time to maturity must be positive.")

    sqrt_t = math.sqrt(time_to_maturity)
    d1 = (
        math.log(spot_price / strike_price)
        + (risk_free_rate - dividend_yield + 0.5 * volatility * volatility)
        * time_to_maturity
    ) / (volatility * sqrt_t)
    d2 = d1 - volatility * sqrt_t
    discounted_spot = spot_price * math.exp(-dividend_yield * time_to_maturity)
    discounted_strike = strike_price * math.exp(-risk_free_rate * time_to_maturity)

    call_price = discounted_spot * _normal_cdf(d1) - discounted_strike * _normal_cdf(d2)
    put_price = discounted_strike * _normal_cdf(-d2) - discounted_spot * _normal_cdf(-d1)
    return call_price if option_type == "call" else put_price


def _classify_moneyness(spot_price, strike_price, option_type):
    ratio = spot_price / strike_price
    if 0.97 <= ratio <= 1.03:
        return "At the money"
    if option_type == "call":
        return "In the money" if ratio > 1.03 else "Out of the money"
    return "In the money" if ratio < 0.97 else "Out of the money"


def _default_barrier_form_data():
    valuation_date = datetime.today().date()
    maturity_date = valuation_date + timedelta(days=365)
    return {
        "ticker": "AAPL",
        "strike_price": 200.0,
        "start_date": valuation_date.isoformat(),
        "end_date": maturity_date.isoformat(),
        "r": 0.04,
        "sigma": 0.25,
        "spot_price": 190.0,
        "dividend_yield": 0.005,
        "notional": 1,
        "contract_multiplier": 100.0,
        "day_count": "ACT/365",
        "option_type": "call",
        "barrier_type": "up_and_out",
        "barrier": 230.0,
        "num_steps": 252,
        "num_paths": 10000,
        "random_type": "sobol",
        "discretization": "euler",
    }


def _build_barrier_form_data_from_instrument(instrument):
    if not instrument:
        return {}

    params = instrument.params_json or {}
    return {
        "ticker": instrument.ticker or "AAPL",
        "strike_price": params.get("strike_price", params.get("K")),
        "start_date": instrument.start_date or "",
        "end_date": instrument.end_date or "",
        "r": params.get("risk_free_rate", params.get("r")),
        "sigma": params.get("volatility", params.get("sigma")),
        "spot_price": params.get("spot_price"),
        "dividend_yield": params.get("dividend_yield", params.get("q", 0.0)),
        "notional": params.get("notional", 1),
        "contract_multiplier": params.get("contract_multiplier", 100.0),
        "day_count": params.get("day_count", "ACT/365"),
        "option_type": params.get("option_type", "call"),
        "barrier_type": params.get("barrier_type", "up_and_out"),
        "barrier": params.get("barrier_level", params.get("barrier")),
        "num_steps": params.get("num_steps", params.get("N", 252)),
        "num_paths": params.get("num_paths", params.get("M", 10000)),
        "random_type": params.get("random_type", "sobol"),
        "discretization": params.get("discretization", "euler"),
    }


def _barrier_direction(barrier_type):
    return "up" if barrier_type.startswith("up") else "down"


def _barrier_activation_style(barrier_type):
    return "knock-in" if barrier_type.endswith("_in") else "knock-out"


def _price_barrier_mc(
    spot_price,
    strike_price,
    time_to_maturity,
    risk_free_rate,
    volatility,
    dividend_yield,
    option_type,
    barrier_type,
    barrier_level,
    num_paths,
    num_steps,
    random_type,
):
    engine = monte_carlo_module.create_monte_carlo_engine(
        S0=spot_price,
        r=risk_free_rate,
        sigma=volatility,
        T=time_to_maturity,
        num_paths=max(100, int(num_paths)),
        num_steps=max(2, int(num_steps)),
        random_type=random_type,
    )
    return float(
        engine.price_barrier_option(
            strike_price=strike_price,
            barrier_level=barrier_level,
            option_type=option_type,
            barrier_type=barrier_type,
            dividend_yield=dividend_yield,
        )
    )


def _barrier_greeks_finite_difference(
    form_data,
    spot_price,
    strike_price,
    time_to_maturity,
    base_price,
):
    num_paths = min(max(int(form_data.get("num_paths", 10000) * 0.25), 1000), 5000)
    num_steps = min(max(int(form_data.get("num_steps", 252)), 25), 252)
    risk_free_rate = form_data["r"]
    volatility = form_data["sigma"]
    dividend_yield = form_data.get("dividend_yield", 0.0)
    option_type = form_data["option_type"]
    barrier_type = form_data["barrier_type"]
    barrier_level = form_data["barrier"]
    random_type = form_data.get("random_type", "sobol")

    def price_at(spot=None, vol=None, rate=None, time=None):
        return _price_barrier_mc(
            spot if spot is not None else spot_price,
            strike_price,
            time if time is not None else time_to_maturity,
            rate if rate is not None else risk_free_rate,
            vol if vol is not None else volatility,
            dividend_yield,
            option_type,
            barrier_type,
            barrier_level,
            num_paths,
            num_steps,
            random_type,
        )

    spot_bump = max(spot_price * 0.01, 0.01)
    vol_bump = 0.01
    rate_bump = 0.0001
    time_bump = min(1.0 / 365.25, max(time_to_maturity / 2.0, 1e-6))

    price_spot_up = price_at(spot=spot_price + spot_bump)
    price_spot_down = price_at(spot=max(spot_price - spot_bump, 0.0001))
    price_vol_up = price_at(vol=volatility + vol_bump)
    price_vol_down = price_at(vol=max(volatility - vol_bump, 0.0001))
    price_rate_up = price_at(rate=risk_free_rate + rate_bump)
    price_rate_down = price_at(rate=risk_free_rate - rate_bump)
    shorter_price = price_at(time=max(time_to_maturity - time_bump, 1e-6))

    return {
        "delta": (price_spot_up - price_spot_down) / (2.0 * spot_bump),
        "gamma": (price_spot_up - 2.0 * base_price + price_spot_down)
        / (spot_bump * spot_bump),
        "vega": (price_vol_up - price_vol_down) / (2.0 * vol_bump),
        "theta": (shorter_price - base_price) / time_bump,
        "rho": (price_rate_up - price_rate_down) / (2.0 * rate_bump),
    }


def _estimate_barrier_breach_probability(
    spot_price,
    barrier_level,
    time_to_maturity,
    risk_free_rate,
    volatility,
    dividend_yield,
    barrier_type,
    num_steps,
    num_paths,
):
    path_count = min(max(int(num_paths * 0.2), 1000), 5000)
    steps = min(max(int(num_steps), 25), 252)
    dt = time_to_maturity / steps
    drift = (risk_free_rate - dividend_yield - 0.5 * volatility * volatility) * dt
    diffusion = volatility * math.sqrt(dt)
    rng = random_module.Random(1729)
    direction = _barrier_direction(barrier_type)
    breached = 0

    for _ in range(path_count):
        price = spot_price
        path_breached = False
        for _step in range(steps):
            price *= math.exp(drift + diffusion * rng.gauss(0.0, 1.0))
            if direction == "up" and price >= barrier_level:
                path_breached = True
                break
            if direction == "down" and price <= barrier_level:
                path_breached = True
                break
        if path_breached:
            breached += 1

    return breached / path_count


def _build_barrier_analytics(
    form_data,
    spot_price,
    strike_price,
    time_to_maturity,
    raw_option_price,
    greeks,
):
    option_type = form_data["option_type"]
    barrier_type = form_data["barrier_type"]
    barrier_level = form_data["barrier"]
    notional = form_data.get("notional", 1) or 1
    contract_multiplier = form_data.get("contract_multiplier", 100.0) or 100.0
    activation_style = _barrier_activation_style(barrier_type)
    direction = _barrier_direction(barrier_type)

    vanilla_price = _price_european_black_scholes(
        spot_price,
        strike_price,
        time_to_maturity,
        form_data["r"],
        form_data["sigma"],
        form_data.get("dividend_yield", 0.0),
        option_type,
    )
    intrinsic_value = (
        max(spot_price - strike_price, 0.0)
        if option_type == "call"
        else max(strike_price - spot_price, 0.0)
    )
    breach_probability = _estimate_barrier_breach_probability(
        spot_price,
        barrier_level,
        time_to_maturity,
        form_data["r"],
        form_data["sigma"],
        form_data.get("dividend_yield", 0.0),
        barrier_type,
        form_data.get("num_steps", 252),
        form_data.get("num_paths", 10000),
    )
    barrier_distance_pct = (barrier_level / spot_price) - 1.0
    premium_gap = vanilla_price - raw_option_price
    if activation_style == "knock-in":
        premium_interpretation = (
            "Knock-in value is expected to be lower than the matching vanilla "
            "option unless the activation event is already likely."
        )
    else:
        premium_interpretation = (
            "Knock-out value is expected to be lower than the matching vanilla "
            "option because a barrier breach extinguishes the payoff."
        )

    if direction == "up" and barrier_level <= spot_price:
        barrier_status = "Barrier is already at or below spot; review this up-barrier setup."
    elif direction == "down" and barrier_level >= spot_price:
        barrier_status = "Barrier is already at or above spot; review this down-barrier setup."
    else:
        barrier_status = (
            f"Barrier is {abs(barrier_distance_pct):.2%} "
            f"{'above' if barrier_distance_pct > 0 else 'below'} spot."
        )

    def local_sensitivity(driver, down_label, down_move, up_label, up_move):
        down = max(raw_option_price + down_move, 0.0)
        up = max(raw_option_price + up_move, 0.0)
        return {
            "driver": driver,
            "down_label": down_label,
            "down": down,
            "base": raw_option_price,
            "up_label": up_label,
            "up": up,
        }

    spot_move = spot_price * 0.10
    sensitivity_rows = [
        local_sensitivity(
            "Spot price",
            "-10%",
            greeks["delta"] * -spot_move + 0.5 * greeks["gamma"] * spot_move * spot_move,
            "+10%",
            greeks["delta"] * spot_move + 0.5 * greeks["gamma"] * spot_move * spot_move,
        ),
        local_sensitivity(
            "Volatility",
            "-5 vol pts",
            greeks["vega"] * -0.05,
            "+5 vol pts",
            greeks["vega"] * 0.05,
        ),
        local_sensitivity(
            "Risk-free rate",
            "-100 bps",
            greeks["rho"] * -0.01,
            "+100 bps",
            greeks["rho"] * 0.01,
        ),
    ]

    payoff_points = []
    for multiplier in [0.7, 0.85, 1.0, 1.15, 1.3]:
        underlying = spot_price * multiplier
        vanilla_payoff = (
            max(underlying - strike_price, 0.0)
            if option_type == "call"
            else max(strike_price - underlying, 0.0)
        )
        if activation_style == "knock-out":
            if direction == "up" and underlying >= barrier_level:
                payoff = 0.0
            elif direction == "down" and underlying <= barrier_level:
                payoff = 0.0
            else:
                payoff = vanilla_payoff
        else:
            if direction == "up" and underlying >= barrier_level:
                payoff = vanilla_payoff
            elif direction == "down" and underlying <= barrier_level:
                payoff = vanilla_payoff
            else:
                payoff = 0.0
        payoff_points.append({
            "underlying": underlying,
            "payoff": payoff,
            "net_payoff": payoff - raw_option_price,
        })

    return {
        "spot_price": spot_price,
        "time_to_maturity": time_to_maturity,
        "calendar_days": max(0, int(round(time_to_maturity * 365.25))),
        "moneyness_ratio": spot_price / strike_price,
        "moneyness_label": _classify_moneyness(spot_price, strike_price, option_type),
        "intrinsic_value": intrinsic_value,
        "time_value_proxy": raw_option_price - intrinsic_value,
        "position_value": raw_option_price * contract_multiplier * notional,
        "vanilla_price": vanilla_price,
        "premium_gap": premium_gap,
        "activation_style": activation_style,
        "barrier_direction": direction,
        "barrier_distance_pct": barrier_distance_pct,
        "barrier_status": barrier_status,
        "breach_probability": breach_probability,
        "survival_probability": 1.0 - breach_probability,
        "premium_interpretation": premium_interpretation,
        "sensitivity_rows": sensitivity_rows,
        "payoff_points": payoff_points,
    }


def _default_asian_form_data():
    valuation_date = datetime.today().date()
    maturity_date = valuation_date + timedelta(days=365)
    averaging_start = valuation_date + timedelta(days=30)
    return {
        "ticker": "AAPL",
        "strike_price": 200.0,
        "start_date": valuation_date.isoformat(),
        "end_date": maturity_date.isoformat(),
        "averaging_start_date": averaging_start.isoformat(),
        "averaging_end_date": maturity_date.isoformat(),
        "averaging_frequency": "monthly",
        "custom_averaging_dates": "",
        "r": 0.04,
        "sigma": 0.25,
        "spot_price": 190.0,
        "dividend_yield": 0.005,
        "notional": 1,
        "contract_multiplier": 100.0,
        "day_count": "ACT/365",
        "option_type": "call",
        "payoff_variant": "average_price",
        "average_type": "arithmetic",
        "num_paths": 10000,
        "seed": 42,
    }


def _build_asian_form_data_from_instrument(instrument):
    if not instrument:
        return {}

    params = instrument.params_json or {}
    return {
        "ticker": instrument.ticker or "AAPL",
        "strike_price": params.get("strike_price"),
        "start_date": instrument.start_date or "",
        "end_date": instrument.end_date or "",
        "averaging_start_date": params.get("averaging_start_date"),
        "averaging_end_date": params.get("averaging_end_date"),
        "averaging_frequency": params.get("averaging_frequency", "monthly"),
        "custom_averaging_dates": params.get("custom_averaging_dates", ""),
        "r": params.get("risk_free_rate"),
        "sigma": params.get("volatility"),
        "spot_price": params.get("spot_price"),
        "dividend_yield": params.get("dividend_yield", 0.0),
        "notional": params.get("notional", 1),
        "contract_multiplier": params.get("contract_multiplier", 100.0),
        "day_count": params.get("day_count", "ACT/365"),
        "option_type": params.get("option_type", "call"),
        "payoff_variant": params.get("payoff_variant", "average_price"),
        "average_type": params.get("average_type", "arithmetic"),
        "num_paths": params.get("num_paths", 10000),
        "seed": params.get("seed", 42),
    }


def _parse_date_value(value, field_name):
    if not value:
        raise ValueError(f"{field_name} is required.")
    return datetime.strptime(str(value), "%Y-%m-%d").date()


def _build_asian_averaging_dates(form_data):
    frequency = form_data.get("averaging_frequency", "monthly")
    if frequency == "custom":
        raw_dates = form_data.get("custom_averaging_dates", "")
        dates = [
            _parse_date_value(item.strip(), "Custom averaging date")
            for item in raw_dates.split(",")
            if item.strip()
        ]
        if not dates:
            raise ValueError("At least one custom averaging date is required.")
        return sorted(set(dates))

    start_date = _parse_date_value(
        form_data.get("averaging_start_date"),
        "Averaging start date",
    )
    end_date = _parse_date_value(
        form_data.get("averaging_end_date"),
        "Averaging end date",
    )
    if end_date < start_date:
        raise ValueError("Averaging end date must be on or after averaging start date.")

    step_days = {
        "daily": 1,
        "weekly": 7,
        "monthly": 30,
        "quarterly": 91,
    }.get(frequency, 30)
    dates = []
    current = start_date
    while current <= end_date:
        dates.append(current)
        current += timedelta(days=step_days)
    if dates[-1] != end_date:
        dates.append(end_date)
    return sorted(set(dates))


def _asian_price_mc(
    spot_price,
    strike_price,
    valuation_date,
    maturity_date,
    averaging_dates,
    risk_free_rate,
    volatility,
    dividend_yield,
    option_type,
    payoff_variant,
    average_type,
    num_paths,
    seed,
    day_count,
):
    if spot_price <= 0:
        raise ValueError("Spot price must be positive.")
    if strike_price <= 0:
        raise ValueError("Strike price must be positive.")
    if volatility <= 0:
        raise ValueError("Volatility must be positive.")
    if not averaging_dates:
        raise ValueError("At least one averaging date is required.")

    maturity_t = _year_fraction(valuation_date, maturity_date, day_count)
    observation_dates = sorted(set(averaging_dates + [maturity_date]))
    times = []
    for observation_date in observation_dates:
        if observation_date < valuation_date:
            raise ValueError("Averaging dates cannot be before valuation date.")
        if observation_date > maturity_date:
            raise ValueError("Averaging dates cannot be after maturity date.")
        times.append(0.0 if observation_date == valuation_date else _year_fraction(valuation_date, observation_date, day_count))

    path_count = min(max(int(num_paths), 100), 250000)
    rng = np.random.default_rng(int(seed))
    paths = np.empty((path_count, len(times)))
    previous_time = 0.0
    previous_spot = np.full(path_count, float(spot_price))
    for index, time_point in enumerate(times):
        dt = time_point - previous_time
        if dt > 0:
            shocks = rng.standard_normal(path_count)
            previous_spot = previous_spot * np.exp(
                (risk_free_rate - dividend_yield - 0.5 * volatility * volatility) * dt
                + volatility * math.sqrt(dt) * shocks
            )
        paths[:, index] = previous_spot
        previous_time = time_point

    averaging_indices = [observation_dates.index(date) for date in averaging_dates]
    averaging_paths = paths[:, averaging_indices]
    if average_type == "geometric":
        average_prices = np.exp(np.mean(np.log(np.maximum(averaging_paths, 1e-12)), axis=1))
    else:
        average_prices = np.mean(averaging_paths, axis=1)
    terminal_prices = paths[:, observation_dates.index(maturity_date)]

    if payoff_variant == "average_strike":
        if option_type == "call":
            payoffs = np.maximum(terminal_prices - average_prices, 0.0)
        else:
            payoffs = np.maximum(average_prices - terminal_prices, 0.0)
    else:
        if option_type == "call":
            payoffs = np.maximum(average_prices - strike_price, 0.0)
        else:
            payoffs = np.maximum(strike_price - average_prices, 0.0)

    discount = math.exp(-risk_free_rate * maturity_t)
    discounted_payoffs = discount * payoffs
    return {
        "price": float(np.mean(discounted_payoffs)),
        "standard_error": float(np.std(discounted_payoffs, ddof=1) / math.sqrt(path_count)),
        "average_underlying": float(np.mean(average_prices)),
        "terminal_underlying": float(np.mean(terminal_prices)),
        "payoff_mean": float(np.mean(payoffs)),
        "path_count": path_count,
        "maturity_t": maturity_t,
    }


def _asian_greeks_finite_difference(
    form_data,
    valuation_date,
    maturity_date,
    averaging_dates,
    base_price,
):
    spot_price = form_data["spot_price"]
    strike_price = form_data["strike_price"]
    volatility = form_data["sigma"]
    risk_free_rate = form_data["r"]
    dividend_yield = form_data.get("dividend_yield", 0.0)
    num_paths = min(max(int(form_data.get("num_paths", 10000) * 0.35), 1000), 7500)
    seed = int(form_data.get("seed", 42))

    def price_at(spot=None, vol=None, rate=None):
        return _asian_price_mc(
            spot if spot is not None else spot_price,
            strike_price,
            valuation_date,
            maturity_date,
            averaging_dates,
            rate if rate is not None else risk_free_rate,
            vol if vol is not None else volatility,
            dividend_yield,
            form_data["option_type"],
            form_data["payoff_variant"],
            form_data["average_type"],
            num_paths,
            seed,
            form_data.get("day_count", "ACT/365"),
        )["price"]

    spot_bump = max(spot_price * 0.01, 0.01)
    vol_bump = 0.01
    rate_bump = 0.0001
    price_spot_up = price_at(spot=spot_price + spot_bump)
    price_spot_down = price_at(spot=max(spot_price - spot_bump, 0.0001))
    price_vol_up = price_at(vol=volatility + vol_bump)
    price_vol_down = price_at(vol=max(volatility - vol_bump, 0.0001))
    price_rate_up = price_at(rate=risk_free_rate + rate_bump)
    price_rate_down = price_at(rate=risk_free_rate - rate_bump)

    return {
        "delta": (price_spot_up - price_spot_down) / (2.0 * spot_bump),
        "gamma": (price_spot_up - 2.0 * base_price + price_spot_down)
        / (spot_bump * spot_bump),
        "vega": (price_vol_up - price_vol_down) / (2.0 * vol_bump),
        "theta": None,
        "rho": (price_rate_up - price_rate_down) / (2.0 * rate_bump),
    }


def _build_asian_analytics(
    form_data,
    valuation_date,
    maturity_date,
    averaging_dates,
    pricing_output,
    greeks,
):
    raw_option_price = pricing_output["price"]
    notional = form_data.get("notional", 1) or 1
    contract_multiplier = form_data.get("contract_multiplier", 100.0) or 100.0
    strike_price = form_data["strike_price"]
    spot_price = form_data["spot_price"]
    option_type = form_data["option_type"]
    payoff_variant = form_data["payoff_variant"]

    vanilla_price = None
    averaging_discount = None
    if payoff_variant == "average_price":
        vanilla_price = _price_european_black_scholes(
            spot_price,
            strike_price,
            pricing_output["maturity_t"],
            form_data["r"],
            form_data["sigma"],
            form_data.get("dividend_yield", 0.0),
            option_type,
        )
        averaging_discount = vanilla_price - raw_option_price

    window_days = (max(averaging_dates) - min(averaging_dates)).days if averaging_dates else 0
    sensitivity_rows = []
    spot_move = spot_price * 0.10
    sensitivity_rows.append({
        "driver": "Spot price",
        "down_label": "-10%",
        "down": max(raw_option_price + greeks["delta"] * -spot_move + 0.5 * greeks["gamma"] * spot_move * spot_move, 0.0),
        "base": raw_option_price,
        "up_label": "+10%",
        "up": max(raw_option_price + greeks["delta"] * spot_move + 0.5 * greeks["gamma"] * spot_move * spot_move, 0.0),
    })
    sensitivity_rows.append({
        "driver": "Volatility",
        "down_label": "-5 vol pts",
        "down": max(raw_option_price + greeks["vega"] * -0.05, 0.0),
        "base": raw_option_price,
        "up_label": "+5 vol pts",
        "up": max(raw_option_price + greeks["vega"] * 0.05, 0.0),
    })
    sensitivity_rows.append({
        "driver": "Risk-free rate",
        "down_label": "-100 bps",
        "down": max(raw_option_price + greeks["rho"] * -0.01, 0.0),
        "base": raw_option_price,
        "up_label": "+100 bps",
        "up": max(raw_option_price + greeks["rho"] * 0.01, 0.0),
    })

    payoff_points = []
    for multiplier in [0.8, 0.9, 1.0, 1.1, 1.2]:
        average_level = pricing_output["average_underlying"] * multiplier
        terminal_level = pricing_output["terminal_underlying"] * multiplier
        if payoff_variant == "average_strike":
            payoff = (
                max(terminal_level - average_level, 0.0)
                if option_type == "call"
                else max(average_level - terminal_level, 0.0)
            )
            label = f"Terminal ${terminal_level:,.2f} / Avg ${average_level:,.2f}"
        else:
            payoff = (
                max(average_level - strike_price, 0.0)
                if option_type == "call"
                else max(strike_price - average_level, 0.0)
            )
            label = f"Average ${average_level:,.2f}"
        payoff_points.append({
            "label": label,
            "payoff": payoff,
            "net_payoff": payoff - raw_option_price,
        })

    variant_label = (
        "Average Strike (floating strike)"
        if payoff_variant == "average_strike"
        else "Average Price (fixed strike)"
    )
    return {
        "spot_price": spot_price,
        "calendar_days": (maturity_date - valuation_date).days,
        "averaging_count": len(averaging_dates),
        "averaging_window_days": window_days,
        "first_averaging_date": min(averaging_dates).isoformat(),
        "last_averaging_date": max(averaging_dates).isoformat(),
        "variant_label": variant_label,
        "average_type_label": form_data["average_type"].title(),
        "average_underlying": pricing_output["average_underlying"],
        "terminal_underlying": pricing_output["terminal_underlying"],
        "payoff_mean": pricing_output["payoff_mean"],
        "standard_error": pricing_output["standard_error"],
        "position_value": raw_option_price * contract_multiplier * notional,
        "vanilla_price": vanilla_price,
        "averaging_discount": averaging_discount,
        "moneyness_label": _classify_moneyness(spot_price, strike_price, option_type),
        "sensitivity_rows": sensitivity_rows,
        "payoff_points": payoff_points,
    }


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
    latest_analysis = None
    latest_pricing_result = None
    form_data = {}
    structured_form_data = {}
    latest_pricing_result_id = None
    latest_analysis_result_id = None

    if current_user.is_authenticated:
        latest_pricing_result_id, latest_analysis_result_id = _get_latest_result_ids(
            current_user.id
        )
        last_result_id = latest_pricing_result_id
        last_analysis_result_id = latest_analysis_result_id

        if last_result_id:
            latest_pricing_result = PricingResult.query.filter_by(
                id=last_result_id,
                user_id=current_user.id,
            ).first()
            if latest_pricing_result and latest_pricing_result.result_json:
                option_price_value = latest_pricing_result.result_json.get(
                    "option_price"
                )
                if option_price_value is not None:
                    option_price = "${:,.4f}".format(float(option_price_value))

        if last_analysis_result_id:
            latest_analysis = AnalysisResult.query.filter_by(
                id=last_analysis_result_id,
                user_id=current_user.id,
            ).first()
            if latest_analysis and latest_analysis.result_json:
                if latest_analysis.analysis_type == "autocallable_sensitivity":
                    sensitivity_results = latest_analysis.result_json
                elif latest_analysis.analysis_type == "autocallable_scenario":
                    scenario_results = latest_analysis.result_json
                elif latest_analysis.analysis_type == "autocallable_convergence":
                    convergence_results = latest_analysis.result_json
                elif latest_analysis.analysis_type == "autocallable_risk_pl":
                    risk_pl_results = latest_analysis.result_json
    else:
        sensitivity_results = None
        scenario_results = None
        convergence_results = None
        risk_pl_results = None

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
                    coupon_rate=float(
                        request.form.get("structured_coupon_rate", 0.025)
                    ),
                    coupon_barrier=float(
                        request.form.get("structured_coupon_barrier", 0.70)
                    ),
                    autocall_barrier=float(
                        request.form.get("structured_autocall_barrier", 1.00)
                    ),
                    protection_barrier=float(
                        request.form.get("structured_protection_barrier", 0.60)
                    ),
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
                    "autocall_probability": _format_percent(
                        raw_results["autocall_probability"]
                    ),
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
                sp_form=structured_form_data,
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

                serialized_values = (
                    values.tolist() if hasattr(values, "tolist") else list(values)
                )
                serialized_outputs = [
                    float(v) if isinstance(v, (np.floating, np.integer)) else v
                    for v in outputs
                ]

                sensitivity_results_data = {
                    "variable": variable,
                    "values": serialized_values,
                    "greek_values": serialized_outputs,
                    "target_variable": target_variable,
                    "plot_filename": plot_filename,
                }

                if current_user.is_authenticated:
                    instrument = Instrument(
                        user_id=current_user.id,
                        product_type="autocallable_option",
                        ticker=ticker,
                        model_name="monte_carlo_v2",
                        start_date=None,
                        end_date=None,
                        params_json={
                            "strike_price": K,
                            "risk_free_rate": r,
                            "volatility": sigma,
                            "time_to_expiry": T,
                            "dividend_yield": q,
                            "num_steps": N,
                            "num_paths": M,
                            "barrier_levels": barrier_levels,
                            "coupon_rates": coupon_rates,
                            "option_type": option_type,
                            "discretization": discretization,
                        },
                    )
                    db.session.add(instrument)
                    db.session.flush()

                    analysis_result = AnalysisResult(
                        user_id=current_user.id,
                        instrument_id=instrument.id,
                        pricing_result_id=latest_pricing_result_id,
                        analysis_type="autocallable_sensitivity",
                        result_json=sensitivity_results_data,
                    )
                    db.session.add(analysis_result)
                    db.session.flush()

                    plot = Plot(
                        user_id=current_user.id,
                        analysis_result_id=analysis_result.id,
                        pricing_result_id=latest_pricing_result_id,
                        plot_type="autocallable_sensitivity",
                        filename=plot_filename,
                        filepath=os.path.join("derivapro", "static", plot_filename),
                    )
                    db.session.add(plot)
                    db.session.commit()
                    latest_analysis = analysis_result
                else:
                    pass

                sensitivity_results = sensitivity_results_data

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

                scenario_results_data = {
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

                if current_user.is_authenticated:
                    instrument = Instrument(
                        user_id=current_user.id,
                        product_type="autocallable_option",
                        ticker=ticker,
                        model_name="monte_carlo_v2",
                        start_date=None,
                        end_date=None,
                        params_json={
                            "strike_price": K,
                            "risk_free_rate": r,
                            "volatility": sigma,
                            "time_to_expiry": T,
                            "dividend_yield": q,
                            "num_steps": N,
                            "num_paths": M,
                            "barrier_levels": barrier_levels,
                            "coupon_rates": coupon_rates,
                            "option_type": option_type,
                            "discretization": discretization,
                            "spot_scenario": spot_change,
                            "vol_scenario": vol_change,
                            "rate_scenario": rate_change,
                        },
                    )
                    db.session.add(instrument)
                    db.session.flush()

                    analysis_result = AnalysisResult(
                        user_id=current_user.id,
                        instrument_id=instrument.id,
                        pricing_result_id=latest_pricing_result_id,
                        analysis_type="autocallable_scenario",
                        result_json=scenario_results_data,
                    )
                    db.session.add(analysis_result)
                    db.session.commit()
                    latest_analysis = analysis_result
                else:
                    pass

                scenario_results = scenario_results_data

                return render_template(
                    "autocallables.html",
                    option_price=None,
                    form_data=form_data,
                    sensitivity_results=sensitivity_results,
                    convergence_results=convergence_results,
                    scenario_results=scenario_results,
                    risk_pl_results=risk_pl_results,
                    structured_results=structured_results,
                    structured_form_data=structured_form_data,
                    sp_form=structured_form_data,
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

                plot_path = monte_carlo_module.plot_convergence(results, mode)
                serialized_results = [(int(x), float(y)) for x, y in results]

                convergence_results_data = {
                    "results": serialized_results,
                    "mode": mode,
                    "plot_filename": os.path.basename(plot_path) if plot_path else None,
                }

                if current_user.is_authenticated:
                    instrument = Instrument(
                        user_id=current_user.id,
                        product_type="autocallable_option",
                        ticker=ticker,
                        model_name="monte_carlo_v2",
                        start_date=None,
                        end_date=None,
                        params_json={
                            "strike_price": K,
                            "risk_free_rate": r,
                            "volatility": sigma,
                            "time_to_expiry": T,
                            "dividend_yield": q,
                            "num_steps": N,
                            "num_paths": M,
                            "barrier_levels": barrier_levels,
                            "coupon_rates": coupon_rates,
                            "option_type": option_type,
                            "discretization": discretization,
                            "mode": mode,
                            "max_steps": num_steps_max,
                            "max_sims": max_sims,
                            "obs": obs,
                        },
                    )
                    db.session.add(instrument)
                    db.session.flush()

                    analysis_result = AnalysisResult(
                        user_id=current_user.id,
                        instrument_id=instrument.id,
                        pricing_result_id=latest_pricing_result_id,
                        analysis_type="autocallable_convergence",
                        result_json=convergence_results_data,
                    )
                    db.session.add(analysis_result)
                    db.session.flush()

                    plot = Plot(
                        user_id=current_user.id,
                        analysis_result_id=analysis_result.id,
                        pricing_result_id=latest_pricing_result_id,
                        plot_type="autocallable_convergence",
                        filename=convergence_results_data["plot_filename"],
                        filepath=os.path.join(
                            "derivapro",
                            "static",
                            convergence_results_data["plot_filename"],
                        ),
                    )
                    db.session.add(plot)
                    db.session.commit()
                    latest_analysis = analysis_result
                else:
                    pass

                convergence_results = convergence_results_data

                return render_template(
                    "autocallables.html",
                    option_price=None,
                    form_data=form_data,
                    sensitivity_results=sensitivity_results,
                    convergence_results=convergence_results,
                    scenario_results=scenario_results,
                    risk_pl_results=risk_pl_results,
                    structured_results=structured_results,
                    structured_form_data=structured_form_data,
                    sp_form=structured_form_data,
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

                if current_user.is_authenticated:
                    instrument = Instrument(
                        user_id=current_user.id,
                        product_type="autocallable_option",
                        ticker=ticker,
                        model_name="monte_carlo_v2",
                        start_date=None,
                        end_date=None,
                        params_json={
                            "strike_price": K,
                            "risk_free_rate": r,
                            "volatility": sigma,
                            "time_to_expiry": T,
                            "dividend_yield": q,
                            "num_steps": N,
                            "num_paths": M,
                            "barrier_levels": barrier_levels,
                            "coupon_rates": coupon_rates,
                            "option_type": option_type,
                            "discretization": discretization,
                            "price_change": price_change,
                            "vol_change": vol_change,
                        },
                    )
                    db.session.add(instrument)
                    db.session.flush()

                    analysis_result = AnalysisResult(
                        user_id=current_user.id,
                        instrument_id=instrument.id,
                        pricing_result_id=latest_pricing_result_id,
                        analysis_type="autocallable_risk_pl",
                        result_json=risk_pl_results,
                    )
                    db.session.add(analysis_result)
                    db.session.commit()
                    latest_analysis = analysis_result
                else:
                    pass

                return render_template(
                    "autocallables.html",
                    option_price=None,
                    form_data=form_data,
                    sensitivity_results=sensitivity_results,
                    convergence_results=convergence_results,
                    scenario_results=scenario_results,
                    risk_pl_results=risk_pl_results,
                    structured_results=structured_results,
                    structured_form_data=structured_form_data,
                    sp_form=structured_form_data,
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
                    structured_results=structured_results,
                    structured_form_data=structured_form_data,
                    sp_form=structured_form_data,
                    md_content=md_content,
                )

        try:
            stock_data = StockData(ticker)
            S0 = float(stock_data.get_current_price())
            mc_engine = monte_carlo_module.create_monte_carlo_engine(
                S0=S0,
                r=r,
                sigma=sigma,
                T=T,
                num_paths=M,
                num_steps=N,
                random_type="sobol",
            )

            raw_option_price = mc_engine.price_autocallable_option(
                strike_price=K,
                barrier_levels=barrier_levels,
                coupon_rates=coupon_rates,
                T=T,
                option_type=option_type,
                discretization=discretization,
                dividend_yield=q,
            )

            if current_user.is_authenticated:
                instrument = Instrument(
                    user_id=current_user.id,
                    product_type="autocallable_option",
                    ticker=ticker,
                    model_name="monte_carlo_v2",
                    start_date=None,
                    end_date=None,
                    params_json={
                        "strike_price": K,
                        "risk_free_rate": r,
                        "volatility": sigma,
                        "time_to_expiry": T,
                        "dividend_yield": q,
                        "num_steps": N,
                        "num_paths": M,
                        "barrier_levels": barrier_levels,
                        "coupon_rates": coupon_rates,
                        "option_type": option_type,
                        "discretization": discretization,
                    },
                )
                db.session.add(instrument)
                db.session.flush()

                pricing_result = PricingResult(
                    user_id=current_user.id,
                    instrument_id=instrument.id,
                    price=float(raw_option_price),
                    delta=None,
                    gamma=None,
                    vega=None,
                    theta=None,
                    rho=None,
                    result_json={
                        "option_price": float(raw_option_price),
                    },
                )
                db.session.add(pricing_result)
                db.session.commit()

            option_price = "${:,.4f}".format(raw_option_price)
        except Exception:
            logger.exception("Error using v2 MC engine for autocallable pricing")
            option_price = None

    return render_template(
        "autocallables.html",
        option_price=option_price,
        form_data=form_data,
        sensitivity_results=sensitivity_results,
        convergence_results=convergence_results,
        scenario_results=scenario_results,
        risk_pl_results=risk_pl_results,
        structured_results=structured_results,
        structured_form_data=structured_form_data,
        sp_form=structured_form_data,
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

    form_data = _default_asian_form_data()
    option_price = None
    delta = None
    gamma = None
    vega = None
    theta = None
    rho = None
    run_summary = None
    pricing_error = None

    if current_user.is_authenticated:
        latest_pricing_result = _get_latest_pricing_result_for_user("asian_option")
        if latest_pricing_result:
            option_price = _format_currency(latest_pricing_result.price)
            delta = "{:.4f}".format(float(latest_pricing_result.delta))
            gamma = "{:.6f}".format(float(latest_pricing_result.gamma))
            vega = "{:.4f}".format(float(latest_pricing_result.vega))
            theta = (
                "{:.4f}".format(float(latest_pricing_result.theta))
                if latest_pricing_result.theta is not None
                else "N/A"
            )
            rho = "{:.4f}".format(float(latest_pricing_result.rho))
            if latest_pricing_result.instrument:
                form_data.update(
                    _build_asian_form_data_from_instrument(
                        latest_pricing_result.instrument
                    )
                )
            if latest_pricing_result.result_json:
                run_summary = latest_pricing_result.result_json.get("run_summary")

    if request.method == "POST":
        try:
            form_data = {
                "ticker": request.form.get("ticker", "AAPL").upper().strip(),
                "strike_price": request.form.get("strike_price", type=float),
                "start_date": request.form.get("start_date"),
                "end_date": request.form.get("end_date"),
                "averaging_start_date": request.form.get("averaging_start_date"),
                "averaging_end_date": request.form.get("averaging_end_date"),
                "averaging_frequency": request.form.get("averaging_frequency", "monthly"),
                "custom_averaging_dates": request.form.get("custom_averaging_dates", ""),
                "r": request.form.get("r", type=float),
                "sigma": request.form.get("sigma", type=float),
                "spot_price": request.form.get("spot_price", type=float),
                "dividend_yield": request.form.get(
                    "dividend_yield", type=float, default=0.0
                ),
                "notional": request.form.get("notional", type=int, default=1),
                "contract_multiplier": request.form.get(
                    "contract_multiplier", type=float, default=100.0
                ),
                "day_count": request.form.get("day_count", "ACT/365"),
                "option_type": request.form.get("option_type", "call"),
                "payoff_variant": request.form.get("payoff_variant", "average_price"),
                "average_type": request.form.get("average_type", "arithmetic"),
                "num_paths": request.form.get("num_paths", type=int, default=10000),
                "seed": request.form.get("seed", type=int, default=42),
            }

            for field_name, label in [
                ("spot_price", "Spot price"),
                ("strike_price", "Strike price"),
                ("sigma", "Volatility"),
            ]:
                if form_data[field_name] is None or form_data[field_name] <= 0:
                    raise ValueError(f"{label} must be positive.")

            if form_data["option_type"] not in {"call", "put"}:
                raise ValueError("Option type must be call or put.")
            if form_data["payoff_variant"] not in {"average_price", "average_strike"}:
                raise ValueError("Unsupported Asian payoff variant.")
            if form_data["average_type"] not in {"arithmetic", "geometric"}:
                raise ValueError("Unsupported averaging type.")

            valuation_date = _parse_date_value(form_data["start_date"], "Valuation date")
            maturity_date = _parse_date_value(form_data["end_date"], "Maturity date")
            if maturity_date <= valuation_date:
                raise ValueError("Maturity date must be after valuation date.")

            averaging_dates = _build_asian_averaging_dates(form_data)
            pricing_output = _asian_price_mc(
                form_data["spot_price"],
                form_data["strike_price"],
                valuation_date,
                maturity_date,
                averaging_dates,
                form_data["r"],
                form_data["sigma"],
                form_data["dividend_yield"],
                form_data["option_type"],
                form_data["payoff_variant"],
                form_data["average_type"],
                form_data["num_paths"],
                form_data["seed"],
                form_data["day_count"],
            )
            raw_option_price = pricing_output["price"]
            raw_greeks = _asian_greeks_finite_difference(
                form_data,
                valuation_date,
                maturity_date,
                averaging_dates,
                raw_option_price,
            )
            run_summary = _build_asian_analytics(
                form_data,
                valuation_date,
                maturity_date,
                averaging_dates,
                pricing_output,
                raw_greeks,
            )
            session["asian_form_data"] = form_data.copy()

            option_price = _format_currency(raw_option_price)
            delta = "{:.4f}".format(raw_greeks["delta"])
            gamma = "{:.6f}".format(raw_greeks["gamma"])
            vega = "{:.4f}".format(raw_greeks["vega"])
            theta = "N/A"
            rho = "{:.4f}".format(raw_greeks["rho"])

            if current_user.is_authenticated:
                instrument = Instrument(
                    user_id=current_user.id,
                    product_type="asian_option",
                    ticker=form_data["ticker"],
                    model_name="monte_carlo_asian_variants",
                    start_date=form_data["start_date"],
                    end_date=form_data["end_date"],
                    params_json={
                        "strike_price": form_data["strike_price"],
                        "risk_free_rate": form_data["r"],
                        "volatility": form_data["sigma"],
                        "spot_price": form_data["spot_price"],
                        "dividend_yield": form_data["dividend_yield"],
                        "notional": form_data["notional"],
                        "contract_multiplier": form_data["contract_multiplier"],
                        "day_count": form_data["day_count"],
                        "option_type": form_data["option_type"],
                        "payoff_variant": form_data["payoff_variant"],
                        "average_type": form_data["average_type"],
                        "averaging_start_date": form_data["averaging_start_date"],
                        "averaging_end_date": form_data["averaging_end_date"],
                        "averaging_frequency": form_data["averaging_frequency"],
                        "custom_averaging_dates": form_data["custom_averaging_dates"],
                        "averaging_dates": [d.isoformat() for d in averaging_dates],
                        "num_paths": form_data["num_paths"],
                        "seed": form_data["seed"],
                    },
                )
                db.session.add(instrument)
                db.session.flush()

                pricing_result = PricingResult(
                    user_id=current_user.id,
                    instrument_id=instrument.id,
                    price=raw_option_price,
                    delta=raw_greeks["delta"],
                    gamma=raw_greeks["gamma"],
                    vega=raw_greeks["vega"],
                    theta=None,
                    rho=raw_greeks["rho"],
                    result_json={
                        "option_price": raw_option_price,
                        "delta": raw_greeks["delta"],
                        "gamma": raw_greeks["gamma"],
                        "vega": raw_greeks["vega"],
                        "theta": None,
                        "rho": raw_greeks["rho"],
                        "run_summary": run_summary,
                    },
                )
                db.session.add(pricing_result)
                db.session.commit()

        except Exception as exc:
            logger.exception("Error using Asian variant Monte Carlo pricing")
            db.session.rollback()
            pricing_error = str(exc)

    return render_template(
        "asian_options.html",
        option_price=option_price,
        form_data=form_data,
        delta=delta,
        gamma=gamma,
        vega=vega,
        theta=theta,
        rho=rho,
        md_content=md_content,
        run_summary=run_summary,
        pricing_error=pricing_error,
    )


@exotic_options_bp.route("/asian-legacy", methods=["GET", "POST"])
def asian_options_legacy():
    readme_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "asian_options.md"
    )
    with open(readme_path, "r") as readme_file:
        content = readme_file.read()
    md_content = markdown.markdown(content)

    option_price = sensitivity_results = scenario_results = convergence_results = (
        risk_pl_results
    ) = None
    latest_analysis = None
    latest_pricing_result = None
    form_data = {}
    latest_pricing_result_id = None
    latest_analysis_result_id = None

    if current_user.is_authenticated:
        latest_pricing_result_id, latest_analysis_result_id = _get_latest_result_ids(
            current_user.id
        )
        last_result_id = latest_pricing_result_id
        last_analysis_result_id = latest_analysis_result_id

        if last_result_id:
            latest_pricing_result = PricingResult.query.filter_by(
                id=last_result_id,
                user_id=current_user.id,
            ).first()
            if latest_pricing_result and latest_pricing_result.result_json:
                option_price_value = latest_pricing_result.result_json.get(
                    "option_price"
                )
                if option_price_value is not None:
                    option_price = "${:,.4f}".format(float(option_price_value))

        if last_analysis_result_id:
            latest_analysis = AnalysisResult.query.filter_by(
                id=last_analysis_result_id,
                user_id=current_user.id,
            ).first()
            if latest_analysis and latest_analysis.result_json:
                if latest_analysis.analysis_type == "asian_sensitivity":
                    sensitivity_results = latest_analysis.result_json
                elif latest_analysis.analysis_type == "asian_scenario":
                    scenario_results = latest_analysis.result_json
                elif latest_analysis.analysis_type == "asian_convergence":
                    convergence_results = latest_analysis.result_json
                elif latest_analysis.analysis_type == "asian_risk_pl":
                    risk_pl_results = latest_analysis.result_json
    else:
        sensitivity_results = None
        scenario_results = None
        convergence_results = None
        risk_pl_results = None

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

                serialized_values = (
                    values.tolist() if hasattr(values, "tolist") else list(values)
                )
                serialized_outputs = [
                    float(v) if isinstance(v, (np.floating, np.integer)) else v
                    for v in outputs
                ]

                sensitivity_results_data = {
                    "variable": variable,
                    "values": serialized_values,
                    "greek_values": serialized_outputs,
                    "target_variable": target_variable,
                    "plot_filename": plot_filename,
                }

                if current_user.is_authenticated:
                    instrument = Instrument(
                        user_id=current_user.id,
                        product_type="asian_option",
                        ticker=ticker,
                        model_name="monte_carlo_v2",
                        start_date=str(T_str),
                        end_date=str(averaging_dates[-1].date())
                        if averaging_dates
                        else None,
                        params_json={
                            "strike_price": K,
                            "risk_free_rate": r,
                            "volatility": sigma,
                            "dividend_yield": q,
                            "averaging_dates": [
                                d.strftime("%Y-%m-%d") for d in averaging_dates
                            ],
                            "num_paths": num_paths,
                            "option_type": option_type,
                        },
                    )
                    db.session.add(instrument)
                    db.session.flush()

                    analysis_result = AnalysisResult(
                        user_id=current_user.id,
                        instrument_id=instrument.id,
                        pricing_result_id=latest_pricing_result_id,
                        analysis_type="asian_sensitivity",
                        result_json=sensitivity_results_data,
                    )
                    db.session.add(analysis_result)
                    db.session.flush()

                    plot = Plot(
                        user_id=current_user.id,
                        analysis_result_id=analysis_result.id,
                        pricing_result_id=latest_pricing_result_id,
                        plot_type="asian_sensitivity",
                        filename=plot_filename,
                        filepath=os.path.join("derivapro", "static", plot_filename),
                    )
                    db.session.add(plot)
                    db.session.commit()
                    latest_analysis = analysis_result
                else:
                    pass

                sensitivity_results = sensitivity_results_data

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

                scenario_results_data = {
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

                if current_user.is_authenticated:
                    instrument = Instrument(
                        user_id=current_user.id,
                        product_type="asian_option",
                        ticker=ticker,
                        model_name="monte_carlo_v2",
                        start_date=str(T_str),
                        end_date=str(averaging_dates[-1].date())
                        if averaging_dates
                        else None,
                        params_json={
                            "strike_price": K,
                            "risk_free_rate": r,
                            "volatility": sigma,
                            "dividend_yield": q,
                            "averaging_dates": [
                                d.strftime("%Y-%m-%d") for d in averaging_dates
                            ],
                            "num_paths": num_paths,
                            "option_type": option_type,
                            "spot_scenario": spot_change,
                            "vol_scenario": vol_change,
                            "rate_scenario": rate_change,
                        },
                    )
                    db.session.add(instrument)
                    db.session.flush()

                    analysis_result = AnalysisResult(
                        user_id=current_user.id,
                        instrument_id=instrument.id,
                        pricing_result_id=latest_pricing_result_id,
                        analysis_type="asian_scenario",
                        result_json=scenario_results_data,
                    )
                    db.session.add(analysis_result)
                    db.session.commit()
                    latest_analysis = analysis_result
                else:
                    pass

                scenario_results = scenario_results_data

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
                max_sims = int(request.form.get("max_sims", num_paths))
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
                    max_sims=max_sims,
                    obs=obs,
                    pricer_class=AsianOption,
                    pricer_params=pricer_params,
                    mode=mode,
                )

                plot_path = asian_plot_convergence(results, mode)
                plot_filename = os.path.basename(plot_path)
                serialized_results = [(int(x), float(y)) for x, y in results]

                convergence_results_data = {
                    "results": serialized_results,
                    "mode": mode,
                    "plot_filename": plot_filename,
                }

                if current_user.is_authenticated:
                    instrument = Instrument(
                        user_id=current_user.id,
                        product_type="asian_option",
                        ticker=ticker,
                        model_name="monte_carlo_v2",
                        start_date=str(T_str),
                        end_date=str(averaging_dates[-1].date())
                        if averaging_dates
                        else None,
                        params_json={
                            "strike_price": K,
                            "risk_free_rate": r,
                            "volatility": sigma,
                            "dividend_yield": q,
                            "averaging_dates": [
                                d.strftime("%Y-%m-%d") for d in averaging_dates
                            ],
                            "num_paths": num_paths,
                            "option_type": option_type,
                            "mode": mode,
                            "max_steps": max_steps,
                            "max_sims": max_sims,
                            "obs": obs,
                        },
                    )
                    db.session.add(instrument)
                    db.session.flush()

                    analysis_result = AnalysisResult(
                        user_id=current_user.id,
                        instrument_id=instrument.id,
                        pricing_result_id=latest_pricing_result_id,
                        analysis_type="asian_convergence",
                        result_json=convergence_results_data,
                    )
                    db.session.add(analysis_result)
                    db.session.flush()

                    plot = Plot(
                        user_id=current_user.id,
                        analysis_result_id=analysis_result.id,
                        pricing_result_id=latest_pricing_result_id,
                        plot_type="asian_convergence",
                        filename=convergence_results_data["plot_filename"],
                        filepath=os.path.join(
                            "derivapro",
                            "static",
                            convergence_results_data["plot_filename"],
                        ),
                    )
                    db.session.add(plot)
                    db.session.commit()
                    latest_analysis = analysis_result
                else:
                    pass

                convergence_results = convergence_results_data

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

                if current_user.is_authenticated:
                    instrument = Instrument(
                        user_id=current_user.id,
                        product_type="asian_option",
                        ticker=ticker,
                        model_name="monte_carlo_v2",
                        start_date=str(T_str),
                        end_date=str(averaging_dates[-1].date())
                        if averaging_dates
                        else None,
                        params_json={
                            "strike_price": K,
                            "risk_free_rate": r,
                            "volatility": sigma,
                            "dividend_yield": q,
                            "averaging_dates": [
                                d.strftime("%Y-%m-%d") for d in averaging_dates
                            ],
                            "num_paths": num_paths,
                            "option_type": option_type,
                            "price_change": price_change,
                            "vol_change": vol_change,
                        },
                    )
                    db.session.add(instrument)
                    db.session.flush()

                    analysis_result = AnalysisResult(
                        user_id=current_user.id,
                        instrument_id=instrument.id,
                        pricing_result_id=latest_pricing_result_id,
                        analysis_type="asian_risk_pl",
                        result_json=risk_pl_results,
                    )
                    db.session.add(analysis_result)
                    db.session.commit()
                    latest_analysis = analysis_result
                else:
                    pass

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

        try:
            stock_data = StockData(ticker)
            S0 = float(stock_data.get_current_price())
            averaging_dates_sorted = sorted(averaging_dates)
            num_steps = len(averaging_dates_sorted) - 1
            T_years = (
                averaging_dates_sorted[-1] - averaging_dates_sorted[0]
            ).days / 365.25

            mc_engine = monte_carlo_module.create_monte_carlo_engine(
                S0=S0,
                r=r,
                sigma=sigma,
                T=T_years,
                num_paths=num_paths,
                num_steps=num_steps,
                random_type="sobol",
            )

            raw_option_price = mc_engine.price_asian_option(
                strike_price=K,
                averaging_dates=averaging_dates_sorted,
                option_type=option_type,
                dividend_yield=q,
            )

            greeks = mc_engine.calculate_greeks_finite_difference(
                strike_price=K,
                option_type=option_type,
                option_style="asian",
                averaging_dates=averaging_dates_sorted,
                dividend_yield=q,
            )

            if current_user.is_authenticated:
                instrument = Instrument(
                    user_id=current_user.id,
                    product_type="asian_option",
                    ticker=ticker,
                    model_name="monte_carlo_v2",
                    start_date=str(T_str),
                    end_date=str(averaging_dates_sorted[-1].date()),
                    params_json={
                        "strike_price": K,
                        "risk_free_rate": r,
                        "volatility": sigma,
                        "dividend_yield": q,
                        "averaging_dates": [
                            d.strftime("%Y-%m-%d") for d in averaging_dates_sorted
                        ],
                        "num_paths": num_paths,
                        "option_type": option_type,
                    },
                )
                db.session.add(instrument)
                db.session.flush()

                pricing_result = PricingResult(
                    user_id=current_user.id,
                    instrument_id=instrument.id,
                    price=float(raw_option_price),
                    delta=float(greeks["Delta"]),
                    gamma=float(greeks["Gamma"]),
                    vega=float(greeks["Vega"]),
                    theta=float(greeks["Theta"]),
                    rho=float(greeks["Rho"]),
                    result_json={
                        "option_price": float(raw_option_price),
                        "delta": float(greeks["Delta"]),
                        "gamma": float(greeks["Gamma"]),
                        "vega": float(greeks["Vega"]),
                        "theta": float(greeks["Theta"]),
                        "rho": float(greeks["Rho"]),
                    },
                )
                db.session.add(pricing_result)
                db.session.commit()

            option_price = "${:,.4f}".format(raw_option_price)
        except Exception:
            logger.exception("Error using v2 MC engine for Asian pricing")
            option_price = "Pricing error"

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

    form_data = _default_barrier_form_data()
    option_price = None
    delta = None
    gamma = None
    vega = None
    theta = None
    rho = None
    run_summary = None
    pricing_error = None
    market_reference = None
    market_error = None
    market_query = {
        "symbol": form_data["ticker"],
        "period": "6mo",
        "option_type": form_data["option_type"],
        "strike": form_data["strike_price"],
        "maturity_date": form_data["end_date"],
        "visual_mode": "none",
    }

    if current_user.is_authenticated:
        latest_pricing_result = _get_latest_pricing_result_for_user("barrier_option")
        if latest_pricing_result:
            option_price = _format_currency(latest_pricing_result.price)
            delta = "{:.4f}".format(float(latest_pricing_result.delta))
            gamma = "{:.6f}".format(float(latest_pricing_result.gamma))
            vega = "{:.4f}".format(float(latest_pricing_result.vega))
            theta = "{:.4f}".format(float(latest_pricing_result.theta))
            rho = "{:.4f}".format(float(latest_pricing_result.rho))

            if latest_pricing_result.instrument:
                form_data.update(
                    _build_barrier_form_data_from_instrument(
                        latest_pricing_result.instrument
                    )
                )
                market_query.update(
                    {
                        "symbol": form_data.get("ticker", market_query["symbol"]),
                        "option_type": form_data.get(
                            "option_type", market_query["option_type"]
                        ),
                        "strike": form_data.get("strike_price", market_query["strike"]),
                        "maturity_date": form_data.get(
                            "end_date", market_query["maturity_date"]
                        ),
                    }
                )
            if latest_pricing_result.result_json:
                run_summary = latest_pricing_result.result_json.get("run_summary")

    if request.method == "POST":
        action = request.form.get("analysis_type")

        if action == "market_reference":
            session_form_data = session.get("barrier_form_data", {})
            if session_form_data:
                form_data.update(session_form_data)

            market_query = {
                "symbol": request.form.get(
                    "market_symbol", form_data.get("ticker", "AAPL")
                )
                .upper()
                .strip(),
                "period": request.form.get("market_period", "6mo"),
                "option_type": request.form.get("market_option_type", "call"),
                "strike": request.form.get("market_strike", type=float),
                "maturity_date": request.form.get("market_maturity_date", ""),
                "visual_mode": request.form.get("visual_mode", "none"),
            }
            if market_query["symbol"]:
                form_data["ticker"] = market_query["symbol"]

            try:
                market_reference = build_equity_market_reference(
                    market_query["symbol"],
                    market_query["period"],
                    market_query["strike"],
                    market_query["maturity_date"],
                    market_query["option_type"],
                    market_query["visual_mode"],
                )
            except Exception as exc:
                logger.warning(
                    "Barrier market reference fetch failed for %s: %s",
                    market_query["symbol"],
                    exc,
                )
                market_error = str(exc)

            return render_template(
                "barrier_options.html",
                option_price=option_price,
                form_data=form_data,
                delta=delta,
                gamma=gamma,
                vega=vega,
                theta=theta,
                rho=rho,
                md_content=md_content,
                run_summary=run_summary,
                pricing_error=pricing_error,
                market_query=market_query,
                market_reference=market_reference,
                market_error=market_error,
            )

        def safe_int(raw_value, default):
            try:
                if raw_value in [None, ""]:
                    return default
                return int(raw_value)
            except (TypeError, ValueError):
                return default

        try:
            form_data = {
                "ticker": request.form.get("ticker", "AAPL").upper().strip(),
                "strike_price": request.form.get("strike_price", type=float),
                "start_date": str(request.form.get("start_date")),
                "end_date": str(request.form.get("end_date")),
                "r": request.form.get("r", type=float),
                "sigma": request.form.get("sigma", type=float),
                "spot_price": request.form.get("spot_price", type=float),
                "dividend_yield": request.form.get(
                    "dividend_yield", type=float, default=0.0
                ),
                "notional": request.form.get("notional", type=int, default=1),
                "contract_multiplier": request.form.get(
                    "contract_multiplier", type=float, default=100.0
                ),
                "day_count": request.form.get("day_count", "ACT/365"),
                "option_type": request.form.get("option_type", "call"),
                "barrier_type": request.form.get("barrier_type", "up_and_out"),
                "barrier": request.form.get("barrier", type=float),
                "num_steps": safe_int(request.form.get("num_steps"), 252),
                "num_paths": safe_int(request.form.get("num_paths"), 10000),
                "random_type": request.form.get("random_type", "sobol"),
                "discretization": request.form.get("discretization", "euler"),
            }

            required_positive_fields = [
                ("spot_price", "Spot price"),
                ("strike_price", "Strike price"),
                ("sigma", "Volatility"),
                ("barrier", "Barrier level"),
            ]
            for field_name, label in required_positive_fields:
                if form_data[field_name] is None or form_data[field_name] <= 0:
                    raise ValueError(f"{label} must be positive.")

            if form_data["option_type"] not in {"call", "put"}:
                raise ValueError("Option type must be call or put.")
            if form_data["barrier_type"] not in {
                "up_and_out",
                "down_and_out",
                "up_and_in",
                "down_and_in",
            }:
                raise ValueError("Unsupported barrier type.")

            start_date_obj = datetime.strptime(
                form_data["start_date"], "%Y-%m-%d"
            ).date()
            end_date_obj = datetime.strptime(form_data["end_date"], "%Y-%m-%d").date()
            time_to_maturity = _year_fraction(
                start_date_obj,
                end_date_obj,
                form_data["day_count"],
            )
            session["barrier_form_data"] = form_data.copy()
            market_query.update(
                {
                    "symbol": form_data.get("ticker", market_query["symbol"]),
                    "option_type": form_data.get(
                        "option_type", market_query["option_type"]
                    ),
                    "strike": form_data.get("strike_price", market_query["strike"]),
                    "maturity_date": form_data.get(
                        "end_date", market_query["maturity_date"]
                    ),
                }
            )

            raw_option_price = _price_barrier_mc(
                form_data["spot_price"],
                form_data["strike_price"],
                time_to_maturity,
                form_data["r"],
                form_data["sigma"],
                form_data["dividend_yield"],
                form_data["option_type"],
                form_data["barrier_type"],
                form_data["barrier"],
                form_data["num_paths"],
                form_data["num_steps"],
                form_data["random_type"],
            )
            raw_greeks = _barrier_greeks_finite_difference(
                form_data,
                form_data["spot_price"],
                form_data["strike_price"],
                time_to_maturity,
                raw_option_price,
            )
            run_summary = _build_barrier_analytics(
                form_data,
                form_data["spot_price"],
                form_data["strike_price"],
                time_to_maturity,
                raw_option_price,
                raw_greeks,
            )

            option_price = _format_currency(raw_option_price)
            delta = "{:.4f}".format(raw_greeks["delta"])
            gamma = "{:.6f}".format(raw_greeks["gamma"])
            vega = "{:.4f}".format(raw_greeks["vega"])
            theta = "{:.4f}".format(raw_greeks["theta"])
            rho = "{:.4f}".format(raw_greeks["rho"])

            if current_user.is_authenticated:
                instrument = Instrument(
                    user_id=current_user.id,
                    product_type="barrier_option",
                    ticker=form_data["ticker"],
                    model_name="monte_carlo_v2",
                    start_date=form_data["start_date"],
                    end_date=form_data["end_date"],
                    params_json={
                        "strike_price": form_data["strike_price"],
                        "risk_free_rate": form_data["r"],
                        "volatility": form_data["sigma"],
                        "spot_price": form_data["spot_price"],
                        "dividend_yield": form_data["dividend_yield"],
                        "notional": form_data["notional"],
                        "contract_multiplier": form_data["contract_multiplier"],
                        "day_count": form_data["day_count"],
                        "option_type": form_data["option_type"],
                        "barrier_type": form_data["barrier_type"],
                        "barrier_level": form_data["barrier"],
                        "num_steps": form_data["num_steps"],
                        "num_paths": form_data["num_paths"],
                        "random_type": form_data["random_type"],
                        "discretization": form_data["discretization"],
                    },
                )
                db.session.add(instrument)
                db.session.flush()

                pricing_result = PricingResult(
                    user_id=current_user.id,
                    instrument_id=instrument.id,
                    price=raw_option_price,
                    delta=raw_greeks["delta"],
                    gamma=raw_greeks["gamma"],
                    vega=raw_greeks["vega"],
                    theta=raw_greeks["theta"],
                    rho=raw_greeks["rho"],
                    result_json={
                        "option_price": raw_option_price,
                        "delta": raw_greeks["delta"],
                        "gamma": raw_greeks["gamma"],
                        "vega": raw_greeks["vega"],
                        "theta": raw_greeks["theta"],
                        "rho": raw_greeks["rho"],
                        "run_summary": run_summary,
                    },
                )
                db.session.add(pricing_result)
                db.session.commit()

        except Exception as exc:
            logger.exception("Error using v2 MC engine for barrier pricing")
            db.session.rollback()
            pricing_error = str(exc)

    return render_template(
        "barrier_options.html",
        option_price=option_price,
        form_data=form_data,
        delta=delta,
        gamma=gamma,
        vega=vega,
        theta=theta,
        rho=rho,
        md_content=md_content,
        run_summary=run_summary,
        pricing_error=pricing_error,
        market_query=market_query,
        market_reference=market_reference,
        market_error=market_error,
    )


@exotic_options_bp.route("/barrier-legacy", methods=["GET", "POST"])
def barrier_options_legacy():
    readme_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "barrier_options.md"
    )
    with open(readme_path, "r") as readme_file:
        content = readme_file.read()
    md_content = markdown.markdown(content)

    option_price = sensitivity_results = scenario_results = convergence_results = (
        risk_pl_results
    ) = None
    latest_analysis = None
    latest_pricing_result = None
    form_data = {}
    latest_pricing_result_id = None
    latest_analysis_result_id = None

    if current_user.is_authenticated:
        latest_pricing_result_id, latest_analysis_result_id = _get_latest_result_ids(
            current_user.id
        )
        last_result_id = latest_pricing_result_id
        last_analysis_result_id = latest_analysis_result_id

        if last_result_id:
            latest_pricing_result = PricingResult.query.filter_by(
                id=last_result_id,
                user_id=current_user.id,
            ).first()
            if latest_pricing_result and latest_pricing_result.result_json:
                option_price_value = latest_pricing_result.result_json.get(
                    "option_price"
                )
                if option_price_value is not None:
                    option_price = "${:,.4f}".format(float(option_price_value))

        if last_analysis_result_id:
            latest_analysis = AnalysisResult.query.filter_by(
                id=last_analysis_result_id,
                user_id=current_user.id,
            ).first()
            if latest_analysis and latest_analysis.result_json:
                if latest_analysis.analysis_type == "barrier_sensitivity":
                    sensitivity_results = latest_analysis.result_json
                elif latest_analysis.analysis_type == "barrier_scenario":
                    scenario_results = latest_analysis.result_json
                elif latest_analysis.analysis_type == "barrier_convergence":
                    convergence_results = latest_analysis.result_json
                elif latest_analysis.analysis_type == "barrier_risk_pl":
                    risk_pl_results = latest_analysis.result_json
    else:
        sensitivity_results = None
        scenario_results = None
        convergence_results = None
        risk_pl_results = None

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

                tester = monte_carlo_module.MonteCarloBarrierSmoothnessTest(**form_data)

                values, greek_values = tester.calculate_greeks_over_range(
                    variable, num_steps, step_range, target_variable
                )

                plot_path = tester.plot_single_greek(
                    values, greek_values, target_variable, variable
                )
                plot_filename = os.path.basename(plot_path)
                logger.debug("Barrier sensitivity plot saved to: %s", plot_path)

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
                        product_type="barrier_option",
                        ticker=form_data["ticker"],
                        model_name="monte_carlo_v2",
                        start_date=str(form_data["start_date"]),
                        end_date=str(form_data["end_date"]),
                        params_json={
                            "strike_price": form_data["K"],
                            "risk_free_rate": form_data["r"],
                            "volatility": form_data["sigma"],
                            "dividend_yield": form_data["q"],
                            "num_steps": form_data["N"],
                            "num_paths": form_data["M"],
                            "barrier_level": form_data["barrier"],
                            "option_type": form_data["option_type"],
                            "barrier_type": form_data["barrier_type"],
                            "discretization": form_data["discretization"],
                        },
                    )
                    db.session.add(instrument)
                    db.session.flush()

                    analysis_result = AnalysisResult(
                        user_id=current_user.id,
                        instrument_id=instrument.id,
                        pricing_result_id=latest_pricing_result_id,
                        analysis_type="barrier_sensitivity",
                        result_json=sensitivity_results_data,
                    )
                    db.session.add(analysis_result)
                    db.session.flush()

                    plot = Plot(
                        user_id=current_user.id,
                        analysis_result_id=analysis_result.id,
                        pricing_result_id=latest_pricing_result_id,
                        plot_type="barrier_sensitivity",
                        filename=plot_filename,
                        filepath=os.path.join("derivapro", "static", plot_filename),
                    )
                    db.session.add(plot)
                    db.session.commit()
                    latest_analysis = analysis_result
                else:
                    pass

                sensitivity_results = sensitivity_results_data

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

            barrier_step_results = monte_carlo_module.barrier_convergence_test(
                max_steps=max_steps,
                max_sims=max_sims,
                obs=obs,
                form_data=form_data,
                mode=mode,
            )

            plot_path = monte_carlo_module.plot_convergence(barrier_step_results, mode)
            plot_filename = os.path.basename(plot_path)
            serialized_results = [(int(x), float(y)) for x, y in barrier_step_results]

            convergence_results_data = {
                "results": serialized_results,
                "mode": mode,
                "plot_filename": plot_filename,
            }

            if current_user.is_authenticated:
                instrument = Instrument(
                    user_id=current_user.id,
                    product_type="barrier_option",
                    ticker=form_data["ticker"],
                    model_name="monte_carlo_v2",
                    start_date=str(form_data["start_date"]),
                    end_date=str(form_data["end_date"]),
                    params_json={
                        "strike_price": form_data["K"],
                        "risk_free_rate": form_data["r"],
                        "volatility": form_data["sigma"],
                        "dividend_yield": form_data["q"],
                        "num_steps": form_data["N"],
                        "num_paths": form_data["M"],
                        "barrier_level": form_data["barrier"],
                        "option_type": form_data["option_type"],
                        "barrier_type": form_data["barrier_type"],
                        "discretization": form_data["discretization"],
                        "mode": mode,
                        "max_steps": max_steps,
                        "max_sims": max_sims,
                        "obs": obs,
                    },
                )
                db.session.add(instrument)
                db.session.flush()

                analysis_result = AnalysisResult(
                    user_id=current_user.id,
                    instrument_id=instrument.id,
                    pricing_result_id=latest_pricing_result_id,
                    analysis_type="barrier_convergence",
                    result_json=convergence_results_data,
                )
                db.session.add(analysis_result)
                db.session.flush()

                plot = Plot(
                    user_id=current_user.id,
                    analysis_result_id=analysis_result.id,
                    pricing_result_id=latest_pricing_result_id,
                    plot_type="barrier_convergence",
                    filename=convergence_results_data["plot_filename"],
                    filepath=os.path.join(
                        "derivapro", "static", convergence_results_data["plot_filename"]
                    ),
                )
                db.session.add(plot)
                db.session.commit()
                latest_analysis = analysis_result
            else:
                pass

            convergence_results = convergence_results_data

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

                stock_data = StockData(
                    form_data["ticker"],
                    form_data["start_date"],
                    form_data["end_date"],
                )
                S0 = float(stock_data.get_closing_price())
                T = stock_data.get_years_difference()

                baseline_engine = monte_carlo_module.create_monte_carlo_engine(
                    S0=S0,
                    r=form_data["r"],
                    sigma=form_data["sigma"],
                    T=T,
                    num_paths=form_data["M"],
                    num_steps=form_data["N"],
                    random_type="sobol",
                )

                baseline_price = baseline_engine.price_barrier_option(
                    strike_price=form_data["K"],
                    barrier_level=form_data["barrier"],
                    option_type=form_data["option_type"],
                    barrier_type=form_data["barrier_type"],
                    dividend_yield=form_data["q"],
                )

                baseline_greeks = baseline_engine.calculate_greeks_finite_difference(
                    strike_price=form_data["K"],
                    option_type=form_data["option_type"],
                    option_style="barrier",
                    barrier_level=form_data["barrier"],
                    barrier_type=form_data["barrier_type"],
                    dividend_yield=form_data["q"],
                )

                baseline_price = "{:.4f}".format(baseline_price)
                baseline_delta = "{:.4f}".format(baseline_greeks["Delta"])
                baseline_gamma = "{:.4f}".format(baseline_greeks["Gamma"])
                baseline_vega = "{:.4f}".format(baseline_greeks["Vega"])
                baseline_theta = "{:.4f}".format(baseline_greeks["Theta"])
                baseline_rho = "{:.4f}".format(baseline_greeks["Rho"])

                stressed_spot = strike_price * (1 + spot_change)
                stressed_vol = volatility + vol_change
                stressed_rate = risk_free_rate + rate_change

                stressed_engine = monte_carlo_module.create_monte_carlo_engine(
                    S0=stressed_spot,
                    r=stressed_rate,
                    sigma=stressed_vol,
                    T=T,
                    num_paths=form_data["M"],
                    num_steps=form_data["N"],
                    random_type="sobol",
                )

                stressed_price = stressed_engine.price_barrier_option(
                    strike_price=form_data["K"],
                    barrier_level=form_data["barrier"],
                    option_type=form_data["option_type"],
                    barrier_type=form_data["barrier_type"],
                    dividend_yield=form_data["q"],
                )

                stressed_greeks = stressed_engine.calculate_greeks_finite_difference(
                    strike_price=form_data["K"],
                    option_type=form_data["option_type"],
                    option_style="barrier",
                    barrier_level=form_data["barrier"],
                    barrier_type=form_data["barrier_type"],
                    dividend_yield=form_data["q"],
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

                logger.debug(
                    "Barrier baseline scenario table: %s", baseline_scenario_table
                )
                logger.debug(
                    "Barrier stressed scenario table: %s", stressed_scenario_table
                )

                scenario_results_data = {
                    "baseline_scenario_table": baseline_scenario_table,
                    "stressed_scenario_table": stressed_scenario_table,
                    "gpt_scenario_assessment": "No assessment yet.",
                }

                if current_user.is_authenticated:
                    instrument = Instrument(
                        user_id=current_user.id,
                        product_type="barrier_option",
                        ticker=form_data["ticker"],
                        model_name="monte_carlo_v2",
                        start_date=str(form_data["start_date"]),
                        end_date=str(form_data["end_date"]),
                        params_json={
                            "strike_price": form_data["K"],
                            "risk_free_rate": form_data["r"],
                            "volatility": form_data["sigma"],
                            "dividend_yield": form_data["q"],
                            "num_steps": form_data["N"],
                            "num_paths": form_data["M"],
                            "barrier_level": form_data["barrier"],
                            "option_type": form_data["option_type"],
                            "barrier_type": form_data["barrier_type"],
                            "discretization": form_data["discretization"],
                            "spot_scenario": spot_change,
                            "vol_scenario": vol_change,
                            "rate_scenario": rate_change,
                        },
                    )
                    db.session.add(instrument)
                    db.session.flush()

                    analysis_result = AnalysisResult(
                        user_id=current_user.id,
                        instrument_id=instrument.id,
                        pricing_result_id=latest_pricing_result_id,
                        analysis_type="barrier_scenario",
                        result_json=scenario_results_data,
                    )
                    db.session.add(analysis_result)
                    db.session.commit()
                    latest_analysis = analysis_result
                else:
                    pass

                scenario_results = scenario_results_data

            except Exception:
                logger.exception("An error occurred during barrier scenario analysis")
                scenario_results = None

        elif action == "ai_scenario_assessment":
            scenario_data = scenario_results or {}

            if current_user.is_authenticated and latest_analysis:
                if latest_analysis.analysis_type == "barrier_scenario":
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
                assessment_input = f"Please assess the scenario analysis of the option price and Greeks based on the following results: {table_text}. Please limit the assessment to be less than 100 words."
                gpt_scenario_assessment = ask_gpt(assessment_input)
                scenario_data["gpt_scenario_assessment"] = gpt_scenario_assessment
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

        elif action == "risk_pl":
            try:
                form_data["price_change"] = float(request.form["price_change"])
                form_data["vol_change"] = float(request.form["vol_change"])

                price_change = form_data["price_change"]
                vol_change = form_data["vol_change"]

                stock_data = StockData(
                    form_data["ticker"],
                    form_data["start_date"],
                    form_data["end_date"],
                )
                S0 = float(stock_data.get_closing_price())
                T = stock_data.get_years_difference()

                base_engine = monte_carlo_module.create_monte_carlo_engine(
                    S0=S0,
                    r=form_data["r"],
                    sigma=form_data["sigma"],
                    T=T,
                    num_paths=form_data["M"],
                    num_steps=form_data["N"],
                    random_type="sobol",
                )

                base_price = base_engine.price_barrier_option(
                    strike_price=form_data["K"],
                    barrier_level=form_data["barrier"],
                    option_type=form_data["option_type"],
                    barrier_type=form_data["barrier_type"],
                    dividend_yield=form_data["q"],
                )

                bumped_engine = monte_carlo_module.create_monte_carlo_engine(
                    S0=S0 * (1 + price_change),
                    r=form_data["r"],
                    sigma=form_data["sigma"] * (1 + vol_change),
                    T=T,
                    num_paths=form_data["M"],
                    num_steps=form_data["N"],
                    random_type="sobol",
                )

                bumped_price = bumped_engine.price_barrier_option(
                    strike_price=form_data["K"],
                    barrier_level=form_data["barrier"],
                    option_type=form_data["option_type"],
                    barrier_type=form_data["barrier_type"],
                    dividend_yield=form_data["q"],
                )

                base_greeks = base_engine.calculate_greeks_finite_difference(
                    strike_price=form_data["K"],
                    option_type=form_data["option_type"],
                    option_style="barrier",
                    barrier_level=form_data["barrier"],
                    barrier_type=form_data["barrier_type"],
                    dividend_yield=form_data["q"],
                )

                delta_pl = base_greeks["Delta"] * (1 + price_change)
                gamma_pl = base_greeks["Gamma"] * 0.5 * (1 + price_change)
                vega_pl = base_greeks["Vega"] * (1 + vol_change)

                risk_pl_results = {
                    "Initial Price": base_price,
                    "Bumped Price": bumped_price,
                    "Actual P&L": bumped_price - base_price,
                    "Delta P&L": delta_pl,
                    "Vega P&L": vega_pl,
                    "Gamma P&L": gamma_pl,
                    "Greek P&L Sum": (delta_pl + vega_pl + gamma_pl),
                    "Difference": (bumped_price - base_price)
                    - (delta_pl + vega_pl + gamma_pl),
                }

                logger.debug("Barrier risk-based P&L results: %s", risk_pl_results)

                if current_user.is_authenticated:
                    instrument = Instrument(
                        user_id=current_user.id,
                        product_type="barrier_option",
                        ticker=form_data["ticker"],
                        model_name="monte_carlo_v2",
                        start_date=str(form_data["start_date"]),
                        end_date=str(form_data["end_date"]),
                        params_json={
                            "strike_price": form_data["K"],
                            "risk_free_rate": form_data["r"],
                            "volatility": form_data["sigma"],
                            "dividend_yield": form_data["q"],
                            "num_steps": form_data["N"],
                            "num_paths": form_data["M"],
                            "barrier_level": form_data["barrier"],
                            "option_type": form_data["option_type"],
                            "barrier_type": form_data["barrier_type"],
                            "discretization": form_data["discretization"],
                            "price_change": price_change,
                            "vol_change": vol_change,
                        },
                    )
                    db.session.add(instrument)
                    db.session.flush()

                    analysis_result = AnalysisResult(
                        user_id=current_user.id,
                        instrument_id=instrument.id,
                        pricing_result_id=latest_pricing_result_id,
                        analysis_type="barrier_risk_pl",
                        result_json=risk_pl_results,
                    )
                    db.session.add(analysis_result)
                    db.session.commit()
                    latest_analysis = analysis_result
                else:
                    pass

            except Exception:
                logger.exception(
                    "An error occurred during barrier Risk-Based P&L analysis"
                )
                risk_pl_results = None

        else:
            try:
                stock_data = StockData(
                    form_data["ticker"],
                    form_data["start_date"],
                    form_data["end_date"],
                )
                S0 = float(stock_data.get_closing_price())
                T = stock_data.get_years_difference()

                mc_engine = monte_carlo_module.create_monte_carlo_engine(
                    S0=S0,
                    r=form_data["r"],
                    sigma=form_data["sigma"],
                    T=T,
                    num_paths=form_data["M"],
                    num_steps=form_data["N"],
                    random_type="sobol",
                )

                raw_option_price = mc_engine.price_barrier_option(
                    strike_price=form_data["K"],
                    barrier_level=form_data["barrier"],
                    option_type=form_data["option_type"],
                    barrier_type=form_data["barrier_type"],
                    dividend_yield=form_data["q"],
                )

                greeks = mc_engine.calculate_greeks_finite_difference(
                    strike_price=form_data["K"],
                    option_type=form_data["option_type"],
                    option_style="barrier",
                    barrier_level=form_data["barrier"],
                    barrier_type=form_data["barrier_type"],
                    dividend_yield=form_data["q"],
                )

                if current_user.is_authenticated:
                    instrument = Instrument(
                        user_id=current_user.id,
                        product_type="barrier_option",
                        ticker=form_data["ticker"],
                        model_name="monte_carlo_v2",
                        start_date=str(form_data["start_date"]),
                        end_date=str(form_data["end_date"]),
                        params_json={
                            "strike_price": form_data["K"],
                            "risk_free_rate": form_data["r"],
                            "volatility": form_data["sigma"],
                            "dividend_yield": form_data["q"],
                            "num_steps": form_data["N"],
                            "num_paths": form_data["M"],
                            "barrier_level": form_data["barrier"],
                            "option_type": form_data["option_type"],
                            "barrier_type": form_data["barrier_type"],
                            "discretization": form_data["discretization"],
                        },
                    )
                    db.session.add(instrument)
                    db.session.flush()

                    pricing_result = PricingResult(
                        user_id=current_user.id,
                        instrument_id=instrument.id,
                        price=float(raw_option_price),
                        delta=float(greeks["Delta"]),
                        gamma=float(greeks["Gamma"]),
                        vega=float(greeks["Vega"]),
                        theta=float(greeks["Theta"]),
                        rho=float(greeks["Rho"]),
                        result_json={
                            "option_price": float(raw_option_price),
                            "delta": float(greeks["Delta"]),
                            "gamma": float(greeks["Gamma"]),
                            "vega": float(greeks["Vega"]),
                            "theta": float(greeks["Theta"]),
                            "rho": float(greeks["Rho"]),
                        },
                    )
                    db.session.add(pricing_result)
                    db.session.commit()

                option_price = "${:,.4f}".format(raw_option_price)
            except Exception:
                logger.exception("Error using v2 MC engine for barrier pricing")
                option_price = None

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
