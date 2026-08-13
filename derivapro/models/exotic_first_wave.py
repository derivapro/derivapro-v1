import math
from dataclasses import dataclass
from typing import Any

import numpy as np


def normal_cdf(value: float) -> float:
    return 0.5 * (1.0 + math.erf(value / math.sqrt(2.0)))


def money(value: float) -> str:
    return f"${value:,.4f}"


def pct(value: float) -> str:
    return f"{value * 100:.2f}%"


def _gbm_paths(
    spot: float,
    rate: float,
    dividend_yield: float,
    volatility: float,
    maturity: float,
    paths: int,
    steps: int,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    dt = maturity / steps
    shocks = rng.normal(size=(paths, steps))
    increments = (
        (rate - dividend_yield - 0.5 * volatility * volatility) * dt
        + volatility * math.sqrt(dt) * shocks
    )
    log_paths = np.cumsum(increments, axis=1)
    paths_arr = spot * np.exp(log_paths)
    return np.column_stack([np.full(paths, spot), paths_arr])


@dataclass
class DigitalTerms:
    spot: float
    strike: float
    maturity: float
    rate: float
    dividend_yield: float
    volatility: float
    option_type: str
    payout: float
    scenario_shock: float


def price_digital_option(terms: DigitalTerms) -> dict[str, Any]:
    def calc(spot: float) -> tuple[float, float]:
        sqrt_t = math.sqrt(terms.maturity)
        d2 = (
            math.log(spot / terms.strike)
            + (terms.rate - terms.dividend_yield - 0.5 * terms.volatility * terms.volatility)
            * terms.maturity
        ) / (terms.volatility * sqrt_t)
        probability = normal_cdf(d2) if terms.option_type == "call" else normal_cdf(-d2)
        return math.exp(-terms.rate * terms.maturity) * terms.payout * probability, probability

    price, probability = calc(terms.spot)

    return _standard_result(
        price,
        "Closed-form cash-or-nothing digital option valuation under Black-Scholes assumptions.",
        [
            {"label": "Base", "pv": price},
            {"label": f"Spot +{terms.scenario_shock:.0%}", "pv": calc(terms.spot * (1 + terms.scenario_shock))[0]},
            {"label": f"Spot -{terms.scenario_shock:.0%}", "pv": calc(terms.spot * max(1 - terms.scenario_shock, 0.01))[0]},
        ],
        [
            {"Metric": "Risk-neutral exercise probability", "Value": probability},
            {"Metric": "Discount factor", "Value": math.exp(-terms.rate * terms.maturity)},
            {"Metric": "Payout", "Value": terms.payout},
        ],
        "Closed-form Black-Scholes",
    )


@dataclass
class LookbackTerms:
    spot: float
    strike: float
    maturity: float
    rate: float
    dividend_yield: float
    volatility: float
    option_type: str
    payoff_variant: str
    paths: int
    steps: int
    seed: int
    scenario_shock: float


def price_lookback_option(terms: LookbackTerms) -> dict[str, Any]:
    price, stderr, path_min, path_max = _lookback_raw(terms)
    up_terms = LookbackTerms(**{**terms.__dict__, "spot": terms.spot * (1 + terms.scenario_shock)})
    down_terms = LookbackTerms(**{**terms.__dict__, "spot": terms.spot * max(1 - terms.scenario_shock, 0.01)})
    return _standard_result(
        price,
        "Monte Carlo valuation of path-extreme payoff using simulated GBM paths.",
        [
            {"label": "Base", "pv": price},
            {"label": f"Spot +{terms.scenario_shock:.0%}", "pv": _lookback_raw(up_terms)[0]},
            {"label": f"Spot -{terms.scenario_shock:.0%}", "pv": _lookback_raw(down_terms)[0]},
        ],
        [
            {"Metric": "Standard error", "Value": stderr},
            {"Metric": "Average path minimum", "Value": float(np.mean(path_min))},
            {"Metric": "Average path maximum", "Value": float(np.mean(path_max))},
        ],
        "Monte Carlo path simulation",
    )


def _lookback_raw(terms: LookbackTerms):
    paths = _gbm_paths(
        terms.spot,
        terms.rate,
        terms.dividend_yield,
        terms.volatility,
        terms.maturity,
        terms.paths,
        terms.steps,
        terms.seed,
    )
    terminal = paths[:, -1]
    path_min = np.min(paths, axis=1)
    path_max = np.max(paths, axis=1)
    if terms.payoff_variant == "floating_strike":
        payoffs = terminal - path_min if terms.option_type == "call" else path_max - terminal
    else:
        payoffs = np.maximum(path_max - terms.strike, 0.0) if terms.option_type == "call" else np.maximum(terms.strike - path_min, 0.0)
    discounted = math.exp(-terms.rate * terms.maturity) * payoffs
    price = float(np.mean(discounted))
    stderr = float(np.std(discounted, ddof=1) / math.sqrt(len(discounted)))
    return price, stderr, path_min, path_max


@dataclass
class BasketTerms:
    spots: list[float]
    weights: list[float]
    volatilities: list[float]
    correlation: float
    strike: float
    maturity: float
    rate: float
    dividend_yield: float
    option_type: str
    paths: int
    seed: int
    scenario_shock: float


def price_basket_option(terms: BasketTerms) -> dict[str, Any]:
    price, initial_basket, terminal_basket = _basket_raw(terms)
    up_terms = BasketTerms(**{**terms.__dict__, "spots": [spot * (1 + terms.scenario_shock) for spot in terms.spots]})
    down_terms = BasketTerms(**{**terms.__dict__, "spots": [spot * max(1 - terms.scenario_shock, 0.01) for spot in terms.spots]})
    return _standard_result(
        price,
        "Monte Carlo valuation of weighted multi-asset basket payoff with constant pairwise correlation.",
        [
            {"label": "Base", "pv": price},
            {"label": f"Basket +{terms.scenario_shock:.0%}", "pv": _basket_raw(up_terms)[0]},
            {"label": f"Basket -{terms.scenario_shock:.0%}", "pv": _basket_raw(down_terms)[0]},
        ],
        [
            {"Metric": "Initial basket level", "Value": initial_basket},
            {"Metric": "Average terminal basket", "Value": terminal_basket},
            {"Metric": "Correlation", "Value": terms.correlation},
        ],
        "Correlated Monte Carlo",
    )


def _basket_raw(terms: BasketTerms):
    spots = np.array(terms.spots, dtype=float)
    weights = np.array(terms.weights, dtype=float)
    if len(terms.spots) != len(terms.weights) or len(terms.spots) != len(terms.volatilities):
        raise ValueError("Basket spots, weights, and volatilities must have the same length.")
    weights = weights / np.sum(weights)
    vols = np.array(terms.volatilities, dtype=float)
    n_assets = len(spots)
    corr = np.full((n_assets, n_assets), terms.correlation)
    np.fill_diagonal(corr, 1.0)
    chol = np.linalg.cholesky(corr)
    rng = np.random.default_rng(terms.seed)
    shocks = rng.normal(size=(terms.paths, n_assets)) @ chol.T
    terminal = spots * np.exp(
        (terms.rate - terms.dividend_yield - 0.5 * vols * vols) * terms.maturity
        + vols * math.sqrt(terms.maturity) * shocks
    )
    basket = terminal @ weights
    payoffs = np.maximum(basket - terms.strike, 0.0) if terms.option_type == "call" else np.maximum(terms.strike - basket, 0.0)
    discounted = math.exp(-terms.rate * terms.maturity) * payoffs
    price = float(np.mean(discounted))
    return price, float(spots @ weights), float(np.mean(basket))


@dataclass
class CliquetTerms:
    spot: float
    maturity: float
    rate: float
    dividend_yield: float
    volatility: float
    notional: float
    periods: int
    local_floor: float
    local_cap: float
    global_floor: float
    global_cap: float
    paths: int
    seed: int
    scenario_shock: float


def price_cliquet_option(terms: CliquetTerms) -> dict[str, Any]:
    price, total_returns = _cliquet_raw(terms)
    up_terms = CliquetTerms(**{**terms.__dict__, "volatility": terms.volatility + terms.scenario_shock})
    down_terms = CliquetTerms(**{**terms.__dict__, "volatility": max(terms.volatility - terms.scenario_shock, 0.001)})
    return _standard_result(
        price,
        "Monte Carlo valuation of periodically reset cliquet payoff with local and global caps/floors.",
        [
            {"label": "Base", "pv": price},
            {"label": f"Vol +{terms.scenario_shock:.0%}", "pv": _cliquet_raw(up_terms)[0]},
            {"label": f"Vol -{terms.scenario_shock:.0%}", "pv": _cliquet_raw(down_terms)[0]},
        ],
        [
            {"Metric": "Average capped return", "Value": float(np.mean(total_returns))},
            {"Metric": "Global floor", "Value": terms.global_floor},
            {"Metric": "Global cap", "Value": terms.global_cap},
        ],
        "Monte Carlo reset simulation",
    )


def _cliquet_raw(terms: CliquetTerms):
    path_arr = _gbm_paths(
        terms.spot,
        terms.rate,
        terms.dividend_yield,
        terms.volatility,
        terms.maturity,
        terms.paths,
        terms.periods,
        terms.seed,
    )
    period_returns = path_arr[:, 1:] / path_arr[:, :-1] - 1.0
    capped_returns = np.clip(period_returns, terms.local_floor, terms.local_cap)
    total_returns = np.clip(np.sum(capped_returns, axis=1), terms.global_floor, terms.global_cap)
    payoffs = terms.notional * np.maximum(total_returns, 0.0)
    discounted = math.exp(-terms.rate * terms.maturity) * payoffs
    price = float(np.mean(discounted))
    return price, total_returns


@dataclass
class QuantoTerms:
    spot: float
    strike: float
    maturity: float
    domestic_rate: float
    foreign_yield: float
    equity_volatility: float
    fx_volatility: float
    equity_fx_correlation: float
    option_type: str
    scenario_shock: float


def price_quanto_option(terms: QuantoTerms) -> dict[str, Any]:
    price, adjusted_yield = _quanto_raw(terms)
    up_terms = QuantoTerms(**{**terms.__dict__, "equity_fx_correlation": min(0.99, terms.equity_fx_correlation + 0.10)})
    down_terms = QuantoTerms(**{**terms.__dict__, "equity_fx_correlation": max(-0.99, terms.equity_fx_correlation - 0.10)})

    return _standard_result(
        price,
        "Closed-form quanto-adjusted Black-Scholes valuation with equity-FX correlation drift adjustment.",
        [
            {"label": "Base", "pv": price},
            {"label": "Correlation +10 pp", "pv": _quanto_raw(up_terms)[0]},
            {"label": "Correlation -10 pp", "pv": _quanto_raw(down_terms)[0]},
        ],
        [
            {"Metric": "Quanto adjusted yield", "Value": adjusted_yield},
            {"Metric": "Equity volatility", "Value": terms.equity_volatility},
            {"Metric": "FX volatility", "Value": terms.fx_volatility},
        ],
        "Quanto-adjusted closed form",
    )


def _quanto_raw(terms: QuantoTerms):
    adjusted_yield = terms.foreign_yield + terms.equity_fx_correlation * terms.equity_volatility * terms.fx_volatility
    sqrt_t = math.sqrt(terms.maturity)
    d1 = (
        math.log(terms.spot / terms.strike)
        + (terms.domestic_rate - adjusted_yield + 0.5 * terms.equity_volatility**2)
        * terms.maturity
    ) / (terms.equity_volatility * sqrt_t)
    d2 = d1 - terms.equity_volatility * sqrt_t
    discounted_spot = terms.spot * math.exp(-adjusted_yield * terms.maturity)
    discounted_strike = terms.strike * math.exp(-terms.domestic_rate * terms.maturity)
    price = (
        discounted_spot * normal_cdf(d1) - discounted_strike * normal_cdf(d2)
        if terms.option_type == "call"
        else discounted_strike * normal_cdf(-d2) - discounted_spot * normal_cdf(-d1)
    )
    return price, adjusted_yield


def _standard_result(price, summary, scenarios, diagnostics, methodology):
    return {
        "raw_price": price,
        "primary_metrics": [
            {"label": "Present Value", "value": money(price)},
            {"label": "Methodology", "value": methodology},
        ],
        "summary": summary,
        "scenarios": scenarios,
        "diagnostics": diagnostics,
    }
