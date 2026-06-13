from dataclasses import dataclass
from typing import List

import numpy as np

from .mdls_monte_carlo_v2 import create_monte_carlo_engine


@dataclass
class AutocallableNoteTerms:
    """Terms for a Phoenix-style autocallable structured note."""

    spot_prices: List[float]
    volatilities: List[float]
    risk_free_rate: float
    dividend_yield: float
    maturity: float
    observation_times: List[float]
    notional: float
    coupon_rate: float
    coupon_barrier: float
    autocall_barrier: float
    protection_barrier: float
    memory_coupon: bool = True
    correlation: float = 0.0
    num_paths: int = 10000
    num_steps: int = 252
    random_type: str = "sobol"


def _equicorrelation_matrix(n_assets: int, correlation: float) -> np.ndarray:
    if n_assets == 1:
        return np.array([[1.0]])
    corr = float(np.clip(correlation, -0.95, 0.95))
    matrix = np.full((n_assets, n_assets), corr)
    np.fill_diagonal(matrix, 1.0)
    return matrix


def _observation_indices(observation_times: List[float], maturity: float, num_steps: int) -> List[int]:
    indices = []
    for obs_time in observation_times:
        clipped = min(max(float(obs_time), 0.0), maturity)
        index = int(round(clipped / maturity * num_steps))
        indices.append(min(max(index, 1), num_steps))
    return sorted(set(indices))


def price_autocallable_note(terms: AutocallableNoteTerms) -> dict:
    """
    Price a Phoenix-style autocallable note with Monte Carlo paths.

    The payoff supports:
    - single underlying or worst-of basket underlying
    - observation schedule
    - autocall barrier
    - coupon barrier
    - knock-in / protection barrier at maturity
    - memory coupon accrual
    - notional redemption
    - flat pairwise correlation for baskets
    """

    spot_prices = np.array(terms.spot_prices, dtype=float)
    volatilities = np.array(terms.volatilities, dtype=float)

    if spot_prices.ndim != 1 or len(spot_prices) == 0:
        raise ValueError("At least one spot price is required.")
    if len(volatilities) != len(spot_prices):
        raise ValueError("Volatility count must match number of underlyings.")
    if np.any(spot_prices <= 0):
        raise ValueError("Spot prices must be positive.")
    if np.any(volatilities <= 0):
        raise ValueError("Volatilities must be positive.")
    if terms.notional <= 0:
        raise ValueError("Notional must be positive.")
    if terms.maturity <= 0:
        raise ValueError("Maturity must be positive.")

    n_assets = len(spot_prices)
    corr_matrix = _equicorrelation_matrix(n_assets, terms.correlation)

    engine = create_monte_carlo_engine(
        S0=spot_prices.tolist(),
        r=terms.risk_free_rate,
        sigma=volatilities.tolist(),
        T=terms.maturity,
        num_paths=terms.num_paths,
        num_steps=terms.num_steps,
        random_type=terms.random_type,
        basket=n_assets > 1,
        cov_matrix=corr_matrix if n_assets > 1 else None,
    )
    engine.q = terms.dividend_yield
    engine.validate_parameters()

    uniform_randoms = engine.generate_uniform_randoms()
    normal_randoms = engine.generate_normal_randoms(uniform_randoms)
    paths = engine.safe_euler_paths(normal_randoms)

    relative_paths = paths / spot_prices.reshape(1, 1, n_assets)
    risk_factor_paths = np.min(relative_paths, axis=2)

    observation_indices = _observation_indices(
        terms.observation_times, terms.maturity, terms.num_steps
    )
    dt = terms.maturity / terms.num_steps
    n_paths = terms.num_paths

    payoffs = np.zeros(n_paths)
    event_times = np.full(n_paths, terms.maturity)
    autocalled = np.zeros(n_paths, dtype=bool)
    coupon_paid_count = np.zeros(n_paths, dtype=int)
    accrued_coupon_count = np.zeros(n_paths, dtype=int)

    for obs_number, step_index in enumerate(observation_indices, start=1):
        active = ~autocalled
        if not np.any(active):
            break

        risk_level = risk_factor_paths[:, step_index]
        coupon_hit = active & (risk_level >= terms.coupon_barrier)
        autocall_hit = active & (risk_level >= terms.autocall_barrier)

        period_coupon_count = np.ones(n_paths, dtype=int)
        payable_coupon_count = np.where(
            terms.memory_coupon,
            accrued_coupon_count + period_coupon_count,
            period_coupon_count,
        )

        coupon_paid_count[coupon_hit] += payable_coupon_count[coupon_hit]
        accrued_coupon_count[coupon_hit] = 0
        missed_coupon = active & (~coupon_hit)
        if terms.memory_coupon:
            accrued_coupon_count[missed_coupon] += 1

        if np.any(autocall_hit):
            payoffs[autocall_hit] = terms.notional * (
                1.0 + terms.coupon_rate * coupon_paid_count[autocall_hit]
            )
            event_times[autocall_hit] = step_index * dt
            autocalled[autocall_hit] = True

    not_autocalled = ~autocalled
    final_risk_level = risk_factor_paths[:, -1]

    if np.any(not_autocalled):
        maturity_coupon_hit = not_autocalled & (final_risk_level >= terms.coupon_barrier)
        maturity_coupon_count = np.where(
            terms.memory_coupon,
            accrued_coupon_count + 1,
            1,
        )
        coupon_paid_count[maturity_coupon_hit] += maturity_coupon_count[maturity_coupon_hit]

        protected = final_risk_level >= terms.protection_barrier
        redemption = np.where(
            protected,
            terms.notional,
            terms.notional * final_risk_level,
        )
        payoffs[not_autocalled] = redemption[not_autocalled] + (
            terms.notional * terms.coupon_rate * coupon_paid_count[not_autocalled]
        )

    discount_factors = np.exp(-terms.risk_free_rate * event_times)
    discounted_payoffs = payoffs * discount_factors
    price = float(np.mean(discounted_payoffs))
    stderr = float(np.std(discounted_payoffs, ddof=1) / np.sqrt(n_paths))

    first_autocall_probability = float(np.mean(autocalled))
    protection_breach_probability = float(np.mean(final_risk_level < terms.protection_barrier))
    average_coupon_count = float(np.mean(coupon_paid_count))

    return {
        "price": price,
        "standard_error": stderr,
        "autocall_probability": first_autocall_probability,
        "protection_breach_probability": protection_breach_probability,
        "average_coupon_count": average_coupon_count,
        "expected_discounted_payoff": price,
        "observation_count": len(observation_indices),
        "underlying_count": n_assets,
        "worst_final_level_mean": float(np.mean(final_risk_level)),
        "worst_final_level_p05": float(np.percentile(final_risk_level, 5)),
        "worst_final_level_p50": float(np.percentile(final_risk_level, 50)),
        "worst_final_level_p95": float(np.percentile(final_risk_level, 95)),
    }
