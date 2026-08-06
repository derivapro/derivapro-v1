from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np


@dataclass
class StructuredNoteTerms:
    product_type: str
    notional: float = 1_000_000.0
    spot_price: float = 100.0
    maturity: float = 1.0
    risk_free_rate: float = 0.045
    dividend_yield: float = 0.0
    volatility: float = 0.22
    coupon_rate: float = 0.08
    coupon_frequency: int = 4
    participation_rate: float = 1.0
    cap_return: float = 0.25
    buffer: float = 0.10
    downside_participation: float = 1.0
    principal_protection: float = 1.0
    protection_barrier: float = 0.70
    coupon_barrier: float = 0.70
    memory_coupon: bool = False
    hazard_rate: float = 0.02
    recovery_rate: float = 0.40
    num_paths: int = 10_000
    num_steps: int = 252
    random_seed: int = 42


def _validate_terms(terms: StructuredNoteTerms) -> None:
    if terms.notional <= 0:
        raise ValueError("Notional must be positive.")
    if terms.maturity <= 0:
        raise ValueError("Maturity must be positive.")
    if terms.risk_free_rate < -0.20:
        raise ValueError("Risk-free rate is outside the supported range.")
    if terms.spot_price <= 0:
        raise ValueError("Spot price must be positive.")
    if terms.volatility <= 0:
        raise ValueError("Volatility must be positive.")
    if terms.num_paths < 100:
        raise ValueError("Number of paths must be at least 100.")
    if terms.num_steps < 1:
        raise ValueError("Number of time steps must be positive.")
    if not 0 <= terms.recovery_rate <= 1:
        raise ValueError("Recovery rate must be between 0 and 1.")
    if terms.hazard_rate < 0:
        raise ValueError("Hazard rate cannot be negative.")


def _discount(rate: float, time: float) -> float:
    return float(np.exp(-rate * time))


def _terminal_relative_levels(terms: StructuredNoteTerms) -> np.ndarray:
    rng = np.random.default_rng(int(terms.random_seed))
    shocks = rng.standard_normal(int(terms.num_paths))
    drift = (
        terms.risk_free_rate
        - terms.dividend_yield
        - 0.5 * terms.volatility * terms.volatility
    ) * terms.maturity
    diffusion = terms.volatility * np.sqrt(terms.maturity) * shocks
    return np.exp(drift + diffusion)


def _observation_relative_paths(terms: StructuredNoteTerms) -> tuple[np.ndarray, np.ndarray]:
    obs_count = max(1, int(round(terms.coupon_frequency * terms.maturity)))
    observation_times = np.linspace(
        terms.maturity / obs_count,
        terms.maturity,
        obs_count,
    )
    rng = np.random.default_rng(int(terms.random_seed))
    paths = np.ones((int(terms.num_paths), obs_count))
    previous_time = 0.0
    previous_level = np.ones(int(terms.num_paths))

    for index, obs_time in enumerate(observation_times):
        dt = float(obs_time - previous_time)
        shocks = rng.standard_normal(int(terms.num_paths))
        previous_level = previous_level * np.exp(
            (
                terms.risk_free_rate
                - terms.dividend_yield
                - 0.5 * terms.volatility * terms.volatility
            )
            * dt
            + terms.volatility * np.sqrt(dt) * shocks
        )
        paths[:, index] = previous_level
        previous_time = float(obs_time)

    return paths, observation_times


def _summarize_discounted_payoffs(discounted_payoffs: np.ndarray) -> Dict[str, float]:
    return {
        "price": float(np.mean(discounted_payoffs)),
        "standard_error": float(
            np.std(discounted_payoffs, ddof=1) / np.sqrt(len(discounted_payoffs))
        ),
        "p05": float(np.percentile(discounted_payoffs, 5)),
        "p50": float(np.percentile(discounted_payoffs, 50)),
        "p95": float(np.percentile(discounted_payoffs, 95)),
    }


def _price_barrier_reverse_convertible(terms: StructuredNoteTerms) -> Dict[str, float]:
    final_rel = _terminal_relative_levels(terms)
    breached = final_rel < terms.protection_barrier
    redemption = np.where(breached, terms.notional * final_rel, terms.notional)
    coupon_cashflow = terms.notional * terms.coupon_rate * terms.maturity
    payoff = redemption + coupon_cashflow
    discounted = payoff * _discount(terms.risk_free_rate, terms.maturity)
    summary = _summarize_discounted_payoffs(discounted)
    summary.update(
        {
            "breach_probability": float(np.mean(breached)),
            "expected_redemption": float(np.mean(redemption)),
            "expected_coupon": float(coupon_cashflow),
            "final_level_mean": float(np.mean(final_rel)),
            "final_level_p05": float(np.percentile(final_rel, 5)),
            "final_level_p50": float(np.percentile(final_rel, 50)),
            "final_level_p95": float(np.percentile(final_rel, 95)),
        }
    )
    return summary


def _price_principal_protected_note(terms: StructuredNoteTerms) -> Dict[str, float]:
    final_rel = _terminal_relative_levels(terms)
    upside_return = np.maximum(final_rel - 1.0, 0.0) * terms.participation_rate
    upside_return = np.minimum(upside_return, terms.cap_return)
    payoff = terms.notional * (terms.principal_protection + upside_return)
    discounted = payoff * _discount(terms.risk_free_rate, terms.maturity)
    summary = _summarize_discounted_payoffs(discounted)
    summary.update(
        {
            "protection_floor": terms.notional * terms.principal_protection,
            "upside_participation_probability": float(np.mean(final_rel > 1.0)),
            "cap_hit_probability": float(
                np.mean(
                    np.maximum(final_rel - 1.0, 0.0) * terms.participation_rate
                    >= terms.cap_return
                )
            ),
            "expected_note_return": float(np.mean(payoff / terms.notional - 1.0)),
            "final_level_mean": float(np.mean(final_rel)),
            "final_level_p05": float(np.percentile(final_rel, 5)),
            "final_level_p50": float(np.percentile(final_rel, 50)),
            "final_level_p95": float(np.percentile(final_rel, 95)),
        }
    )
    return summary


def _price_enhanced_participation_note(terms: StructuredNoteTerms) -> Dict[str, float]:
    final_rel = _terminal_relative_levels(terms)
    raw_return = final_rel - 1.0
    upside = np.minimum(np.maximum(raw_return, 0.0) * terms.participation_rate, terms.cap_return)
    downside = np.where(
        raw_return >= -terms.buffer,
        0.0,
        terms.downside_participation * (raw_return + terms.buffer),
    )
    note_return = upside + downside
    payoff = terms.notional * np.maximum(1.0 + note_return, 0.0)
    discounted = payoff * _discount(terms.risk_free_rate, terms.maturity)
    summary = _summarize_discounted_payoffs(discounted)
    summary.update(
        {
            "buffer_breach_probability": float(np.mean(raw_return < -terms.buffer)),
            "cap_hit_probability": float(np.mean(upside >= terms.cap_return)),
            "expected_note_return": float(np.mean(payoff / terms.notional - 1.0)),
            "final_level_mean": float(np.mean(final_rel)),
            "final_level_p05": float(np.percentile(final_rel, 5)),
            "final_level_p50": float(np.percentile(final_rel, 50)),
            "final_level_p95": float(np.percentile(final_rel, 95)),
        }
    )
    return summary


def _price_contingent_income_note(terms: StructuredNoteTerms) -> Dict[str, float]:
    paths, observation_times = _observation_relative_paths(terms)
    final_rel = paths[:, -1]
    discounted_cashflows = np.zeros(int(terms.num_paths))
    coupon_paid_count = np.zeros(int(terms.num_paths), dtype=int)
    missed_coupon_count = np.zeros(int(terms.num_paths), dtype=int)

    period_coupon = terms.notional * terms.coupon_rate / max(1, terms.coupon_frequency)
    for index, obs_time in enumerate(observation_times):
        coupon_hit = paths[:, index] >= terms.coupon_barrier
        if terms.memory_coupon:
            payable_count = missed_coupon_count + 1
        else:
            payable_count = np.ones(int(terms.num_paths), dtype=int)

        discounted_cashflows[coupon_hit] += (
            period_coupon * payable_count[coupon_hit] * _discount(terms.risk_free_rate, float(obs_time))
        )
        coupon_paid_count[coupon_hit] += payable_count[coupon_hit]
        missed_coupon_count[coupon_hit] = 0
        missed_coupon_count[~coupon_hit] += 1

    protected = final_rel >= terms.protection_barrier
    redemption = np.where(protected, terms.notional, terms.notional * final_rel)
    discounted_cashflows += redemption * _discount(terms.risk_free_rate, terms.maturity)
    summary = _summarize_discounted_payoffs(discounted_cashflows)
    summary.update(
        {
            "coupon_payment_probability": float(np.mean(coupon_paid_count > 0)),
            "average_coupon_count": float(np.mean(coupon_paid_count)),
            "protection_breach_probability": float(np.mean(~protected)),
            "expected_redemption": float(np.mean(redemption)),
            "final_level_mean": float(np.mean(final_rel)),
            "final_level_p05": float(np.percentile(final_rel, 5)),
            "final_level_p50": float(np.percentile(final_rel, 50)),
            "final_level_p95": float(np.percentile(final_rel, 95)),
        }
    )
    return summary


def _price_credit_linked_note(terms: StructuredNoteTerms) -> Dict[str, float]:
    rng = np.random.default_rng(int(terms.random_seed))
    uniform = rng.uniform(size=int(terms.num_paths))
    default_times = -np.log(np.maximum(1.0 - uniform, 1e-12)) / max(terms.hazard_rate, 1e-12)
    survived = default_times > terms.maturity
    coupon_count = max(1, int(round(terms.coupon_frequency * terms.maturity)))
    coupon_times = np.linspace(terms.maturity / coupon_count, terms.maturity, coupon_count)
    period_coupon = terms.notional * terms.coupon_rate / max(1, terms.coupon_frequency)
    discounted_cashflows = np.zeros(int(terms.num_paths))

    for coupon_time in coupon_times:
        paid = default_times > coupon_time
        discounted_cashflows[paid] += period_coupon * _discount(
            terms.risk_free_rate,
            float(coupon_time),
        )

    discounted_cashflows[survived] += terms.notional * _discount(
        terms.risk_free_rate,
        terms.maturity,
    )
    defaulted = ~survived
    discounted_cashflows[defaulted] += (
        terms.notional
        * terms.recovery_rate
        * np.exp(-terms.risk_free_rate * default_times[defaulted])
    )

    summary = _summarize_discounted_payoffs(discounted_cashflows)
    summary.update(
        {
            "default_probability": float(np.mean(defaulted)),
            "survival_probability": float(np.mean(survived)),
            "expected_loss": float(terms.notional * (1.0 - terms.recovery_rate) * np.mean(defaulted)),
            "expected_recovery": float(terms.notional * terms.recovery_rate * np.mean(defaulted)),
            "average_coupon_count": float(
                np.mean([np.sum(default_time > coupon_times) for default_time in default_times])
            ),
        }
    )
    return summary


def price_structured_note(terms: StructuredNoteTerms) -> Dict[str, float]:
    _validate_terms(terms)
    pricing_functions = {
        "barrier_reverse_convertible": _price_barrier_reverse_convertible,
        "principal_protected_note": _price_principal_protected_note,
        "enhanced_participation_note": _price_enhanced_participation_note,
        "contingent_income_note": _price_contingent_income_note,
        "credit_linked_note": _price_credit_linked_note,
    }
    if terms.product_type not in pricing_functions:
        raise ValueError(f"Unsupported structured product type: {terms.product_type}")

    result = pricing_functions[terms.product_type](terms)
    result.update(
        {
            "product_type": terms.product_type,
            "price_pct_notional": float(result["price"] / terms.notional),
            "standard_error_pct_notional": float(result["standard_error"] / terms.notional),
            "num_paths": int(terms.num_paths),
            "num_steps": int(terms.num_steps),
        }
    )
    return result


def structured_note_scenarios(terms: StructuredNoteTerms) -> List[Dict[str, float | str]]:
    base = price_structured_note(terms)
    scenarios = [{"name": "Base", "price": base["price"], "change": 0.0}]

    if terms.product_type == "credit_linked_note":
        shocks = [
            ("Hazard +100bp", {"hazard_rate": terms.hazard_rate + 0.01}),
            ("Hazard -100bp", {"hazard_rate": max(terms.hazard_rate - 0.01, 0.0)}),
            ("Recovery -10pp", {"recovery_rate": max(terms.recovery_rate - 0.10, 0.0)}),
            ("Rate +100bp", {"risk_free_rate": terms.risk_free_rate + 0.01}),
        ]
    else:
        shocks = [
            ("Spot -10%", {"spot_price": terms.spot_price * 0.90}),
            ("Spot +10%", {"spot_price": terms.spot_price * 1.10}),
            ("Vol +5pp", {"volatility": terms.volatility + 0.05}),
            ("Rate +100bp", {"risk_free_rate": terms.risk_free_rate + 0.01}),
        ]

    for name, updates in shocks:
        scenario_terms = StructuredNoteTerms(**{**terms.__dict__, **updates})
        scenario_price = price_structured_note(scenario_terms)["price"]
        scenarios.append(
            {
                "name": name,
                "price": scenario_price,
                "change": scenario_price - base["price"],
                "change_pct_notional": (scenario_price - base["price"]) / terms.notional,
            }
        )

    return scenarios
