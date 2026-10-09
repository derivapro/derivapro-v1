from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

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


@dataclass
class StructuredAnalysisConfig:
    scenario_package: str = "standard"
    run_scenario_analysis: bool = True
    run_driver_sensitivity: bool = True
    run_pnl_attribution: bool = True
    run_payoff_profile: bool = True
    run_risk_diagnostics: bool = True
    include_visuals: bool = True
    scenario_volatility_shock: float = 0.10
    scenario_barrier_shock: float = 0.05
    scenario_coupon_shock: float = 0.02
    scenario_rate_shock: float = 0.005
    sensitivity_volatility_shock: float = 0.05
    sensitivity_barrier_shock: float = 0.05
    sensitivity_coupon_shock: float = 0.01
    sensitivity_rate_shock: float = 0.01
    payoff_floor: float = 0.40
    payoff_ceiling: float = 1.20
    payoff_points: int = 7


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


def _terms_with_updates(terms: StructuredNoteTerms, updates: Dict[str, float]) -> StructuredNoteTerms:
    return StructuredNoteTerms(**{**terms.__dict__, **updates})


def _format_pp(value: float) -> str:
    return f"{value * 100:.0f} pp"


def _format_bp(value: float) -> str:
    return f"{value * 10000:.0f} bp"


def _validate_analysis_config(config: StructuredAnalysisConfig) -> StructuredAnalysisConfig:
    if config.scenario_package not in {"standard", "downside", "defensive"}:
        raise ValueError("Unsupported scenario package.")
    if not any(
        [
            config.run_scenario_analysis,
            config.run_driver_sensitivity,
            config.run_pnl_attribution,
            config.run_payoff_profile,
            config.run_risk_diagnostics,
        ]
    ):
        raise ValueError("Select at least one analysis module to run.")
    for value in [
        config.scenario_volatility_shock,
        config.scenario_barrier_shock,
        config.scenario_coupon_shock,
        config.scenario_rate_shock,
        config.sensitivity_volatility_shock,
        config.sensitivity_barrier_shock,
        config.sensitivity_coupon_shock,
        config.sensitivity_rate_shock,
    ]:
        if value < 0:
            raise ValueError("Analysis shock sizes must be non-negative.")
    if config.payoff_ceiling <= config.payoff_floor:
        raise ValueError("Payoff profile ceiling must be above the floor.")
    if config.payoff_points < 3:
        raise ValueError("Payoff profile requires at least 3 points.")
    if config.payoff_points > 21:
        raise ValueError("Payoff profile supports at most 21 points.")
    if config.payoff_floor < 0:
        raise ValueError("Payoff profile floor cannot be negative.")
    return config


def _scenario_result(
    terms: StructuredNoteTerms,
    name: str,
    updates: Dict[str, float],
    description: str,
    base_price: float,
) -> Dict[str, float | str]:
    scenario_terms = _terms_with_updates(terms, updates)
    scenario_result = price_structured_note(scenario_terms)
    row: Dict[str, float | str] = {
        "name": name,
        "description": description,
        "price": scenario_result["price"],
        "change": scenario_result["price"] - base_price,
        "change_pct_notional": (scenario_result["price"] - base_price) / terms.notional,
        "standard_error": scenario_result["standard_error"],
    }
    for key in [
        "breach_probability",
        "default_probability",
        "protection_breach_probability",
        "coupon_payment_probability",
        "cap_hit_probability",
        "buffer_breach_probability",
    ]:
        if key in scenario_result:
            row[key] = scenario_result[key]
    return row


def _barrier_reverse_convertible_scenarios(
    terms: StructuredNoteTerms,
    base: Dict[str, float],
    config: StructuredAnalysisConfig,
) -> List[Dict[str, float | str]]:
    standard_rows = [
        {
            "name": "Base",
            "description": "Current submitted assumptions.",
            "price": base["price"],
            "change": 0.0,
            "change_pct_notional": 0.0,
            "standard_error": base["standard_error"],
            "breach_probability": base["breach_probability"],
        },
        _scenario_result(
            terms,
            "Volatility shock",
            {"volatility": terms.volatility + config.scenario_volatility_shock},
            f"Volatility increases by {_format_pp(config.scenario_volatility_shock)}, raising downside-tail risk.",
            base["price"],
        ),
        _scenario_result(
            terms,
            "Barrier step-up",
            {"protection_barrier": min(0.98, terms.protection_barrier + config.scenario_barrier_shock)},
            f"Protection barrier is {_format_pp(config.scenario_barrier_shock)} closer to par.",
            base["price"],
        ),
        _scenario_result(
            terms,
            "Carry compression",
            {
                "coupon_rate": max(0.0, terms.coupon_rate - config.scenario_coupon_shock),
                "risk_free_rate": terms.risk_free_rate - config.scenario_rate_shock,
            },
            f"Coupon falls by {_format_bp(config.scenario_coupon_shock)} and discount rate falls by {_format_bp(config.scenario_rate_shock)}.",
            base["price"],
        ),
        _scenario_result(
            terms,
            "Defensive terms",
            {
                "volatility": max(0.001, terms.volatility - config.sensitivity_volatility_shock),
                "protection_barrier": max(0.01, terms.protection_barrier - config.sensitivity_barrier_shock),
            },
            f"Volatility falls by {_format_pp(config.sensitivity_volatility_shock)} and barrier moves {_format_pp(config.sensitivity_barrier_shock)} lower.",
            base["price"],
        ),
        _scenario_result(
            terms,
            f"Rate +{_format_bp(config.sensitivity_rate_shock)}",
            {"risk_free_rate": terms.risk_free_rate + config.sensitivity_rate_shock},
            f"Parallel discount-rate increase of {_format_bp(config.sensitivity_rate_shock)}.",
            base["price"],
        ),
    ]

    if config.scenario_package == "standard":
        return standard_rows

    if config.scenario_package == "downside":
        return [
            standard_rows[0],
            _scenario_result(
                terms,
                "Combined downside stress",
                {
                    "volatility": terms.volatility + config.scenario_volatility_shock,
                    "protection_barrier": min(0.98, terms.protection_barrier + config.scenario_barrier_shock),
                    "coupon_rate": max(0.0, terms.coupon_rate - config.scenario_coupon_shock),
                    "risk_free_rate": terms.risk_free_rate - config.scenario_rate_shock,
                },
                "Volatility, barrier proximity, coupon carry, and rates move together in a downside repricing package.",
                base["price"],
            ),
            standard_rows[1],
            standard_rows[2],
            standard_rows[3],
        ]

    return [
        standard_rows[0],
        standard_rows[4],
        _scenario_result(
            terms,
            "Coupon richening",
            {"coupon_rate": terms.coupon_rate + config.scenario_coupon_shock},
            f"Coupon increases by {_format_bp(config.scenario_coupon_shock)} with other assumptions unchanged.",
            base["price"],
        ),
        _scenario_result(
            terms,
            "Barrier relief",
            {"protection_barrier": max(0.01, terms.protection_barrier - config.scenario_barrier_shock)},
            f"Protection barrier moves {_format_pp(config.scenario_barrier_shock)} lower.",
            base["price"],
        ),
        standard_rows[5],
    ]


def _barrier_reverse_convertible_sensitivity(
    terms: StructuredNoteTerms,
    base_price: float,
    config: StructuredAnalysisConfig,
) -> List[Dict[str, float | str]]:
    vol = config.sensitivity_volatility_shock
    barrier = config.sensitivity_barrier_shock
    coupon = config.sensitivity_coupon_shock
    rate = config.sensitivity_rate_shock
    shocks = [
        ("Volatility", f"-{_format_pp(vol)}", {"volatility": max(0.001, terms.volatility - vol)}),
        ("Volatility", f"+{_format_pp(vol)}", {"volatility": terms.volatility + vol}),
        ("Volatility", f"+{_format_pp(2 * vol)}", {"volatility": terms.volatility + 2 * vol}),
        (
            "Protection Barrier",
            f"-{_format_pp(barrier)}",
            {"protection_barrier": max(0.01, terms.protection_barrier - barrier)},
        ),
        (
            "Protection Barrier",
            f"+{_format_pp(barrier)}",
            {"protection_barrier": min(0.98, terms.protection_barrier + barrier)},
        ),
        ("Coupon Rate", f"-{_format_bp(coupon)}", {"coupon_rate": max(0.0, terms.coupon_rate - coupon)}),
        ("Coupon Rate", f"+{_format_bp(coupon)}", {"coupon_rate": terms.coupon_rate + coupon}),
        ("Risk-Free Rate", f"-{_format_bp(rate)}", {"risk_free_rate": terms.risk_free_rate - rate}),
        ("Risk-Free Rate", f"+{_format_bp(rate)}", {"risk_free_rate": terms.risk_free_rate + rate}),
    ]
    rows = []
    for driver, shock, updates in shocks:
        shocked_result = price_structured_note(_terms_with_updates(terms, updates))
        rows.append(
            {
                "driver": driver,
                "shock": shock,
                "price": shocked_result["price"],
                "change": shocked_result["price"] - base_price,
                "change_pct_notional": (shocked_result["price"] - base_price)
                / terms.notional,
                "breach_probability": shocked_result["breach_probability"],
            }
        )
    return rows


def _barrier_reverse_convertible_pnl_attribution(
    terms: StructuredNoteTerms,
    base_price: float,
    config: StructuredAnalysisConfig,
) -> Dict[str, Any]:
    attribution_steps = [
        (
            f"Volatility +{_format_pp(config.scenario_volatility_shock)}",
            {"volatility": terms.volatility + config.scenario_volatility_shock},
        ),
        (
            f"Barrier +{_format_pp(config.scenario_barrier_shock)}",
            {"protection_barrier": min(0.98, terms.protection_barrier + config.scenario_barrier_shock)},
        ),
        (
            f"Coupon -{_format_bp(config.scenario_coupon_shock)}",
            {"coupon_rate": max(0.0, terms.coupon_rate - config.scenario_coupon_shock)},
        ),
        (
            f"Rate -{_format_bp(config.scenario_rate_shock)}",
            {"risk_free_rate": terms.risk_free_rate - config.scenario_rate_shock},
        ),
    ]
    cumulative_updates: Dict[str, float] = {}
    previous_price = base_price
    rows = []
    for step, updates in attribution_steps:
        cumulative_updates.update(updates)
        step_price = price_structured_note(_terms_with_updates(terms, cumulative_updates))[
            "price"
        ]
        contribution = step_price - previous_price
        rows.append(
            {
                "step": step,
                "price": step_price,
                "contribution": contribution,
                "contribution_pct_notional": contribution / terms.notional,
            }
        )
        previous_price = step_price
    return {
        "scenario_name": "Downside-risk repricing",
        "description": (
            f"Sequential attribution for a combined stress: volatility +{_format_pp(config.scenario_volatility_shock)}, "
            f"barrier +{_format_pp(config.scenario_barrier_shock)}, coupon -{_format_bp(config.scenario_coupon_shock)}, "
            f"and rate -{_format_bp(config.scenario_rate_shock)}."
        ),
        "rows": rows,
        "total_change": previous_price - base_price,
        "total_change_pct_notional": (previous_price - base_price) / terms.notional,
    }


def _barrier_reverse_convertible_payoff_profile(
    terms: StructuredNoteTerms,
    config: StructuredAnalysisConfig,
) -> List[Dict[str, float | str]]:
    rows = []
    coupon_pct = terms.coupon_rate * terms.maturity
    final_levels = np.linspace(
        config.payoff_floor,
        config.payoff_ceiling,
        int(config.payoff_points),
    )
    for final_level in final_levels:
        final_level = float(final_level)
        breached = final_level < terms.protection_barrier
        redemption_pct = final_level if breached else 1.0
        payoff_pct = redemption_pct + coupon_pct
        rows.append(
            {
                "final_level": final_level,
                "state": "Barrier breached" if breached else "Full redemption",
                "redemption_pct": redemption_pct,
                "coupon_pct": coupon_pct,
                "payoff_pct": payoff_pct,
            }
        )
    return rows


def _barrier_reverse_convertible_risk_indicators(
    terms: StructuredNoteTerms,
    base: Dict[str, float],
) -> List[Dict[str, str | float]]:
    expected_shortfall = terms.notional - base["expected_redemption"]
    return [
        {
            "label": "Initial Cushion to Barrier",
            "value": max(0.0, 1.0 - terms.protection_barrier),
            "value_type": "percent",
            "comment": "Distance between initial reference level and protection barrier.",
        },
        {
            "label": "Estimated Barrier Breach",
            "value": base["breach_probability"],
            "value_type": "percent",
            "comment": "Monte Carlo probability that final level is below the protection barrier.",
        },
        {
            "label": "Expected Redemption Shortfall",
            "value": expected_shortfall,
            "value_type": "currency",
            "comment": "Expected notional loss before coupon and discounting effects.",
        },
        {
            "label": "5th Percentile Final Level",
            "value": base["final_level_p05"],
            "value_type": "percent",
            "comment": "Left-tail terminal underlying level under submitted assumptions.",
        },
    ]


def _barrier_reverse_convertible_visuals(report: Dict[str, Any]) -> Dict[str, Any]:
    visuals: Dict[str, Any] = {}
    sensitivity_rows = report.get("sensitivity_rows") or []
    if sensitivity_rows:
        max_abs_change = max(
            abs(float(row["change_pct_notional"])) for row in sensitivity_rows
        ) or 1.0
        visuals["sensitivity_chart"] = [
            {
                "label": f"{row['driver']} {row['shock']}",
                "value": float(row["change_pct_notional"]),
                "width": abs(float(row["change_pct_notional"])) / max_abs_change,
            }
            for row in sensitivity_rows
        ]

    attribution = report.get("pnl_attribution")
    if attribution and attribution.get("rows"):
        max_abs_contribution = max(
            abs(float(row["contribution_pct_notional"]))
            for row in attribution["rows"]
        ) or 1.0
        visuals["pnl_chart"] = [
            {
                "label": row["step"],
                "value": float(row["contribution_pct_notional"]),
                "width": abs(float(row["contribution_pct_notional"]))
                / max_abs_contribution,
            }
            for row in attribution["rows"]
        ]

    payoff_profile = report.get("payoff_profile") or []
    if payoff_profile:
        final_levels = [float(row["final_level"]) for row in payoff_profile]
        payoff_values = [float(row["payoff_pct"]) for row in payoff_profile]
        x_min, x_max = min(final_levels), max(final_levels)
        y_min, y_max = min(payoff_values), max(payoff_values)
        y_span = y_max - y_min or 1.0
        x_span = x_max - x_min or 1.0
        points = []
        for final_level, payoff_value in zip(final_levels, payoff_values):
            x = (final_level - x_min) / x_span * 100.0
            y = 100.0 - ((payoff_value - y_min) / y_span * 90.0 + 5.0)
            points.append(f"{x:.2f},{y:.2f}")
        visuals["payoff_chart"] = {
            "points": " ".join(points),
            "x_min": x_min,
            "x_max": x_max,
            "y_min": y_min,
            "y_max": y_max,
        }

    return visuals


def build_structured_note_analysis(
    terms: StructuredNoteTerms,
    config: StructuredAnalysisConfig | None = None,
) -> Dict[str, Any]:
    config = _validate_analysis_config(config or StructuredAnalysisConfig())
    base = price_structured_note(terms)
    if terms.product_type != "barrier_reverse_convertible":
        return {
            "product_type": terms.product_type,
            "is_product_specific": False,
            "scenario_rows": structured_note_scenarios(terms),
        }

    report: Dict[str, Any] = {
        "product_type": terms.product_type,
        "is_product_specific": True,
        "title": "Barrier Reverse Convertible Risk Review",
        "summary": (
            "Review downside barrier exposure, coupon carry, volatility/rate sensitivity, "
            "and sequential P&L attribution for a combined downside-risk repricing scenario."
        ),
        "analysis_config": config.__dict__,
        "scenario_rows": [],
        "sensitivity_rows": [],
        "pnl_attribution": None,
        "payoff_profile": [],
        "risk_indicators": [],
        "visuals": {},
    }
    if config.run_scenario_analysis:
        report["scenario_rows"] = _barrier_reverse_convertible_scenarios(
            terms,
            base,
            config,
        )
    if config.run_driver_sensitivity:
        report["sensitivity_rows"] = _barrier_reverse_convertible_sensitivity(
            terms,
            base["price"],
            config,
        )
    if config.run_pnl_attribution:
        report["pnl_attribution"] = _barrier_reverse_convertible_pnl_attribution(
            terms,
            base["price"],
            config,
        )
    if config.run_payoff_profile:
        report["payoff_profile"] = _barrier_reverse_convertible_payoff_profile(
            terms,
            config,
        )
    if config.run_risk_diagnostics:
        report["risk_indicators"] = _barrier_reverse_convertible_risk_indicators(
            terms,
            base,
        )
    if config.include_visuals:
        report["visuals"] = _barrier_reverse_convertible_visuals(report)

    return report


def structured_note_scenarios(terms: StructuredNoteTerms) -> List[Dict[str, float | str]]:
    base = price_structured_note(terms)
    if terms.product_type == "barrier_reverse_convertible":
        return _barrier_reverse_convertible_scenarios(
            terms,
            base,
            StructuredAnalysisConfig(),
        )

    scenarios = [
        {
            "name": "Base",
            "description": "Current submitted assumptions.",
            "price": base["price"],
            "change": 0.0,
            "change_pct_notional": 0.0,
        }
    ]

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
