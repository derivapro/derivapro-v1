# Barrier Option Methodology

## 1. Purpose

This document describes the methodology used by DerivaPro for the Barrier Option workflow. It supports pricing transparency, model review, implementation consistency, and future model governance documentation.

The current product-standard page covers single-underlying barrier calls and puts with knock-in or knock-out activation. The workflow includes market reference data, explicit pricing assumptions, Monte Carlo valuation, finite-difference Greeks, vanilla benchmark comparison, barrier diagnostics, first-pass driver sensitivity, illustrative payoff views, and a structured run summary.

This document is not a trading recommendation, investment advisory document, or independent model validation report.

## 2. Product Definition

A barrier option is a path-dependent option whose payoff depends on whether the underlying price breaches a specified barrier level during the monitoring period.

| Term | Description |
|---|---|
| Up-and-out | Option is extinguished if the underlying trades at or above the barrier. |
| Down-and-out | Option is extinguished if the underlying trades at or below the barrier. |
| Up-and-in | Option becomes active only if the underlying trades at or above the barrier. |
| Down-and-in | Option becomes active only if the underlying trades at or below the barrier. |
| Call payoff | `max(S_T - K, 0)` if the barrier condition permits payoff. |
| Put payoff | `max(K - S_T, 0)` if the barrier condition permits payoff. |

where `S_T` is the terminal underlying price and `K` is the strike.

The current implementation assumes discrete monitoring at simulation time steps. Continuous-monitoring adjustment is not yet included in the product-standard page.

## 3. User Inputs

### Contract Terms

| Input | Description |
|---|---|
| Underlying ticker | Equity ticker used for labeling and optional market reference lookups. |
| Option type | Call or put. |
| Strike price | Contract strike. Must be positive. |
| Number of contracts | Position count used for position-value scaling. |
| Contract multiplier | Units per contract, commonly `100` for listed US equity options. |

### Barrier Terms

| Input | Description |
|---|---|
| Barrier type | Up-and-out, down-and-out, up-and-in, or down-and-in. |
| Barrier level | Price level used to test the path-dependent barrier condition. |
| Monitoring convention | Discrete monitoring at Monte Carlo simulation time steps. |

### Market Assumptions

| Input | Description |
|---|---|
| Valuation date | Pricing date. |
| Maturity date | Option maturity. Must be after valuation date. |
| Spot price | Current underlying price assumption used directly in pricing. |
| Risk-free rate | Continuously compounded risk-free rate assumption. |
| Annualized volatility | Constant volatility assumption used in path simulation. |
| Dividend yield | Continuous dividend yield used in risk-neutral drift. |
| Day count | `ACT/365` or `ACT/360` year-fraction convention. |

### Simulation Settings

| Setting | Description |
|---|---|
| Time steps | Number of discrete monitoring and simulation steps. |
| Simulation paths | Number of Monte Carlo paths for base valuation. |
| Random sequence | Sobol or pseudo-random sequence depending on engine support. |
| Discretization | Governance field retained for model transparency; current path generation applies safe Euler paths. |

## 4. Market Reference Data

The market reference section helps users select assumptions. It does not automatically override pricing inputs unless the user explicitly applies a displayed value.

Current market reference data is fetched on demand from Yahoo Finance through the open-source `yfinance` package. Provider availability, data latency, corporate-action adjustments, option-chain availability, and licensing terms may affect runtime results.

Price-history metrics require only ticker and lookback period. Listed-option implied-volatility metrics additionally use option type, target strike, and target maturity to select the nearest provider option-chain contract.

The optional 3D IV surface is a listed-option data snapshot by expiration and strike moneyness. It is not a calibrated production volatility surface.

## 5. Monte Carlo Valuation

Barrier options are valued by simulating underlying price paths under a risk-neutral geometric Brownian motion assumption:

```text
dS_t / S_t = (r - q) dt + sigma dW_t
```

where:

| Symbol | Meaning |
|---|---|
| `S_t` | Underlying price at time `t` |
| `r` | Risk-free rate |
| `q` | Continuous dividend yield |
| `sigma` | Annualized volatility |
| `W_t` | Brownian motion |

For each simulated path, DerivaPro checks whether the barrier condition is breached. Terminal vanilla payoff is then retained or set to zero depending on the barrier type:

```text
Knock-out payoff = vanilla payoff if barrier was not breached, otherwise 0
Knock-in payoff  = vanilla payoff if barrier was breached, otherwise 0
```

The option premium is the discounted average payoff across paths.

## 6. Greeks

The page displays finite-difference Greeks around the selected barrier setup:

| Greek | Interpretation |
|---|---|
| Delta | Sensitivity of option value to underlying price. |
| Gamma | Sensitivity of delta to underlying price. |
| Vega | Sensitivity of option value to volatility. |
| Theta | Sensitivity of option value to passage of time. |
| Rho | Sensitivity of option value to risk-free rate. |

To keep browser interactions usable, page-level Greek calculations use capped Monte Carlo reruns rather than forcing very large path counts. Production usage should validate Greek stability across path count, time-step count, random seed or sequence, and bump size.

## 7. Vanilla Benchmark

The page calculates a European Black-Scholes-Merton benchmark using the same spot, strike, maturity, rate, volatility, dividend yield, and option type.

For knock-out structures, the barrier premium is generally below the matching vanilla option premium because a barrier breach can extinguish the payoff. For knock-in structures, the barrier premium is generally below the matching vanilla premium unless activation is already highly likely.

The displayed premium gap is:

```text
Premium gap = European vanilla premium - barrier premium
```

## 8. Diagnostics and Analysis

The Barrier Option page currently includes:

| Diagnostic | Purpose |
|---|---|
| Barrier distance | Relative distance between spot and barrier. |
| Breach probability | Estimated probability that the simulated path reaches the barrier. |
| Survival probability | `1 - breach probability`; most relevant for knock-out options. |
| Premium decomposition | Intrinsic value, premium less intrinsic, vanilla benchmark, and premium gap. |
| Driver sensitivity | Local Greek-based response to selected spot, volatility, and rate shocks. |
| Payoff strip | Representative terminal underlying levels with a path-dependency caveat. |
| Run summary | Compact record of terms, assumptions, model settings, outputs, and limitations. |

## 9. Validation Expectations

Before using this workflow for production or commercial decision-making, the following controls should be added or reviewed:

| Area | Recommended Control |
|---|---|
| In/out parity | For matching inputs, knock-in plus knock-out value should reconcile to the vanilla value within numerical tolerance. |
| Convergence | Prices should stabilize as path count and time-step count increase. |
| Barrier placement | Validate behavior when the barrier is close to spot, far from spot, or already breached. |
| Monitoring convention | Compare discrete-monitoring results to continuous-monitoring approximations where applicable. |
| Greeks | Review finite-difference stability under alternate bump sizes and random sequences. |
| Market data | Add provider error handling, timestamp display, caching, and data-quality controls. |

## 10. Limitations

| Limitation | Implication |
|---|---|
| Constant volatility | Does not capture volatility skew, local volatility, stochastic volatility, or jumps. |
| Constant rate and dividend yield | Does not yet use full curves or discrete dividend schedules. |
| Discrete monitoring | Continuous barrier correction is not yet included. |
| Single underlying | Does not cover basket barriers, FX barriers, quanto barriers, or double barriers. |
| No rebate | Current product-standard page does not model barrier rebates. |
| Greek noise | Monte Carlo finite-difference Greeks can be noisy and need convergence controls. |
| Reporting | The page provides a structured run summary; customized institutional reports remain future work. |

## 11. Planned Enhancements

1. Add in/out parity checks directly to the page.
2. Add full scenario repricing for spot, volatility, rate, and barrier shifts.
3. Add convergence diagnostics by path count and monitoring step count.
4. Add continuous-monitoring correction or Brownian bridge adjustment.
5. Add rebate support for knock-out and knock-in structures.
6. Add double-barrier, partial-time barrier, FX barrier, and basket barrier extensions.
7. Add scaled Greeks and downloadable report output.
8. Add automated regression tests for pricing, Greeks, convergence, and in/out parity.

## 12. Implementation References

| File | Purpose |
|---|---|
| `derivapro/templates/barrier_options.html` | Barrier option workflow UI. |
| `derivapro/routes/exotic_options.py` | Route logic, pricing orchestration, and page analytics. |
| `derivapro/models/mdls_monte_carlo_v2.py` | Monte Carlo path engine and barrier payoff logic. |
| `derivapro/services/market_reference.py` | Market reference data and optional visuals. |
