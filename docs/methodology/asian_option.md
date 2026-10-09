# Asian Option Methodology

## 1. Purpose

This document describes the methodology used by DerivaPro for the Asian Option workflow. It supports pricing transparency, model review, implementation consistency, and future model governance documentation.

The current product-standard page covers single-underlying Asian calls and puts with average-price and average-strike payoff variants. The workflow includes explicit pricing assumptions, configurable averaging windows, arithmetic or geometric averaging, Monte Carlo valuation, finite-difference Greeks, benchmark diagnostics, first-pass sensitivities, payoff views, and a structured run summary.

This document is not a trading recommendation, investment advisory document, or independent model validation report.

## 2. Product Definition

Asian options are path-dependent options whose payoff depends on an average underlying level observed over a specified window.

| Variant | Description |
|---|---|
| Average price / fixed strike | Payoff compares the average underlying price to a fixed strike. |
| Average strike / floating strike | Payoff compares terminal spot to the average underlying price. |
| Arithmetic average | Average is the simple mean of observed prices. |
| Geometric average | Average is the exponential of the mean log observed price. |

The fixed-strike payoffs are:

```text
Average-price call = max(A - K, 0)
Average-price put  = max(K - A, 0)
```

The floating-strike payoffs are:

```text
Average-strike call = max(S_T - A, 0)
Average-strike put  = max(A - S_T, 0)
```

where `A` is the observed average, `K` is the fixed strike, and `S_T` is terminal spot.

## 3. User Inputs

### Contract Terms

| Input | Description |
|---|---|
| Underlying ticker | Equity ticker used for labeling. |
| Payoff variant | Average price / fixed strike or average strike / floating strike. |
| Option type | Call or put. |
| Strike price | Fixed strike for average-price variants; retained as reference for floating-strike variants. |
| Number of contracts | Position count used for position-value scaling. |
| Contract multiplier | Units per contract, commonly `100` for listed US equity options. |

### Averaging Terms

| Input | Description |
|---|---|
| Average type | Arithmetic or geometric. |
| Averaging frequency | Daily, weekly, monthly, quarterly, or custom dates. |
| Averaging start and end | Date range used to generate scheduled observations. |
| Custom averaging dates | Comma-separated observation dates when custom mode is selected. |

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
| Simulation paths | Number of Monte Carlo paths for valuation. |
| Random seed | Seed used for repeatable pseudo-random path generation. |

## 4. Market Convention Context

Asian options are commonly grouped into average-price and average-strike structures. Average-price options use a fixed strike and are often useful where exposure is accumulated over time. Average-strike options use the average as the strike and are useful when the economics depend on the difference between terminal price and realized average.

Arithmetic averaging is the most direct commercial convention for realized price exposure. Geometric averaging is also useful because it can provide analytical benchmarks or control-variate style checks under Black-Scholes assumptions.

## 5. Monte Carlo Valuation

DerivaPro simulates the underlying under a risk-neutral geometric Brownian motion assumption:

```text
dS_t / S_t = (r - q) dt + sigma dW_t
```

At each averaging observation date, the simulated underlying level is stored. The selected arithmetic or geometric average is calculated path by path. The payoff is then discounted from maturity:

```text
Price = exp(-rT) * mean(path payoff)
```

The page displays the Monte Carlo standard error of the discounted payoff estimator.

## 6. Greeks

The page displays finite-difference Greeks around the selected Asian setup:

| Greek | Interpretation |
|---|---|
| Delta | Sensitivity of option value to underlying price. |
| Gamma | Sensitivity of delta to underlying price. |
| Vega | Sensitivity of option value to volatility. |
| Rho | Sensitivity of option value to risk-free rate. |

Theta is marked as unavailable in the current product-standard page because a robust theta requires jointly rolling valuation date, maturity, and remaining averaging observations.

## 7. Benchmark Comparison

For fixed-strike average-price variants, the page calculates a European Black-Scholes-Merton benchmark with the same spot, strike, maturity, rate, volatility, dividend yield, and option type.

The benchmark gap is:

```text
Benchmark gap = European vanilla premium - Asian premium
```

Average-price options often have lower value than comparable vanilla options because averaging dampens terminal spot variance. The benchmark is not directly comparable for floating-strike average-strike variants.

## 8. Diagnostics and Analysis

The Asian Option page currently includes:

| Diagnostic | Purpose |
|---|---|
| Payoff structure | Variant, average type, mean payoff, and Monte Carlo standard error. |
| Averaging window | Observation count, first and last observations, and window length. |
| Vanilla benchmark | European benchmark and averaging discount for fixed-strike variants. |
| Driver sensitivity | Local Greek-based response to selected spot, volatility, and rate shocks. |
| Payoff strip | Representative average or terminal-average payoff levels. |
| Run summary | Compact record of terms, assumptions, model settings, outputs, and limitations. |

## 9. Validation Expectations

Before using this workflow for production or commercial decision-making, the following controls should be added or reviewed:

| Area | Recommended Control |
|---|---|
| Monte Carlo convergence | Price and standard error should stabilize as path count increases. |
| Averaging schedule | Observation dates should reconcile to term sheet conventions and business-day calendars. |
| Arithmetic/geometric consistency | Geometric variants can be used as a benchmark where analytical formulas are available. |
| Greeks | Review finite-difference stability under alternate bump sizes and seeds. |
| Fixed/floating variant checks | Validate payoff formulas against independent examples. |
| Market data | Add provider-linked spot, realized volatility, and listed-option IV references where appropriate. |

## 10. Limitations

| Limitation | Implication |
|---|---|
| Constant volatility | Does not capture local volatility, stochastic volatility, skew, or jumps. |
| Constant rate and dividend yield | Does not yet use full curves or discrete dividends. |
| Calendar simplification | Generated monthly and quarterly schedules use day increments rather than full business-day calendars. |
| Single underlying | Does not cover basket Asians, commodity baskets, FX quanto Asians, or spread Asians. |
| No partial fixing history | Current page does not yet support already-realized fixings for in-season averaging windows. |
| Reporting | The page provides a structured run summary; customized institutional reports remain future work. |

## 11. Planned Enhancements

1. Add realized fixing history for partially averaged trades.
2. Add business-day calendar and holiday adjustment for observation schedules.
3. Add control-variate valuation using geometric Asian benchmarks.
4. Add convergence diagnostics by path count and observation count.
5. Add market reference data and volatility context consistent with the European and American pages.
6. Add basket, FX, and commodity Asian extensions.
7. Add scaled Greeks and downloadable report output.
8. Add automated regression tests for payoff variants, Greeks, and benchmark checks.

## 12. Implementation References

| File | Purpose |
|---|---|
| `derivapro/templates/asian_options.html` | Asian option workflow UI. |
| `derivapro/routes/exotic_options.py` | Route logic, pricing orchestration, and page analytics. |
| `derivapro/models/mdls_monte_carlo_v2.py` | Existing Monte Carlo Asian fixed-strike implementation retained for legacy workflows. |
| `derivapro/models/mdls_asian_options.py` | Existing QuantLib-based Asian implementation retained for legacy workflows. |
