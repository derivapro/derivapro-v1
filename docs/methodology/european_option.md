# European Option Methodology

## 1. Purpose

This document describes the methodology used by DerivaPro for the European vanilla option workflow. It is intended to support model transparency, user review, pricing reproducibility, and future model governance documentation.

The current implementation covers European call and put options on a single equity-style underlying. The workflow supports analytical Black-Scholes-Merton pricing, Greeks, first-pass diagnostics, market reference data, and Monte Carlo comparison for registered users.

This document is not a trading recommendation, investment advisory document, or independent model validation report.

## 2. Product Definition

A European option gives the holder the right, but not the obligation, to buy or sell an underlying asset at a fixed strike price on a fixed maturity date.

| Term | Description |
|---|---|
| Call option | Right to buy the underlying at strike `K` at maturity. |
| Put option | Right to sell the underlying at strike `K` at maturity. |
| Exercise style | European; exercise occurs only at maturity. |
| Settlement | Modeled as a standard cash-equivalent payoff at maturity. |

At maturity `T`, the payoff is:

```text
Call payoff = max(S_T - K, 0)
Put payoff  = max(K - S_T, 0)
```

where:

| Symbol | Meaning |
|---|---|
| `S_T` | Underlying price at maturity |
| `K` | Strike price |
| `T` | Time to maturity in years |

## 3. User Inputs

The European option page separates contract terms, market assumptions, model settings, and market reference data.

### Contract Terms

| Input | Description |
|---|---|
| Underlying ticker | Equity ticker used for labeling and optional market reference lookups. |
| Option type | Call or put. |
| Strike price | Contract strike. Must be positive. |
| Number of contracts | Position count used for position-value scaling. |
| Contract multiplier | Units per contract, commonly `100` for listed US equity options. |

### Market Assumptions

| Input | Description |
|---|---|
| Valuation date | Pricing date. |
| Maturity date | Option maturity date. Must be after valuation date. |
| Spot price | Current underlying price assumption used directly in pricing. |
| Risk-free rate | Continuously compounded risk-free rate assumption, entered in absolute format such as `0.04` for 4%. |
| Annualized volatility | Volatility assumption used by Black-Scholes-Merton and Monte Carlo. |
| Dividend yield | Continuously compounded dividend yield assumption. |
| Day count | `ACT/365` or `ACT/360` year-fraction convention. |

### Model Settings

| Setting | Description |
|---|---|
| Black-Scholes | Analytical benchmark available to all users. |
| Monte Carlo | Registered-user comparison workflow using simulated terminal prices. |
| Number of paths | Monte Carlo sample count, bounded in the application for responsiveness. |
| Number of time steps | Retained as a standard simulation setting; the current European Monte Carlo implementation uses the exact terminal distribution. |

## 4. Market Reference Data

The market reference section is designed to help users choose pricing assumptions. It does not automatically override pricing inputs unless the user explicitly applies a displayed value.

Current market reference data is fetched on demand from Yahoo Finance through the open-source `yfinance` package. Provider availability, data latency, corporate-action adjustments, option-chain availability, and licensing terms may affect runtime results.

### Price History Metrics

Price-history metrics require only:

```text
Ticker symbol
Lookback period
```

The workflow displays:

| Metric | Method |
|---|---|
| Latest close | Most recent close returned by the provider. |
| Period return | `latest close / first close - 1`. |
| Realized volatility | Standard deviation of daily close-to-close returns annualized by `sqrt(252)`. |
| Period range | Minimum low and maximum high over the returned history when available. |
| Average volume | Average daily volume over the returned history when available. |

### Listed-Option Implied Volatility

Option type, target strike, and target maturity are used only for the listed-option implied-volatility lookup.

The current procedure is:

1. Fetch the provider option-chain expirations for the ticker.
2. Select the listed expiration nearest to the target maturity.
3. Select calls or puts based on the selected option type.
4. Select the listed strike nearest to the target strike.
5. Display the provider-reported implied volatility for that contract.
6. Display nearby strike implied volatilities as a light smile/skew reference.

This is a point implied-volatility lookup from listed option-chain data. It is not a calibrated volatility surface.

### Optional Visuals

The market reference section supports visuals that are generated only when requested:

| Visual | Description |
|---|---|
| Historical price chart | Lightweight price trend based on sampled daily closes. |
| IV smile / skew | Nearby listed-option implied volatilities around the target strike. |
| 3D IV surface | Snapshot of listed option-chain implied volatility across nearby strikes and expirations. |

The 3D IV surface is a provider data visualization. It should not be treated as a production-quality volatility surface calibration without additional cleaning, filtering, interpolation, arbitrage checks, and governance review.

## 5. Analytical Pricing Model

The Black-Scholes-Merton implementation assumes:

| Assumption | Description |
|---|---|
| Underlying dynamics | Geometric Brownian motion. |
| Volatility | Constant over the option life. |
| Interest rate | Constant risk-free rate. |
| Dividend yield | Constant continuous dividend yield. |
| Exercise | European exercise only. |
| Markets | Frictionless markets with no transaction costs, taxes, or funding constraints. |

The risk-neutral process is:

```text
dS_t = (r - q) S_t dt + sigma S_t dW_t
```

where:

| Symbol | Meaning |
|---|---|
| `S_t` | Underlying price at time `t` |
| `r` | Risk-free rate |
| `q` | Continuous dividend yield |
| `sigma` | Annualized volatility |
| `W_t` | Brownian motion under the risk-neutral measure |

The Black-Scholes-Merton terms are:

```text
d1 = [ln(S / K) + (r - q + 0.5 * sigma^2) * T] / [sigma * sqrt(T)]
d2 = d1 - sigma * sqrt(T)
```

Call price:

```text
C = S * exp(-qT) * N(d1) - K * exp(-rT) * N(d2)
```

Put price:

```text
P = K * exp(-rT) * N(-d2) - S * exp(-qT) * N(-d1)
```

where `N(.)` is the standard normal cumulative distribution function.

## 6. Greeks

The European option workflow displays standard first-order and second-order Greeks per option unit.

| Greek | Interpretation |
|---|---|
| Delta | Sensitivity of option value to underlying price. |
| Gamma | Sensitivity of delta to underlying price. |
| Vega | Sensitivity of option value to volatility. |
| Theta | Sensitivity of option value to time decay. |
| Rho | Sensitivity of option value to risk-free rate. |

Current page convention:

| Output | Convention |
|---|---|
| Premium | Per option unit. |
| Position value | Premium multiplied by contract multiplier and number of contracts. |
| Vega | Displayed for a 1.00 absolute volatility change in the analytical formula output. |
| Rho | Displayed for a 1.00 absolute rate change in the analytical formula output. |

Future report views may add scaled Greeks, such as vega per 1 volatility point and rho per 1 basis point, to align with desk reporting conventions.

## 7. Monte Carlo Comparison

For registered users, the European option page supports a Monte Carlo comparison. The current implementation uses the exact terminal distribution of geometric Brownian motion:

```text
S_T = S_0 * exp((r - q - 0.5 * sigma^2) * T + sigma * sqrt(T) * Z)
```

where `Z` is a standard normal random variable.

The discounted payoff is averaged across paths:

```text
Monte Carlo price = exp(-rT) * average(payoff(S_T))
```

The implementation uses antithetic sampling and a deterministic seed for stable preview behavior. The workflow also displays standard error and a 95% confidence interval when Monte Carlo is selected.

For plain European options, the analytical Black-Scholes-Merton result remains the benchmark. The Monte Carlo mode is primarily useful for demonstrating the simulation workflow and preparing the platform for more complex payoff types.

## 8. Diagnostics and Analysis

The European option page currently includes:

| Diagnostic | Purpose |
|---|---|
| Premium decomposition | Intrinsic value, time value, breakeven, and tenor. |
| Moneyness | Simple in-the-money, at-the-money, or out-of-the-money classification. |
| Put-call parity check | Internal consistency check against analytical call and put prices. |
| Driver sensitivity | Price response to selected spot, volatility, and rate shocks. |
| Payoff strip | Gross and net payoff at representative maturity spot levels. |
| Run summary | Compact record of contract terms, market assumptions, outputs, and limitations. |

## 9. Validation Expectations

Before using this workflow for production or commercial decision-making, the following controls should be added or reviewed:

| Area | Recommended Control |
|---|---|
| Analytical benchmark | Unit tests against known Black-Scholes-Merton values. |
| Put-call parity | Automated regression test across representative scenarios. |
| Boundary behavior | Tests for near-zero volatility, short maturity, deep in/out-of-the-money cases, and invalid inputs. |
| Monte Carlo convergence | Convergence tests against the analytical price. |
| Market data | Provider error handling, timestamp display, caching, and data-quality checks. |
| Reporting | Explicit convention disclosure for Greek scaling, day count, rate compounding, and dividend treatment. |

## 10. Limitations

The current European option implementation has the following limitations:

| Limitation | Implication |
|---|---|
| Constant volatility | Does not capture stochastic volatility, local volatility, jumps, or volatility skew in pricing unless the user manually inputs an adjusted volatility. |
| Constant rate and dividend yield | Does not yet use full curves for discounting or carry. |
| Single underlying | Does not cover basket, quanto, spread, or cross-asset vanilla structures. |
| European exercise only | Does not represent early-exercise products. |
| Market data source | Free public-data feeds may be incomplete, delayed, revised, or unavailable. |
| IV surface | Current 3D visualization is a listed-option snapshot, not a fully calibrated volatility surface. |
| Reporting | The page provides a structured run summary; customized institutional reports remain a future enhancement. |

## 11. Planned Enhancements

Near-term enhancements for this workflow include:

1. Add scaled Greeks for desk-style reporting.
2. Add configurable shock templates for scenario analysis.
3. Promote market-data logic into a reusable provider layer with caching.
4. Add a calibrated volatility-surface service with interpolation and cleaning rules.
5. Add downloadable methodology-linked pricing reports.
6. Add automated regression tests for pricing, Greeks, parity, and Monte Carlo convergence.
7. Add richer user-specific saved instruments and assumption sets.

## 12. Implementation References

Primary implementation files:

| File | Purpose |
|---|---|
| `derivapro/templates/european_options.html` | European option workflow UI. |
| `derivapro/routes/vanilla_options.py` | Route logic, pricing orchestration, and page analytics. |
| `derivapro/services/market_reference.py` | Market reference data and optional visuals. |
| `derivapro/models/mdls_vanilla_options.py` | Existing vanilla option model utilities. |
| `derivapro/models/market_data.py` | Existing yfinance market-data helper. |

