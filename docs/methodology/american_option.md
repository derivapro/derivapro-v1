# American Option Methodology

## 1. Purpose

This document describes the methodology used by DerivaPro for the American vanilla option workflow. It supports pricing transparency, model review, implementation consistency, and future model governance documentation.

The current product-standard page covers American call and put options on a single equity-style underlying. The workflow includes market reference data, explicit pricing assumptions, recombining tree valuation, finite-difference Greeks, a European benchmark comparison, early-exercise diagnostics, first-pass sensitivities, payoff views, and a structured run summary.

This document is not a trading recommendation, investment advisory document, or independent model validation report.

## 2. Product Definition

An American option gives the holder the right, but not the obligation, to buy or sell the underlying asset at a fixed strike price at any time up to and including maturity.

| Term | Description |
|---|---|
| American call | Right to buy the underlying at strike `K` on or before maturity. |
| American put | Right to sell the underlying at strike `K` on or before maturity. |
| Exercise style | American; exercise may occur before maturity. |
| Settlement | Modeled as a standard cash-equivalent payoff when exercised. |

If exercised at time `t`, the payoff is:

```text
Call payoff = max(S_t - K, 0)
Put payoff  = max(K - S_t, 0)
```

where `S_t` is the underlying price at the exercise time and `K` is the strike.

## 3. User Inputs

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
| Maturity date | Option maturity. Must be after valuation date. |
| Spot price | Current underlying price assumption used directly in pricing. |
| Risk-free rate | Continuously compounded risk-free rate assumption. |
| Annualized volatility | Volatility assumption used in the tree. |
| Dividend yield | Continuous dividend yield used in risk-neutral carry. |
| Day count | `ACT/365` or `ACT/360` year-fraction convention. |

### Model Settings

| Setting | Description |
|---|---|
| Cox Ross Rubinstein Tree | Recombining binomial tree with CRR up/down factors. |
| Jarrow Rudd Tree | Recombining tree with equal branch probabilities and drift-adjusted up/down factors. |
| Tree steps | Number of time steps used in backward induction. |

## 4. Market Reference Data

The market reference section is designed to help users choose assumptions. It does not automatically override pricing inputs unless the user explicitly applies a displayed value.

Current market reference data is fetched on demand from Yahoo Finance through the open-source `yfinance` package. Provider availability, data latency, corporate-action adjustments, option-chain availability, and licensing terms may affect runtime results.

Price-history metrics require only ticker and lookback period. Listed-option implied-volatility metrics additionally use option type, target strike, and target maturity to select the nearest provider option-chain contract.

The optional 3D IV surface is a listed-option data snapshot by expiration and strike moneyness. It is not a calibrated production volatility surface.

## 5. Tree Valuation

American options generally do not have a universal closed-form solution because of the early-exercise decision. DerivaPro values the option using backward induction through a recombining tree.

At each node, the model compares:

```text
Continuation value = discounted expected next-step value
Exercise value     = immediate payoff at the node
Node value         = max(Continuation value, Exercise value)
```

This max operation is the key difference from a European tree valuation.

For the CRR model:

```text
u = exp(sigma * sqrt(dt))
d = 1 / u
p = [exp((r - q) * dt) - d] / (u - d)
discount = exp(-r * dt)
```

where:

| Symbol | Meaning |
|---|---|
| `u` | Up factor |
| `d` | Down factor |
| `p` | Risk-neutral up probability |
| `r` | Risk-free rate |
| `q` | Dividend yield |
| `sigma` | Annualized volatility |
| `dt` | Time-step size |

For the Jarrow Rudd model, the tree uses equal branch probability and drift-adjusted up/down factors.

## 6. Greeks

The page displays finite-difference Greeks around the selected American tree:

| Greek | Interpretation |
|---|---|
| Delta | Sensitivity of option value to underlying price. |
| Gamma | Sensitivity of delta to underlying price. |
| Vega | Sensitivity of option value to volatility. |
| Theta | Sensitivity of option value to passage of time. |
| Rho | Sensitivity of option value to risk-free rate. |

Future report views may add desk-style scaled Greeks, such as vega per 1 volatility point and rho per 1 basis point.

## 7. European Benchmark and Early-Exercise Premium

The page calculates a European Black-Scholes-Merton benchmark using the same spot, strike, maturity, rate, volatility, dividend yield, and option type.

The early-exercise premium is:

```text
Early-exercise premium = American premium - European benchmark premium
```

This metric helps users understand the incremental value of American exercise rights under the selected assumptions.

## 8. Diagnostics and Analysis

The American option page currently includes:

| Diagnostic | Purpose |
|---|---|
| Premium decomposition | Intrinsic value, time value, European benchmark, and breakeven. |
| Early-exercise premium | Incremental value over the European benchmark. |
| Exercise node count | Number of backward-induction nodes where immediate exercise exceeds continuation value. |
| Moneyness | Simple in-the-money, at-the-money, or out-of-the-money classification. |
| Driver sensitivity | Price response to selected spot, volatility, and rate shocks. |
| Payoff strip | Gross and net payoff at representative maturity spot levels. |
| Run summary | Compact record of contract terms, market assumptions, model settings, outputs, and limitations. |

## 9. Validation Expectations

Before using this workflow for production or commercial decision-making, the following controls should be added or reviewed:

| Area | Recommended Control |
|---|---|
| European consistency | American value should be greater than or equal to the European benchmark within numerical tolerance. |
| Convergence | Prices should stabilize as tree steps increase. |
| Boundary cases | Test deep in/out-of-the-money, near-expiry, low-volatility, and high-dividend scenarios. |
| Early-exercise logic | Validate exercise boundary behavior against known references. |
| Greeks | Compare finite-difference stability under different bump sizes. |
| Market data | Add provider error handling, timestamp display, caching, and data-quality controls. |

## 10. Limitations

| Limitation | Implication |
|---|---|
| Constant volatility | Does not capture stochastic volatility, jumps, or local volatility. |
| Constant rate and dividend yield | Does not yet use full curves or discrete dividend schedules in the product-standard page. |
| Single underlying | Does not cover basket, quanto, or spread structures. |
| Tree approximation | Accuracy depends on step count and numerical stability. |
| IV surface | Current 3D visualization is a listed-option snapshot, not a calibrated volatility surface. |
| Reporting | The page provides a structured run summary; customized institutional reports remain future work. |

## 11. Planned Enhancements

1. Add tree convergence diagnostics directly to the product-standard page.
2. Add early-exercise boundary visualization.
3. Restore and standardize LSMC after regression testing.
4. Add discrete dividend support in the product-standard pricing workflow.
5. Add scaled Greeks and downloadable report output.
6. Add automated regression tests for pricing, Greeks, convergence, and early-exercise premium.

## 12. Implementation References

| File | Purpose |
|---|---|
| `derivapro/templates/american_options.html` | American option workflow UI. |
| `derivapro/routes/vanilla_options.py` | Route logic, pricing orchestration, and page analytics. |
| `derivapro/services/market_reference.py` | Market reference data and optional visuals. |
| `derivapro/models/mdls_lattice_trees.py` | Existing lattice model utilities retained for legacy analysis workflows. |
| `derivapro/models/mdls_binomial_tree.py` | Existing binomial CRR engine retained for legacy dividend workflows. |

