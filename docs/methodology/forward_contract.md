# Forward Contract Methodology

## 1. Purpose

This document describes the methodology used by DerivaPro for forward-contract pricing, settlement profit and loss, sensitivity analysis, scenario analysis, and risk-based P&L.

The workflow supports long and short bilateral forward positions under risk-adjusted and cost-of-carry assumptions. This note supports model transparency and user review; it is not investment advice or an independent model-validation report.

## 2. Product Definition

A forward is a bilateral over-the-counter agreement to buy or sell an underlying asset at a contractual price on a future settlement date. Its terms can be customized, but it generally carries more direct counterparty exposure than a centrally cleared futures contract.

| Term | Description |
|---|---|
| Long forward | Agreement to buy the underlying at settlement. |
| Short forward | Agreement to sell the underlying at settlement. |
| Settlement price | Contractual delivery price, denoted by `K`. |
| Multiplier | Underlying units represented by one contract. |
| Contract count | Number of contracts in the position. |

## 3. Pricing Models

### Risk-Adjusted Model

The risk-adjusted model uses financing and dividend-yield assumptions:

$$F = S_0 \cdot e^{(r-q)T}$$

### Cost-of-Carry Model

The cost-of-carry model extends the calculation with convenience-yield and storage-cost assumptions:

$$F = S_0 \cdot e^{(r-q-c+s)T}$$

| Symbol | Meaning |
|---|---|
| `F` | Model-implied forward price |
| `S_0` | Current underlying level obtained by the model's market-data process |
| `r` | Risk-free rate |
| `q` | Continuous dividend yield |
| `c` | Convenience yield |
| `s` | Storage-cost input |
| `T` | Time from entry to settlement |

Inputs must use the decimal conventions shown on the product page, such as `0.05` for 5%.

## 4. Position Payoff and P&L

At settlement, the unscaled payoff is:

```text
Long forward payoff  = S_T - K
Short forward payoff = K - S_T
```

where `S_T` is the underlying level at settlement and `K` is the contractual settlement price. Position P&L also reflects contract count, multiplier, position direction, and contract fee as implemented by the pricing model.

## 5. User Inputs

| Input group | Inputs |
|---|---|
| Contract terms | Ticker, position, entry date, settlement date, settlement price, contract count, multiplier, contract fee |
| Carry assumptions | Risk-free rate, dividend yield, convenience yield, storage cost |
| Model settings | Risk-adjusted or cost-of-carry model |

## 6. Analysis

The workflow retains three analysis modes:

| Analysis | Purpose |
|---|---|
| Sensitivity | Reprices across a user-selected range for risk-free rate, dividend yield, or convenience yield. |
| Scenario | Compares baseline and stressed price/P&L using selected rate, dividend-yield, and convenience-yield shocks. |
| Risk-based P&L | Bumps the spot input and reports actual P&L with intermediate prices. |

## 7. Key Assumptions and Limitations

- Rates and carry inputs are treated as constant over the contract life.
- The model does not construct full financing, dividend, or commodity forward curves.
- Market-data availability and quality can affect the underlying level used by the model.
- Counterparty credit risk, collateral, funding valuation adjustments, liquidity, taxes, and legal terms are not explicitly modeled.
- Storage cost and convenience yield are simplified user assumptions rather than calibrated commodity-curve components.
- Sensitivity and scenario results depend on user-entered shock size and should not be interpreted as forecasts.

## 8. Validation Expectations

Production use should include benchmark comparisons, boundary and sign tests, long/short symmetry checks, market-data controls, date validation, and regression tests for all pricing and analysis modes.

## 9. Implementation References

| File | Purpose |
|---|---|
| `derivapro/templates/forwards.html` | Forward pricing and analysis interface |
| `derivapro/routes/futures_forwards.py` | Request handling and pricing orchestration |
| `derivapro/models/mdls_futures_forwards.py` | Forward pricing, P&L, and analysis models |
