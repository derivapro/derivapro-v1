# Futures Contract Methodology

## 1. Purpose

This document describes the methodology used by DerivaPro for futures pricing, settlement profit and loss, margin requirements, simulated daily mark-to-market, sensitivity analysis, scenario analysis, and risk-based P&L.

The workflow supports long and short futures positions under risk-adjusted and cost-of-carry assumptions. This note supports model transparency and user review; it is not investment advice or an independent model-validation report.

## 2. Product Definition

A futures contract is a standardized exchange-traded agreement to buy or sell an underlying asset at a future date. Clearing and daily settlement distinguish futures from bilateral forward contracts.

| Term | Description |
|---|---|
| Long futures | Position benefiting from an increase in the underlying relative to the contract price. |
| Short futures | Position benefiting from a decrease in the underlying relative to the contract price. |
| Initial margin | Opening collateral requirement calculated from the configured percentage. |
| Maintenance margin | Minimum ongoing balance before a margin call is indicated. |
| Daily mark-to-market | Daily recognition of gains and losses through the margin balance. |

## 3. Pricing Models

### Risk-Adjusted Model

The risk-adjusted model uses financing and dividend-yield assumptions:

$$F = S_0 \cdot e^{(r-q)T}$$

### Cost-of-Carry Model

The cost-of-carry model extends the calculation with convenience-yield and storage-cost assumptions:

$$F = S_0 \cdot e^{(r-q-c+s)T}$$

| Symbol | Meaning |
|---|---|
| `F` | Model-implied futures price |
| `S_0` | Current underlying level obtained by the model's market-data process |
| `r` | Risk-free rate |
| `q` | Continuous dividend yield |
| `c` | Convenience yield |
| `s` | Storage-cost input |
| `T` | Time from entry to settlement |

Inputs must use the decimal conventions shown on the product page, such as `0.05` for 5%.

## 4. Position Payoff and P&L

The terminal unscaled payoff is:

```text
Long futures payoff  = S_T - K
Short futures payoff = K - S_T
```

where `S_T` is the underlying level at settlement and `K` is the contractual settlement price. Position P&L also reflects contract count, multiplier, position direction, and contract fee.

In practice, futures gains and losses are generally realized through variation margin during the contract life rather than only at final settlement.

## 5. Margin and Daily Settlement

The application calculates initial and maintenance margin requirements from user-entered percentages. It also generates a daily price path and reports:

- daily P&L;
- margin balance;
- margin-call status; and
- corresponding dates.

The daily path is a model simulation for workflow analysis. It is not an exchange statement or a forecast of realized settlement prices.

## 6. User Inputs

| Input group | Inputs |
|---|---|
| Contract terms | Ticker, position, entry date, settlement date, settlement price, contract count, multiplier, contract fee |
| Market and carry | Risk-free rate, dividend yield, convenience yield, storage cost |
| Margin | Initial-margin percentage, maintenance-margin percentage |
| Model settings | Risk-adjusted or cost-of-carry model |

## 7. Analysis

The workflow retains three analysis modes:

| Analysis | Purpose |
|---|---|
| Sensitivity | Reprices across a selected range for risk-free rate, dividend yield, or convenience yield. |
| Scenario | Compares baseline and stressed price/P&L using selected rate, dividend-yield, and convenience-yield shocks. |
| Risk-based P&L | Bumps the spot input and reports actual P&L with intermediate prices. |

## 8. Key Assumptions and Limitations

- Rates and carry inputs are treated as constant over the contract life.
- The model does not construct full financing, dividend, or commodity forward curves.
- Futures and forwards can diverge when rates and underlying prices are correlated; that convexity effect is not explicitly calibrated here.
- The daily mark-to-market path is simulated and does not reproduce an exchange's official settlement history.
- Exchange-specific margin methodology, variation-margin timing, intraday calls, collateral remuneration, and default-fund mechanics are simplified.
- Market-data availability and quality can affect the underlying level used by the model.
- Transaction costs, liquidity, taxes, and basis risk are not fully modeled.

## 9. Validation Expectations

Production use should include exchange or vendor benchmark comparisons, long/short symmetry checks, margin reconciliation, deterministic simulation controls, date and boundary testing, and regression tests for all pricing and analysis modes.

## 10. Implementation References

| File | Purpose |
|---|---|
| `derivapro/templates/futures.html` | Futures pricing, margin, M2M, and analysis interface |
| `derivapro/routes/futures_forwards.py` | Request handling and pricing orchestration |
| `derivapro/models/mdls_futures_forwards.py` | Futures pricing, margin, M2M, and analysis models |
