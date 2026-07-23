**Description:**
A variance swap is an over-the-counter derivative that pays the difference between realized variance over an observation window and a pre-agreed variance strike, scaled by variance notional.

Typical settlement logic is:

* $Settlement = Variance\ Notional \times (Realized\ Variance - Strike\ Variance)$

where strike variance is derived from strike volatility (square of volatility, in consistent units). This workflow supports realized variance analysis, expected variance scenarios, and settlement valuation for long/short positions.

**Instructions:**
Use this page to evaluate variance swap exposures under current and simulated market assumptions.

1. Enter core contract and market inputs (ticker, start/end/as-of dates, strike vol, vega notional, risk-free rate, position).
2. Provide scenario parameters (for example, new strike volatility and model parameters used in expected-variance simulation).
3. Run the calculator to generate realized variance, expected variance, and settlement outputs.
4. Review current value and simulated settlement for stress-testing and risk reporting.
5. Use the results to compare sensitivity across alternative volatility assumptions.